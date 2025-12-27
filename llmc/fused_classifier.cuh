/*
 * ============================================================================
 * Fused Classifier - 融合分类器
 * ============================================================================
 * 
 * 功能:
 * - 计算 Cross Entropy Loss (交叉熵损失)
 * - 从不完整存储归一化后的 logits，只在目标标签位置计算
 * - (融合优化) 同时执行反向传播，因为数据已在缓存中
 * 
 * 核心优化思想:
 * - 传统做法: softmax -> 存储 probs -> 读取 probs -> 计算 loss/grad (3次显存访问)
 * - 融合做法: 一个 kernel 完成 softmax + loss + grad，复用 L2 缓存 (1次显存读+1次写)
 * 
 * 数值稳定性 (防止 exp 溢出):
 * - softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)
 */
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

/**
 * SoftmaxParams - Softmax 归一化参数
 * 
 * 存储 block-wide reduction 后的结果，用于计算: softmax(x_i) = exp(x_i - Offset) * Scale
 * 
 * @member Scale:  1.0 / Σexp(x_j - Offset)，归一化因子的倒数
 * @member Offset: max(x_j)，数值稳定性偏移量
 */
struct SoftmaxParams {
    float Scale;   // = 1 / Σexp(x_i - max)
    float Offset;  // = max(x_i)
};

/**
 * prepare_softmax_blockwide3 - Block 级别 Softmax 预处理
 * 
 * 使用 "在线算法" (Online Algorithm) 一次遍历同时计算 max 和 sum:
 *   当遇到新 max 时: new_sum = old_sum * exp(old_max - new_max) + exp(v - new_max)
 * 
 * @param idx: 行索引 (对应 batch*seq 位置)，int64_t 防止 idx*P 溢出
 * @param inp: 输入 logits [B*T, P]
 * @param V:   词汇表实际大小 (Vocabulary)
 * @param P:   填充后大小 (P >= V，通常是 128 的倍数以优化访存)
 * @return     SoftmaxParams{Scale, Offset}
 * 
 * 访存优化:
 * - 从后向前遍历，与后续梯度计算的访问顺序匹配，提高缓存命中
 * - 使用 x128 向量化加载 (BF16 下一次读 8 个元素)
 */
__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    // 获取当前行指针: inp[idx, 0:P]
    const floatX* x = inp + idx * P;
    
    // 每个线程维护局部 max 和 sum
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    
    // 计算起始索引: 从末尾开始，线程交错分布
    // ceil(V / x128::size) + threadIdx.x - blockDim.x
    // 例: V=50257, x128::size=8, blockDim.x=1024
    //     -> 线程0从第 6282-1024=5258 块开始
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // ========== 阶段1: 处理尾部不对齐元素 (需要边界检查) ==========
    // 当 (i+1)*8 > V 时，该块可能包含越界元素
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // 超出实际词汇表范围，停止
            }
            float v = (float)x[i*x128::size+k];
            
            // 在线 Softmax 算法核心:
            // 1. 更新 max
            // 2. 调整之前的 sum: old_sum * exp(old_max - new_max)
            // 3. 累加当前项: + exp(v - new_max)
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;  // 移动到下一个块 (stride = blockDim.x)
    }

    // ========== 阶段2: 主循环 (无边界检查，性能最优) ==========
    for (; i >= 0; i -= blockDim.x) {
        // 向量化加载 128 位，数据会留在 L2 缓存供后续使用
        x128 packed_x = load128(x + i * x128::size);
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // ========== 阶段3: Block 级归约 ==========
    // 步骤1: 所有线程的 max 归约得到全局 max
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
    
    // 步骤2: 将每个线程的 sum 调整到统一的 max 基准
    //        thread_sum_adjusted = thread_sum * exp(thread_max - block_max)
    thread_sumval *= expf(thread_maxval - block_maxval);
    
    // 步骤3: 所有线程的 sum 归约得到全局 sum
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // 返回: Scale = 1/sum, Offset = max
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

/**
 * fused_classifier_kernel5 - 融合的前向 Loss + 反向梯度计算
 * 
 * 功能 (融合在一个 kernel 中):
 * 1. 计算 softmax 参数 (通过 prepare_softmax_blockwide3)
 * 2. 计算交叉熵损失: loss = -log(softmax[target])
 * 3. 计算 dlogits 梯度: dlogits[i] = (softmax[i] - one_hot[i]) * dloss
 * 4. (可选) 输出 softmax 概率用于推理
 * 
 * 数学推导 (Softmax + CrossEntropy 联合梯度):
 *   L = -log(p_target), 其中 p_i = softmax(logit_i)
 *   ∂L/∂logit_i = p_i - 1{i == target}
 *   即: 目标位置梯度 = prob - 1, 其他位置梯度 = prob
 * 
 * 模板参数:
 * @tparam WriteDLogits: true=写回梯度 (训练), false=只计算loss (推理)
 * @tparam WriteProbs:   true=输出概率 (调试/推理)
 * 
 * Kernel 参数:
 * @param logits:  [B*T, P] 输入 logits，会被原地覆盖为梯度
 * @param losses:  [B*T] 输出损失 (累加到现有值)
 * @param probs:   [B*T, P] 输出概率 (仅 WriteProbs=true 时使用)
 * @param dloss:   损失缩放因子 (通常 = 1/(B*T) 用于取平均)
 * @param targets: [B*T] 每个位置的正确 token ID
 * @param B, T:    批次大小, 序列长度
 * @param V:       词汇表实际大小
 * @param P:       词汇表填充大小 (P >= V)
 * 
 * Grid/Block: (B*T) blocks × 1024 threads
 */
template <bool WriteDLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(floatX* logits, float* losses, floatX* probs,
                                const float dloss, const int* targets,
                                int B, int T, int V, int P, std::bool_constant<WriteDLogits>) {
    // 使用 int64_t 确保 idx * P 不会溢出 (P 可能很大，如 50304)
    // 反向遍历: gridDim.x-1, gridDim.x-2, ..., 0
    // 原因: matmul 输出最后的行最热，反向遍历提高缓存命中率
    int64_t idx = gridDim.x - (blockIdx.x+1);
    
    // 获取当前位置的正确 token ID
    int ix = targets[idx];

    // ========== 阶段1: 计算 Softmax 参数 ==========
    // 读取整行 logits [idx, 0:V]，计算 max 和 sum
    // 数据会留在 L2 缓存，供下面的梯度计算复用
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // ========== 阶段2: 计算 Loss (仅线程0执行) ==========
    if(threadIdx.x == 0) {
        // 计算目标 token 的概率: p = exp(logit[ix] - max) * (1/sum)
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        // 交叉熵: -log(p)，累加到 losses[idx]
        losses[idx] -= logf(prob);
    }

    // ========== 关键同步点 ==========
    // 必须等待 loss 计算完成后才能修改 logits!
    // 竞态条件说明:
    //   - 上面计算 loss 需要读取 logits[idx*P + ix]
    //   - 下面会把 logits 覆盖为梯度 (范围 [-1, 1])
    //   - 如果不同步，可能读到已被修改的梯度值
    //   - 由于 sp.Offset 可能 < -90，会计算 exp(90+) = inf，导致 loss 爆炸
    __syncthreads();

    // ========== 阶段3: 计算并写回梯度 ==========
    // 梯度公式: dlogits[i] = (prob[i] - indicator[i]) * dloss
    //   - indicator[i] = 1 当 i == target, 否则 = 0
    
    const floatX* logits_vec = logits + idx * P;
    
    // 主循环: 处理 V/x128::size 个完整向量块
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // 第二次读取 logits (第一次在 prepare_softmax 中)
        // 由于刚读过，很可能还在 L2 缓存中
        x128 packed_logits_vec = load128(logits_vec + i * x128::size);
        x128 packed_probs;
        
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            
            // 计算 softmax 概率
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            
            // 计算梯度: 目标位置 = (prob-1)*dloss, 其他位置 = prob*dloss
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        
        if (WriteDLogits){
            // store128cs: cache streaming hint
            // 提示 GPU 这些数据不会很快再次读取，可以较快驱逐出缓存
            // 让缓存优先保留 prepare_softmax 到这里之间需要复用的 logits
            store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // 尾部处理: V 不是 x128::size 倍数时的剩余元素
    // 例: V=50257, x128::size=8 -> 需要处理最后 50257 % 8 = 1 个元素
    int unaligned_start = V & ~(x128::size - 1); // 向下取整: 50257 -> 50256
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        
        if (WriteDLogits){
            __stcs(logits + idx * P + i, (floatX)dlogit);  // cache streaming store
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel Launcher - 主机端调用接口
// ----------------------------------------------------------------------------

/**
 * fused_classifier - 融合分类器入口
 * 
 * 将 logits 原地替换为梯度，同时累加 loss
 * 
 * @tparam Type:        数据类型 (floatX)
 * @tparam WriteDLogits: 是否写回梯度
 * 
 * @param logits:  [B, T, P] logits，会被原地覆盖为梯度
 * @param losses:  [B*T] 损失输出
 * @param dloss:   梯度缩放因子
 * @param targets: [B*T] 目标标签
 * @param B, T:    批次大小, 序列长度
 * @param V, P:    词汇表大小, 填充大小
 * @param stream:  CUDA 流
 */
template <typename Type, bool WriteDLogits>
void fused_classifier(Type* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 1024;    // 每 block 1024 线程
    const int N = B * T;            // 总行数
    const int grid_size = N;        // 每行一个 block
    fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (floatX*)NULL, dloss, targets, B, T, V, P, write_dlogits);
    cudaCheck(cudaGetLastError());
}
