/*
 * ============================================================================
 * Attention 注意力机制实现
 * ============================================================================
 * 
 * 本文件实现了 Transformer 模型中的多头自注意力(Multi-Head Self-Attention)机制。
 * 当 cuDNN 的 Flash Attention 不可用时，作为备选实现使用。
 * 
 * 注意力计算公式:
 *   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 * 
 * 主要包含以下 CUDA kernel:
 *   1. permute_kernel        - 将 QKV 张量从 (B,T,3,NH,HS) 重排为 3个 (B,NH,T,HS)
 *   2. permute_kernel_backward - permute_kernel 的反向传播
 *   3. unpermute_kernel      - 将注意力输出从 (B,NH,T,HS) 重排为 (B,T,NH,HS)
 *   4. unpermute_kernel_backward - unpermute_kernel 的反向传播
 *   5. softmax_forward_kernel5 - 带 scale 的因果 softmax 前向（在线算法）
 *   6. softmax_autoregressive_backward_inplace_kernel - softmax 反向传播(原地操作)
 * 
 * 以及两个主要的 launcher 函数:
 *   - attention_forward()  - 注意力前向传播
 *   - attention_backward() - 注意力反向传播
 */
#include <assert.h>
// llmc 内部头文件
#include "cuda_common.h"   // CUDA 通用定义，包括 floatX 类型定义
#include "cuda_utils.cuh"  // CUDA 工具函数，包括 warpReduce 等
#include "cublas_common.h" // cuBLAS/cuBLASLt 矩阵乘法封装

// ============================================================================
// CUDA Kernels - 核心计算内核
// ============================================================================

/**
 * permute_kernel - QKV 张量重排内核（前向传播）
 * 
 * 功能描述:
 *   将融合的 QKV 张量拆分并重新排列，使其适合后续的批量矩阵乘法计算。
 *   输入张量的头维度(NH)在序列维度(N)之后，需要交换这两个维度。
 * 
 * 张量变换:
 *   输入: inp  - shape (B, N, 3, NH, d) - 融合的 QKV 张量
 *   输出: q    - shape (B, NH, N, d)    - Query 张量
 *         k    - shape (B, NH, N, d)    - Key 张量  
 *         v    - shape (B, NH, N, d)    - Value 张量
 * 
 * 索引映射:
 *   Q[b][nh][n][d] = inp[b][n][0][nh][d]  (QKV中的第0个)
 *   K[b][nh][n][d] = inp[b][n][1][nh][d]  (QKV中的第1个)
 *   V[b][nh][n][d] = inp[b][n][2][nh][d]  (QKV中的第2个)
 * 
 * @param q    [out] Query 输出张量，shape (B, NH, N, d)
 * @param k    [out] Key 输出张量，shape (B, NH, N, d)
 * @param v    [out] Value 输出张量，shape (B, NH, N, d)
 * @param inp  [in]  融合的 QKV 输入张量，shape (B, N, 3, NH, d)
 * @param B    [in]  批次大小 (Batch size)
 * @param N    [in]  序列长度 (Sequence length, 即 T)
 * @param NH   [in]  注意力头数量 (Number of heads)
 * @param d    [in]  每个头的维度 (Head size, 即 HS = C/NH)
 * 
 * 线程配置:
 *   总线程数 = B * NH * N * d
 *   每个线程处理输出张量中的一个元素
 * 
 * 内存访问优化:
 *   使用 __ldcs (load cached streaming) 提示编译器这是流式访问模式
 */
__global__ void permute_kernel(floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int d) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查：确保不越界
    if (idx >= B * NH * N * d) { return; }

    // 将一维索引 idx 分解为四维坐标 (b, nh_, n, d_)
    // idx 的布局: [b][nh_][n][d_] 对应输出张量的布局
    int b = idx / (NH * N * d);           // 批次索引
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);             // 注意力头索引
    rest = rest % (N * d);
    int n = rest / d;                      // 序列位置索引
    int d_ = rest % d;                     // 头内维度索引
    
    // 计算输入张量中对应 Q 的索引
    // inp 布局: [b][n][qkv][nh_][d_]，其中 qkv=0 对应 Q
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    
    // 使用流式加载从输入张量读取 Q, K, V
    // K 在 Q 之后偏移 NH*d，V 在 Q 之后偏移 2*NH*d
    q[idx] = __ldcs(&inp[inp_idx]);
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}

/**
 * permute_kernel_backward - QKV 张量重排的反向传播内核
 * 
 * 功能描述:
 *   permute_kernel 的反向传播。将分离的 dQ, dK, dV 梯度重新合并回
 *   融合的 dinp 梯度张量。这是 permute_kernel 的逆操作。
 * 
 * 梯度流向:
 *   输入梯度: dq, dk, dv - 各自 shape (B, NH, N, d)
 *   输出梯度: dinp       - shape (B, N, 3, NH, d)
 * 
 * @param dinp [out] 输入梯度，shape (B, N, 3, NH, d)，存放合并后的 dQ/dK/dV
 * @param dq   [in]  Query 的梯度，shape (B, NH, N, d)
 * @param dk   [in]  Key 的梯度，shape (B, NH, N, d)
 * @param dv   [in]  Value 的梯度，shape (B, NH, N, d)
 * @param B    [in]  批次大小
 * @param N    [in]  序列长度
 * @param NH   [in]  注意力头数量
 * @param d    [in]  每个头的维度
 * 
 * 线程配置:
 *   总线程数 = B * NH * N * d
 *   每个线程将一个梯度元素写回对应位置
 */
__global__ void permute_kernel_backward(floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    // 分解索引到四维坐标（与 permute_kernel 相同）
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    // 计算输出（dinp）中的目标索引
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    
    // 将 dq, dk, dv 写回到 dinp 的对应位置
    dinp[inp_idx] = dq[idx];                    // dQ 写入 qkv=0 的位置
    dinp[inp_idx + NH * d] = dk[idx];           // dK 写入 qkv=1 的位置
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];     // dV 写入 qkv=2 的位置
}

/**
 * unpermute_kernel - 注意力输出重排内核（前向传播）
 * 
 * 功能描述:
 *   将注意力计算的输出从 (B, NH, N, d) 重排为 (B, N, NH, d)。
 *   这是将多头注意力的结果"重新组装"回原始序列维度优先的布局，
 *   以便后续的线性投影层处理。
 * 
 * 张量变换:
 *   输入: inp - shape (B, NH, N, d) - 各头的注意力输出（头维度在前）
 *   输出: out - shape (B, N, NH, d) - 重排后的输出（序列维度在前）
 * 
 * 索引映射:
 *   out[b][n][nh][d] = inp[b][nh][n][d]
 * 
 * @param inp [in]  输入张量，shape (B, NH, N, d)，注意力计算结果
 * @param out [out] 输出张量，shape (B, N, NH, d)，重排后用于后续线性层
 * @param B   [in]  批次大小
 * @param N   [in]  序列长度
 * @param NH  [in]  注意力头数量
 * @param d   [in]  每个头的维度
 * 
 * 线程配置:
 *   总线程数 = B * NH * N * d
 *   每个线程读取 inp 的一个元素，写入 out 的对应位置
 */
__global__ void unpermute_kernel(floatX* inp, floatX *out, int B, int N, int NH, int d) {
    // 计算全局线程索引
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= B * NH * N * d) { return; }

    // 分解索引：idx 对应 inp[b][nh_][n][d_] 的布局
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    
    // 计算输出索引：out[b][n][nh_][d_] 的布局
    // 注意：这里维度顺序变了，N 和 NH 交换位置
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    
    // 流式加载并写入
    out[other_idx] = __ldcs(&inp[idx]);
}

/**
 * unpermute_kernel_backward - 注意力输出重排的反向传播内核
 * 
 * 功能描述:
 *   unpermute_kernel 的反向传播。将梯度从 (B, N, NH, d) 布局
 *   转换回 (B, NH, N, d) 布局，以便继续向后传播到注意力计算。
 * 
 * 梯度流向:
 *   输入梯度: dout - shape (B, N, NH, d) - 来自后续层的梯度
 *   输出梯度: dinp - shape (B, NH, N, d) - 传递给注意力计算的梯度
 * 
 * @param dinp [out] 输出梯度，shape (B, NH, N, d)
 * @param dout [in]  输入梯度，shape (B, N, NH, d)
 * @param B    [in]  批次大小
 * @param N    [in]  序列长度
 * @param NH   [in]  注意力头数量
 * @param d    [in]  每个头的维度
 * 
 * 线程配置:
 *   总线程数 = B * NH * N * d
 *   每个线程处理一个梯度元素的重排
 */
__global__ void unpermute_kernel_backward(floatX* dinp, const floatX *dout, int B, int N, int NH, int d) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    // 分解索引到四维坐标
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    
    // 计算 dout 中对应元素的索引（布局为 [b][n][nh_][d_]）
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    
    // 从 dout 读取梯度并写入 dinp
    dinp[idx] = (floatX)dout[other_idx];
}

/**
 * softmax_forward_kernel5 - 因果注意力的 Softmax 前向内核（带 scale）
 * 
 * 功能描述:
 *   实现带温度缩放的因果（自回归）softmax。使用"在线 softmax"算法，
 *   在单次遍历中同时计算最大值和指数和，避免多次读取内存。
 *   
 *   由于是自回归（因果）注意力，只计算下三角部分：
 *   位置 t 只能看到位置 0..t 的信息。
 * 
 * 算法原理（在线 Softmax）:
 *   传统 softmax 需要两次遍历：第一次找 max，第二次计算 exp 和归一化。
 *   在线算法在一次遍历中完成：
 *   1. 维护当前最大值 maxval 和缩放后的部分和 sumval
 *   2. 遇到新的最大值时，调整 sumval: sumval *= exp(old_max - new_max)
 *   3. 最后 warp 内规约得到全局 max 和 sum
 * 
 * 计算公式:
 *   out[i][j] = exp(scale * (inp[i][j] - max)) / sum(exp(scale * (inp[i][:j+1] - max)))
 *   其中 scale = inv_temperature = 1/sqrt(head_size)
 * 
 * @param out             [out] 输出张量，shape (N, T, T)，归一化后的注意力权重
 * @param inv_temperature [in]  温度系数的倒数，即 1/sqrt(d_k)，用于缩放
 * @param inp             [in]  输入张量，shape (N, T, T)，Q@K^T 的结果（preatt）
 * @param N               [in]  批次数 × 头数，即 B * NH
 * @param T               [in]  序列长度，必须是 4 的倍数
 * 
 * 线程配置:
 *   - 每个 warp（32线程）处理一行
 *   - 总共需要 N * T 个 warp
 *   - grid 和 block 配置需保证覆盖所有行
 * 
 * 缓存优化:
 *   反向遍历 block（从高地址到低地址），使得后续的矩阵乘法
 *   可以利用缓存中残留的靠近左上角的数据。
 */
__global__ void softmax_forward_kernel5(floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    // 要求 T 是 4 的倍数，以便使用 4 元素向量化加载
    assert(T % 4  == 0);
    
    // 计算线程在 warp 内的位置和 warp ID
    int lane_id = threadIdx.x % WARP_SIZE;   // warp 内的 lane 索引 (0-31)
    int warp_id = threadIdx.x / WARP_SIZE;   // block 内的 warp 索引
    int num_warps = blockDim.x / WARP_SIZE;  // 每个 block 的 warp 数量

    // 【缓存优化】反向遍历：从最后一行开始处理
    // 这样 softmax 完成后，缓存中保留的是靠近矩阵左上角的数据，
    // 有利于后续 att @ V 矩阵乘法的缓存命中率
    int idx = (gridDim.x - blockIdx.x - 1) * num_warps + warp_id;
    if(idx >= N * T) {
        return;
    }
    
    // 当前行在序列中的位置（决定因果 mask 的边界）
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;  // 可以完整处理的 4 元素组数

    // 定位到当前行的起始位置
    const floatX* x = inp + idx * T;

    // 使用足够大的值作为初始最小值，避免使用 float.h
    const float flt_max = 340282346638528859811704183484516925440.0f;
    float maxval = -flt_max;
    float sumval = 0.0f;

    // 【主循环】每次处理 4 个元素，使用在线 softmax 算法
    // 假设内存对齐，提示编译器进行优化
    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        // 加载 4 个连续元素到寄存器
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        
        // 更新最大值
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, regarray[k]);
        }
        
        // 调整累积和：因为 max 变了，需要重新缩放之前的 sum
        // sumval_new = sumval_old * exp(scale * (old_max - new_max))
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        
        // 累加当前 4 个元素的 exp 值
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (regarray[k] - maxval));
        }
    }

    // 【尾部处理】处理不足 4 个元素的剩余部分
    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    // 【Warp 规约】在 warp 内找全局最大值
    float global_maxval = warpReduceMax(maxval);
    // 根据全局最大值调整每个线程的部分和
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    // 【Warp 规约】累加所有线程的部分和得到总和
    float sum = warpReduceSum(sumval);
    float norm = 1.f / sum;  // 归一化因子

    // 【输出】计算并写入归一化后的 softmax 值
    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        // 重新计算 exp 值比从内存读取更快（计算换内存）
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));
    }
}

/**
 * softmax_autoregressive_backward_inplace_kernel - Softmax 反向传播内核（原地操作）
 * 
 * 功能描述:
 *   计算因果 softmax 的反向传播梯度。这是一个原地操作：
 *   输入的 datt (d_attention) 会被覆盖为 dpreatt (d_pre_attention)。
 * 
 * 数学推导:
 *   设 softmax 输出为 p = softmax(x)，损失对 p 的梯度为 dp
 *   则损失对 x 的梯度为:
 *     dx[i] = p[i] * (dp[i] - sum(p * dp))
 *   
 *   带 scale 的版本:
 *     dpreatt[i] = scale * att[i] * (datt[i] - sum(att * datt))
 * 
 * 因果掩码:
 *   只有位置 j <= t 的元素参与计算，其余设为 0
 * 
 * @param datt  [in/out] 输入时为 d_attention，输出时为 d_pre_attention
 *                       shape: (B*NH, T, T)，会被原地修改
 * @param att   [in]     前向传播时保存的 attention 权重，shape: (B*NH, T, T)
 * @param B     [in]     批次大小
 * @param T     [in]     序列长度
 * @param C     [in]     通道数/隐藏维度（此处用于计算 NH = C/HS，但实际未使用）
 * @param scale [in]     缩放因子，即 1/sqrt(head_size)
 * 
 * 线程配置:
 *   - gridDim.x = ceil(T/4)，每个 block 处理 4 行
 *   - gridDim.y = B * NH，每个 y 维度处理一个 (batch, head) 对
 *   - blockDim.x = 256
 * 
 * 优化策略:
 *   1. 每个 block 处理 4 行，提高 SM 占用率
 *   2. 反向遍历行（从大 t 到小 t），让耗时最长的行先开始
 *   3. 使用流式内存访问，利用前一行计算留下的缓存
 */
__global__ void softmax_autoregressive_backward_inplace_kernel(floatX* datt, const floatX* att,
                                                               int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;  // 每个 block 的线程数
    constexpr int T_per_block = 4;        // 每个 block 处理的行数

    // 【负载均衡】反向遍历：让需要处理更多元素的行先开始
    // 位置 t 需要处理 t+1 个元素，所以从大 t 开始可以更好地隐藏延迟
    int t0 = T - 1 - T_per_block*blockIdx.x;
    int idx = blockIdx.y;  // 当前处理的 (batch, head) 索引

    // 偏移到当前 (batch, head) 对应的 T×T 矩阵
    att += idx * T * T;
    datt += idx * T * T;

    // 每个 block 连续处理 4 行
    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;  // 当前处理的行号
        if(t < 0) return;  // 超出范围则提前返回
        
        // 定位到当前行
        const floatX* att_bth = att + t * T;      // attention 权重的第 t 行
        const floatX* datt_bth = datt + t * T;    // 输入梯度的第 t 行
        floatX* dpreatt_bth = datt + t * T;       // 输出梯度的第 t 行（原地覆盖）

        // 【第一步】计算 sum(att * datt)，只考虑因果部分 (j <= t)
        float local_sum = 0;
        for (int t2 = threadIdx.x; t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        // block 内规约求和
        local_sum = blockReduce<warpReduceSum>(local_sum);

        // 【第二步】计算 dpreatt = scale * att * (datt - sum)
        for (int t3 = threadIdx.x; t3 < T; t3 += BlockSize) {
            // 使用流式加载/存储，不污染缓存
            // 前一行的数据可能还在缓存中，我们可以利用它
            if(t3 <= t) {
                // 因果部分：正常计算梯度
                float acc = (float) __ldcs(att_bth + t3) * ((float) __ldcs(datt_bth + t3) - local_sum);
                __stcs(dpreatt_bth + t3, (floatX) (scale * acc));
            } else {
                // 非因果部分：显式设为 0（这些位置在前向时被 mask 掉了）
                __stcs(dpreatt_bth + t3, (floatX)0.f);
            }
        }
    }
}

// ============================================================================
// Kernel Launchers - 内核启动器（主要 API）
// ============================================================================

/**
 * attention_forward - 多头自注意力前向传播
 * 
 * 功能描述:
 *   执行完整的多头自注意力计算：
 *   1. 将融合的 QKV 张量拆分并重排
 *   2. 计算注意力分数：preatt = K^T @ Q
 *   3. 应用因果 mask 和 softmax：att = softmax(preatt / sqrt(d))
 *   4. 计算注意力输出：vaccum = V @ att
 *   5. 重排输出维度
 * 
 * 计算流程图:
 *   inp (B,T,3C)
 *     │ permute_kernel
 *     ▼
 *   Q, K, V (各 B,NH,T,HS)
 *     │ matmul: K^T @ Q
 *     ▼
 *   preatt (B,NH,T,T)  ──► 注意：复用 inp 作为 scratch
 *     │ softmax_forward_kernel5 (带 scale 和因果 mask)
 *     ▼
 *   att (B,NH,T,T)
 *     │ matmul: V @ att
 *     ▼
 *   vaccum (B,NH,T,HS) ──► 注意：复用 inp 作为 scratch
 *     │ unpermute_kernel
 *     ▼
 *   out (B,T,C)
 * 
 * @param out    [out] 注意力输出，shape (B, T, C)
 * @param qkvr   [out] 重排后的 Q,K,V 缓冲区，shape (3, B, NH, T, HS)
 *                     会在反向传播时使用，需要保存
 * @param att    [out] 注意力权重矩阵，shape (B, NH, T, T)
 *                     会在反向传播时使用，需要保存
 * @param inp    [in/scratch] 输入的融合 QKV，shape (B, T, 3*C)
 *                            注意：会被覆盖用作 scratch buffer！
 * @param B      [in] 批次大小 (Batch size)
 * @param T      [in] 序列长度 (Sequence length)
 * @param C      [in] 隐藏维度 (Hidden dimension)，等于 NH * HS
 * @param NH     [in] 注意力头数量 (Number of heads)
 * @param stream [in] CUDA 流
 * 
 * 内存复用:
 *   inp 在前向传播后不再需要，因此被复用为 scratch buffer，
 *   用于存储 preatt 和 vaccum 中间结果。
 */
void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();  // NVTX 性能分析标记
    const int block_size = 256;  // CUDA block 大小

    // 张量维度说明:
    // inp: (B, T, 3C) - 融合的 QKV，其中 3C = 3 * NH * HS
    // preatt, att: (B, NH, T, T) - 注意力分数矩阵
    // output: (B, T, C) - 注意力输出
    const int HS = C / NH;  // 每个头的维度 (Head Size)

    // ==================== Step 1: QKV 重排 ====================
    // 将 inp 从 (B, T, 3, NH, HS) 重排为 3 个 (B, NH, T, HS) 张量
    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;  // Query: 偏移 0
    k = qkvr + 1 * B * T * C;  // Key:   偏移 B*T*C
    v = qkvr + 2 * B * T * C;  // Value: 偏移 2*B*T*C
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH, HS);

    // ==================== Step 2: 计算注意力分数 ====================
    // preatt = K^T @ Q，结果 shape: (B*NH, T, T)
    // 复用 inp 作为 preatt 的存储空间
    floatX* preatt = inp;
    matmul_cublaslt(preatt, k, q, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);

    // ==================== Step 3: Softmax（带 scale） ====================
    // att = softmax(preatt / sqrt(HS))
    // 使用因果 mask：每个位置只能看到之前的位置
    float scale = 1.f / sqrtf(HS);  // 缩放因子 = 1/sqrt(d_k)
    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    softmax_forward_kernel5<<<grid_size, block_size, 0, stream>>>(att, scale, preatt, B * NH, T);

    // ==================== Step 4: 加权求和 ====================
    // vaccum = V @ att，即 attention 输出
    // 复用 inp 作为 vaccum 的存储空间
    floatX* vaccum = inp;
    // (B*NH, HS, T) @ (B*NH, T, T) -> (B*NH, HS, T)
    // 但由于 matmul_cublaslt 的参数顺序，实际计算的是:
    // (B*NH, T, T) @ (B*NH, T, HS) -> (B*NH, T, HS)
    matmul_cublaslt(vaccum, v, att, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);

    // ==================== Step 5: 输出重排 ====================
    // 将 vaccum 从 (B, NH, T, HS) 重排为 (B, T, NH, HS)
    // 等价于 PyTorch 的: y.transpose(1, 2).contiguous().view(B, T, C)
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH, HS);
    
    cudaCheck(cudaGetLastError());  // 检查 CUDA 错误
}

/**softmax_autogressive_backward_inplace_ke
 * attention_backward - 多头自注意力反向传播
 * 
 * 功能描述:
 *   执行多头自注意力的完整反向传播，计算输入的梯度。
 *   反向传播按照前向传播的逆序进行。
 * 
 * 前向传播路径（用于理解反向传播）:
 *   inp (B,T,3C) → qkvr (B,T,3C) → preatt (B,NH,T,T) → att (B,NH,T,T) → vaccum (B,T,C) → out (B,T,C)
 * 
 * 反向传播流程图（逆序）:
 *   dout (B,T,C)
 *     │ unpermute_kernel_backward
 *     ▼
 *   d_vaccum (B,NH,T,HS) ──► 存储在 scratch 中
 *     │
 *     ├──► matmul: V^T @ d_vaccum ──► datt (B,NH,T,T)
 *     │
 *     └──► matmul: d_vaccum @ att^T ──► dV (B,NH,T,HS)
 *     
 *   datt (B,NH,T,T)
 *     │ softmax_backward (原地操作)
 *     ▼
 *   dpreatt (B,NH,T,T) ──► 复用 datt 内存
 *     │
 *     ├──► matmul: K @ dpreatt ──► dQ (B,NH,T,HS)
 *     │
 *     └──► matmul: Q @ dpreatt^T ──► dK (B,NH,T,HS)
 *     
 *   dQ, dK, dV (各 B,NH,T,HS)
 *     │ permute_kernel_backward
 *     ▼
 *   dinp (B,T,3C)
 * 
 * @param dinp    [out] 输入梯度，shape (B, T, 3*C)
 * @param dqkvr   [out] 重排 QKV 的梯度缓冲区，shape (3, B, NH, T, HS)
 *                      存储 dQ, dK, dV
 * @param datt    [in/out] 注意力权重的梯度，shape (B, NH, T, T)
 *                         会被 softmax backward 原地修改为 dpreatt
 * @param scratch [temp] 临时缓冲区，shape (B, NH, T, HS)
 *                       用于存储 d_vaccum（unpermute 的反向结果）
 * @param dout    [in] 输出梯度，shape (B, T, C)
 * @param qkvr    [in] 前向时保存的重排后 QKV，shape (3, B, NH, T, HS)
 * @param att     [in] 前向时保存的注意力权重，shape (B, NH, T, T)
 * @param B       [in] 批次大小
 * @param T       [in] 序列长度
 * @param C       [in] 隐藏维度
 * @param NH      [in] 注意力头数量
 * @param stream  [in] CUDA 流
 * 
 * 梯度计算公式:
 *   1. d_vaccum = unpermute^{-1}(dout)
 *   2. datt = V^T @ d_vaccum
 *   3. dV = d_vaccum @ att^T
 *   4. dpreatt = softmax_backward(datt, att) * scale
 *   5. dQ = K @ dpreatt
 *   6. dK = Q @ dpreatt^T
 *   7. dinp = permute^{-1}(dQ, dK, dV)
 */
void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const floatX* att,
                        int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();  // NVTX 性能分析标记
    const int block_size = 256;
    const int HS = C / NH;  // 每个头的维度

    // 从 qkvr 缓冲区中提取 Q, K, V 指针（前向时保存的）
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    
    // 分配 dQ, dK, dV 的存储位置
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // ==================== Step 1: unpermute 反向 ====================
    // 将 dout 从 (B, T, NH, HS) 重排回 (B, NH, T, HS)
    // 结果存入 scratch 作为 d_vaccum
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(scratch, dout, B, T, NH, HS);
    
    // ==================== Step 2: 计算 datt ====================
    // d_vaccum = att @ V，所以 datt = V^T @ d_vaccum
    // datt 的 shape: (B*NH, T, T)
    matmul_cublaslt(datt, v, scratch, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);
    
    // ==================== Step 3: 计算 dV ====================
    // d_vaccum = att @ V，所以 dV = d_vaccum @ att^T
    // 注意这里用的是 att^T
    matmul_cublaslt(dv, scratch, att, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    
    // ==================== Step 4: Softmax 反向（原地操作）====================
    // att = softmax(preatt * scale)
    // 这里将 datt 原地转换为 dpreatt
    const float scale = 1.0f / sqrtf((float)HS);
    softmax_autoregressive_backward_inplace_kernel<<<dim3(T / 4, B * NH), 256>>>(datt, att, B, T, C, scale);
    const floatX* dpreatt = datt;  // datt 现在包含 dpreatt
    
    // ==================== Step 5: 计算 dQ ====================
    // preatt = K^T @ Q，所以 dQ = K @ dpreatt
    matmul_cublaslt(dq, k, dpreatt, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);
    
    // ==================== Step 6: 计算 dK ====================
    // preatt = K^T @ Q，所以 dK = Q @ dpreatt^T
    matmul_cublaslt(dk, q, dpreatt, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    
    // ==================== Step 7: permute 反向 ====================
    // 将 dQ, dK, dV 合并回 dinp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(dinp, dq, dk, dv, B, T, NH, HS);
    
    cudaCheck(cudaGetLastError());  // 检查 CUDA 错误
}
