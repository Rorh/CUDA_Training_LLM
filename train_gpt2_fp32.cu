/*
================================================================================
GPT-2 Transformer Neural Net trained in raw CUDA (FP32版本)
================================================================================

【架构概述】
GPT-2是一个基于Transformer Decoder的自回归语言模型，其核心架构包括：

1. Token Embedding (wte): 将词汇表中的token映射到连续向量空间
   - 输入: token_id ∈ [0, V)，其中V是词汇表大小
   - 输出: embedding向量 ∈ R^C，其中C是通道数/隐藏维度

2. Position Embedding (wpe): 为序列中的每个位置添加位置信息
   - 输入: position ∈ [0, T)，其中T是最大序列长度
   - 输出: position向量 ∈ R^C

3. Transformer Block × L层，每层包含:
   a) Layer Normalization 1 (Pre-Norm架构)
   b) Multi-Head Self-Attention (MHSA)
      - Q, K, V 投影: Linear(C → 3C)
      - Attention计算: softmax(QK^T / sqrt(d)) @ V，其中d = C/NH
      - Output投影: Linear(C → C)
   c) 残差连接
   d) Layer Normalization 2
   e) Feed-Forward Network (MLP)
      - 扩展层: Linear(C → 4C) + GELU激活
      - 收缩层: Linear(4C → C)
   f) 残差连接

4. Final Layer Normalization

5. Language Model Head: 将隐藏状态投影到词汇表空间
   - 与wte共享权重 (weight tying)
   - 输出logits ∈ R^V

【符号约定】
- B: batch size (批次大小)
- T: sequence length (序列长度)
- C: channels/hidden dimension (隐藏维度，如768)
- V: vocabulary size (词汇表大小，如50257)
- Vp: padded vocabulary size (填充后的词汇表大小，如50304，为了对齐)
- L: number of layers (Transformer层数，如12)
- NH: number of attention heads (注意力头数，如12)
- HS: head size = C / NH (每个头的维度，如64)

【反向传播内存优化说明】
在反向传播中我们采用了内存优化策略：
- 所有参数梯度使用 += 操作，支持后续的梯度累积
- 大部分激活值梯度使用 = 操作（只写不读，更快）
- 残差流中的梯度必须使用 +=，因为梯度需要累加
- LayerNorm与残差连接，因此LayerNorm反向传播中使用 +=

【CUDA优化技术】
1. 使用float4进行向量化内存访问（128位读写）
2. 使用cooperative_groups进行warp级别的归约操作
3. 使用shared memory减少全局内存访问
4. 使用__ldcs/__stcs进行流式缓存提示
5. 使用cuBLAS进行高效矩阵乘法
6. 使用在线softmax算法避免数值溢出
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

// ============================================================================
// CUDA 工具函数和宏定义
// ============================================================================

/*
 * CEIL_DIV 宏: 向上取整除法
 * 用途: 计算CUDA kernel的grid维度
 * 公式: ceil(M / N) = (M + N - 1) / N
 * 
 * 示例: 如果有1000个元素，每个block处理256个
 *       则需要 CEIL_DIV(1000, 256) = 4 个block
 *
 * 参数:
 *   M: 被除数（如总元素数）
 *   N: 除数（如每block处理的元素数）
 */
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;

namespace cg = cooperative_groups;

// ----------------------------------------------------------------------------
// all the kernels

/*
 * float4加法辅助函数
 * 用于向量化操作，一次处理四个float值
 */
__device__ inline float4 add_float4(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/*
 * ============================================================================
 * Encoder前向传播 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 将输入的token序列转换为嵌入向量，结合词token嵌入和位置嵌入
 * 
 * 【计算公式】
 * 对于每个位置 (b, t) 和每个通道 c:
 *   out[b, t, c] = wte[inp[b, t], c] + wpe[t, c]
 * 
 * 其中:
 *   - wte: Token Embedding矩阵，形状 (V, C)
 *   - wpe: Position Embedding矩阵，形状 (T, C)
 *   - inp[b, t]: 第b个样本的第t个位置的token ID
 * 
 * 【CUDA优化】
 * - 使用float4向量化读写，产生128位的LDG/STG指令
 * - 每个线程处理连续的4个float值，提高内存带宽利用率
 * - 对于内存受限(memory-bound)的操作特别有效
 *
 * 【参数说明】
 * @param out  输出张量，形状 (B, T, C)，存储嵌入后的序列
 * @param inp  输入token ID数组，形状 (B, T)，每个元素∈[0, V)
 * @param wte  Token Embedding权重，形状 (V, C)
 * @param wpe  Position Embedding权重，形状 (T, C)
 * @param B    批次大小 (batch size)
 * @param T    序列长度 (sequence length)
 * @param C    嵌入维度/通道数 (embedding dimension)
 */
__global__ void encoder_forward_kernel3(float4* out,
                               const int* inp, const float4* wte, const float4* wpe,
                               int B, int T, int C) {
    int C4 = C / 4;  // 每个float4包含4个float，所以需要C/4个float4
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程索引
    int N = B * T * C4;  // 总共需要处理的float4元素数
    
    if (idx < N) {
        // 从线性索引计算多维索引
        int bt = idx / C4;       // batch-time 维度索引
        int b = bt / T;          // batch 索引
        int t = bt % T;          // time/position 索引
        int c4 = idx % C4;       // channel 索引 (在float4单位中)
        
        int ix = inp[b * T + t]; // 获取当前位置的token ID
        
        // 将token embedding和position embedding相加
        out[b * T * C4 + t * C4 + c4] = add_float4(
            wte[ix * C4 + c4],   // token embedding: wte[token_id, :]
            wpe[t * C4 + c4]     // position embedding: wpe[position, :]
        );
    }
}

/*
 * ============================================================================
 * Encoder反向传播 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 计算Token Embedding和Position Embedding的梯度
 * 
 * 【反向传播公式】
 * 前向: out[b,t,c] = wte[inp[b,t], c] + wpe[t, c]
 * 
 * 梯度传播:
 *   d(wte[ix, c]) += dout[b, t, c]   对于所有(b,t)使用了ix的位置
 *   d(wpe[t, c])  += dout[b, t, c]   对于所有batch
 * 
 * 【注意事项】
 * - 这是一个朴素实现，使用atomicAdd处理索引冲突
 * - 同一个token可能出现在多个位置，必须用原子加法累加梯度
 * - 性能较差，但实现简单，适合理解算法
 * 
 * 【参数说明】
 * @param dwte  Token Embedding梯度输出，形状 (V, C)
 * @param dwpe  Position Embedding梯度输出，形状 (T, C)
 * @param dout  上游传来的梯度，形状 (B, T, C)
 * @param inp   输入token ID数组，形状 (B, T)
 * @param B     批次大小
 * @param T     序列长度
 * @param C     通道数
 */
__global__ void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程索引
    int N = B * T * C;  // 总元素数

    if (idx < N) {
        // 从线性索引解码多维索引
        int bt = idx / C;        // batch*time 索引
        int b = bt / T;          // batch 索引
        int t = bt % T;          // time 索引
        int c = idx % C;         // channel 索引

        int ix = inp[b * T + t]; // 当前位置的token ID

        // 计算指针偏移
        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;  // 索引到对应token的embedding
        float* dwpe_tc = dwpe + t * C + c;   // 索引到对应position的embedding

        // 使用原子加法累加梯度（因为同一个token/position可能被多次引用）
        atomicAdd(dwte_ix, *dout_btc);
        atomicAdd(dwpe_tc, *dout_btc);
    }
}

/*
 * ============================================================================
 * Layer Normalization 前向传播 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 对输入张量进行层归一化，稳定训练过程中的数值分布
 * 
 * 【计算公式】
 * 对于每一行 x ∈ R^C:
 * 
 * Step 1: 计算均值
 *   μ = (1/C) * Σ_i x_i
 * 
 * Step 2: 计算方差和标准差的倒数
 *   σ² = (1/C) * Σ_i (x_i - μ)²
 *   rstd = 1 / sqrt(σ² + ε)    其中 ε = 1e-5 防止除零
 * 
 * Step 3: 归一化并应用仿射变换
 *   y_i = γ_i * (x_i - μ) * rstd + β_i
 * 
 * 其中 γ (weight) 和 β (bias) 是可学习参数
 * 
 * 【CUDA优化】
 * - 使用warp级别的并行归约计算均值和方差
 * - 每个warp处理一行数据，warp内的线程共同归约
 * - 使用cooperative_groups进行warp内归约
 * - 使用__ldcs/__stcs进行流式内存访问，优化缓存使用
 * 
 * 【参数说明】
 * @param out    输出张量，形状 (N, C)，归一化后的结果
 * @param mean   输出均值，形状 (N,)，可为NULL
 * @param rstd   输出标准差倒数，形状 (N,)，可为NULL
 * @param inp    输入张量，形状 (N, C)
 * @param weight 缩放参数 γ，形状 (C,)
 * @param bias   偏移参数 β，形状 (C,)
 * @param N      行数 (= B * T 在Transformer中)
 * @param C      列数/特征维度
 */
__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    // 创建cooperative_groups对象
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);  // 32线程一个warp
    
    // 计算当前warp负责的行索引
    // meta_group_size: block中有多少个warp
    // meta_group_rank: 当前warp在block中的索引
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // 当前warp负责的输入行
    const float* x = inp + idx * C;

    // ================== Step 1: 计算均值 μ ==================
    float sum = 0.0f;
    // warp内的每个线程处理部分元素，步长为warp大小(32)
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    // warp内归约求和
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;  // 均值
    // 只有0号线程写入结果
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);  // 流式存储
    }

    // ================== Step 2: 计算标准差倒数 rstd ==================
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;    // x_i - μ
        sum += diff * diff;       // (x_i - μ)²
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    // rsqrtf 计算 1/sqrt(x)，加上ε防止除零
    float s = rsqrtf(sum / C + 1e-5f);  // rstd = 1 / sqrt(σ² + ε)
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // ================== Step 3: 归一化并应用仿射变换 ==================
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // 使用.cs"流式"提示加载和存储
        // 表示这些数据不会很快被重用，可以通过缓存流式处理
        // 这样可以为(共享的)weight和bias参数保留更多缓存
        float n = s * (__ldcs(x+c) - m);     // 归一化: (x - μ) * rstd
        __stcs(o+c, n * weight[c] + bias[c]); // 仿射变换: γ * norm + β
    }
}

/*
 * ============================================================================
 * QKV Permute Kernel - 注意力头重排列
 * ============================================================================
 * 
 * 【功能】
 * 将合并的QKV张量分离并重新排列维度，以适配cuBLAS批量矩阵乘法
 * 
 * 【维度转换】
 * 输入:  inp 形状 (B, N, 3, NH, d)  - 合并的QKV
 *        其中3表示Q/K/V三个矩阵
 * 
 * 输出:  Q 形状 (B, NH, N, d)
 *        K 形状 (B, NH, N, d)
 *        V 形状 (B, NH, N, d)
 * 
 * 【索引映射】
 * Q[b][nh][n][d] = inp[b][n][0][nh][d]
 * K[b][nh][n][d] = inp[b][n][1][nh][d]
 * V[b][nh][n][d] = inp[b][n][2][nh][d]
 * 
 * 【为什么需要重排列】
 * cuBLAS批量矩阵乘法需要连续的batch维度
 * 重排列后，每个注意力头的数据在内存中连续存储
 * 
 * 【参数说明】
 * @param q    Q矩阵输出，形状 (B, NH, N, d)
 * @param k    K矩阵输出，形状 (B, NH, N, d)
 * @param v    V矩阵输出，形状 (B, NH, N, d)
 * @param inp  输入QKV合并张量，形状 (B, N, 3, NH, d)
 * @param B    批次大小
 * @param N    序列长度 (= T)
 * @param NH   注意力头数
 * @param d    每个头的维度 (= C / NH)
 */
__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程索引
    
    if (idx < B * NH * N * d) {
        // 从输出的线性索引解码多维索引
        // 输出形状: (B, NH, N, d)
        int b = idx / (NH * N * d);        // batch 索引
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);          // head 索引
        rest = rest % (N * d);
        int n = rest / d;                  // sequence 索引
        int d_ = rest % d;                 // dimension 索引
        
        // 计算输入索引: inp形状为 (B, N, 3, NH, d)
        // Q在第3维度的索引0位置
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        
        // 使用流式加载，K和V分别偏移 NH*d 和 2*NH*d
        q[idx] = __ldcs(&inp[inp_idx]);                   // Q: offset = 0
        k[idx] = __ldcs(&inp[inp_idx + NH * d]);          // K: offset = NH * d
        v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);    // V: offset = 2 * NH * d
    }
}

/*
 * ============================================================================
 * Permute Backward Kernel - 注意力头重排列反向传播
 * ============================================================================
 * 
 * 【功能】
 * permute_kernel的反向操作，将Q/K/V的梯度合并回原始形状
 * 
 * 【梯度流向】
 * dinp[b][n][0][nh][d] = dq[b][nh][n][d]
 * dinp[b][n][1][nh][d] = dk[b][nh][n][d]
 * dinp[b][n][2][nh][d] = dv[b][nh][n][d]
 */
__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        // 解码输出维度索引 (B, NH, N, d)
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        // 计算输入索引 (B, N, 3, NH, d)
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        
        // 将三个梯度合并写回
        dinp[inp_idx] = dq[idx];                  // Q梯度
        dinp[inp_idx + NH * d] = dk[idx];         // K梯度
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];   // V梯度
    }
}

/*
 * ============================================================================
 * Unpermute Kernel - 注意力输出重排列
 * ============================================================================
 * 
 * 【功能】
 * 将注意力输出从头优先格式转换回序列优先格式
 * 
 * 【维度转换】
 * 输入:  inp 形状 (B, NH, N, d)  - 注意力头优先
 * 输出:  out 形状 (B, N, NH, d)  - 序列优先，可视为 (B, N, C)
 * 
 * 【索引映射】
 * out[b][n][nh][d] = inp[b][nh][n][d]
 */
__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < B * NH * N * d) {
        // 解码输入维度索引 (B, NH, N, d)
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        
        // 计算输出索引 (B, N, NH, d)
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = __ldcs(&inp[idx]);
    }
}

/*
 * Unpermute反向传播 - unpermute_kernel的梯度计算
 */
__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];  // 梯度流向相反
    }
}

/* float4向量索引访问辅助函数 */
__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

/*
 * ============================================================================
 * Softmax 前向传播 Kernel (自回归注意力专用)
 * ============================================================================
 * 
 * 【功能】
 * 对注意力分数应用缩放softmax，支持自回归掩码
 * 
 * 【计算公式】
 * 对于每一行(位置t对应的attention scores):
 * 
 * 1. 应用温度缩放:
 *    scaled_score[i] = score[i] * inv_temperature
 *    其中 inv_temperature = 1 / sqrt(d_head)
 * 
 * 2. 应用自回归掩码:
 *    只计算 i <= t 的位置 (下三角形部分)
 * 
 * 3. 使用在线softmax算法避免数值溢出:
 *    max_val = max(scaled_scores)
 *    sum_exp = Σ exp(scaled_score[i] - max_val)
 *    softmax[i] = exp(scaled_score[i] - max_val) / sum_exp
 * 
 * 【在线softmax算法】
 * 传统算法需要三次遍历: 1)求max 2)求sum 3)归一化
 * 在线算法只需一次遍历，边遍历边更新max和sum:
 *   - 当发现新的最大值时，调整已累加的sum
 *   - sum_new = sum_old * exp(old_max - new_max) + exp(x - new_max)
 * 
 * 【CUDA优化】
 * - 使用warp级并行归约
 * - float4向量化读取提高内存效率
 * - 反向遍历优化缓存命中率
 * 
 * 【参数说明】
 * @param out             输出softmax概率，形状 (N, T, T)
 * @param inv_temperature 温度的倒数 = 1/sqrt(head_dim)
 * @param inp             输入attention scores，形状 (N, T, T)
 * @param N               batch*num_heads (= B * NH)
 * @param T               序列长度
 */
__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    assert(T % 4 == 0);  // T必须是4的倍数以支持float4向量化
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // 微优化: 反向遍历block
    // 这样在softmax反向完成后，缓存中保留的是矩阵左上角的数据
    // 这有利于紧接着的矩阵乘法操作
    int idx = (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N * T) {
        return;
    }
    
    int own_pos = idx % T;  // 当前位置，决定掩码范围
    int pos_by_4 = own_pos / 4;  // float4边界

    // 当前行的输入数据
    const float* x = inp + idx * T;

    // 初始化为-FLT_MAX而不是-INF，避免减法产生NaN
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    // =============== 在线softmax算法: 向量化部分 ===============
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        // 更新最大值
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        // 调整已累加的sum
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        // 累加新值
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    // 处理剩余元素(非float4对齐部分)
    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    // =============== warp内归约 ===============
    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));  // 调整到全局最大值
    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;  // 归一化因子

    // =============== 计算最终的softmax值 ===============
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // 重新计算exp比从内存加载更快
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}

/*
 * ============================================================================
 * 残差连接前向传播 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 实现残差连接: out = inp1 + inp2
 * 
 * 【作用】
 * 残差连接是Transformer的核心组件，它允许梯度直接流过网络，
 * 缓解梯度消失问题，使得训练更深层的网络成为可能
 * 
 * 【参数说明】
 * @param out   输出，形状 (N,)
 * @param inp1  第一个输入(通常是残差流)
 * @param inp2  第二个输入(通常是子层输出)
 * @param N     元素总数
 */
__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __ldcs(&inp1[idx]) + __ldcs(&inp2[idx]);
    }
}

/*
 * GELU缩放因子: sqrt(2/π) ≈ 0.7978845608
 */
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

/*
 * ============================================================================
 * GELU 激活函数前向传播 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 计算Gaussian Error Linear Unit (GELU)激活函数
 * 
 * 【计算公式】
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 * 
 * 这是GELU的近似实现，原始的GELU定义是:
 * GELU(x) = x * Φ(x)
 * 其中Φ(x)是标准正态分布的累积分布函数
 * 
 * 【为什么用GELU而不是ReLU】
 * - GELU是平滑的，处处可微
 * - 在零附近有正则化效果，小负值不会被完全置零
 * - GPT系列模型中广泛使用
 * 
 * 【参数说明】
 * @param out  输出，形状 (N,)
 * @param inp  输入，形状 (N,)
 * @param N    元素总数
 */
__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;  // 0.044715 * x³
        // GELU = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

/*
 * ============================================================================
 * GELU 激活函数反向传播 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 计算GELU的梯度
 * 
 * 【反向传播公式】
 * 设 u = sqrt(2/π) * (x + 0.044715 * x³)
 * 设 t = tanh(u)
 * 
 * dGELU/dx = 0.5 * (1 + t) + 0.5 * x * sech²(u) * sqrt(2/π) * (1 + 3*0.044715*x²)
 * 
 * 其中 sech(u) = 1/cosh(u)，sech²(u) = 1/cosh²(u)
 * 
 * 最终梯度: dinp = dGELU/dx * dout
 * 
 * 【参数说明】
 * @param dinp  输入梯度输出，形状 (N,)
 * @param inp   前向传播的输入（需要用于计算梯度）
 * @param dout  上游梯度，形状 (N,)
 * @param N     元素总数
 */
__global__ void gelu_backward_kernel(float* dinp, const float* inp, const float* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);  // u
        float tanh_out = tanhf(tanh_arg);                   // tanh(u)
        float coshf_out = coshf(tanh_arg);                  // cosh(u)
        float sech_out = 1.0f / (coshf_out * coshf_out);    // sech²(u) = 1/cosh²(u)
        
        // dGELU/dx = 0.5*(1+tanh) + x*0.5*sech²*sqrt(2/π)*(1+3*0.044715*x²)
        float local_grad = 0.5f * (1.0f + tanh_out) + 
                          x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];  // 链式法则
    }
}

/*
 * ============================================================================
 * 矩阵乘法反向传播 - Bias梯度计算 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 计算对bias的梯度，等价于PyTorch中的:
 * dbias = dout.sum((0, 1))  # 对batch和sequence维度求和
 * 
 * 【计算公式】
 * 前向: out[b,t,c] = inp[b,t,:] @ weight[:, c] + bias[c]
 * 反向: dbias[c] = Σ_{b,t} dout[b,t,c]
 * 
 * 【算法设计】
 * - 每个block处理32列(一个warp宽度)，保证合并访问
 * - block内的多个warp共同处理所有行
 * - 使用shared memory累加各warp的部分和
 * 
 * 【内存访问模式】
 * 连续的线程访问连续的列，实现内存合并
 * 
 * 【参数说明】
 * @param dbias  bias梯度输出，形状 (OC,)
 * @param dout   上游梯度，形状 (B, T, OC)
 * @param B      批次大小
 * @param T      序列长度
 * @param OC     输出通道数
 */
__global__ void matmul_backward_bias_kernel4(float* dbias, const float* dout, int B, int T, int OC) {
    // 这个kernel的grid维度是 OC/32
    extern __shared__ float smem[];  // 大小为block_size (e.g., 128)
    
    const int warp_id = threadIdx.x / warpSize;  // block内的warp索引: 0,1,2,3
    const int lane_id = threadIdx.x % warpSize;  // warp内的线程索引: 0-31
    const int tl = blockIdx.x * warpSize;        // 当前block负责的32列的起始索引
    const int vstep = blockDim.x / warpSize;     // block中的warp数量 (e.g., 4)

    // 指向当前lane负责的列的起始位置
    // 每个warp中的同一lane_id的线程共同处理同一列
    const float* dout_col = dout + tl + lane_id;

    // 列归约: 遍历所有行
    // 每个线程从warp_id开始，步长为vstep
    // 这样vstep个warp共同覆盖所有B*T行
    // 连续的线程访问连续的列，实现内存合并
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += dout_col[row * OC];
    }
    // 将各warp的部分和存入shared memory
    smem[lane_id + warp_id * warpSize] = dout_sum;
    __syncthreads();

    // warp 0对shared memory进行最终归约
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] += dout_sum;  // 使用 += 支持梯度累积
    }
}

// uses shared memory instead for the reduces
/*
 * ============================================================================
 * 层归一化反向传播 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 计算层归一化的反向传播梯度
 * 
 * 【计算公式】
 * 前向: out = (inp - mean) / std
 *      mean = Σ_{i} inp[i] / N
 *      std = sqrt(Σ_{i} (inp[i] - mean)² / N)
 * 
 * 反向: 
 *      dinp = (dout / std) - (dout * (inp - mean) / std³) * (Σ_{i} (dout * (inp[i] - mean)) / N)
 *      dweight = Σ_{i} dout[i] * (inp[i] - mean) / std
 *      dbias = Σ_{i} dout[i]
 * 
 * 【算法设计】
 * - 使用shared memory存储中间结果
 * - 使用warp内的线程并行计算
 * 
 * 【参数说明】
 * @param dinp   输入梯度输出，形状 (B, T, C)
 * @param dweight 权重梯度输出，形状 (C,)
 * @param dbias  偏置梯度输出，形状 (C,)
 * @param dout   上游梯度，形状 (B, T, C)
 * @param inp    输入，形状 (B, T, C)
 * @param weight 权重，形状 (C,)
 * @param mean   均值，形状 (B, T)
 * @param rstd   标准差的倒数，形状 (B, T)
 * @param B      批次大小
 * @param T      序列长度
 * @param C      通道数
 */
__global__ void layernorm_backward_kernel2(float* dinp, float* dweight, float* dbias,
                                           const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                                           int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if(idx >= N) { return; } // thread guards

    int b = idx / T;
    int t = idx % T;

    // 定位到当前 (b, t) 位置的数据指针，每个位置有 C 个通道
    const float* dout_bt = dout + b * T * C + t * C;   // 输出梯度，形状 [B, T, C] 中的 [b, t, :] 切片
    const float* inp_bt = inp + b * T * C + t * C;     // 前向输入，形状 [B, T, C] 中的 [b, t, :] 切片
    float* dinp_bt = dinp + b * T * C + t * C;         // 输入梯度，形状 [B, T, C] 中的 [b, t, :] 切片
    // 定位到当前 (b, t) 位置的统计量，mean 和 rstd 形状为 [B, T]
    const float mean_bt = mean[b * T + t];             // 当前位置的均值
    const float rstd_bt = rstd[b * T + t];             // 当前位置的标准差倒数 (1/std)

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll
	for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
    __syncthreads();

    // write to global memory
	for(int i = threadIdx.x; i < C; i+= blockDim.x){
        atomicAdd(&dbias[i], dbias_shared[i]);
        atomicAdd(&dweight[i], dweight_shared[i]);
	}
}

/*
 * ============================================================================
 * Softmax 反向传播 Kernel (自回归注意力专用)
 * ============================================================================
 * 
 * 【功能】
 * 计算自回归softmax的反向传播梯度
 * 
 * 【计算公式】
 * 前向: att[i] = exp(preatt[i] * scale) / sum(exp(preatt[:] * scale))
 * 
 * 反向: 对于softmax的反向传播，设 y = softmax(x)，则:
 *   dx_i = y_i * (dy_i - Σ_j y_j * dy_j)
 * 
 * 即: dpreatt[i] = scale * att[i] * (datt[i] - Σ_j att[j] * datt[j])
 * 
 * 【算法设计】
 * - 每个block处理T_per_block行，反向遍历以优化缓存
 * - 使用两阶段归约计算Σ_j att[j] * datt[j]
 * 
 * 【参数说明】
 * @param dpreatt  pre-attention梯度输出，形状 (B*NH, T, T)
 * @param datt     attention梯度输入，形状 (B*NH, T, T)
 * @param att      attention权重，形状 (B*NH, T, T)
 * @param B        批次大小
 * @param T        序列长度
 * @param C        通道数
 * @param scale    缩放因子 = 1/sqrt(head_dim)
 */
/**
 * CUDA kernel for computing the backward pass of softmax in an autoregressive attention mechanism.
 * This computes the gradient of the loss with respect to the pre-softmax attention scores.
 * 
 * @param dpreatt Output gradient for the pre-softmax attention scores (B, T, T)
 * @param datt Gradient of the loss with respect to the attention weights (B, T, T)
 * @param att Attention weights from forward pass (B, T, T)
 * @param B Batch size
 * @param T Sequence length
 * @param C Feature dimension (unused in this kernel)
 * @param scale Scaling factor for the output gradient
 */
__global__ void softmax_autoregressive_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, float scale) {
    // Configuration constants
    constexpr const int BlockSize = 256;  // Number of threads per block
    constexpr int T_per_block = 4;        // Number of rows processed per block
    
    // Cooperative groups for efficient thread synchronization
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Shared memory for partial reduction results
    __shared__ float block_acc[32];

    // Get the batch index for this block
    int idx = blockIdx.y;
    
    // Process rows in reverse order to improve cache locality for autoregressive masking
    // Each block processes T_per_block rows, starting from the end of the sequence
    int t0 = T - 1 - T_per_block * blockIdx.x;

    // Offset pointers for this batch
    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    // Initialize shared memory for reduction
    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    // Process each row assigned to this block
    for(int to = 0; to < T_per_block; ++to) {
        // Get the current timestep (processing in reverse)
        int t = t0 - to;
        if(t < 0) return;  // Skip if we've gone past the start of the sequence
        
        // Get pointers to the current row in each tensor
        const float* att_bth = att + t * T;        // Attention weights for current timestep
        const float* datt_bth = datt + t * T;      // Input gradients for current timestep
        float* dpreatt_bth = dpreatt + t * T;      // Output gradients for current timestep

        // First pass: compute local sum of att * datt for this row
        float local_sum = 0;
        for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        // Reduce the partial sums within the warp
        block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        block.sync();
        
        // Final reduction across warps in the block
        local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

        // Second pass: compute and store the final gradients
        // Using __ldcs and __stcs for cache control (streaming load/store)
        for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
            // Gradient formula for softmax backward: att * (datt - sum(att * datt))
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);  // Apply scaling factor and store
        }
    }
}

/*
 * 线性插值函数 (lerp)
 * 使用两个浮点操作实现（朴素实现需要三个）
 * 公式: lerp(a, b, w) = a + w * (b - a) = (1-w)*a + w*b
 * 优化: fma(w, b, fma(-w, a, a)) = w*b + (-w*a + a) = w*b + a*(1-w)
 * 参考: https://developer.nvidia.com/blog/lerp-faster-cuda
 */
__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

/*
 * ============================================================================
 * AdamW 优化器 Kernel
 * ============================================================================
 * 
 * 【功能】
 * 实现AdamW优化算法，带权重衰减解耦的Adam变体
 * 
 * 【算法公式】
 * AdamW结合了Adam的自适应学习率和L2正则化:
 * 
 * 1. 更新一阶矩估计 (momentum):
 *    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 *    实现: m = lerp(grad, m, β₁) = β₁*m + (1-β₁)*grad
 * 
 * 2. 更新二阶矩估计 (RMSprop风格):
 *    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 *    实现: v = lerp(grad², v, β₂) = β₂*v + (1-β₂)*grad²
 * 
 * 3. 偏差修正:
 *    m̂_t = m_t / (1 - β₁^t)
 *    v̂_t = v_t / (1 - β₂^t)
 * 
 * 4. 参数更新 (包含解耦的权重衰减):
 *    θ_t = θ_{t-1} - lr * (m̂_t / (sqrt(v̂_t) + ε) + λ * θ_{t-1})
 * 
 * 【AdamW vs Adam】
 * - Adam: 权重衰减被包含在梯度中，会被自适应学习率调整
 * - AdamW: 权重衰减与梯度更新解耦，理论上更合理
 * 
 * 【参数说明】
 * @param params_memory     模型参数，形状 (num_parameters,)
 * @param grads_memory      参数梯度，形状 (num_parameters,)
 * @param m_memory          一阶矩缓存，形状 (num_parameters,)
 * @param v_memory          二阶矩缓存，形状 (num_parameters,)
 * @param num_parameters    参数总数
 * @param learning_rate     学习率 lr
 * @param beta1             一阶矩衰减系数 β₁ (通常0.9)
 * @param beta2             二阶矩衰减系数 β₂ (通常0.999)
 * @param beta1_correction  偏差修正项 1 - β₁^t
 * @param beta2_correction  偏差修正项 1 - β₂^t
 * @param eps               数值稳定性常数 ε (通常1e-8)
 * @param weight_decay      权重衰减系数 λ
 */
__global__ void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // 线程越界检查
   
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   
   // 更新一阶矩 (momentum): m = β₁*m + (1-β₁)*grad
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   
   // 更新二阶矩 (RMSprop): v = β₂*v + (1-β₂)*grad²
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   
   // 偏差修正
   m /= beta1_correction;  // m̂ = m / (1 - β₁^t)
   v /= beta2_correction;  // v̂ = v / (1 - β₂^t)
   
   // 参数更新: θ -= lr * (m̂ / (sqrt(v̂) + ε) + λ * θ)
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

/*
 * Softmax参数结构体
 * 用于存储在线softmax算法的中间结果
 */
struct SoftmaxParams {
    float Scale;   // 1 / sum(exp(x - max))  归一化因子
    float Offset;  // max(x)  用于数值稳定性
};

/*
 * ============================================================================
 * Softmax 参数准备函数 (block级别归约)
 * ============================================================================
 * 
 * 【功能】
 * 为softmax计算准备最大值和归一化因子，使用在线算法
 * 
 * 【算法说明】
 * 1. 各线程并行计算局部最大值和指数和
 * 2. warp内归约，然后通过shared memory跨warp归约
 * 3. 返回缩放因子和偏移量供后续使用
 * 
 * 【参数说明】
 * @param warp  warp协作组对象
 * @param idx   当前处理的行索引
 * @param inp   输入logits，形状 (*, P)
 * @param V     实际词汇表大小
 * @param P     填充后的词汇表大小 (padded)
 * @return      SoftmaxParams {scale, offset}
 */
__device__ SoftmaxParams prepare_softmax_blockwide_nofloat4(cg::thread_block_tile<32>& warp,
                                                   int idx, const float* inp, int V, int P) {
    // 这是非float4版本，处理一行inp[idx, :]，形状为(V,)

    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf((old_maxval - thread_maxval));
        thread_sumval += expf(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

/*
 * ============================================================================
 * 融合分类器 Kernel (Softmax + Cross-Entropy Loss + 反向传播)
 * ============================================================================
 * 
 * 【功能】
 * 将softmax、交叉熵损失计算和梯度计算融合在一个kernel中
 * 这避免了存储完整的概率分布，大大节省内存带宽
 * 
 * 【计算公式】
 * 1. Softmax:
 *    prob[i] = exp(logit[i] - max) / sum(exp(logit - max))
 * 
 * 2. Cross-Entropy Loss:
 *    loss = -log(prob[target])
 * 
 * 3. Softmax + Cross-Entropy 梯度 (融合计算):
 *    dlogit[i] = prob[i] - 1_{i=target}   (当i是目标类别时为1，否则为0)
 *    缩放: dlogit[i] *= dloss  (dloss默认为1/(B*T)表示平均损失)
 * 
 * 【为什么要融合】
 * - 避免存储完整的prob张量 (B*T*V 可能非常大)
 * - 减少全局内存访问次数
 * - 利用缓存局部性
 * 
 * 【参数说明】
 * @param logits   输入logits，会被原地替换为梯度，形状 (B*T, P)
 * @param losses   输出损失值，形状 (B*T,)
 * @param probs    可选的概率输出（调试/推理用），可为NULL
 * @param dlosses  损失梯度，默认为1/(B*T)
 * @param targets  目标token ID，形状 (B*T,)
 * @param B        批次大小
 * @param T        序列长度
 * @param V        词汇表大小
 * @param P        填充后的词汇表大小
 */
__global__ void fused_classifier_kernel3(float* logits, float* losses, float* probs,
                                         const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x;        // 当前处理的样本索引
    int ix = targets[idx];       // 目标token ID

    // ============ Step 1: 计算softmax参数 ============
    // 读取logits并计算max和sum，希望后续访问仍在缓存中
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

    // ============ Step 2: 计算损失 ============
    // 只有90号线程计算并写入损失值
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);  // cross-entropy: -log(prob[target])
    }

    // ============ Step 3: 计算梯度 ============
    // dloss默认为1/(B*T)，表示平均损失
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    
    // 直接计算梯度，避免存储完整的probs
    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // 第二次读取logits，使用流式加载因为不会再用
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        
        // 可选地存储概率(用于调试或推理)
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        
        // 梯度: dlogit = (prob - indicator) * dloss
        // indicator = 1 当 i == target，否则 = 0
        float indicator = (i == ix) ? 1.0f : 0.0f;
        logits[idx * P + i] = (prob - indicator) * dloss;
    }
}

/* float4向量化加载辅助函数 */
__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

/* float4向量化存储辅助函数 */
__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

/*
 * ============================================================================
 * 矩阵乘法前向传播 Kernel (Tiled GEMM)
 * ============================================================================
 * 
 * 【功能】
 * 实现高效的矩阵乘法: out = inp @ weight.T + bias
 * 
 * 【计算公式】
 * out[b,t,oc] = Σ_c inp[b,t,c] * weight[oc,c] + bias[oc]
 * 
 * 【算法设计 - Tiled GEMM】
 * 1. 每个block处理128x128的输出块
 * 2. 每个线程处禆8x8的输出元素
 * 3. 使用shared memory缓存输入块，每次加载32列
 * 4. 使用float4向量化读写提高内存带宽利用率
 * 
 * 【内存布局】
 * - lhs_s[128][32]: 缓存输入矩阵块
 * - rhs_s[128][32]: 缓存权重矩阵块
 * - vals[8][8]: 每个线程的输出累加器
 * 
 * 【Launch Bounds】
 * __launch_bounds__(256, 2): 每block 256线程，最小2个block/SM
 * 这有助于编译器优化寄存器分配
 * 
 * 【参数说明】
 * @param out     输出矩阵，形状 (B*T, OC)
 * @param inp     输入矩阵，形状 (B*T, C)
 * @param weight  权重矩阵，形状 (OC, C)
 * @param bias    偏置向量，形状 (OC,)，可为NULL
 * @param C       输入通道数
 * @param OC      输出通道数 (e.g., 4*C 对于MLP)
 */
__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel4(float* out,
                                                                   const float* inp, const float* weight, const float* bias,
                                                                   int C, int OC) {
    // ==================== 线程与输出映射 ====================
    // 每个线程负责计算8x8个输出元素
    // 线程块大小: 16x16 = 256个线程
    // 每个线程块处理: 128x128的输出子矩阵 (16*8 x 16*8)
    int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);  // 当前线程负责的输出通道起始索引

    // ==================== 共享内存分配 ====================
    // 用于缓存输入矩阵和权重矩阵的分块
    // lhs_s: 左矩阵(输入)的共享内存缓存，128行 x 32列
    // rhs_s: 右矩阵(权重)的共享内存缓存，128行 x 32列
    // 每次迭代处理K维度的32个元素
    __shared__ float lhs_s[128][32];
    __shared__ float rhs_s[128][32];

    // ==================== 指针偏移调整 ====================
    // 将指针移动到当前线程块负责的数据区域
    // inp: 移动到第 (128 * blockIdx.x) 行
    // weight: 移动到第 (128 * blockIdx.y) 行
    // out: 移动到输出矩阵的对应位置 (128*blockIdx.x, 128*blockIdx.y)
    inp += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    // ==================== 累加器初始化 ====================
    // vals[8][8]: 每个线程的局部累加器，存储8x8个输出结果
    // 如果有偏置，用偏置值初始化；否则初始化为0
    float vals[8][8] = {};
    if(bias != NULL) {
        // 使用float4向量化加载偏置值（每次加载4个float）
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j+0] = b.x;
                vals[i][j+1] = b.y;
                vals[i][j+2] = b.z;
                vals[i][j+3] = b.w;
            }
        }
    }

    // ==================== 主计算循环 ====================
    // 沿K维度(C)进行分块迭代，每次处理32个元素
    // si_start: 当前线程在共享内存中读取的起始位置
    // 使用循环移位确保不同线程读取不同数据，减少bank冲突
    int si_start = 4*(16 * threadIdx.y + threadIdx.x);
    for (int so = 0; so < C; so += 32) {
        // ---------- 阶段1: 协作加载数据到共享内存 ----------
        __syncthreads();  // 确保上一轮计算完成后再加载新数据
        
        // 计算当前线程负责加载的列偏移
        int xmod8 = threadIdx.x % 8;   // 线程x索引对8取模
        int xby8 = threadIdx.x / 8;    // 线程x索引除以8
        int xo = 4 * xmod8;            // 列偏移（0, 4, 8, 12, 16, 20, 24, 28）
        
        // 每个线程加载多行数据，步长为32（256线程/8列 = 32）
        // 使用float4向量化加载，每次加载4个连续的float
        for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));     // 加载输入矩阵块
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));  // 加载权重矩阵块
        }
        __syncthreads();  // 确保所有线程完成加载后再开始计算

        // ---------- 阶段2: 计算矩阵乘法 ----------
        // 每个线程遍历共享内存中的32列数据
        // 使用循环移位(si % 32)避免bank冲突
        for (int si = si_start; si < si_start + 32; si += 4) {
            // 预加载当前线程需要的8行权重数据（右矩阵）
            float4 rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            // 遍历8行输入数据（左矩阵），与权重做外积累加
            for (int ii = 0; ii < 8; ++ii) {
                // 加载左矩阵的一行（4个元素）
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                
                // 对8列权重数据进行点积累加
                // vals[ii][ji] += dot(lhs, rhs[ji])
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    // ==================== 结果写回全局内存 ====================
    // 将8x8的计算结果写回输出矩阵
    // 使用float4向量化存储，每次写入4个连续的float
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            // 输出位置: (8*threadIdx.x + i, 8*threadIdx.y + j) 相对于块起始位置
            st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
        }
    }
}


// ============================================================================
// Kernel 启动函数 (Kernel Launchers)
// ============================================================================
// 这些函数封装了CUDA kernel的启动逻辑，包括:
// - 计算grid和block维度
// - 启动kernel
// - 检查CUDA错误

/*
 * Encoder前向传播
 * 将token ID序列转换为嵌入向量
 * out = wte[inp] + wpe[0:T]
 */
void encoder_forward(float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    assert(C % 4 == 0);
    const int block_size = 512;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N / 4, block_size);
    encoder_forward_kernel3<<<grid_size, block_size>>>((float4*) out, inp, (float4*) wte, (float4*) wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_backward(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

/*
 * 矩阵乘法前向传播
 * out = inp @ weight.T + bias
 * 其中: inp形状(B,T,C), weight形状(OC,C), out形状(B,T,OC)
 */
void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // 使用Tiled GEMM kernel，每block 16x16线程，每线程处礆8x8元素
    int sqrt_block_size = 16;

    dim3 gridDim(CEIL_DIV(B * T, 8*sqrt_block_size), CEIL_DIV(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}

/*
 * ============================================================================
 * 多头自注意力前向传播
 * ============================================================================
 * 
 * 【数据流】
 * inp (B,T,3C) -> 分离Q,K,V (B,NH,T,HS) -> attention -> out (B,T,C)
 * 
 * 【计算步骤】
 * 1. Permute: 将QKV从inp中分离并重排为头优先格式
 * 2. Q @ K.T: 计算注意力分数 (cuBLAS)
 * 3. Softmax: 应用缩放softmax，带自回归掩码
 * 4. Att @ V: 加权求和 (cuBLAS)
 * 5. Unpermute: 重排回序列优先格式
 * 
 * 【注意力公式】
 * Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) @ V
 * 
 * 【参数说明】
 * @param out   输出，形状 (B, T, C)
 * @param qkvr  Q/K/V存储缓冲区，形状 (3, B, T, C)
 * @param att   注意力权重输出，形状 (B, NH, T, T)
 * @param inp   输入QKV，形状 (B, T, 3C)，会被用作临时缓冲区
 * @param B     批次大小
 * @param T     序列长度
 * @param C     通道数
 * @param NH    注意力头数
 */
void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // inp在反向传播中不需要，因此复用作为临时缓冲区
    const int block_size = 256;
    const int softmax_block_size = 256;

    int HS = C / NH;  // 每个头的维度 (head size)

    // ============ Step 1: Permute ============
    // 将inp从(B, T, 3, NH, HS)分离为3个(B, NH, T, HS)的张量
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;  // Q: 偏移0
    k = qkvr + 1 * B * T * C;  // K: 偏移B*T*C
    v = qkvr + 2 * B * T * C;  // V: 偏移2*B*T*C
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // ============ Step 2: Q @ K.T (cuBLAS批量矩阵乘法) ============
    // preatt = Q @ K.T, 形状 (B, NH, T, T)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* preatt = inp;  // 复用inp存储pre-attention scores
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, 
        CUBLAS_OP_T, CUBLAS_OP_N,  // K要转置
        T, T, HS,                   // M, N, K
        &alpha, k, HS, T * HS,     // K矩阵
        q, HS, T * HS,             // Q矩阵
        &beta, preatt, T, T * T,   // 输出
        B * NH));                   // batch数

    // ============ Step 3: Softmax ============
    // att = softmax(preatt * scale), scale = 1/sqrt(head_size)
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // ============ Step 4: Att @ V (cuBLAS批量矩阵乘法) ============
    // vaccum = att @ v, 形状 (B, NH, T, HS)
    float* vaccum = inp;  // 复用inp
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        HS, T, T, &alpha, 
        v, HS, T * HS,
        att, T, T * T, 
        &beta, vaccum, HS, T * HS, 
        B * NH));

    // ============ Step 5: Unpermute ============
    // 将(B, NH, T, HS)重排为(B, T, NH, HS) = (B, T, C)
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward(float* out, const float* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &one, weight, C, dout, OC, &zero, dinp, C));
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &one, inp, C, dout, OC, &one, dweight, C));
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        const int block_size = 1024;
        const int grid_size = OC / 32; // for now, OC must be divisible by 32 for this kernel to work
        matmul_backward_bias_kernel4<<<grid_size, block_size, block_size * sizeof(float)>>>(dbias, dout, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const  float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(32*N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    layernorm_backward_kernel2<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* scratch,
                        const float* dout,
                        const float* qkvr, const float* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;
    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));
    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    softmax_autoregressive_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());
    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// replaces logits with logit gradients
void fused_classifier3(float* logits, float* losses,
                      const float* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(logits, losses, NULL, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// GPT-2 模型定义
// ============================================================================

/*
 * GPT-2 模型配置结构体
 * 存储模型的超参数
 */
typedef struct {
    int max_seq_len;       // 最大序列长度，如 1024
    int vocab_size;        // 词汇表大小，如 50257
    int padded_vocab_size; // 填充后的词汇表大小，如 50304 (为了对齐到128的倍数)
    int num_layers;        // Transformer层数，如 12
    int num_heads;         // 注意力头数，如 12
    int channels;          // 隐藏层维度/通道数，如 768
} GPT2Config;

/*
 * GPT-2 参数张量结构体
 * 包含模型的16种参数张量
 * 
 * 【符号说明】
 * V = vocab_size, C = channels, L = num_layers
 * maxT = max_seq_len, NH = num_heads
 */
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    // ============ Embedding层 ============
    float* wte;       // Token Embedding权重，形状 (V, C)
    float* wpe;       // Position Embedding权重，形状 (maxT, C)
    
    // ============ 每层Transformer Block的参数 ============
    // 注意力块 (Attention Block)
    float* ln1w;      // LayerNorm1权重，形状 (L, C)
    float* ln1b;      // LayerNorm1偏置，形状 (L, C)
    float* qkvw;      // QKV投影权重，形状 (L, 3*C, C) - 将输入投影为Q,K,V
    float* qkvb;      // QKV投影偏置，形状 (L, 3*C)
    float* attprojw;  // 注意力输出投影权重，形状 (L, C, C)
    float* attprojb;  // 注意力输出投影偏置，形状 (L, C)
    
    // MLP块 (Feed-Forward Block)
    float* ln2w;      // LayerNorm2权重，形状 (L, C)
    float* ln2b;      // LayerNorm2偏置，形状 (L, C)
    float* fcw;       // MLP扩展层权重，形状 (L, 4*C, C) - 扩展4倍
    float* fcb;       // MLP扩展层偏置，形状 (L, 4*C)
    float* fcprojw;   // MLP收缩层权重，形状 (L, C, 4*C) - 收缩回原始维度
    float* fcprojb;   // MLP收缩层偏置，形状 (L, C)
    
    // ============ 最终LayerNorm ============
    float* lnfw;      // 最终LayerNorm权重，形状 (C,)
    float* lnfb;      // 最终LayerNorm偏置，形状 (C,)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    int Vp = config.padded_vocab_size;
    int C = config.channels;
    int maxT = config.max_seq_len;
    int L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, int on_device) {
    // on_device: 0 = CPU, 1 = GPU
    // calculate the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once on the device
    float* params_memory;
    if (on_device) {
        cudaCheck(cudaMalloc((void**)&params_memory, num_parameters * sizeof(float)));
    } else {
        params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    }
    // assign all the tensors their place in the array
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

/*
 * 激活值张量结构体
 * 存储前向传播过程中的21种中间结果
 * 这些激活值在反向传播时也需要使用
 */
#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    // ============ Embedding层输出 ============
    float* encoded;     // token + position embedding结果，形状 (B, T, C)
    
    // ============ 每层Transformer Block的激活值 ============
    // 注意力块
    float* ln1;         // LayerNorm1输出，形状 (L, B, T, C)
    float* ln1_mean;    // LayerNorm1均值，形状 (L, B, T) - 用于反向传播
    float* ln1_rstd;    // LayerNorm1标准差倒数，形状 (L, B, T)
    float* atty;        // 注意力输出，形状 (L, B, T, C)
    float* att;         // 注意力权重，形状 (L, B, NH, T, T)
    float* attproj;     // 注意力投影输出，形状 (L, B, T, C)
    float* residual2;   // 第一个残差连接后，形状 (L, B, T, C)
    
    // MLP块
    float* ln2;         // LayerNorm2输出，形状 (L, B, T, C)
    float* ln2_mean;    // LayerNorm2均值，形状 (L, B, T)
    float* ln2_rstd;    // LayerNorm2标准差倒数，形状 (L, B, T)
    float* fch;         // MLP扩展层输出，形状 (L, B, T, 4*C)
    float* fch_gelu;    // GELU激活后，形状 (L, B, T, 4*C)
    float* fcproj;      // MLP收缩层输出，形状 (L, B, T, C)
    float* residual3;   // 第二个残差连接后，形状 (L, B, T, C)
    
    // ============ 最终层 ============
    float* lnf;         // 最终LayerNorm输出，形状 (B, T, C)
    float* lnf_mean;    // 最终LayerNorm均值，形状 (B, T)
    float* lnf_rstd;    // 最终LayerNorm标准差倒数，形状 (B, T)
    
    // ============ 损失和输出 ============
    float* losses;      // 每个位置的损失，形状 (B, T)
    float* qkvr;        // QKV重排列缓冲区，形状 (L, B, T, 3*C)
    
    /*
     * output缓冲区 - 多用途:
     * - 推理模式: 存储logits，形状 (B, T, V)
     * - 训练模式: 存储logits的梯度
     * - Transformer块处理时: 用作临时缓冲区
     * 分配大小: max((B,T,3C), (B,NH,T,T), (B,T,V))
     */
    float* output;
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * C; // atty
    act_sizes[5] = L * B * NH * T * T; // att
    act_sizes[6] = L * B * T * C; // attproj
    act_sizes[7] = L * B * T * C; // residual2
    act_sizes[8] = L * B * T * C; // ln2
    act_sizes[9] = L * B * T; // ln2_mean
    act_sizes[10] = L * B * T; // ln2_rstd
    act_sizes[11] = L * B * T * 4*C; // fch
    act_sizes[12] = L * B * T * 4*C; // fch_gelu
    act_sizes[13] = L * B * T * C; // fcproj
    act_sizes[14] = L * B * T * C; // residual3
    act_sizes[15] = B * T * C; // lnf
    act_sizes[16] = B * T; // lnf_mean
    act_sizes[17] = B * T; // lnf_rstd
    act_sizes[18] = B * T; // losses
    act_sizes[19] = L * B * T * 3*C; // qkvr
    act_sizes[20] = B * T * max(3*C, max(NH*T, Vp)); // output / scratch
}

/*
 * 反向传播激活值梯度结构体
 * 
 * 【内存优化说明】
 * 反向传播与前向传播不同，我们可以在处理完一层后立即丢弃其激活值
 * 这允许我们激进地重用内存，所以反向传播只需要3个张量
 */
#define NUM_BACKWARD_TENSORS 3
typedef struct {
    float* bt4c;      // MLP层的梯度缓冲区，形状 (B, T, 4*C)
    float* preatt;    // pre-attention梯度，形状 (B, NH, T, T)
    float* residual3; // 残差流梯度，形状 (B, T, C)
} GradActTensors;


void fill_in_grad_act_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * 4 * C; // bt4c
    act_sizes[1] = B * NH * T * T; // preatt
    act_sizes[2] = B * T * C; // residual3
}


float* malloc_and_point(float** targets[], const size_t* act_sizes, int n) {
    size_t num_activations = 0;
    for (size_t i = 0; i < n; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, num_activations * sizeof(float)));
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

float* malloc_and_point_activations(ActivationTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output
    };
    return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

float* malloc_and_point_backward(GradActTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->bt4c, &acts->preatt, &acts->residual3
    };
    return malloc_and_point(ptrs, act_sizes, NUM_BACKWARD_TENSORS);
}

/*
 * GPT-2 模型主结构体
 * 包含模型的所有状态:配置、参数、梯度、激活值、优化器状态等
 */
typedef struct {
    // ============ 模型配置 ============
    GPT2Config config;  // 模型超参数
    
    // ============ 模型参数 ============
    ParameterTensors params;                    // 参数张量指针
    size_t param_sizes[NUM_PARAMETER_TENSORS];  // 每个参数张量的大小
    float* params_memory;                       // GPU上的参数内存
    size_t num_parameters;                      // 参数总数
    
    // ============ 参数梯度 ============
    ParameterTensors grads;   // 梯度张量指针 (与params结构相同)
    float* grads_memory;      // GPU上的梯度内存
    
    // ============ AdamW优化器状态 ============
    float* m_memory;  // 一阶矩估计 (momentum)，形状同参数
    float* v_memory;  // 二阶矩估计 (RMSprop)，形状同参数
    
    // ============ 激活值 ============
    ActivationTensors acts;                   // 激活值张量指针
    size_t act_sizes[NUM_ACTIVATION_TENSORS]; // 每个激活值张量的大小
    float* acts_memory;                       // GPU上的激活值内存
    size_t num_activations;                   // 激活值总数
    
    // ============ 激活值梯度 ============
    GradActTensors grads_acts;   // 激活值梯度张量指针
    size_t num_grad_acts;        // 激活值梯度总数
    float* grads_acts_memory;    // GPU上的激活值梯度内存
    
    // ============ 运行状态 ============
    int batch_size;       // 当前前向传播的批次大小 B
    int seq_len;          // 当前前向传播的序列长度 T
    int* inputs;          // GPU上的输入token IDs，形状 (B, T)
    int* targets;         // GPU上的目标token IDs，形状 (B, T)
    float mean_loss;      // 前向传播后的平均损失，-1表示未计算
    float* cpu_losses;    // CPU端的损失数组 (用cudaMallocHost分配，用于快速拷贝)
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "Bad magic model file\n"); exit(EXIT_FAILURE); }
    if (model_header[1] != 3) {
        // was bumped from 1 -> 3 to incorporate the padded vocab size
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // read in all the parameters from file and copy them to device
    float* params_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    free(params_memory_cpu);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

/*
 * ============================================================================
 * GPT-2 前向传播
 * ============================================================================
 * 
 * 【数据流】
 * inputs (B,T) → Embedding → L层Transformer Block → Final LN → LM Head → logits
 * 
 * 【每层Transformer Block】
 * residual → LN1 → Attention → +residual → LN2 → MLP → +residual
 * 
 * 【参数说明】
 * @param model   GPT2模型结构体
 * @param inputs  输入token IDs，CPU上的数组，形状 (B, T)
 * @param targets 目标token IDs，CPU上的数组，形状 (B, T)，可为NULL
 * @param B       批次大小
 * @param T       序列长度
 */
void gpt2_forward(GPT2 *model, int* inputs, int* targets, int B, int T) {
    // targets可选，如果为NULL则不计算损失

    // 检查模型是否已初始化
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // 提取配置参数
    int V = model->config.vocab_size;         // 词汇表大小
    int Vp = model->config.padded_vocab_size; // 填充后的词汇表大小
    int L = model->config.num_layers;         // 层数
    int NH = model->config.num_heads;         // 注意力头数
    int C = model->config.channels;           // 通道数

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        printf("allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20); // >> 20 is /(1024*1024)
        // also create memory for caching inputs and targets
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    if (targets != NULL) {
        cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        float* scratch = acts.output;

        // ============ 执行前向传播 ============
        // Step 1: LayerNorm1
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        
        // Step 2: QKV投影 (C → 3C)
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        
        // Step 3: 多头自注意力
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        
        // Step 4: 注意力输出投影 (C → C)
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        
        // Step 5: 第一个残差连接
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        
        // Step 6: LayerNorm2
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        
        // Step 7: MLP扩展层 (C → 4C)
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        
        // Step 8: GELU激活
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        
        // Step 9: MLP收缩层 (4C → C)
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        
        // Step 10: 第二个残差连接
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        // fused classifier: does the forward pass and first part of the backward pass
        // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->cpu_losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;

    } else {
        // if we don't have targets, we don't have loss
        model->mean_loss = -1.0f;
    }
}

/*
 * 清零所有梯度
 * 在新的训练步骤开始前调用
 */
void gpt2_zero_grad(GPT2 *model) {
    if (model->grads_acts_memory != NULL) { 
        cudaCheck(cudaMemset(model->grads_acts_memory, 0, model->num_grad_acts * sizeof(float))); 
    }
    if (model->grads_memory != NULL) { 
        cudaCheck(cudaMemset(model->grads_memory, 0, model->num_parameters * sizeof(float))); 
    }
}

/*
 * ============================================================================
 * GPT-2 反向传播
 * ============================================================================
 * 
 * 【数据流】
 * 与前向传播相反的顺序，从损失开始，通过链式法则传播梯度
 * dLoss → LM Head → Final LN → L层Transformer Block → Embedding
 * 
 * 【梯度累积策略】
 * - 参数梯度使用 += ，支持梯度累积
 * - 激活值梯度大部分使用 = ，程序更快
 * - 残差流梯度使用 +=
 * 
 * 【内存优化】
 * 反向传播时可以立即丢弃处理完的层的激活值梯度，
 * 因此只需要3个梯度缓冲区，大大节省内存
 */
void gpt2_backward(GPT2 *model) {

    // 检查是否已经执行过带targets的前向传播
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(EXIT_FAILURE);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        // allocate buffers for weight gradients
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes, 1);
        printf("allocated %zu MiB for parameter gradients\n", (model->num_parameters * sizeof(float)) >> 20);
        // we're going to be clever for the activations backward pass. we don't need to exactly
        // mirror the forward pass acrtivations and we will save memory.
        size_t bw_act_sizes[NUM_ACTIVATION_TENSORS];
        GPT2Config cfg = model->config;
        cfg.num_layers = 1; // copy the configuration but override number of layers to 1
        fill_in_grad_act_sizes(bw_act_sizes, model->batch_size, model->seq_len, cfg);
        // count up and allocate the space
        model->grads_acts_memory = malloc_and_point_backward(&model->grads_acts, bw_act_sizes);
        model->num_grad_acts = 0;
        for (int i = 0; i < NUM_BACKWARD_TENSORS; i++) {
            model->num_grad_acts += bw_act_sizes[i];
        }
        printf("allocated %zu MiB for activation gradients\n", (model->num_grad_acts * sizeof(float)) >> 20);
        // init gradients of parameters and activations to zero
        gpt2_zero_grad(model);
    }

    // convenience shortcuts
    int B = model->batch_size;
    int T = model->seq_len;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    GradActTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    matmul_backward(grads_acts.bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp);
    // backward the final layernorm
    float* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    float* dresidual = grads_acts.residual3; // the main buffer holding the gradient in the backward pass
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        // we need a B x T x C buffer; thankfully, the forward activation for lnf isn't needed anymore,
        // so we can co-opt it here.
        float* dl_btc = acts.lnf;
        float* dl_bt4c = grads_acts.bt4c;
        float* dl_preatt = grads_acts.preatt;

        // re-use scratch buffer of the forward pass
        float* scratch = acts.output;

        // backprop this layer
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_bt4c, l_fch, dl_bt4c, B*T*4*C);
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
        // we more B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        float* buffer_a = l_atty;
        float* buffer_b = l_fch;        // this is B x T x 4C, so even larger than what we need

        attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH);
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs, B, T, C);
}

/*
 * ============================================================================
 * GPT-2 参数更新 (AdamW优化器)
 * ============================================================================
 * 
 * 【算法】
 * AdamW优化器，结合momentum、RMSprop和解耦的权重衰减
 * 
 * 【参数说明】
 * @param model         GPT2模型结构体
 * @param learning_rate 学习率
 * @param beta1         一阶矩衰减系数 (通常0.9)
 * @param beta2         二阶矩衰减系数 (通常0.999)
 * @param eps           数值稳定性常数 (通常1e-8)
 * @param weight_decay  权重衰减系数
 * @param t             当前训练步数 (从1开始)
 * 
 * 参考: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
 */
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {

    // 延迟分配优化器状态内存
    if (model->m_memory == NULL) {
        cudaCheck(cudaMalloc((void**)&model->m_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&model->v_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->m_memory, 0, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, model->num_parameters * sizeof(float)));
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    
    // 计算偏差修正项: 1 - β^t
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    
    // 启动AdamW更新kernel
    adamw_kernel2<<<num_blocks, block_size>>>(
        model->params_memory, model->grads_memory, 
        model->m_memory, model->v_memory,
        model->num_parameters,
        learning_rate, beta1, beta2, 
        beta1_correction, beta2_correction, 
        eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

/*
 * 释放模型占用的所有GPU内存
 */
void gpt2_free(GPT2 *model) {
    cudaCheck(cudaFree(model->params_memory));       // 参数
    cudaCheck(cudaFree(model->grads_memory));        // 梯度
    cudaCheck(cudaFree(model->m_memory));            // AdamW一阶矩
    cudaCheck(cudaFree(model->v_memory));            // AdamW二阶矩
    cudaCheck(cudaFree(model->acts_memory));         // 激活值
    cudaCheck(cudaFree(model->grads_acts_memory));   // 激活值梯度
    cudaCheck(cudaFree(model->inputs));              // 输入缓冲区
    cudaCheck(cudaFree(model->targets));             // 目标缓冲区
    cudaFreeHost(model->cpu_losses);                 // CPU端损失数组
}

#ifndef TESTING
// 如果是TESTING模式（见test_gpt2.cu），则跳过下面的main函数

// ============================================================================
// 采样器: 从概率分布中采样token
// ============================================================================

#define GPT2_EOT 50256  // GPT-2的结束符token ID

/*
 * Xorshift随机数生成器
 * 输出32位无符号整数
 * 参考: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
 */
unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

/*
 * 生成[0,1)范围内的随机float32
 */
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

/*
 * 从softmax概率分布中采样
 * 
 * 【算法】
 * 1. 计算归一化常数 sum(exp(logits))
 * 2. 使用累积分布函数(CDF)采样
 * 
 * @param logits  未归一化的logits数组
 * @param n       logits数组长度
 * @param coin    [0,1)范围内的随机数
 * @return        采样得到的索引
 */
int sample_softmax(const float* logits, int n, float coin) {
    // 计算归一化常数
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // 不需要对每个exp(logit)除以norm，直接让coin乘以norm
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;  // 处理舍入误差
}

// ----------------------------------------------------------------------------
// Logger lite, will probably grow/change some over time

typedef struct {
    FILE *logfile;
    int flush_every; // every how many steps to flush the log
} Logger;

void logger_init(Logger *logger, const char *filename) {
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL) { logger->logfile = fopenCheck(filename, "w"); }
}

void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
    }
}

void logger_log_train(Logger *logger, int step, float train_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
        if (step % 10 == 0) { fflush(logger->logfile); }
    }
}

void logger_free(Logger *logger) {
    if (logger->logfile != NULL) { fclose(logger->logfile); }
}

// ----------------------------------------------------------------------------
// CLI, poor man's argparse

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2fp32cu [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {

    // read in the (optional) command line arguments
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* output_log_file = NULL;
    int B = 4; // batch size
    int T = 1024; // sequence length max
    float learning_rate = 3e-4f;
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_file = argv[i+1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); }
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else { error_usage(); }
    }
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Parameter             | Value                                              |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| train data pattern    | %-50s |\n", train_data_pattern);
    printf("| val data pattern      | %-50s |\n", val_data_pattern);
    printf("| output log file       | %-50s |\n", output_log_file == NULL ? "NULL" : output_log_file);
    printf("| batch size B          | %-50d |\n", B);
    printf("| sequence length T     | %-50d |\n", T);
    printf("| learning rate         | %-50f |\n", learning_rate);
    printf("| val_loss_every        | %-50d |\n", val_loss_every);
    printf("| val_max_steps         | %-50d |\n", val_max_steps);
    printf("| sample_every          | %-50d |\n", sample_every);
    printf("| genT                  | %-50d |\n", genT);
    printf("+-----------------------+----------------------------------------------------+\n");

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    printf("| device                | %-50s |\n", deviceProp.name);
    printf("| TF32                  | %-50s |\n", enable_tf32 ? "enabled" : "disabled");
    printf("+-----------------------+----------------------------------------------------+\n");

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    printf("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf("| channels C            | %-50d |\n", model.config.channels);
    printf("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf("+-----------------------+----------------------------------------------------+\n");

    // build DataLoaders for both train and val
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
    int train_num_batches = train_loader.num_tokens / (B*T); // let's do 1 epoch by default for now
    int val_num_batches = val_loader.num_tokens / (B*T);
    if (val_num_batches > val_max_steps) { val_num_batches = val_max_steps; }
    printf("| train_num_batches     | %-50d |\n", train_num_batches);
    printf("| val_num_batches       | %-50d |\n", val_num_batches);
    printf("+-----------------------+----------------------------------------------------+\n");

    // print model parameter allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    printf("allocated %d MiB for model parameters\n", (int)round(model.num_parameters * sizeof(float) / (1024 * 1024)));

    // set up the Logger
    Logger logger;
    logger_init(&logger, output_log_file);

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    float* cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));

    // train
    struct timespec start, end;
    double total_sum_iteration_time_s = 0.0;
    for (int step = 0; step <= train_num_batches; step++) {
        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss
        if (step % val_loss_every == 0 || last_step) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % sample_every == 0 || last_step) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = GPT2_EOT;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                float* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                cudaCheck(cudaMemcpy(cpu_logits, logits, model.config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
                float coin = random_f32(&rng_state);
                int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        cudaCheck(cudaDeviceSynchronize()); // finish all CUDA work to get correct precise timings
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_sum_iteration_time_s += time_elapsed_s;
        int tokens_per_second = (B * T) / time_elapsed_s;
        printf("step %4d/%d: train loss %f (%f ms, %d tok/s)\n", step + 1, train_num_batches, model.mean_loss, time_elapsed_s * 1000, tokens_per_second);
        logger_log_train(&logger, step, model.mean_loss);
    }
    // add a total average, for optimizations that are only mild improvements
    printf("total average iteration time: %f ms\n", total_sum_iteration_time_s / train_num_batches * 1000);

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(cpu_logits);
    free(gen_tokens);
    cublasCheck(cublasDestroy(cublas_handle));
    logger_free(&logger);

    return 0;
}
#endif