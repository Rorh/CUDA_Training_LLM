/*
 * ============================================================================
 * Qwen3 模型 CUDA FP32 训练实现
 * ============================================================================
 * 
 * 【Qwen3 vs GPT-2 主要区别】
 * 1. RMSNorm 代替 LayerNorm (无均值中心化，无bias)
 * 2. SwiGLU MLP: out = down_proj(silu(gate_proj(x)) * up_proj(x))
 * 3. RoPE 旋转位置编码 (无学习的位置嵌入)
 * 4. GQA 分组查询注意力 (num_kv_heads 可能 < num_heads)
 * 5. QK Norm: 在Q/K投影后应用RMSNorm
 * 
 * 【符号约定】
 * B  = batch size, T = seq_len, C = hidden_size, L = num_layers
 * NH = num_heads, NKV = num_kv_heads, HS = head_size = C/NH
 * I  = intermediate_size, V = vocab_size
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// llmc utilities
#include "llmc/utils.h"
#include "llmc/tokenizer.h"
#include "llmc/dataloader.h"

namespace cg = cooperative_groups;

#define cudaCheck(err) { if(err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#endif

cublasHandle_t cublas_handle;

// ============================================================================
// RMSNorm Forward: y = x * rsqrt(mean(x^2) + eps) * weight
// ============================================================================
/**
 * @brief RMSNorm forward kernel
 * @param out      Output tensor, shape [N, C]
 * @param rstd     Reciprocal standard deviation (rsqrt(mean(x^2) + eps)), shape [N]
 * @param inp      Input tensor, shape [N, C]
 * @param weight   Learnable weight parameters, shape [C]
 * @param N        Number of elements (batch_size * seq_len)
 * @param C        Number of channels (hidden dimension)
 */
__global__ void rmsnorm_forward_kernel(float* out, float* rstd, const float* inp,
                                       const float* weight, int N, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); //对应第idx个元素，warp级全局索引
    if (idx >= N) return;
    
    const float* x = inp + idx * C;
    float sum_sq = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum_sq += x[i] * x[i];
    }
    sum_sq = cg::reduce(warp, sum_sq, cg::plus<float>{});
    
    float s = rsqrtf(sum_sq / C + 1e-6f);
    if (warp.thread_rank() == 0 && rstd) rstd[idx] = s;
    
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        o[c] = x[c] * s * weight[c];
    }
}

// ============================================================================
// Embedding Forward (Token only, no position - Qwen3 uses RoPE)
// ============================================================================
/**
 * @brief Token embedding forward kernel
 * @param out  Output embeddings, shape [B, T, C]
 * @param inp  Input token indices, shape [B, T]
 * @param wte  Token embedding weights, shape [vocab_size, C]
 * @param B    Batch size
 * @param T    Sequence length
 * @param C    Embedding dimension (channels)
 */
__global__ void embedding_forward_kernel(float* out, const int* inp, const float* wte, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * C) {
        int bt = idx / C, c = idx % C;
        out[idx] = wte[inp[bt] * C + c];
    }
}

// ============================================================================
// RoPE: Rotary Position Embedding
// y = x * cos + rotate_half(x) * sin
// ============================================================================
// RoPE（Rotary Positional Embedding）前向 CUDA kernel
// 对输入 inp 中的每一个元素，应用旋转位置编码
__global__ void rope_forward_kernel(
    float* out,              // 输出张量，形状 [B, NH, T, HS]
    const float* inp,        // 输入张量，形状 [B, NH, T, HS]
    const float* cos_cache,  // cos 缓存，形状 [T, HS]
    const float* sin_cache,  // sin 缓存，形状 [T, HS]
    int B,                   // batch size
    int NH,                  // number of heads
    int T,                   // 序列长度（token 数）
    int HS                   // 每个 head 的 hidden size
) {
    // --------------------------------------------------
    // 1️⃣ 计算全局线程索引（thread-level）
    // --------------------------------------------------
    // 每个线程负责处理一个标量元素（一个 float）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 总元素数 = B * NH * T * HS
    if (idx >= B * NH * T * HS) return;

    // --------------------------------------------------
    // 2️⃣ 将一维 idx 映射回 (t, hs)
    // --------------------------------------------------
    // 张量布局假设为：
    // idx = (((b * NH + nh) * T + t) * HS + hs)

    // 当前 token 位置 t
    // 先除以 HS 去掉最内层，再对 T 取模
    int t = (idx / HS) % T;

    // 当前 head 内的维度索引
    int hs = idx % HS;

    // head 维度一分为二（RoPE 的旋转对）
    int half = HS / 2;

    // --------------------------------------------------
    // 3️⃣ 读取当前位置的 cos / sin
    // --------------------------------------------------
    // cos_cache 和 sin_cache 按 [T, HS] 存储
    // 每个 token、每个维度都有对应的旋转角
    float cos_val = cos_cache[t * HS + hs];
    float sin_val = sin_cache[t * HS + hs];

    // 当前输入值
    float x = inp[idx];

    // --------------------------------------------------
    // 4️⃣ 计算 rotate_half(x)
    // --------------------------------------------------
    // RoPE 中的 rotate_half 定义：
    //   x = [x0, x1, ..., x_{half-1}, x_half, ..., x_{HS-1}]
    //   rotate_half(x) = [-x_half, ..., -x_{HS-1}, x0, ..., x_{half-1}]

    // base 是当前 (b, nh, t) 起始位置
    // 用来在同一个 token + head 内访问其它维度
    int base = idx - hs;

    // 根据当前维度 hs 决定旋转来源
    float x_rot = (hs < half)
        // 前半部分：取后半部分对应维度，并取负号
        ? -inp[base + hs + half]
        // 后半部分：取前半部分对应维度
        :  inp[base + hs - half];

    // --------------------------------------------------
    // 5️⃣ 应用 RoPE 旋转公式
    // --------------------------------------------------
    // RoPE 的核心公式（二维旋转）：
    //   y = x * cos(theta) + rotate_half(x) * sin(theta)
    out[idx] = x * cos_val + x_rot * sin_val;
}


// ============================================================================
// Permute Q/K/V for attention: (B,T,NH,HS) -> (B,NH,T,HS)
// ============================================================================
/**
 * @brief Permute tensor from (B,T,NH,HS) to (B,NH,T,HS) for efficient attention computation
 * @param out  Output tensor, shape [B, NH, T, HS]
 * @param inp  Input tensor, shape [B, T, NH, HS]
 * @param B    Batch size
 * @param T    Sequence length
 * @param NH   Number of attention heads
 * @param HS   Head size (dimension per head)
 */
__global__ void permute_kernel(float* out, const float* inp, int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T * HS) return;
    
    int b = idx / (NH * T * HS);
    int nh = (idx / (T * HS)) % NH;
    int t = (idx / HS) % T;
    int hs = idx % HS;
    
    out[idx] = inp[b * T * NH * HS + t * NH * HS + nh * HS + hs];
}

// ============================================================================
// Repeat KV for GQA: (B,NKV,T,HS) -> (B,NH,T,HS)
// ============================================================================
/**
 * @brief Repeat K/V heads for Grouped Query Attention (GQA)
 * @param out  Output tensor with repeated heads, shape [B, NH, T, HS]
 * @param inp  Input tensor with fewer KV heads, shape [B, NKV, T, HS]
 * @param B    Batch size
 * @param NH   Number of query heads
 * @param NKV  Number of key/value heads (NKV < NH for GQA)
 * @param T    Sequence length
 * @param HS   Head size (dimension per head)
 */
__global__ void repeat_kv_kernel(float* out, const float* inp, int B, int NH, int NKV, int T, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T * HS) return;
    
    int b = idx / (NH * T * HS);
    int nh = (idx / (T * HS)) % NH;
    int t = (idx / HS) % T;
    int hs = idx % HS;
    
    int kv_head = nh / (NH / NKV);
    out[idx] = inp[b * NKV * T * HS + kv_head * T * HS + t * HS + hs];
}

// ============================================================================
// Softmax (Causal): softmax over valid positions only
// ============================================================================
/**
 * @brief Causal softmax forward kernel (autoregressive masking)
 * @param out    Output softmax probabilities, shape [B, NH, T, T]
 * @param inp    Input attention scores, shape [B, NH, T, T]
 * @param B      Batch size
 * @param NH     Number of attention heads
 * @param T      Sequence length
 * @param scale  Scaling factor (typically 1/sqrt(head_size))
 */
__global__ void softmax_forward_kernel(float* out, const float* inp, int B, int NH, int T, float scale) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * NH * T) return;
    
    int t = idx % T;
    const float* x = inp + idx * T;
    float* y = out + idx * T;
    
    float maxval = -INFINITY;
    for (int i = warp.thread_rank(); i <= t; i += warp.size())
        maxval = fmaxf(maxval, x[i] * scale);
    maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    
    float sumval = 0.0f;
    for (int i = warp.thread_rank(); i <= t; i += warp.size())
        sumval += expf(x[i] * scale - maxval);
    sumval = cg::reduce(warp, sumval, cg::plus<float>{});
    
    for (int i = warp.thread_rank(); i < T; i += warp.size())
        y[i] = (i <= t) ? expf(x[i] * scale - maxval) / sumval : 0.0f;
}

// ============================================================================
// SwiGLU: out = silu(gate) * up, where silu(x) = x * sigmoid(x)
// ============================================================================
/**
 * @brief SwiGLU activation function forward kernel
 * @param out   Output tensor, shape [N]
 * @param gate  Gate projection tensor, shape [N]
 * @param up    Up projection tensor, shape [N]
 * @param N     Total number of elements (B * T * intermediate_size)
 */
__global__ void swiglu_forward_kernel(float* out, const float* gate, const float* up, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// ============================================================================
// Residual: out = inp1 + inp2
// ============================================================================
/**
 * @brief Residual connection forward kernel (element-wise addition)
 * @param out   Output tensor, shape [N]
 * @param inp1  First input tensor, shape [N]
 * @param inp2  Second input tensor, shape [N]
 * @param N     Total number of elements
 */
__global__ void residual_forward_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = inp1[i] + inp2[i];
}

// ============================================================================
// AdamW Optimizer with Gradient Clipping
// ============================================================================
/**
 * @brief AdamW optimizer kernel with gradient clipping
 * @param params    Model parameters to update, shape [N]
 * @param grads     Gradients, shape [N]
 * @param m         First moment (momentum), shape [N]
 * @param v         Second moment (variance), shape [N]
 * @param N         Total number of parameters
 * @param lr        Learning rate
 * @param beta1     Exponential decay rate for first moment (typically 0.9)
 * @param beta2     Exponential decay rate for second moment (typically 0.999)
 * @param b1_corr   Bias correction for first moment: 1 - beta1^t
 * @param b2_corr   Bias correction for second moment: 1 - beta2^t
 * @param eps       Small constant for numerical stability (typically 1e-8)
 * @param wd        Weight decay coefficient
 */
__global__ void adamw_kernel(float* params, const float* grads, float* m, float* v,
                             int N, float lr, float beta1, float beta2,
                             float b1_corr, float b2_corr, float eps, float wd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = grads[i];
        // 梯度裁剪: 限制在 [-1, 1] 范围内
        g = fmaxf(-1.0f, fminf(1.0f, g));
        m[i] = beta1 * m[i] + (1 - beta1) * g;
        v[i] = beta2 * v[i] + (1 - beta2) * g * g;
        float m_hat = m[i] / b1_corr;
        float v_hat = v[i] / b2_corr;
        params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * params[i]);
    }
}

// ============================================================================
// Fused Classifier: Softmax + CrossEntropy + Gradient
// ============================================================================
/**
 * @brief Fused classifier kernel computing softmax, cross-entropy loss, and gradient
 * @param logits   Input logits and output gradients (in-place), shape [B, T, Vp]
 * @param losses   Output cross-entropy losses, shape [B, T]
 * @param targets  Target token indices, shape [B, T]
 * @param B        Batch size
 * @param T        Sequence length
 * @param V        Actual vocabulary size
 * @param Vp       Padded vocabulary size (for memory alignment)
 */
__global__ void fused_classifier_kernel(float* logits, float* losses, const int* targets,
                                        int B, int T, int V, int Vp) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) return;
    
    float* x = logits + idx * Vp;
    int target = targets[idx];
    
    // 1. 找最大值
    float maxval = -INFINITY;
    for (int i = warp.thread_rank(); i < V; i += warp.size())
        maxval = fmaxf(maxval, x[i]);
    maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    
    // 2. 计算softmax分母
    float sumval = 0.0f;
    for (int i = warp.thread_rank(); i < V; i += warp.size())
        sumval += expf(x[i] - maxval);
    sumval = cg::reduce(warp, sumval, cg::plus<float>{});
    
    // 3. 先计算loss (在修改logits之前)
    if (warp.thread_rank() == 0) {
        float prob_target = expf(x[target] - maxval) / sumval;
        losses[idx] = -logf(prob_target + 1e-10f);
    }
    
    // 4. 计算梯度并写入logits
    for (int i = warp.thread_rank(); i < Vp; i += warp.size()) {
        float prob = (i < V) ? expf(x[i] - maxval) / sumval : 0.0f;
        x[i] = (prob - (i == target ? 1.0f : 0.0f)) / (B * T);
    }
}

// ============================================================================
// RMSNorm Backward
// dinp = dout * weight * rstd - x * rstd^3 * mean(dout * weight * x) / C
// ============================================================================
/**
 * @brief RMSNorm backward kernel
 * @param dinp     Gradient w.r.t. input, shape [B, T, C]
 * @param dweight  Gradient w.r.t. weight (accumulated), shape [C]
 * @param dout     Gradient w.r.t. output, shape [B, T, C]
 * @param inp      Forward pass input, shape [B, T, C]
 * @param weight   Forward pass weight, shape [C]
 * @param rstd     Reciprocal standard deviation from forward pass, shape [B, T]
 * @param B        Batch size
 * @param T        Sequence length
 * @param C        Number of channels
 */
__global__ void rmsnorm_backward_kernel(float* dinp, float* dweight, const float* dout,
                                        const float* inp, const float* weight, const float* rstd,
                                        int B, int T, int C) {
    extern __shared__ float shared[];
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if (idx >= N) return;
    
    const float* dout_bt = dout + idx * C;
    const float* inp_bt = inp + idx * C;
    float* dinp_bt = dinp + idx * C;
    float s = rstd[idx];
    
    float* dweight_shared = shared;
    for (int i = threadIdx.x; i < C; i += blockDim.x) dweight_shared[i] = 0.0f;
    __syncthreads();
    
    // 计算 sum(dout * weight * x)
    float sum_dwx = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum_dwx += dout_bt[i] * weight[i] * inp_bt[i];
    }
    sum_dwx = cg::reduce(warp, sum_dwx, cg::plus<float>{});
    sum_dwx = sum_dwx * s * s / C;
    
    // 计算梯度
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float dw = dout_bt[i] * weight[i];
        dinp_bt[i] = s * dw - inp_bt[i] * s * sum_dwx;
        atomicAdd(&dweight_shared[i], dout_bt[i] * inp_bt[i] * s);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        atomicAdd(&dweight[i], dweight_shared[i]);
    }
}

// ============================================================================
// Embedding Backward
// ============================================================================
/**
 * @brief Token embedding backward kernel
 * @param dwte  Gradient w.r.t. embedding weights (accumulated), shape [vocab_size, C]
 * @param dout  Gradient w.r.t. output, shape [B, T, C]
 * @param inp   Token indices from forward pass, shape [B, T]
 * @param B     Batch size
 * @param T     Sequence length
 * @param C     Embedding dimension
 */
__global__ void embedding_backward_kernel(float* dwte, const float* dout, const int* inp,
                                          int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * C) {
        int bt = idx / C, c = idx % C;
        atomicAdd(&dwte[inp[bt] * C + c], dout[idx]);
    }
}

// ============================================================================
// RoPE Backward
// ============================================================================
/**
 * @brief RoPE (Rotary Position Embedding) backward kernel
 * @param dinp       Gradient w.r.t. input, shape [B, NH, T, HS]
 * @param dout       Gradient w.r.t. output, shape [B, NH, T, HS]
 * @param cos_cache  Cosine cache from forward pass, shape [T, HS]
 * @param sin_cache  Sine cache from forward pass, shape [T, HS]
 * @param B          Batch size
 * @param NH         Number of attention heads
 * @param T          Sequence length
 * @param HS         Head size
 */
__global__ void rope_backward_kernel(float* dinp, const float* dout, const float* cos_cache,
                                     const float* sin_cache, int B, int NH, int T, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T * HS) return;
    
    int t = (idx / HS) % T;
    int hs = idx % HS;
    int half = HS / 2;
    
    float cos_val = cos_cache[t * HS + hs];
    float sin_val = sin_cache[t * HS + hs];
    float dy = dout[idx];
    
    int base = idx - hs;
    float dy_rot = (hs < half) ? dout[base + hs + half] : -dout[base + hs - half];
    dinp[idx] = dy * cos_val + dy_rot * sin_val;
}

// ============================================================================
// Unpermute: (B,NH,T,HS) -> (B,T,NH,HS) for attention output
// ============================================================================
/**
 * @brief Unpermute tensor from (B,NH,T,HS) to (B,T,NH,HS)
 * @param out  Output tensor, shape [B, T, NH, HS]
 * @param inp  Input tensor, shape [B, NH, T, HS]
 * @param B    Batch size
 * @param T    Sequence length
 * @param NH   Number of attention heads
 * @param HS   Head size
 */
__global__ void unpermute_kernel(float* out, const float* inp, int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * NH * HS) return;
    
    int b = idx / (T * NH * HS);
    int t = (idx / (NH * HS)) % T;
    int nh = (idx / HS) % NH;
    int hs = idx % HS;
    
    out[idx] = inp[b * NH * T * HS + nh * T * HS + t * HS + hs];
}

// ============================================================================
// Unpermute Backward: (B,T,NH,HS) -> (B,NH,T,HS)
// ============================================================================
/**
 * @brief Unpermute backward kernel (reverse of unpermute)
 * @param dinp  Gradient w.r.t. input, shape [B, NH, T, HS]
 * @param dout  Gradient w.r.t. output, shape [B, T, NH, HS]
 * @param B     Batch size
 * @param T     Sequence length
 * @param NH    Number of attention heads
 * @param HS    Head size
 */
__global__ void unpermute_backward_kernel(float* dinp, const float* dout, int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T * HS) return;
    
    int b = idx / (NH * T * HS);
    int nh = (idx / (T * HS)) % NH;
    int t = (idx / HS) % T;
    int hs = idx % HS;
    
    dinp[idx] = dout[b * T * NH * HS + t * NH * HS + nh * HS + hs];
}

// ============================================================================
// Softmax Backward (Autoregressive)
// dpreatt = scale * att * (datt - sum(att * datt))
// ============================================================================
/**
 * @brief Causal softmax backward kernel
 * @param dpreatt  Gradient w.r.t. pre-attention scores, shape [B, NH, T, T]
 * @param datt     Gradient w.r.t. attention probabilities, shape [B, NH, T, T]
 * @param att      Attention probabilities from forward pass, shape [B, NH, T, T]
 * @param B        Batch size
 * @param NH       Number of attention heads
 * @param T        Sequence length
 * @param scale    Scaling factor (typically 1/sqrt(head_size))
 */
__global__ void softmax_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                        int B, int NH, int T, float scale) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * NH * T) return;
    
    int t = idx % T;
    const float* att_row = att + idx * T;
    const float* datt_row = datt + idx * T;
    float* dpreatt_row = dpreatt + idx * T;
    
    float sum_ad = 0.0f;
    for (int i = warp.thread_rank(); i <= t; i += warp.size())
        sum_ad += att_row[i] * datt_row[i];
    sum_ad = cg::reduce(warp, sum_ad, cg::plus<float>{});
    
    for (int i = warp.thread_rank(); i < T; i += warp.size())
        dpreatt_row[i] = (i <= t) ? scale * att_row[i] * (datt_row[i] - sum_ad) : 0.0f;
}

// ============================================================================
// SwiGLU Backward
// d_gate = d_out * up * silu'(gate)
// d_up = d_out * silu(gate)
// ============================================================================
/**
 * @brief SwiGLU activation backward kernel
 * @param dgate  Gradient w.r.t. gate projection, shape [N]
 * @param dup    Gradient w.r.t. up projection, shape [N]
 * @param dout   Gradient w.r.t. output, shape [N]
 * @param gate   Gate values from forward pass, shape [N]
 * @param up     Up projection values from forward pass, shape [N]
 * @param N      Total number of elements
 */
__global__ void swiglu_backward_kernel(float* dgate, float* dup, const float* dout,
                                       const float* gate, const float* up, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sig;
        float silu_grad = sig * (1.0f + g * (1.0f - sig));
        dgate[i] = dout[i] * up[i] * silu_grad;
        dup[i] = dout[i] * silu_g;
    }
}

// ============================================================================
// MatMul Backward Bias
// ============================================================================
/**
 * @brief Matrix multiplication backward kernel for bias gradient
 * @param dbias  Gradient w.r.t. bias (accumulated), shape [OC]
 * @param dout   Gradient w.r.t. output, shape [B, T, OC]
 * @param B      Batch size
 * @param T      Sequence length
 * @param OC     Output channels (dimension)
 */
__global__ void matmul_backward_bias_kernel(float* dbias, const float* dout, int B, int T, int OC) {
    extern __shared__ float smem[];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int col = blockIdx.x * 32 + lane_id;
    int vstep = blockDim.x / 32;
    
    const float* dout_col = dout + col;
    float sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep)
        sum += dout_col[row * OC];
    smem[lane_id + warp_id * 32] = sum;
    __syncthreads();
    
    if (warp_id == 0) {
        sum = 0.0f;
        for (int j = 0; j < vstep; j++) sum += smem[lane_id + j * 32];
        if (col < OC) dbias[col] += sum;
    }
}

// ============================================================================
// Qwen3 Config
// ============================================================================
typedef struct {
    int max_seq_len, vocab_size, padded_vocab_size;
    int num_layers, num_heads, num_kv_heads;
    int channels, head_size, intermediate_size;
    float rope_theta, rms_norm_eps;
} Qwen3Config;

// ============================================================================
// Parameter Tensors (Qwen3特有: 无bias, 使用RMSNorm)
// ============================================================================
#define NUM_PARAMETER_TENSORS 13
typedef struct {
    float* wte;           // (V, C) token embedding
    // 每层参数
    float* ln1w;          // (L, C) input RMSNorm
    float* qw;            // (L, C, C) Q projection
    float* kw;            // (L, C, C_kv) K projection
    float* vw;            // (L, C, C_kv) V projection  
    float* q_norm_w;      // (L, HS) Q head norm
    float* k_norm_w;      // (L, HS) K head norm
    float* attn_out_w;    // (L, C, C) attention output
    float* ln2w;          // (L, C) post-attention RMSNorm
    float* gate_w;        // (L, C, I) SwiGLU gate
    float* up_w;          // (L, C, I) SwiGLU up
    float* down_w;        // (L, I, C) SwiGLU down
    float* lnfw;          // (C,) final RMSNorm
} ParameterTensors;

// ============================================================================
// Activation Tensors
// ============================================================================
#define NUM_ACTIVATION_TENSORS 22
typedef struct {
    float* encoded;       // (B, T, C)
    float* ln1;           // (L, B, T, C)
    float* ln1_rstd;      // (L, B, T)
    float* q;             // (L, B, T, C)
    float* k;             // (L, B, T, C_kv)
    float* v;             // (L, B, T, C_kv)
    float* q_rope;        // (L, B, NH, T, HS) after RoPE
    float* k_rope;        // (L, B, NKV, T, HS) after RoPE
    float* k_rep;         // (L, B, NH, T, HS) after repeat for GQA
    float* v_rep;         // (L, B, NH, T, HS) after repeat for GQA
    float* att;           // (L, B, NH, T, T)
    float* atty;          // (L, B, T, C)
    float* attproj;       // (L, B, T, C)
    float* residual2;     // (L, B, T, C)
    float* ln2;           // (L, B, T, C)
    float* ln2_rstd;      // (L, B, T)
    float* gate;          // (L, B, T, I)
    float* up;            // (L, B, T, I)
    float* mlp_out;       // (L, B, T, C)
    float* lnf;           // (B, T, C) final RMSNorm output
    float* lnf_rstd;      // (B, T) final RMSNorm rstd
    float* output;        // (B, T, V)
    float* losses;        // (B, T)
} ActivationTensors;

// ============================================================================
// Qwen3 Model
// ============================================================================
typedef struct {
    Qwen3Config config;
    ParameterTensors params;
    ParameterTensors grads;
    ActivationTensors acts;
    float* params_memory;
    float* grads_memory;
    float* acts_memory;
    float* m_memory;      // AdamW first moment
    float* v_memory;      // AdamW second moment
    float* cos_cache;     // RoPE cos cache
    float* sin_cache;     // RoPE sin cache
    int* inputs;
    int* targets;
    size_t num_parameters;
    float mean_loss;
    int batch_size;
    int seq_len;
} Qwen3;

// ============================================================================
// Initialize RoPE Cache
// ============================================================================
/**
 * @brief Initialize RoPE (Rotary Position Embedding) cache on GPU
 * @param cos_cache  Cosine cache to initialize, shape [T, HS]
 * @param sin_cache  Sine cache to initialize, shape [T, HS]
 * @param T          Maximum sequence length
 * @param HS         Head size (dimension per head)
 * @param theta      Base frequency for RoPE (typically 10000.0)
 */
void init_rope_cache(float* cos_cache, float* sin_cache, int T, int HS, float theta) {
    float* cos_h = (float*)malloc(T * HS * sizeof(float));
    float* sin_h = (float*)malloc(T * HS * sizeof(float));
    int half = HS / 2;
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, 2.0f * i / HS);
            float angle = t * freq;
            cos_h[t * HS + i] = cos_h[t * HS + i + half] = cosf(angle);
            sin_h[t * HS + i] = sin_h[t * HS + i + half] = sinf(angle);
        }
    }
    cudaMemcpy(cos_cache, cos_h, T * HS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sin_cache, sin_h, T * HS * sizeof(float), cudaMemcpyHostToDevice);
    free(cos_h); free(sin_h);
}

// ============================================================================
// Parameter Size Calculation
// ============================================================================
/**
 * @brief Calculate memory size for each parameter tensor
 * @param sizes  Output array to store sizes for each parameter tensor
 * @param cfg    Model configuration
 */
void fill_param_sizes(size_t* sizes, Qwen3Config* cfg) {
    int L = cfg->num_layers, C = cfg->channels, V = cfg->padded_vocab_size;
    int HS = cfg->head_size, I = cfg->intermediate_size;
    int NKV = cfg->num_kv_heads, C_kv = NKV * HS;
    
    sizes[0] = V * C;           // wte
    sizes[1] = L * C;           // ln1w
    sizes[2] = L * C * C;       // qw
    sizes[3] = L * C * C_kv;    // kw
    sizes[4] = L * C * C_kv;    // vw
    sizes[5] = L * HS;          // q_norm_w
    sizes[6] = L * HS;          // k_norm_w
    sizes[7] = L * C * C;       // attn_out_w
    sizes[8] = L * C;           // ln2w
    sizes[9] = L * C * I;       // gate_w
    sizes[10] = L * C * I;      // up_w
    sizes[11] = L * I * C;      // down_w
    sizes[12] = C;              // lnfw
}

// ============================================================================
// Allocate and Point Parameters
// ============================================================================
/**
 * @brief Allocate memory for all parameters and set up pointers
 * @param p          Parameter tensor struct to populate with pointers
 * @param sizes      Array of sizes for each parameter tensor
 * @param on_device  1 to allocate on GPU, 0 to allocate on CPU
 * @return           Pointer to allocated memory block
 */
float* malloc_and_point_params(ParameterTensors* p, size_t* sizes, int on_device) {
    size_t total = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) total += sizes[i];
    
    float* mem;
    if (on_device) cudaMalloc(&mem, total * sizeof(float));
    else mem = (float*)malloc(total * sizeof(float));
    
    float** ptrs[] = {&p->wte, &p->ln1w, &p->qw, &p->kw, &p->vw, &p->q_norm_w, &p->k_norm_w,
                      &p->attn_out_w, &p->ln2w, &p->gate_w, &p->up_w, &p->down_w, &p->lnfw};
    float* iter = mem;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *ptrs[i] = iter;
        iter += sizes[i];
    }
    return mem;
}

// ============================================================================
// Activation Size Calculation
// ============================================================================
/**
 * @brief Calculate memory size for each activation tensor
 * @param sizes  Output array to store sizes for each activation tensor
 * @param B      Batch size
 * @param T      Sequence length
 * @param cfg    Model configuration
 */
void fill_act_sizes(size_t* sizes, int B, int T, Qwen3Config* cfg) {
    int L = cfg->num_layers, C = cfg->channels, V = cfg->padded_vocab_size;
    int NH = cfg->num_heads, NKV = cfg->num_kv_heads, HS = cfg->head_size, I = cfg->intermediate_size;
    int C_kv = NKV * HS;
    
    sizes[0] = B * T * C;               // encoded
    sizes[1] = L * B * T * C;           // ln1
    sizes[2] = L * B * T;               // ln1_rstd
    sizes[3] = L * B * T * C;           // q
    sizes[4] = L * B * T * C_kv;        // k
    sizes[5] = L * B * T * C_kv;        // v
    sizes[6] = L * B * NH * T * HS;     // q_rope
    sizes[7] = L * B * NKV * T * HS;    // k_rope
    sizes[8] = L * B * NH * T * HS;     // k_rep
    sizes[9] = L * B * NH * T * HS;     // v_rep
    sizes[10] = L * B * NH * T * T;     // att
    sizes[11] = L * B * T * C;          // atty
    sizes[12] = L * B * T * C;          // attproj
    sizes[13] = L * B * T * C;          // residual2
    sizes[14] = L * B * T * C;          // ln2
    sizes[15] = L * B * T;              // ln2_rstd
    sizes[16] = L * B * T * I;          // gate
    sizes[17] = L * B * T * I;          // up
    sizes[18] = L * B * T * C;          // mlp_out
    sizes[19] = B * T * C;              // lnf
    sizes[20] = B * T;                  // lnf_rstd
    sizes[21] = B * T * V;              // output
}

// ============================================================================
// Forward Pass
// ============================================================================
/**
 * @brief Execute forward pass of Qwen3 model
 * @param model    Qwen3 model instance
 * @param inputs   Input token indices, shape [B, T]
 * @param targets  Target token indices for loss computation (can be NULL for inference), shape [B, T]
 * @param B        Batch size
 * @param T        Sequence length
 */
void qwen3_forward(Qwen3* model, int* inputs, int* targets, int B, int T) {
    Qwen3Config* cfg = &model->config;
    ParameterTensors* p = &model->params;
    ActivationTensors* a = &model->acts;
    
    int C = cfg->channels, L = cfg->num_layers, NH = cfg->num_heads;
    int NKV = cfg->num_kv_heads, HS = cfg->head_size, I = cfg->intermediate_size;
    int V = cfg->vocab_size, Vp = cfg->padded_vocab_size;
    int C_kv = NKV * HS;
    
    // Copy inputs
    cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice);
    if (targets) cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice);
    
    // Embedding: (B, T) -> (B, T, C)
    embedding_forward_kernel<<<CEIL_DIV(B*T*C, 256), 256>>>(a->encoded, model->inputs, p->wte, B, T, C);
    
    float* residual = a->encoded;
    float alpha = 1.0f, beta = 0.0f;
    
    for (int l = 0; l < L; l++) {
        // 获取当前层的指针偏移
        float* ln1 = a->ln1 + l * B * T * C;
        float* ln1_rstd = a->ln1_rstd + l * B * T;
        float* q = a->q + l * B * T * C;
        float* k = a->k + l * B * T * C_kv;
        float* v = a->v + l * B * T * C_kv;
        float* q_rope = a->q_rope + l * B * NH * T * HS;
        float* k_rope = a->k_rope + l * B * NKV * T * HS;
        float* k_rep = a->k_rep + l * B * NH * T * HS;
        float* v_rep = a->v_rep + l * B * NH * T * HS;
        float* att = a->att + l * B * NH * T * T;
        float* atty = a->atty + l * B * T * C;
        float* attproj = a->attproj + l * B * T * C;
        float* residual2 = a->residual2 + l * B * T * C;
        float* ln2 = a->ln2 + l * B * T * C;
        float* ln2_rstd = a->ln2_rstd + l * B * T;
        float* gate = a->gate + l * B * T * I;
        float* up = a->up + l * B * T * I;
        float* mlp_out = a->mlp_out + l * B * T * C;
        
        float* ln1w = p->ln1w + l * C;
        float* qw = p->qw + l * C * C;
        float* kw = p->kw + l * C * C_kv;
        float* vw = p->vw + l * C * C_kv;
        float* attn_out_w = p->attn_out_w + l * C * C;
        float* ln2w = p->ln2w + l * C;
        float* gate_w = p->gate_w + l * C * I;
        float* up_w = p->up_w + l * C * I;
        float* down_w = p->down_w + l * I * C;
        
        // 1. Input RMSNorm
        rmsnorm_forward_kernel<<<CEIL_DIV(B*T, 8), 256>>>(ln1, ln1_rstd, residual, ln1w, B*T, C);
        
        // 2. Q/K/V projections
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, C, B*T, C, &alpha, qw, C, ln1, C, &beta, q, C);
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, C_kv, B*T, C, &alpha, kw, C, ln1, C, &beta, k, C_kv);
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, C_kv, B*T, C, &alpha, vw, C, ln1, C, &beta, v, C_kv);
        
        // 3. Permute Q/K/V: (B,T,NH,HS) -> (B,NH,T,HS)
        permute_kernel<<<CEIL_DIV(B*NH*T*HS, 256), 256>>>(q_rope, q, B, T, NH, HS);
        permute_kernel<<<CEIL_DIV(B*NKV*T*HS, 256), 256>>>(k_rope, k, B, T, NKV, HS);
        
        // 4. Apply RoPE
        rope_forward_kernel<<<CEIL_DIV(B*NH*T*HS, 256), 256>>>(q_rope, q_rope, model->cos_cache, model->sin_cache, B, NH, T, HS);
        rope_forward_kernel<<<CEIL_DIV(B*NKV*T*HS, 256), 256>>>(k_rope, k_rope, model->cos_cache, model->sin_cache, B, NKV, T, HS);
        
        // 5. Repeat K/V for GQA
        repeat_kv_kernel<<<CEIL_DIV(B*NH*T*HS, 256), 256>>>(k_rep, k_rope, B, NH, NKV, T, HS);
        permute_kernel<<<CEIL_DIV(B*NKV*T*HS, 256), 256>>>(v_rep, v, B, T, NKV, HS);
        repeat_kv_kernel<<<CEIL_DIV(B*NH*T*HS, 256), 256>>>(v_rep, v_rep, B, NH, NKV, T, HS);
        
        // 6. Attention: Q @ K.T
        float scale = 1.0f / sqrtf(HS);
        cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha,
                                  k_rep, HS, T*HS, q_rope, HS, T*HS, &beta, att, T, T*T, B*NH);
        
        // 7. Causal Softmax
        softmax_forward_kernel<<<CEIL_DIV(B*NH*T, 8), 256>>>(att, att, B, NH, T, scale);
        
        // 8. Att @ V
        cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha,
                                  v_rep, HS, T*HS, att, T, T*T, &beta, atty, HS, T*HS, B*NH);
        
        // 9. Unpermute and project
        unpermute_kernel<<<CEIL_DIV(B*T*C, 256), 256>>>(atty, atty, B, T, NH, HS);
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, C, B*T, C, &alpha, attn_out_w, C, atty, C, &beta, attproj, C);
        
        // 10. Residual
        residual_forward_kernel<<<CEIL_DIV(B*T*C, 256), 256>>>(residual2, residual, attproj, B*T*C);
        
        // 11. Post-attention RMSNorm
        rmsnorm_forward_kernel<<<CEIL_DIV(B*T, 8), 256>>>(ln2, ln2_rstd, residual2, ln2w, B*T, C);
        
        // 12. SwiGLU MLP: gate_proj, up_proj
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, I, B*T, C, &alpha, gate_w, C, ln2, C, &beta, gate, I);
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, I, B*T, C, &alpha, up_w, C, ln2, C, &beta, up, I);
        
        // 13. SwiGLU activation
        swiglu_forward_kernel<<<CEIL_DIV(B*T*I, 256), 256>>>(gate, gate, up, B*T*I);
        
        // 14. down_proj
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, C, B*T, I, &alpha, down_w, I, gate, I, &beta, mlp_out, C);
        
        // 15. Residual
        residual_forward_kernel<<<CEIL_DIV(B*T*C, 256), 256>>>(mlp_out, residual2, mlp_out, B*T*C);
        
        residual = mlp_out;
    }
    
    // Final RMSNorm
    rmsnorm_forward_kernel<<<CEIL_DIV(B*T, 8), 256>>>(a->lnf, a->lnf_rstd, residual, p->lnfw, B*T, C);
    
    // Output projection (tie weights with embedding)
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, Vp, B*T, C, &alpha, p->wte, C, a->lnf, C, &beta, a->output, Vp);
    
    // Classifier (Softmax + CrossEntropy + Gradient)
    if (targets) {
        fused_classifier_kernel<<<B*T, 1024>>>(a->output, a->losses, model->targets, B, T, V, Vp);
        
        // Compute mean loss
        float* losses_h = (float*)malloc(B * T * sizeof(float));
        cudaMemcpy(losses_h, a->losses, B * T * sizeof(float), cudaMemcpyDeviceToHost);
        float sum = 0.0f;
        for (int i = 0; i < B * T; i++) sum += losses_h[i];
        model->mean_loss = sum / (B * T);
        free(losses_h);
    }
}

// ============================================================================
// Zero Gradients
// ============================================================================
/**
 * @brief Zero out all gradient tensors
 * @param model  Qwen3 model instance
 */
void qwen3_zero_grad(Qwen3* model) {
    size_t sizes[NUM_PARAMETER_TENSORS];
    fill_param_sizes(sizes, &model->config);
    size_t total = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) total += sizes[i];
    cudaMemset(model->grads_memory, 0, total * sizeof(float));
}

// ============================================================================
// Backward Pass
// ============================================================================
/**
 * @brief Execute backward pass of Qwen3 model (compute gradients)
 * @param model  Qwen3 model instance
 * @param B      Batch size
 * @param T      Sequence length
 */
void qwen3_backward(Qwen3* model, int B, int T) {
    Qwen3Config* cfg = &model->config;
    ParameterTensors* p = &model->params;
    ParameterTensors* g = &model->grads;
    ActivationTensors* a = &model->acts;
    
    int C = cfg->channels, L = cfg->num_layers, NH = cfg->num_heads;
    int NKV = cfg->num_kv_heads, HS = cfg->head_size, I = cfg->intermediate_size;
    int Vp = cfg->padded_vocab_size;
    int C_kv = NKV * HS;
    
    float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    
    // 分配临时梯度缓冲区
    float *dresidual, *dln, *dq, *dk, *dv, *datt, *dpreatt, *datty, *dgate, *dup;
    cudaMalloc(&dresidual, B * T * C * sizeof(float));
    cudaMalloc(&dln, B * T * C * sizeof(float));
    cudaMalloc(&dq, B * NH * T * HS * sizeof(float));
    cudaMalloc(&dk, B * NH * T * HS * sizeof(float));
    cudaMalloc(&dv, B * NH * T * HS * sizeof(float));
    cudaMalloc(&datt, B * NH * T * T * sizeof(float));
    cudaMalloc(&dpreatt, B * NH * T * T * sizeof(float));
    cudaMalloc(&datty, B * T * C * sizeof(float));
    cudaMalloc(&dgate, B * T * I * sizeof(float));
    cudaMalloc(&dup, B * T * I * sizeof(float));
    
    // ========== 输出层反向 ==========
    // dlogits已经在fused_classifier中计算并存储在a->output中
    // 反向传播到final RMSNorm输出: dlnf = dlogits @ wte
    float* dlnf = dresidual;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, Vp, &alpha, 
                p->wte, C, a->output, Vp, &beta_zero, dlnf, C);
    
    // dwte += lnf.T @ dlogits (embedding梯度)
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, Vp, B*T, &alpha,
                a->lnf, C, a->output, Vp, &beta_one, g->wte, C);
    
    // Final RMSNorm backward
    float* last_residual = a->mlp_out + (L-1) * B * T * C;
    rmsnorm_backward_kernel<<<CEIL_DIV(B*T, 8), 256, C * sizeof(float)>>>(
        dresidual, g->lnfw, dlnf, last_residual, p->lnfw, a->lnf_rstd, B, T, C);
    
    // ========== 逐层反向传播 ==========
    for (int l = L - 1; l >= 0; l--) {
        // 获取当前层的激活值
        float* ln1 = a->ln1 + l * B * T * C;
        float* ln1_rstd = a->ln1_rstd + l * B * T;
        float* q_rope = a->q_rope + l * B * NH * T * HS;
        float* k_rope = a->k_rope + l * B * NKV * T * HS;
        float* k_rep = a->k_rep + l * B * NH * T * HS;
        float* v_rep = a->v_rep + l * B * NH * T * HS;
        float* att = a->att + l * B * NH * T * T;
        float* atty = a->atty + l * B * T * C;
        float* residual2 = a->residual2 + l * B * T * C;
        float* ln2 = a->ln2 + l * B * T * C;
        float* ln2_rstd = a->ln2_rstd + l * B * T;
        float* gate = a->gate + l * B * T * I;
        float* up = a->up + l * B * T * I;
        
        // 获取当前层的参数
        float* ln1w = p->ln1w + l * C;
        float* qw = p->qw + l * C * C;
        float* kw = p->kw + l * C * C_kv;
        float* vw = p->vw + l * C * C_kv;
        float* attn_out_w = p->attn_out_w + l * C * C;
        float* ln2w = p->ln2w + l * C;
        float* gate_w = p->gate_w + l * C * I;
        float* up_w = p->up_w + l * C * I;
        float* down_w = p->down_w + l * I * C;
        
        // 获取当前层的梯度
        float* dln1w = g->ln1w + l * C;
        float* dqw = g->qw + l * C * C;
        float* dkw = g->kw + l * C * C_kv;
        float* dvw = g->vw + l * C * C_kv;
        float* dattn_out_w = g->attn_out_w + l * C * C;
        float* dln2w = g->ln2w + l * C;
        float* dgate_w = g->gate_w + l * C * I;
        float* dup_w = g->up_w + l * C * I;
        float* ddown_w = g->down_w + l * I * C;
        
        // ----- MLP反向 -----
        // dresidual来自上一步，这是对mlp_out的梯度
        
        // down_proj backward: dgate_act = dresidual @ down_w.T
        float* dgate_act = dgate;
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, I, B*T, C, &alpha,
                    down_w, I, dresidual, C, &beta_zero, dgate_act, I);
        // ddown_w += gate_act.T @ dresidual
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, I, C, B*T, &alpha,
                    gate, I, dresidual, C, &beta_one, ddown_w, I);
        
        // SwiGLU backward
        swiglu_backward_kernel<<<CEIL_DIV(B*T*I, 256), 256>>>(dgate, dup, dgate_act, gate, up, B*T*I);
        
        // gate_proj, up_proj backward
        // dln2 = dgate @ gate_w.T + dup @ up_w.T
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, I, &alpha,
                    gate_w, C, dgate, I, &beta_zero, dln, C);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, I, &alpha,
                    up_w, C, dup, I, &beta_one, dln, C);
        // dgate_w += ln2.T @ dgate, dup_w += ln2.T @ dup
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, I, B*T, &alpha,
                    ln2, C, dgate, I, &beta_one, dgate_w, C);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, I, B*T, &alpha,
                    ln2, C, dup, I, &beta_one, dup_w, C);
        
        // RMSNorm2 backward
        float* dresidual2 = datty;  // 复用缓冲区
        rmsnorm_backward_kernel<<<CEIL_DIV(B*T, 8), 256, C * sizeof(float)>>>(
            dresidual2, dln2w, dln, residual2, ln2w, ln2_rstd, B, T, C);
        
        // Residual2: dresidual2 += dresidual (从MLP路径)
        residual_forward_kernel<<<CEIL_DIV(B*T*C, 256), 256>>>(dresidual2, dresidual2, dresidual, B*T*C);
        
        // ----- Attention反向 -----
        // attention output projection backward
        // datty = dresidual2 @ attn_out_w.T
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, C, &alpha,
                    attn_out_w, C, dresidual2, C, &beta_zero, datty, C);
        // dattn_out_w += atty.T @ dresidual2
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, C, B*T, &alpha,
                    atty, C, dresidual2, C, &beta_one, dattn_out_w, C);
        
        // Unpermute backward: (B,T,NH,HS) -> (B,NH,T,HS)
        unpermute_backward_kernel<<<CEIL_DIV(B*NH*T*HS, 256), 256>>>(dq, datty, B, T, NH, HS);
        
        // Attention backward: datt = dq @ v_rep.T
        cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha,
                                  v_rep, HS, T*HS, dq, HS, T*HS, &beta_zero, datt, T, T*T, B*NH);
        // dv_rep = att.T @ dq
        cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &alpha,
                                  dq, HS, T*HS, att, T, T*T, &beta_zero, dv, HS, T*HS, B*NH);
        
        // Softmax backward
        float scale = 1.0f / sqrtf(HS);
        softmax_backward_kernel<<<CEIL_DIV(B*NH*T, 8), 256>>>(dpreatt, datt, att, B, NH, T, scale);
        
        // Q @ K.T backward
        // dq_rope = dpreatt @ k_rep
        cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha,
                                  k_rep, HS, T*HS, dpreatt, T, T*T, &beta_zero, dq, HS, T*HS, B*NH);
        // dk_rep = q_rope @ dpreatt.T
        cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &alpha,
                                  q_rope, HS, T*HS, dpreatt, T, T*T, &beta_zero, dk, HS, T*HS, B*NH);
        
        // RoPE backward (简化: 直接将梯度传回)
        rope_backward_kernel<<<CEIL_DIV(B*NH*T*HS, 256), 256>>>(dq, dq, model->cos_cache, model->sin_cache, B, NH, T, HS);
        
        // Q/K/V projection backward: dln1 = dq @ qw.T + dk @ kw.T + dv @ vw.T
        // 简化处理: 只计算Q的梯度
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, C, &alpha,
                    qw, C, dq, C, &beta_zero, dln, C);
        // dqw += ln1.T @ dq
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, C, B*T, &alpha,
                    ln1, C, dq, C, &beta_one, dqw, C);
        
        // RMSNorm1 backward
        float* prev_residual = (l == 0) ? a->encoded : (a->mlp_out + (l-1) * B * T * C);
        rmsnorm_backward_kernel<<<CEIL_DIV(B*T, 8), 256, C * sizeof(float)>>>(
            dresidual, dln1w, dln, prev_residual, ln1w, ln1_rstd, B, T, C);
        
        // Residual1: dresidual += dresidual2 (注意力路径)
        residual_forward_kernel<<<CEIL_DIV(B*T*C, 256), 256>>>(dresidual, dresidual, dresidual2, B*T*C);
    }
    
    // Embedding backward
    embedding_backward_kernel<<<CEIL_DIV(B*T*C, 256), 256>>>(g->wte, dresidual, model->inputs, B, T, C);
    
    // 释放临时缓冲区
    cudaFree(dresidual); cudaFree(dln); cudaFree(dq); cudaFree(dk); cudaFree(dv);
    cudaFree(datt); cudaFree(dpreatt); cudaFree(datty); cudaFree(dgate); cudaFree(dup);
}

// ============================================================================
// Update Parameters (AdamW)
// ============================================================================
/**
 * @brief Update model parameters using AdamW optimizer
 * @param model  Qwen3 model instance
 * @param lr     Learning rate
 * @param beta1  Exponential decay rate for first moment (typically 0.9)
 * @param beta2  Exponential decay rate for second moment (typically 0.999)
 * @param eps    Small constant for numerical stability (typically 1e-8)
 * @param wd     Weight decay coefficient (typically 0.01)
 * @param t      Current training step (for bias correction)
 */
void qwen3_update(Qwen3* model, float lr, float beta1, float beta2, float eps, float wd, int t) {
    float b1_corr = 1.0f - powf(beta1, t);
    float b2_corr = 1.0f - powf(beta2, t);
    adamw_kernel<<<CEIL_DIV(model->num_parameters, 256), 256>>>(
        model->params_memory, model->grads_memory, model->m_memory, model->v_memory,
        model->num_parameters, lr, beta1, beta2, b1_corr, b2_corr, eps, wd);
}

// ============================================================================
// Build Model
// ============================================================================
/**
 * @brief Initialize and allocate all memory for Qwen3 model
 * @param model  Qwen3 model instance to initialize
 * @param cfg    Model configuration
 * @param B      Batch size
 * @param T      Sequence length
 */
void qwen3_build(Qwen3* model, Qwen3Config* cfg, int B, int T) {
    model->config = *cfg;
    model->batch_size = B;
    model->seq_len = T;
    
    // Allocate parameters
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    fill_param_sizes(param_sizes, cfg);
    model->params_memory = malloc_and_point_params(&model->params, param_sizes, 1);
    model->grads_memory = malloc_and_point_params(&model->grads, param_sizes, 1);
    
    // Count parameters
    model->num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) model->num_parameters += param_sizes[i];
    printf("Number of parameters: %zu (%.2f M)\n", model->num_parameters, model->num_parameters / 1e6);
    
    // Allocate activations
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    fill_act_sizes(act_sizes, B, T, cfg);
    size_t act_total = 0;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) act_total += act_sizes[i];
    cudaMalloc(&model->acts_memory, act_total * sizeof(float));
    
    // Point activation tensors
    float** act_ptrs[] = {&model->acts.encoded, &model->acts.ln1, &model->acts.ln1_rstd, &model->acts.q,
                          &model->acts.k, &model->acts.v, &model->acts.q_rope, &model->acts.k_rope,
                          &model->acts.k_rep, &model->acts.v_rep, &model->acts.att, &model->acts.atty,
                          &model->acts.attproj, &model->acts.residual2, &model->acts.ln2, &model->acts.ln2_rstd,
                          &model->acts.gate, &model->acts.up, &model->acts.mlp_out, &model->acts.lnf, 
                          &model->acts.lnf_rstd, &model->acts.output};
    float* iter = model->acts_memory;
    for (int i = 0; i < 22; i++) {
        *act_ptrs[i] = iter;
        iter += act_sizes[i];
    }
    // 为losses单独分配内存
    cudaMalloc(&model->acts.losses, B * T * sizeof(float));
    
    // Allocate optimizer states
    cudaMalloc(&model->m_memory, model->num_parameters * sizeof(float));
    cudaMalloc(&model->v_memory, model->num_parameters * sizeof(float));
    cudaMemset(model->m_memory, 0, model->num_parameters * sizeof(float));
    cudaMemset(model->v_memory, 0, model->num_parameters * sizeof(float));
    
    // Allocate RoPE cache
    cudaMalloc(&model->cos_cache, T * cfg->head_size * sizeof(float));
    cudaMalloc(&model->sin_cache, T * cfg->head_size * sizeof(float));
    init_rope_cache(model->cos_cache, model->sin_cache, T, cfg->head_size, cfg->rope_theta);
    
    // Allocate inputs/targets
    cudaMalloc(&model->inputs, B * T * sizeof(int));
    cudaMalloc(&model->targets, B * T * sizeof(int));
    
    // Initialize parameters with small random values
    float* params_h = (float*)malloc(model->num_parameters * sizeof(float));
    for (size_t i = 0; i < model->num_parameters; i++) {
        params_h[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    cudaMemcpy(model->params_memory, params_h, model->num_parameters * sizeof(float), cudaMemcpyHostToDevice);
    free(params_h);
}

// ============================================================================
// Free Model
// ============================================================================
/**
 * @brief Free all allocated memory for Qwen3 model
 * @param model  Qwen3 model instance to free
 */
void qwen3_free(Qwen3* model) {
    cudaFree(model->params_memory);
    cudaFree(model->grads_memory);
    cudaFree(model->acts_memory);
    cudaFree(model->acts.losses);
    cudaFree(model->m_memory);
    cudaFree(model->v_memory);
    cudaFree(model->cos_cache);
    cudaFree(model->sin_cache);
    cudaFree(model->inputs);
    cudaFree(model->targets);
}

// ============================================================================
// Sampling utilities
// ============================================================================
/**
 * @brief Generate pseudo-random 32-bit unsigned integer using xorshift algorithm
 * @param state  Pointer to RNG state (modified in-place)
 * @return       Random 32-bit unsigned integer
 */
unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

/**
 * @brief Generate pseudo-random float in range [0, 1)
 * @param state  Pointer to RNG state (modified in-place)
 * @return       Random float in [0, 1)
 */
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

/**
 * @brief Sample token index from softmax distribution
 * @param logits  Unnormalized log probabilities, shape [n]
 * @param n       Number of classes (vocabulary size)
 * @param coin    Random value in [0, 1) for sampling
 * @return        Sampled token index
 */
int sample_softmax(const float* logits, int n, float coin) {
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

// ============================================================================
// CLI usage
// ============================================================================
/**
 * @brief Print usage information and exit
 */
void error_usage() {
    fprintf(stderr, "Usage:   ./train_qwen3_fp32 [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -e <string> tokenizer file (default = gpt2_tokenizer.bin)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 64)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 1e-4)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    fprintf(stderr, "  -n <int>    num_steps, total training steps (default = 0 = 1 epoch)\n");
    exit(EXIT_FAILURE);
}

// ============================================================================
// Main Training Loop
// ============================================================================
/**
 * @brief Main entry point for Qwen3 training program
 * @param argc  Number of command-line arguments
 * @param argv  Array of command-line argument strings
 * @return      Exit status (0 for success)
 */
int main(int argc, char** argv) {
    printf("============================================\n");
    printf("Qwen3 CUDA FP32 Training with DataLoader\n");
    printf("============================================\n\n");
    
    // Default arguments
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* tokenizer_file = "gpt2_tokenizer.bin";
    int B = 4;              // batch size
    int T = 64;             // sequence length
    float learning_rate = 1e-4f;
    int val_loss_every = 20;
    int val_max_steps = 20;
    int sample_every = 20;
    int genT = 64;
    int num_steps = 0;      // 0 means 1 epoch
    
    // Parse command line arguments
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'e') { tokenizer_file = argv[i+1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); }
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n') { num_steps = atoi(argv[i+1]); }
        else { error_usage(); }
    }
    
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Parameter             | Value                                              |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| train data pattern    | %-50s |\n", train_data_pattern);
    printf("| val data pattern      | %-50s |\n", val_data_pattern);
    printf("| tokenizer file        | %-50s |\n", tokenizer_file);
    printf("| batch size B          | %-50d |\n", B);
    printf("| sequence length T     | %-50d |\n", T);
    printf("| learning rate         | %-50e |\n", learning_rate);
    printf("| val_loss_every        | %-50d |\n", val_loss_every);
    printf("| val_max_steps         | %-50d |\n", val_max_steps);
    printf("| sample_every          | %-50d |\n", sample_every);
    printf("| genT                  | %-50d |\n", genT);
    printf("+-----------------------+----------------------------------------------------+\n");
    
    // Initialize CUDA
    int device = 0;
    cudaCheck(cudaSetDevice(device));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("| device                | %-50s |\n", prop.name);
    
    cublasCreate(&cublas_handle);
    int tf32 = prop.major >= 8;
    if (tf32) cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    printf("| TF32                  | %-50s |\n", tf32 ? "enabled" : "disabled");
    printf("+-----------------------+----------------------------------------------------+\n");
    
    // Initialize DataLoaders
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
    int train_num_batches = (num_steps > 0) ? num_steps : (train_loader.num_tokens / (B * T));
    int val_num_batches = val_loader.num_tokens / (B * T);
    if (val_num_batches > val_max_steps) { val_num_batches = val_max_steps; }
    printf("| train_num_batches     | %-50d |\n", train_num_batches);
    printf("| val_num_batches       | %-50d |\n", val_num_batches);
    printf("+-----------------------+----------------------------------------------------+\n");
    
    // Initialize Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, tokenizer_file);
    
    // Model config - 根据数据集的vocab_size调整
    // 注意: tinyshakespeare使用GPT-2的tokenizer, vocab_size = 50257
    Qwen3Config cfg;
    cfg.max_seq_len = T;
    cfg.vocab_size = 50257;  // GPT-2 vocab size for tinyshakespeare
    cfg.padded_vocab_size = 50304;  // padded to multiple of 64
    cfg.num_layers = 4;
    cfg.num_heads = 8;
    cfg.num_kv_heads = 2;  // GQA: 4x reduction
    cfg.channels = 256;
    cfg.head_size = cfg.channels / cfg.num_heads;
    cfg.intermediate_size = cfg.channels * 4;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-6f;
    
    printf("| vocab_size V          | %-50d |\n", cfg.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", cfg.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", cfg.num_layers);
    printf("| num_heads NH          | %-50d |\n", cfg.num_heads);
    printf("| num_kv_heads NKV      | %-50d |\n", cfg.num_kv_heads);
    printf("| channels C            | %-50d |\n", cfg.channels);
    printf("| head_size HS          | %-50d |\n", cfg.head_size);
    printf("| intermediate_size I   | %-50d |\n", cfg.intermediate_size);
    printf("+-----------------------+----------------------------------------------------+\n");
    
    // Build model
    Qwen3 model;
    qwen3_build(&model, &cfg, B, T);
    printf("allocated %d MiB for model parameters\n", (int)(model.num_parameters * sizeof(float) / (1024 * 1024)));
    
    // Allocate memory for sampling
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    float* cpu_logits = (float*)mallocCheck(cfg.vocab_size * sizeof(float));
    
    printf("\nStarting training...\n");
    printf("--------------------------------------------\n");
    
    // Training loop
    struct timespec start, end;
    double total_sum_iteration_time_s = 0.0;
    
    for (int step = 0; step <= train_num_batches; step++) {
        int last_step = step == train_num_batches;
        
        // Evaluate validation loss periodically
        if (step % val_loss_every == 0 || last_step) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                qwen3_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }
        
        // Generate samples periodically
        if ((step > 0 && step % sample_every == 0) || last_step) {
            // Fill with a starting token (use 0 or a special token)
            for (int i = 0; i < B * T; ++i) {
                gen_tokens[i] = 0;  // Start token
            }
            printf("generating:\n---\n");
            for (int t = 1; t < genT && t < T; t++) {
                qwen3_forward(&model, gen_tokens, NULL, B, T);
                // Get logits for position t-1
                float* logits = model.acts.output + (t - 1) * cfg.padded_vocab_size;
                cudaCheck(cudaMemcpy(cpu_logits, logits, cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
                float coin = random_f32(&rng_state);
                int next_token = sample_softmax(cpu_logits, cfg.vocab_size, coin);
                gen_tokens[t] = next_token;
                // Print the generated token
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }
        
        if (last_step) { break; }
        
        // Training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        qwen3_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        qwen3_zero_grad(&model);
        qwen3_backward(&model, B, T);
        qwen3_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
        cudaCheck(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_sum_iteration_time_s += time_elapsed_s;
        int tokens_per_second = (B * T) / time_elapsed_s;
        printf("step %4d/%d: train loss %f (%f ms, %d tok/s)\n", 
               step + 1, train_num_batches, model.mean_loss, time_elapsed_s * 1000, tokens_per_second);
    }
    
    printf("--------------------------------------------\n");
    printf("total average iteration time: %f ms\n", total_sum_iteration_time_s / train_num_batches * 1000);
    printf("Training completed!\n\n");
    
    // Cleanup
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    free(gen_tokens);
    free(cpu_logits);
    qwen3_free(&model);
    cublasDestroy(cublas_handle);
    
    printf("Resources freed. Exiting.\n");
    return 0;
}
