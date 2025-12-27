/*
 * ============================================================================
 * LayerNorm 和 Residual 的 CUDA 实现
 * ============================================================================
 * 
 * 本文件包含 Layer Normalization 和 Residual Connection 的 CUDA kernel 实现
 * 这两个操作经常被融合以提高性能 (减少内存访问)
 * 
 * Layer Normalization 数学公式:
 *   y = (x - mean) / sqrt(var + eps) * gamma + beta
 * 其中:
 *   - mean = sum(x) / C           均值
 *   - var = sum((x - mean)^2) / C  方差
 *   - gamma (weight): 可学习的缩放参数
 *   - beta (bias): 可学习的偏移参数
 *   - eps = 1e-5: 防止除零的小常数
 * 
 * 内存优化策略:
 * - 反向传播中，参数梯度使用 += (支持梯度累积)
 * - 激活值梯度通常使用 = (更快，只读不写)
 * - 但残差流中的激活值必须使用 += (因为梯度会累加)
 * - LayerNorm 连接到残差，所以反向传播使用 +=
 * 
 * 性能优化技术:
 * - 向量化加载/存储 (x128 = 128位)
 * - Warp 级别的 reduce 操作
 * - 共享内存缓存 weight/bias
 * - 流式内存访问提示 (__ldcs/__stcs)
 * - 多级归约减少全局内存原子操作
 */

#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// CUDA Kernels - 前向传播
// ============================================================================

/**
 * layernorm_forward_kernel3 - 基础版 LayerNorm 前向 kernel (无共享内存)
 * 
 * 每个 warp 处理一行数据，适用于共享内存不足时的 fallback
 * 
 * @param out:    输出张量 [N, C]，归一化后的结果
 * @param mean:   均值输出 [N]，每行的均值 (可为 nullptr)
 * @param rstd:   逆标准差输出 [N]，1/sqrt(var+eps) (可为 nullptr)
 * @param inp:    输入张量 [N, C]
 * @param weight: 缩放参数 gamma [C]
 * @param bias:   偏移参数 beta [C]
 * @param N:      行数 (batch_size * seq_len)
 * @param C:      列数 (hidden_size)
 * 
 * 线程组织: 每个 warp (32线程) 处理一行，多个 warp 共享一个 block
 */
__global__ void layernorm_forward_kernel3(floatX* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    const floatX* __restrict__ bias, int N, int C) {
    // 计算当前线程在 warp 内的位置和 warp 在 block 内的位置
    int lane_id = threadIdx.x % WARP_SIZE;   // warp 内线程索引 [0, 31]
    int warp_id = threadIdx.x / WARP_SIZE;   // block 内 warp 索引
    int num_warps = blockDim.x / WARP_SIZE;  // block 内 warp 数量

    // 每个 warp 处理一行，计算当前 warp 负责的行索引
    int idx = blockIdx.x * num_warps + warp_id;
    if(idx >= N) { return; } // 边界检查

    // 获取当前行的输入指针
    const floatX* x = inp + idx * C;

    // ========== 第一步: 计算均值 mean ==========
    float sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        sum += (float)x[i];
    }
    sum = warpReduceSum(sum);
    float m = sum / C;
    if(lane_id == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // 计算方差: var = sum((x - mean)^2) / C
    sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float diff = (float)x[i] - m;  // 中心化
        sum += diff * diff;            // 平方和
    }
    sum = warpReduceSum(sum);         // Warp 归约
    float s = rsqrtf(sum / C + 1e-5f); // rstd = 1/sqrt(var + eps)
    if(lane_id == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);         // 流式写出
    }

    // ========== 第三步: 归一化并应用 weight/bias ==========
    floatX* o = out + idx * C;
    for (int c = lane_id; c < C; c += WARP_SIZE) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * ((float)__ldcs(x+c) - m);
        __stcs(o+c, (floatX)(n * (float)weight[c] + (float)bias[c]));
    }
}

/**
 * layernorm_forward_kernel6 - 优化版 LayerNorm 前向 kernel (使用共享内存)
 * 
 * 使用共享内存缓存 weight/bias 和输入数据，提高缓存命中率
 * 使用向量化加载 (x128) 提高内存带宽利用率
 * 
 * @param out:    输出张量 [N, C]
 * @param mean:   均值输出 [N] (可为 nullptr)
 * @param rstd:   逆标准差输出 [N] (可为 nullptr)
 * @param inp:    输入张量 [N, C]
 * @param weight: 缩放参数 [C]
 * @param bias:   偏移参数 [C]
 * @param N:      行数 (B * T)
 * @param C:      列数 (隐藏维度)
 * 
 * 共享内存布局: [weight(C)] [bias(C)] [inp_row_0(C)] [inp_row_1(C)] ...
 * 线程组织: blockDim = (WARP_SIZE, block_y), 每行由一个 warp 处理
 */
__global__ void layernorm_forward_kernel6(floatX* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    const floatX* __restrict__ bias, int N, int C) {
    assert(blockDim.x == WARP_SIZE);  // 要求 x 维度为 warp 大小

    // ========== 加载 weight/bias 到共享内存 ==========
    // 必须在边界检查之前完成，因为所有线程都参与加载
    extern __shared__ char* params[];  // 动态共享内存
    
    // 使用 x128 类型确保 128 位向量化加载/存储
    // (使用 floatX* 时编译器有时生成多条指令)
    x128* s_weight = reinterpret_cast<x128*>(params);                              // weight 缓存
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);             // bias 缓存  
    x128* s_in = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size); // 输入行缓存

    // 协作加载: 所有线程一起加载 weight 和 bias 到共享内存
    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);  // 向量化加载 weight
        s_bias[i/x128::size] = load128(bias + i);      // 向量化加载 bias
    }
    __syncthreads();  // 确保加载完成

    // 计算当前 warp 负责的行索引
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) { return; }  // 边界检查

    // 调整指针到当前 token 的位置
    inp += idx * C;
    out += idx * C;

    // ========== 第一步: 计算均值并缓存输入 ==========
    const float eps = 1e-5f;  // 防除零常数
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        // 向量化加载输入 (128位 = 8个BF16 或 8个FP16)
        const x128 in_data = load128cs(inp + c);  // cache streaming 加载
        for(int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];  // 累加求和
        }
        s_in[c / x128::size] = in_data;  // 缓存到共享内存供后续使用
    }

    // Warp 归约计算均值
    sum = warpReduceSum(sum);
    float m = sum / C;  // mean
    float v = 0.f;      // 方差累加器

    // ========== 第二步: 计算方差 ==========
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];  // 从共享内存读取 (避免重复全局内存访问)
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);  // (x - mean)^2
        }
    }

    // Warp 归约计算方差，然后计算逆标准差
    v = warpReduceSum(v) / C;   // variance
    float s = rsqrtf(v + eps);  // rstd = 1/sqrt(var + eps)

    // ========== 第三步: 归一化并应用 weight/bias ==========
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];     // 输入 (共享内存)
        const x128 w = s_weight[c / x128::size];       // weight (共享内存)
        const x128 b = s_bias[c / x128::size];         // bias (共享内存)
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)in_data[k] - m);   // 归一化: (x - mean) * rstd
            float o = n * (float)w[k] + (float)b[k]; // 缩放+偏移: n * weight + bias
            out_data[k] = (floatX)o;
        }
        store128cs(out + c, out_data);  // 向量化流式存储
    }
    
    // ========== 保存统计量供反向传播使用 ==========
    // 只有 lane 0 写出 (避免重复写入)
    if(threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);  // 缓存均值
    }
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);  // 缓存逆标准差
    }
}

/**
 * fused_residual_forward_kernel5 - 融合的残差连接 + LayerNorm 前向 kernel
 * 
 * 将残差加法和 LayerNorm 融合为单个 kernel，减少内存访问:
 *   residual = inp1 + inp2
 *   normed = LayerNorm(residual)
 * 
 * @param residual: 残差输出 [N, C]，inp1 + inp2
 * @param normed:   归一化输出 [N, C]
 * @param mean:     均值输出 [N]
 * @param rstd:     逆标准差输出 [N]
 * @param inp1:     输入1 [N, C] (例如: 上一层输出)
 * @param inp2:     输入2 [N, C] (例如: 注意力/FFN 输出)
 * @param weight:   LayerNorm 缩放参数 [C]
 * @param bias:     LayerNorm 偏移参数 [C]
 * @param N:        行数 (B * T)
 * @param C:        列数 (隐藏维度)
 * 
 * 融合优点:
 * - 避免 residual 的全局内存往返 (先写后读)
 * - 共享内存缓存减少带宽需求
 */

//                          threadIdx.x
//               0    1    2    3   ...   30   31
//             ┌────┬────┬────┬────┬────┬────┬────┐
//         0   │ 0  │ 1  │ 2  │ 3  │ ...│ 30 │ 31 │  ← Warp 0 (全局ID 0~31)
//             ├────┼────┼────┼────┼────┼────┼────┤
//         1   │ 32 │ 33 │ 34 │ 35 │ ...│ 62 │ 63 │  ← Warp 1 (全局ID 32~63)
// threadIdx.y ├────┼────┼────┼────┼────┼────┼────┤
//         2   │ 64 │ 65 │ 66 │ 67 │ ...│ 94 │ 95 │  ← Warp 2 (全局ID 64~95)
//             ├────┼────┼────┼────┼────┼────┼────┤
//         3   │ 96 │ 97 │ 98 │ 99 │ ...│126 │127 │  ← Warp 3 (全局ID 96~127)
//             └────┴────┴────┴────┴────┴────┴────┘
//                  │←─────── 32 threads ───────→│
//                          = 1 Warp
__global__ void fused_residual_forward_kernel5(floatX* residual, floatX* normed, float* mean, float* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    assert(blockDim.x == WARP_SIZE);

    // ========== 共享内存设置 ==========
    // 加载 weight/bias 到共享内存 (在边界检查前完成)
    extern __shared__ char* params[];
    x128* s_weight = reinterpret_cast<x128*>(params);                               // weight 缓存
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);              // bias 缓存
    x128* s_res = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size); // 残差缓存

    // 协作加载 weight 和 bias
    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;  // 边界检查

    // 调整指针到当前 token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    // ========== 第一步: 残差加法 + 均值计算 ==========
    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in1 = load128cs(inp1 + c);  // 输入1
        const x128 in2 = load128cs(inp2 + c);  // 输入2
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (float)in1[k] + (float)in2[k];  // 残差加法
            sum += (float)out[k];                     // 累加求均值
        }
        store128cs(residual + c, out);  // 写出残差结果
        s_res[c / x128::size] = out;    // 缓存到共享内存
    }

    // ========== 第二步: 计算方差 ==========
    sum = warpReduceSum(sum);
    float m = sum / C;  // mean
    float v = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];  // 从共享内存读取残差
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);  // (x - mean)^2
        }
    }

    v = warpReduceSum(v) / C;   // variance
    float s = rsqrtf(v + eps);  // rstd

    // ========== 第三步: 归一化 ==========
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];  // 残差 (共享内存)
        const x128 w = s_weight[c / x128::size]; // weight (共享内存)
        const x128 b = s_bias[c / x128::size];   // bias (共享内存)
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m);   // 归一化
            float o = n * (float)w[k] + (float)b[k]; // 缩放+偏移
            out[k] = o;
        }
        store128cs(normed + c, out);  // 写出归一化结果
    }
    
    // ========== 保存统计量 ==========
    if(threadIdx.x == 0) {
        mean[idx] = m;  // 供反向传播使用
        rstd[idx] = s;
    }
}

/**
 * residual_forward_kernel - 简单残差加法 kernel
 * 
 * out = inp1 + inp2
 * 
 * @param out:  输出 [N] (展平后的总元素数)
 * @param inp1: 输入1 [N]
 * @param inp2: 输入2 [N]
 * 
 * 使用 x128 向量化加载/存储提高带宽利用率
 * 每个线程处理 x128::size 个元素 (通常是8个)
 */
__global__ void residual_forward_kernel(floatX* out, const floatX* inp1, const floatX* inp2) {
    // 每个线程处理 x128::size 个连续元素
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    // 向量化加载、计算、存储
    x128 packed_out;
    x128 packed_inp1 = load128cs(inp1 + idx);  // 流式加载
    x128 packed_inp2 = load128cs(inp2 + idx);
    for (int k = 0; k < packed_inp1.size; k++) {
        packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
    }
    store128(out + idx, packed_out);  // 向量化存储
}

// ============================================================================
// CUDA Kernels - 反向传播
// ============================================================================

/**
 * layernorm_backward_kernel10 - LayerNorm 反向传播 kernel
 * 
 * 计算三个梯度:
 *   1. dinp (输入梯度): 反向传播到上一层
 *   2. dweight (gamma梯度): 缩放参数的梯度
 *   3. dbias (beta梯度): 偏移参数的梯度
 * 
 * ================================================================================
 * LayerNorm 反向传播完整数学推导
 * ================================================================================
 * 
 * 【前向传播公式】
 *   mean = (1/C) * Σx_i
 *   var  = (1/C) * Σ(x_i - mean)²
 *   rstd = 1 / √(var + ε)
 *   norm = (x - mean) * rstd
 *   y    = norm * weight + bias
 * 
 * 【符号定义】
 *   dout = ∂L/∂y          -- 从下游传回的梯度 (已知输入)
 *   dinp = ∂L/∂x          -- 要计算的输入梯度 (传给上游)
 *   dnorm = dout * weight -- 对归一化结果的梯度
 * 
 * --------------------------------------------------------------------------------
 * 【1. dbias 推导】
 * --------------------------------------------------------------------------------
 *   y = ... + bias
 *   ∂L/∂bias = ∂L/∂y * ∂y/∂bias = dout * 1 = dout
 *   
 *   由于 bias 被所有样本共享，需要对所有样本求和:
 *   >>> dbias = Σ(dout)
 * 
 * --------------------------------------------------------------------------------
 * 【2. dweight 推导】
 * --------------------------------------------------------------------------------
 *   y = norm * weight + ...
 *   ∂L/∂weight = ∂L/∂y * ∂y/∂weight = dout * norm
 *   
 *   由于 weight 被所有样本共享:
 *   >>> dweight = Σ(dout * norm)
 * 
 * --------------------------------------------------------------------------------
 * 【3. dinp 推导】(最复杂，x 通过三条路径影响 y)
 * --------------------------------------------------------------------------------
 * 
 *   路径1: x → norm → y (直接路径)
 *   路径2: x → mean → norm → y (通过均值)
 *   路径3: x → var → rstd → norm → y (通过方差)
 * 
 *   设 dnorm = dout * weight
 * 
 *   [路径1: 直接路径]
 *     norm = (x - mean) * rstd
 *     ∂norm/∂x|直接 = rstd
 *     贡献: dnorm * rstd
 * 
 *   [路径2: 通过 mean]
 *     ∂mean/∂x_i = 1/C
 *     ∂norm/∂mean = -rstd
 *     ∂L/∂x|via_mean = Σ(dnorm) * (-rstd) * (1/C) = -mean(dnorm) * rstd
 *     
 *     令 dnorm_mean = mean(dnorm) = mean(weight * dout)
 *     贡献: -dnorm_mean * rstd
 * 
 *   [路径3: 通过 rstd/var]
 *     rstd = (var + ε)^(-1/2)
 *     ∂rstd/∂var = -1/2 * (var + ε)^(-3/2) = -rstd³/2
 *     
 *     var = (1/C) * Σ(x - mean)²
 *     ∂var/∂x_i = 2(x_i - mean)/C = 2 * norm_i / (C * rstd)
 *     
 *     ∂L/∂x|via_rstd = Σ(dnorm * (x-mean)) * ∂rstd/∂var * ∂var/∂x_i
 *                    = Σ(dnorm * norm/rstd) * (-rstd³/2) * (2 * norm_i / (C * rstd))
 *                    = -mean(dnorm * norm) * norm_i * rstd
 *     
 *     令 dnorm_norm_mean = mean(dnorm * norm) = mean(weight * dout * norm)
 *     贡献: -dnorm_norm_mean * norm * rstd
 * 
 *   [合并三条路径]
 *     dinp = rstd * dnorm - dnorm_mean * rstd - dnorm_norm_mean * norm * rstd
 *          = rstd * (dnorm - dnorm_mean - dnorm_norm_mean * norm)
 *     >>> dinp = rstd * (weight * dout - dnorm_mean - norm * dnorm_norm_mean)
 * 
 * --------------------------------------------------------------------------------
 * 【最终公式汇总】
 * --------------------------------------------------------------------------------
 *   dbias   = Σ(dout)
 *   dweight = Σ(dout * norm)
 *   dinp    = rstd * (weight * dout - dnorm_mean - norm * dnorm_norm_mean)
 *   
 *   其中:
 *   - dnorm_mean      = mean(weight * dout)           -- 梯度均值修正项
 *   - dnorm_norm_mean = mean(weight * dout * norm)    -- 方差修正项
 * 
 * 【直觉理解】
 *   - weight * dout: 直接从输出传回的梯度
 *   - -dnorm_mean: 修正 mean 带来的影响 (所有元素均摊)
 *   - -norm * dnorm_norm_mean: 修正 variance 带来的影响 (与归一化值成比例)
 *   这两个减法项确保梯度的"均值为零"性质，保持归一化层的数值稳定性
 * ================================================================================
 * 
 * @param dinp:    输出，输入梯度 [B, T, C]，使用 += 累加
 * @param dweight: 输出，weight 梯度 [C]，使用 += 累加
 * @param dbias:   输出，bias 梯度 [C]，使用 += 累加
 * @param scratch: 临时存储，用于跨 block 归约 [gridDim.x * 2 * C + 32]
 * @param dout:    输入，输出梯度 [B, T, C]
 * @param inp:     输入，前向的输入 [B, T, C]
 * @param weight:  输入，缩放参数 [C]
 * @param mean:    输入，前向计算的均值 [B, T]
 * @param rstd:    输入，前向计算的逆标准差 [B, T]
 * @param B:       批次大小
 * @param T:       序列长度
 * @param C:       隐藏维度
 * 
 * 优化策略:
 * - 多级归约: block 内归约 -> 写到 scratch -> 最后一个 block 做全局归约
 * - 使用原子操作确定最后一个 block
 * - 共享内存缓存中间结果
 * - 向量化加载/存储 (f128/x128)
 * 
 * 线程组织: 每个 block 512 线程，每 SM 最多 2 个 block
 */
__global__ void __launch_bounds__(512, 2)  // 每 block 512 线程，每 SM 最多 2 block
    layernorm_backward_kernel10(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                                const floatX* dout, const floatX* inp, const floatX* weight,
                                const float* mean, const float* rstd,
                                int B, int T, int C) {
    
    // ========== 线程索引和常量计算 ==========
    int BLOCK_SIZE = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE;  // block 内 warp 数量
    extern __shared__ float shared[];           // 动态共享内存

    int warpId = threadIdx.x / WARP_SIZE;           // block 内 warp 索引
    int baseIdx = blockIdx.x * warpsInBlock + warpId; // 全局 warp 索引 (作为 token 起始索引)
    int warpThreadIdx = threadIdx.x % WARP_SIZE;    // warp 内线程索引
    int warpsInGrid = gridDim.x * warpsInBlock;     // grid 内总 warp 数
    int C_per_iteration = WARP_SIZE * x128::size;   // 每轮迭代处理的 C 元素数
    int iterations_C = CEIL_DIV(C, C_per_iteration); // 处理完整 C 需要的迭代次数

    // ========== 共享内存布局 ==========
    // [dbias_shared (rounded_C)] [dweight_shared (rounded_C)] [tmp buffers]
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);  // 对齐到 warp*向量大小
    float* dbias_shared = shared;                    // dbias 累加器
    float* dweight_shared = shared + rounded_C;      // dweight 累加器
    
    // 临时缓冲区用于 warp 间通信 (warp 0 不写入，所以基地址可以往回调)
    // 这种奇怪的寻址避免了寄存器溢出
    float* dbias_tmp_shared = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // ========== 初始化共享内存为零 ==========
    for(int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    __syncthreads();

    // ========== 主循环: 处理每个 token (bt = batch*seq 索引) ==========
    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;  // 当前 token 的输出梯度
        const floatX* inp_bt = inp + bt * C;    // 当前 token 的输入
        floatX* dinp_bt = dinp + bt * C;        // 当前 token 的输入梯度

        // ========== 第一步: 计算两个 reduce 量 ==========
        // dnorm_mean = mean(weight * dout)
        // dnorm_norm_mean 用于计算 dinp
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);   // 输出梯度
            x128 inp128_i    = load128(inp_bt  + i);   // 输入值
            x128 weight128_i = load128(weight  + i);   // weight 参数
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];  // weight * dout
                dnorm_mean += dnorm_i;                                        // 累加
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];              // weight * dout * inp
            }
        }

        // 获取前向传播保存的统计量
        const float mean_bt = mean[bt];  // 均值
        const float rstd_bt = rstd[bt];  // 逆标准差
        
        // Warp 归约并计算最终的 dnorm 统计量
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        // ========== 第二步: 计算各梯度并累加 ==========
        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            // 加载数据 (边界外填零)
            x128 dout128   = x128::zeros();  // 输出梯度
            x128 inp128    = x128::zeros();  // 输入
            x128 dinp128   = x128::zeros();  // 输入梯度 (需要累加)
            x128 weight128 = x128::zeros();  // weight

            if(global_index < C) {
                dout128 = load128cs(dout_bt + global_index);   // 流式加载
                inp128 = load128cs(inp_bt + global_index);
                dinp128 = load128(dinp_bt + global_index);     // 需要累加，普通加载
                weight128 = load128(weight + global_index);
            }

            // 分块处理 (x128 分成多个 f128 块)
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dbias_f;    // 当前块的 dbias
                f128 dweight_f;  // 当前块的 dweight
                
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;  // 归一化值
                    
                    // dbias = dout (直接传递)
                    dbias_f[i] = dout_i;
                    // dweight = norm * dout
                    dweight_f[i] = norm_bti * dout_i;

                    // dinp 计算 (反向传播公式的三项)
                    float dval = 0.0f;
                    dval += (float)weight128[x] * (float)dout128[x];  // 项1: weight * dout
                    dval -= dnorm_mean;                                // 项2: -dnorm_mean
                    dval -= norm_bti * dnorm_norm_mean;                // 项3: -norm * dnorm_norm_mean
                    dval *= rstd_bt;                                   // 最终缩放
                    dinp128[x] = (floatX)((float)dinp128[x] + dval);   // 累加到 dinp
                }

                // ========== Warp 间归约 (block 内) ==========
                // 非 warp 0 的线程将结果写到临时共享内存
                if (warpId != 0) {
                    store128(dbias_tmp_shared + threadIdx.x * f128::size, dbias_f);
                    store128(dweight_tmp_shared + threadIdx.x * f128::size, dweight_f);
                }
                __syncthreads();
                
                // Warp 0 负责汇总所有 warp 的结果
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp = load128(dbias_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                
                // Warp 0 累加到 block 级别的共享内存累加器
                if (warpId == 0) {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            
            // 写出 dinp (使用 cache global 提示，缓存到 L2 但绕过 L1)
            if(global_index < C) {
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    __syncthreads();
    // ========== 第三步: 跨 block 归约 ==========
    // 策略: 每个 block 写部分和到全局 scratch，最后一个完成的 block 做最终归约
    // 使用原子操作确定哪个 block 是最后一个
    
    unsigned int* scratchFlag = (unsigned int*)(scratch);  // 原子计数器
    scratch += 32;  // 跳过一个 cacheline 保持对齐
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    
    // 每个 block 写出自己的部分和到 scratch
    for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        store128(scratch_dbias + i + 2*C*blockIdx.x, load128(dbias_shared + i));
        store128(scratch_dweight + i + 2*C*blockIdx.x, load128(dweight_shared + i));
    }
    __syncthreads();
    
    // 原子递增计数器，返回值表示当前是第几个完成的 block
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*rounded_C);  // 复用共享内存
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);  // 返回旧值
    }
    __syncthreads();
    
    // 最后一个完成的 block (tmp_flag == gridDim.x-1) 负责最终归约
    if (*tmp_flag == gridDim.x-1) {
        // 从所有 block 的 scratch 区域读取并累加
        for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        __syncthreads();

        // ========== 第四步: 转换精度并写出最终结果 ==========
        // FP32 -> BF16/FP16 转换，并累加到全局 dweight/dbias
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            // 读取现有的全局梯度 (需要累加)
            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            
            // 从共享内存读取本轮计算的梯度，累加并写回
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    dbias128[x] = (floatX)(s_db[i] + (float)dbias128[x]);    // 累加 dbias
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]); // 累加 dweight
                }
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

// ============================================================================
// Kernel 启动器 (Host 端接口)
// ============================================================================

/**
 * layernorm_forward - LayerNorm 前向传播启动器
 * 
 * y = (x - mean) / sqrt(var + eps) * weight + bias
 * 
 * @param out:    输出张量 [B, T, C]，归一化后的结果
 * @param mean:   均值输出 [B * T]，供反向传播使用 (可为 nullptr 仅推理时)
 * @param rstd:   逆标准差输出 [B * T]，供反向传播使用 (可为 nullptr)
 * @param inp:    输入张量 [B, T, C]
 * @param weight: 缩放参数 gamma [C]
 * @param bias:   偏移参数 beta [C]
 * @param B:      批次大小
 * @param T:      序列长度
 * @param C:      隐藏维度
 * @param stream: CUDA 流
 * 
 * 实现策略:
 * - 优先使用 kernel6 (共享内存优化版)
 * - 若共享内存不足 (>48KB 需要特殊设置)，退回 kernel3
 */
void layernorm_forward(floatX* out, float* mean, float* rstd,
                       floatX* inp, const floatX* weight, const floatX* bias,
                       int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();  // NVTX 性能标记
    
    const int block_size = 256;
    int block_y = block_size / WARP_SIZE;  // 每 block 处理的行数
    const int N = B * T;                    // 总 token 数
    const int grid_size = CEIL_DIV(N, block_y);
    
    // 共享内存: [weight(C)] + [bias(C)] + [block_y 行输入缓存]
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // 尝试设置大于 48KB 的共享内存
    // 如果失败则回退到无共享内存版本
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(layernorm_forward_kernel6, 
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaCheck(cudaGetLastError());
    
    if (status == cudaSuccess) {
        // 使用优化版 kernel (共享内存缓存 weight/bias)
        layernorm_forward_kernel6<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(
            out, mean, rstd, inp, weight, bias, N, C);
    } else {
        // 回退到无共享内存版本
        const int grid_size_fb = CEIL_DIV(N * WARP_SIZE, block_size);
        layernorm_forward_kernel3<<<grid_size_fb, block_size, 0, stream>>>(
            out, mean, rstd, inp, weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

/**
 * residual_forward - 残差连接前向传播
 * 
 * out = inp1 + inp2
 * 
 * @param out:    输出张量 [N] (展平后的元素总数)
 * @param inp1:   输入1 [N]
 * @param inp2:   输入2 [N]
 * @param N:      元素总数 (必须是 block_size * x128::size 的倍数)
 * @param stream: CUDA 流
 */
void residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    assert(N % (block_size * x128::size) == 0);  // 确保对齐
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    residual_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2);
    cudaCheck(cudaGetLastError());
}

/**
 * fused_residual_forward5 - 融合残差+LayerNorm 前向传播
 * 
 * residual = inp1 + inp2
 * normed = LayerNorm(residual)
 * 
 * @param residual: 残差输出 [N, C]
 * @param normed:   归一化输出 [N, C]
 * @param mean:     均值输出 [N]
 * @param rstd:     逆标准差输出 [N]
 * @param inp1:     输入1 [N, C]
 * @param inp2:     输入2 [N, C]
 * @param weight:   LayerNorm 缩放参数 [C]
 * @param bias:     LayerNorm 偏移参数 [C]
 * @param N:        token 数量 (B * T)
 * @param C:        隐藏维度
 * @param stream:   CUDA 流
 * 
 * 融合优势:
 * - 减少全局内存访问 (residual 不需要先写后读)
 * - 共享内存缓存提高效率
 * 
 * 回退策略:
 * - 若共享内存不足，分别调用 residual_forward + layernorm_forward
 */
void fused_residual_forward5(floatX* residual, floatX* normed, float* mean, float* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, cudaStream_t stream) {
    const int block_size = 256;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = CEIL_DIV(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // 尝试设置大共享内存，失败则回退
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_residual_forward_kernel5, 
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaCheck(cudaGetLastError());
    
    if(status == cudaSuccess) {
        // 使用融合 kernel
        fused_residual_forward_kernel5<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(
            residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    } else {
        // 回退: 分开执行两个操作
        residual_forward(residual, inp1, inp2, N*C, stream);
        layernorm_forward(normed, mean, rstd, residual, weight, bias, N, 1, C, stream);
    }
    cudaCheck(cudaGetLastError());
}

/**
 * layernorm_backward - LayerNorm 反向传播启动器
 * 
 * 计算三个梯度:
 *   - dinp: 输入梯度，反向传播到前一层
 *   - dweight: gamma 参数梯度
 *   - dbias: beta 参数梯度
 * 
 * @param dinp:    输出，输入梯度 [B, T, C]，累加模式 (+=)
 * @param dweight: 输出，weight 梯度 [C]，累加模式 (+=)
 * @param dbias:   输出，bias 梯度 [C]，累加模式 (+=)
 * @param scratch: 临时存储 [需要 gridDim.x * 2 * C + 32 floats]
 * @param dout:    输入，输出梯度 [B, T, C]
 * @param inp:     输入，前向的输入 [B, T, C]
 * @param weight:  输入，缩放参数 [C]
 * @param mean:    输入，前向保存的均值 [B * T]
 * @param rstd:    输入，前向保存的逆标准差 [B * T]
 * @param B:       批次大小
 * @param T:       序列长度
 * @param C:       隐藏维度
 * @param stream:  CUDA 流
 * 
 * 注意:
 * - 调用前需确保 scratch 已分配足够空间
 * - dinp/dweight/dbias 使用 += 累加，支持梯度累积
 * - grid_size 基于 SM 数量，每 SM 2 个 block
 */
void layernorm_backward(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, 
                        const float* mean, const float* rstd,
                        int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    
    const int block_size = 512;
    const int blocks_per_sm = 2;  // 每 SM 2 个 block，平衡占用率和缓存
    const int grid_size = blocks_per_sm * deviceProp.multiProcessorCount;
    
    // 共享内存: [dbias_accum] + [dweight_accum] + [warp 间通信缓冲]
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    size_t shared_mem_size = (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);

    // 重置原子计数器 (用于确定最后一个完成的 block)
    cudaCheck(cudaMemsetAsync(scratch, 0, 1 * sizeof(float), stream));
    
    layernorm_backward_kernel10<<<grid_size, block_size, shared_mem_size, stream>>>(
        dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}
