/*
 * GeLU (Gaussian Error Linear Unit) 激活函数层
 * 
 * 数学定义:
 *   GELU(x) = x * Φ(x)，其中 Φ(x) 是标准正态分布的累积分布函数(CDF)
 * 
 * 近似公式 (本实现使用):
 *   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * 
 * 这个近似计算高效，广泛用于 Transformer 模型 (GPT, BERT 等)
 */

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA 内核函数

// √(2/π) ≈ 0.7978845608，用于 GeLU 近似公式
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

/*
 * GeLU 前向传播内核 (向量化 x128 版本)
 * 
 * 公式: GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * 
 * 令: cube = 0.044715 * x³
 *     tanh_arg = √(2/π) * (x + cube)
 * 则: GELU(x) = 0.5 * x * (1 + tanh(tanh_arg))
 * 
 * @param out: 输出张量
 * @param inp: 输入张量
 */
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp) {
    // 每个线程处理 x128::size 个元素 (128位 = 8个fp16 或 4个fp32)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_out;
    // load128cs: 缓存流式加载，绕过 L1 缓存 (用于不会重复使用的数据)
    x128 packed_inp = load128cs(inp + idx);
    
    for (int k = 0; k < packed_inp.size; k++) {
        float xi = (float)packed_inp[k];           // 转换为 float 进行计算
        float cube = 0.044715f * xi * xi * xi;     // 0.044715 * x³
        // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + cube)))
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
    }
    // store128: 普通存储 (保留在缓存中，可能对后续操作有用)
    store128(out + idx, packed_out);
}

/*
 * GeLU 反向传播内核 (原地计算梯度)
 * 
 * 使用链式法则求导:
 *   令: t = tanh(√(2/π) * (x + 0.044715 * x³))
 *       sech² = 1 / cosh²(...)  (tanh 的导数是 sech²)
 * 
 *   d(GELU)/dx = 0.5 * (1 + t) + 0.5 * x * sech² * √(2/π) * (1 + 3 * 0.044715 * x²)
 * 
 * 最终梯度: d_inp = local_grad * d_out  (与上游梯度的链式法则)
 * 
 * @param d_in_out: 输入时为 dout (上游梯度)，输出时为 dinp (本层梯度)
 * @param inp: 前向传播时的输入值 (用于计算局部梯度)
 */
__global__ void gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_dinp;
    x128 packed_inp = load128cs(inp + idx);        // 加载原始输入
    x128 packed_dout = load128(d_in_out + idx);    // 加载上游梯度
    
    for (int k = 0; k < packed_inp.size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;                    // 0.044715 * x³
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);     // √(2/π) * (x + cube)
        float tanh_out = tanhf(tanh_arg);                      // t = tanh(tanh_arg)
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);       // sech² = 1/cosh²
        
        // local_grad = d(GELU)/dx = 0.5*(1+t) + x*0.5*sech²*√(2/π)*(1 + 3*0.044715*x²)
        float local_grad = 0.5f * (1.0f + tanh_out) 
                         + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        
        // 链式法则: dinp = local_grad * dout
        packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}

// ----------------------------------------------------------------------------
// 内核启动器

/*
 * GeLU 前向传播
 * @param out: 输出张量 [N]
 * @param inp: 输入张量 [N]
 * @param N: 元素总数
 * @param stream: CUDA 流
 */
void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    // N 必须能被 block_size * x128::size 整除 (确保所有元素都被处理)
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

/*
 * GeLU 反向传播 (原地计算)
 * @param d_in_out: 输入为 dout，输出为 dinp (原地修改)
 * @param inp: 前向传播的输入值
 * @param N: 元素总数
 * @param stream: CUDA 流
 */
void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp);
    cudaCheck(cudaGetLastError());
}