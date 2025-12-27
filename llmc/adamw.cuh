/*
AdamW kernel
AdamW优化器的CUDA实现

AdamW是Adam优化器的变体，将权重衰减(weight decay)从梯度更新中解耦出来。
标准Adam中，权重衰减被添加到梯度中，导致L2正则化效果受学习率影响。
AdamW直接在参数更新时应用权重衰减，实现真正的权重衰减正则化。

AdamW更新公式:
  m_t = β1 * m_{t-1} + (1 - β1) * g_t           // 一阶矩估计(动量)
  v_t = β2 * v_{t-1} + (1 - β2) * g_t^2         // 二阶矩估计(RMSprop)
  m̂_t = m_t / (1 - β1^t)                        // 偏差修正后的一阶矩
  v̂_t = v_t / (1 - β2^t)                        // 偏差修正后的二阶矩
  θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})  // 参数更新(含权重衰减)

其中:
  - g_t: 当前梯度
  - m_t, v_t: 一阶和二阶矩估计
  - β1, β2: 指数衰减率(通常β1=0.9, β2=0.999)
  - ε: 数值稳定性常数(防止除零)
  - λ: 权重衰减系数
  - lr: 学习率
*/

// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

/*
线性插值函数(lerp - Linear Interpolation)
使用FMA(Fused Multiply-Add)指令优化，仅需2次浮点运算(朴素实现需要3次)

数学公式: lerp(start, end, weight) = start + weight * (end - start)
优化形式: fma(weight, end, fma(-weight, start, start))
        = weight * end + (-weight * start + start)
        = weight * end + start * (1 - weight)
        = start + weight * (end - start)  ✓

注意: 在AdamW中，lerp用于计算指数移动平均:
  m = lerp(grad, m, beta1) 等价于 m = beta1 * m + (1 - beta1) * grad

Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
*/
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

/*
AdamW单参数更新的设备函数(Device Function)
每个CUDA线程负责更新一个参数

模板参数:
  - Tp: 参数存储类型(可以是float/half/bfloat16等低精度类型，用于前向传播)
  - Tg: 梯度存储类型

函数参数:
  - params_memory:        模型参数数组(低精度版本，用于前向传播)
  - master_params_memory: 主参数数组(float32高精度版本，用于优化器更新，可为NULL)
  - grads_memory:         梯度数组
  - m_memory:             一阶矩估计数组(动量)
  - v_memory:             二阶矩估计数组(自适应学习率)
  - num_parameters:       参数总数
  - learning_rate:        学习率
  - beta1:                一阶矩衰减率(通常0.9)
  - beta2:                二阶矩衰减率(通常0.999)
  - beta1_correction:     一阶矩偏差修正因子 = 1 - β1^t
  - beta2_correction:     二阶矩偏差修正因子 = 1 - β2^t
  - eps:                  数值稳定性常数(防止除零，通常1e-8)
  - weight_decay:         权重衰减系数(L2正则化强度)
  - grad_scale:           梯度缩放因子(用于混合精度训练中的梯度缩放)
  - seed:                 随机数种子(用于随机舍入)

混合精度训练策略:
  - 前向传播使用低精度参数(params_memory)以节省显存和提高计算速度
  - 优化器使用高精度主参数(master_params_memory)以保证数值精度
  - 使用随机舍入(stochastic rounding)将高精度参数转换为低精度
*/
template <typename Tp, typename Tg>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed) {
    // 计算当前线程负责的参数索引
    // blockIdx.x * blockDim.x: 当前block的起始索引
    // threadIdx.x: 线程在block内的偏移
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：确保不访问越界内存
    if (idx >= num_parameters) { return; }

    // ========== 第1步: 获取当前参数的梯度和矩估计 ==========
    // grad_scale用于混合精度训练中的梯度缩放(loss scaling)
    // 将梯度类型转换为float进行高精度计算
    float grad = grad_scale * (float)grads_memory[idx];
    float m = m_memory[idx];  // 一阶矩(动量)
    float v = v_memory[idx];  // 二阶矩(自适应学习率分母)

    // ========== 第2步: 更新一阶矩估计(动量项) ==========
    // m_t = β1 * m_{t-1} + (1 - β1) * g_t
    // lerp(grad, m, beta1) = beta1 * m + (1 - beta1) * grad
    m = lerp(grad, m, beta1);
    m_memory[idx] = m;  // 写回显存，供下一次迭代使用

    // ========== 第3步: 更新二阶矩估计(RMSprop项) ==========
    // v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
    // lerp(grad*grad, v, beta2) = beta2 * v + (1 - beta2) * grad^2
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;  // 写回显存

    // ========== 第4步: 偏差修正 ==========
    // 由于m和v初始化为0，在训练初期会偏向0
    // 通过除以(1 - β^t)来修正这个偏差
    m /= beta1_correction;  // m̂_t = m_t / (1 - β1^t)
    v /= beta2_correction;  // v̂_t = v_t / (1 - β2^t)

    // ========== 第5步: 获取当前参数值 ==========
    // 优先使用master_params(高精度)，否则使用低精度params
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];

    // ========== 第6步: 计算新参数值 ==========
    // AdamW更新公式: θ = θ - lr * (m̂ / (√v̂ + ε) + λ * θ)
    // 其中 m̂/(√v̂ + ε) 是自适应梯度项，λ*θ 是权重衰减项
    float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));

    // ========== 第7步: 更新低精度参数(用于前向传播) ==========
    // 使用随机舍入(stochastic rounding)将float32转换为低精度类型
    // 随机舍入可以在统计意义上保持无偏，比简单截断更好地保留梯度信息
    stochastic_rounding(param, &params_memory[idx], seed);

    // ========== 第8步: 更新高精度主参数(用于下次优化器更新) ==========
    // master_params保持float32精度，确保优化器更新的数值稳定性
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

/*
AdamW CUDA核函数(Kernel Function) - 版本3
支持多层(slices)并行更新，通过2D grid实现

网格配置: dim3(num_blocks, num_slices)
  - blockIdx.x: 参数块索引(沿参数维度划分)
  - blockIdx.y: 层/切片索引(用于多层并行处理)

步长参数(stride):
  - w_stride: 参数数组的层间步长(params_memory中每层的偏移量)
  - g_stride: 梯度数组的层间步长
  - s_stride: 状态数组(m, v, master_params)的层间步长

使用blockIdx.y来索引不同的层，实现多层参数的并行更新
这种设计可以让不同层的参数更新在同一个kernel调用中完成
*/
template <typename Tp, typename Tg>
__global__ void adamw_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    // 调用adamw_update设备函数，根据blockIdx.y偏移到对应层的数据
    // blockIdx.y * stride: 计算当前层在各数组中的起始位置
    adamw_update(params_memory + blockIdx.y * w_stride,                                    // 当前层的参数
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,  // 当前层的主参数
                 grads_memory + blockIdx.y * g_stride,                                     // 当前层的梯度
                 m_memory + blockIdx.y * s_stride,                                         // 当前层的一阶矩
                 v_memory + blockIdx.y * s_stride,                                         // 当前层的二阶矩
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed
                 );
}

/*
从主参数初始化低精度参数的CUDA核函数

用途:
  在训练开始时或加载checkpoint后，将高精度主参数(float32)转换为
  低精度参数(half/bfloat16等)，用于前向传播

参数:
  - params_memory:        目标低精度参数数组
  - master_params_memory: 源高精度参数数组(float32)
  - num_parameters:       每层的参数数量
  - w_stride:             低精度参数的层间步长
  - s_stride:             高精度参数的层间步长
  - seed:                 随机舍入的种子

网格配置: dim3(num_blocks, num_slices)
  - 使用与adamw_update相同的block_size(512)以确保随机数生成一致性
*/
template <typename Tp>
__global__ void init_from_master_kernel(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                                          ptrdiff_t w_stride, ptrdiff_t s_stride, unsigned int seed) {
    // 计算当前线程处理的参数索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx >= num_parameters) { return; }

    // 根据层索引(blockIdx.y)调整指针到对应层的起始位置
    params_memory += blockIdx.y * w_stride;
    master_params_memory += blockIdx.y * s_stride;

    // 使用随机舍入将float32转换为低精度类型
    // 随机舍入确保转换在统计上无偏，有助于保持训练精度
    stochastic_rounding(master_params_memory[idx], &params_memory[idx], seed);
}

// ----------------------------------------------------------------------------
// Host端包装函数(Host Wrapper Functions)
// 这些函数在CPU端调用，负责配置kernel启动参数并调用CUDA kernel

/*
AdamW优化器更新 - Host端入口函数

功能:
  执行一步AdamW优化器更新，更新所有层的模型参数

参数:
  - params_memory:        模型参数(低精度，用于前向传播)
  - master_params_memory: 主参数(float32，用于优化器更新，可为NULL)
  - grads_memory:         梯度数组
  - m_memory:             一阶矩估计(动量)
  - v_memory:             二阶矩估计
  - num_parameters:       每层的参数数量
  - w_stride, g_stride, s_stride: 各数组的层间步长
  - num_slices:           层/切片数量
  - learning_rate:        学习率
  - beta1, beta2:         动量衰减率
  - t:                    当前训练步数(用于计算偏差修正)
  - eps:                  数值稳定性常数
  - weight_decay:         权重衰减系数
  - grad_scale:           梯度缩放因子
  - seed:                 随机数种子
  - stream:               CUDA流(用于异步执行)
*/
template <typename Tp, typename Tg>
void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, cudaStream_t stream) {
    // 配置kernel启动参数
    int block_size = 512;  // 每个block的线程数
    int num_blocks = CEIL_DIV(num_parameters, block_size);  // 需要的block数(向上取整)

    // 计算偏差修正因子
    // 在训练初期，由于m和v初始化为0，估计值会偏小
    // 通过除以(1 - β^t)来修正这个偏差，随着t增大，修正因子趋近于1
    float beta1_correction = 1.0f - powf(beta1, t);  // 1 - β1^t
    float beta2_correction = 1.0f - powf(beta2, t);  // 1 - β2^t

    // 启动CUDA kernel
    // grid: dim3(num_blocks, num_slices) - 2D网格，x维度处理参数，y维度处理不同层
    // block: block_size - 每个block有512个线程
    // shared memory: 0 - 不使用共享内存
    // stream: 指定CUDA流，支持异步执行
    adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                                                         m_memory, v_memory, num_parameters, w_stride, g_stride, s_stride,
                                                         learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                         grad_scale, seed);
    // 检查kernel启动是否成功
    cudaCheck(cudaGetLastError());
}

/*
从主参数初始化低精度参数 - Host端入口函数

功能:
  将高精度主参数(float32)转换为低精度参数，用于混合精度训练
  通常在以下场景调用:
    1. 训练开始时初始化低精度参数
    2. 加载checkpoint后同步低精度参数
    3. 某些情况下需要重新同步参数时

参数:
  - params_memory:        目标低精度参数数组
  - master_params_memory: 源高精度参数数组
  - num_parameters:       每层的参数数量
  - w_stride, s_stride:   低精度/高精度参数的层间步长
  - num_slices:           层/切片数量
  - seed:                 随机数种子(用于随机舍入)
  - stream:               CUDA流

注意: block_size必须与adamw_update保持一致(512)
      这确保了随机数生成器(RNG)的状态与优化器更新时一致
      从而保证相同seed在不同调用间产生相同的随机舍入结果
*/
template <typename Tp>
void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudaStream_t stream) {
    // block_size必须与adamw_update匹配，确保RNG行为一致
    int block_size = 512;
    int num_blocks = CEIL_DIV(num_parameters, block_size);

    // 启动kernel: 2D网格配置与adamw_kernel3相同
    init_from_master_kernel<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>
                             (params_memory, master_params_memory, num_parameters, w_stride, s_stride, seed);

    // 错误检查
    cudaCheck(cudaGetLastError());
}
