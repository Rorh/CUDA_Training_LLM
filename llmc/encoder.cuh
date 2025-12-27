/*
 * GPT-2 编码器 (Encoder) 实现
 * ============================
 * 
 * 编码器将两种嵌入组合在一起:
 * 1. 词嵌入 (Token Embedding, wte): 将词汇表中的token映射到向量空间
 * 2. 位置嵌入 (Position Embedding, wpe): 编码token在序列中的位置信息
 * 
 * 前向传播: output = wte[token_id] + wpe[position]
 * 反向传播: 梯度分别流向 dwte 和 dwpe，由不同的kernel处理
 * 
 * 关键设计:
 * - 使用128位向量化加载/存储 (x128) 提高内存带宽利用率
 * - wte反向传播使用bucket分组策略保证确定性 (避免atomicAdd的不确定性)
 * - 使用随机舍入 (stochastic rounding) 从FP32转换到BF16
 */
#include <assert.h>
#include <stdint.h>
#include <utility>              // std::pair - 用于bucket排序
#include <vector>               // 存储bucket数据
#include <algorithm>            // std::sort
#include <unordered_map>        // bucket哈希表
// llmc 内部头文件
#include "cuda_common.h"        // floatX, CUDA宏定义
#include "cuda_utils.cuh"       // x128向量类型, load128, store128等

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * encoder_forward_kernel3 - 编码器前向传播核函数
 * 
 * 功能: 将词嵌入和位置嵌入相加得到最终的token表示
 *       out[b,t,:] = wte[inp[b,t],:] + wpe[t,:]
 * 
 * 向量化策略:
 * - 每个线程处理 x128::size 个连续通道 (BF16下为8个元素)
 * - 使用128位向量化加载 (load128cs) 和存储 (store128)
 * - cs = cache streaming hint, 提示数据不会被重复读取
 * 
 * @param out:  输出张量 [B, T, C]
 * @param inp:  输入token索引 [B, T], 每个元素是词汇表中的索引
 * @param wte:  词嵌入矩阵 [V, C], V是词汇表大小
 * @param wpe:  位置嵌入矩阵 [T, C]
 * @param B:    批次大小
 * @param T:    序列长度
 * @param C:    嵌入维度 (hidden size)
 */
__global__ void encoder_forward_kernel3(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    // 每个线程处理x128::size个连续元素，计算全局起始索引
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;  // 总元素数
    if (idx >= N) { return; }  // 边界检查

    // 从线性索引计算三维坐标 (b, t, c)
    int bt = idx / C;       // batch*time 的组合索引
    int b = bt / T;         // batch 索引
    int t = bt % T;         // time (position) 索引  
    int c = idx % C;        // channel 起始索引

    // 获取当前位置的token id，用于索引词嵌入
    int ix = inp[b * T + t];

    // 计算各数组的指针偏移
    floatX* out_btc = out + b * T * C + t * C + c;   // 输出位置
    const floatX* wte_ix = wte + ix * C + c;         // 词嵌入位置 (按token id索引)
    const floatX* wpe_tc = wpe + t * C + c;          // 位置嵌入位置 (按position索引)

    // 向量化加载和计算
    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);  // 加载8个词嵌入元素 (cs=cache streaming)
    x128 wpe128 = load128cs(wpe_tc);  // 加载8个位置嵌入元素
    
    // 逐元素相加 (在float精度下计算，避免精度损失)
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
    }
    store128(out_btc, packed_out);  // 向量化存储结果
}

/**
 * wte_backward_kernel - 词嵌入反向传播核函数
 * 
 * 核心挑战:
 * - 多个位置可能使用相同的token，导致梯度需要累加到同一个dwte位置
 * - 使用atomicAdd会导致非确定性 (执行顺序不固定)
 * 
 * 解决方案 - Bucket分组策略:
 * 1. CPU预处理: 将所有 (token_id, channel_group) 相同的位置分到同一个bucket
 * 2. GPU处理: 每个block处理一个bucket，内部无竞争，结果确定性
 * 
 * Bucket结构:
 * - 每个bucket对应一个 (token_id, channel_group) 组合
 * - channel_group = WARP_SIZE * x128::size 个连续通道
 * - bucket内所有dout梯度累加后写入对应的dwte位置
 * 
 * 并行策略:
 * - 每个block处理一个bucket
 * - block内有 BLOCK_SIZE/WARP_SIZE 个warp
 * - 每个warp的每个lane处理 x128::size 个通道
 * - bucket按大小降序排列，大bucket先处理，避免GPU空闲
 * 
 * @param dwte:            词嵌入梯度输出 [V, C]
 * @param bucket_info:     bucket元信息 (start_idx, size, token_id, c_group)
 * @param workload_indices: bucket内的bt索引列表
 * @param dout:            上游梯度 [B*T, C]
 * @param inp:             输入token索引 [B, T]
 * @param seed:            随机舍入种子 (保证确定性)
 * @param B, T, C:         维度参数
 */
template <int BLOCK_SIZE=256>
__global__ void wte_backward_kernel(floatX* dwte,
                                    const int4* bucket_info, const int* workload_indices, const floatX* dout, const int* inp,
                                    unsigned int seed, int B, int T, int C) {
    // ========== 1. 解析线程和bucket信息 ==========
    int bucket = blockIdx.x;                    // 当前block处理的bucket索引
    int warp_id = threadIdx.x / WARP_SIZE;      // warp编号 (0 ~ BLOCK_SIZE/32-1)
    int lane_id = threadIdx.x % WARP_SIZE;      // lane编号 (0 ~ 31)
    int c_per_warp = WARP_SIZE * x128::size;    // 每个warp处理的通道数

    // 从bucket_info解析bucket元信息
    // int4.x = bucket在workload_indices中的起始索引
    // int4.y = bucket中的元素数量
    // int4.z = token_id (词汇表索引)
    // int4.w = channel_group索引
    int bucket_start_idx = bucket_info[bucket].x;
    int bucket_size = bucket_info[bucket].y;
    int bucket_ix = bucket_info[bucket].z;  // token_id
    int c = bucket_info[bucket].w * c_per_warp + (lane_id * x128::size);  // 该线程处理的通道起始位置

    // ========== 2. 边界检查 ==========
    // 如果C不是c_per_warp的倍数，某些lane可能超出范围
    if (c >= C) { return; }
    // 小bucket时，部分warp没有工作，直接返回
    if (warp_id >= bucket_size) { return; }

    // ========== 3. 累加梯度 (每个warp独立处理部分元素) ==========
    float accum[x128::size] = {0.0f};  // 本线程的累加器
    __shared__ float accum_shared[x128::size * BLOCK_SIZE];  // warp间共享内存

    // 每个warp处理bucket中的一部分元素，stride为block中的warp数
    for(int item = warp_id; item < bucket_size; item += BLOCK_SIZE/WARP_SIZE) {
        int bt = workload_indices[bucket_start_idx + item];  // 获取bt索引

        // 加载对应位置的上游梯度
        const floatX* dout_btc = dout + bt * C + c;
        x128 packed_inp1 = load128cs(dout_btc);
        for (int k = 0; k < packed_inp1.size; k++) {
            accum[k] += (float)packed_inp1[k];  // 累加到FP32
        }
    }

    // ========== 4. Warp间规约 ==========
    if (warp_id != 0) {
        // 非warp0: 将结果写入shared memory后退出
        for (int k = 0; k < x128::size; k++) {
            accum_shared[threadIdx.x + k * BLOCK_SIZE] = accum[k];
        }
        return;  // 只有warp 0需要继续执行
    }

    // Warp 0: 提前发起dwte的读取，隐藏内存延迟
    floatX* dwte_ix = dwte + bucket_ix * C + c;
    x128 packed_in_out = load128(dwte_ix);

    // 等待其他warp写入shared memory (已返回的线程视为同步)
    __syncthreads();

    // 将其他warp的结果累加到warp 0的寄存器中
    for (int i = threadIdx.x+WARP_SIZE; i < min(BLOCK_SIZE, bucket_size*WARP_SIZE); i += WARP_SIZE) {
        for (int k = 0; k < x128::size; k++) {
            accum[k] += accum_shared[i + k * BLOCK_SIZE];
        }
    }

    // ========== 5. 写回结果 (read-modify-write) ==========
    for (unsigned int k = 0; k < x128::size; k++) {
        // 使用随机舍入: FP32 -> BF16
        // seed保证确定性且每个参数位置唯一，避免SquirrelNoise5溢出问题
        stochastic_rounding(accum[k] + (float)packed_in_out[k], &packed_in_out[k], 
                           seed + bucket * WARP_SIZE + threadIdx.x + k);
    }
    store128(dwte_ix, packed_in_out);
}

/**
 * wpe_backward_kernel - 位置嵌入反向传播核函数
 * 
 * 与wte不同，wpe的确定性更容易保证:
 * - 位置嵌入只与位置t相关，与batch无关
 * - 每个(t, c)位置只有一个线程负责，无竞争
 * - 该线程循环累加所有batch的梯度后一次性写入
 * 
 * 数学公式: dwpe[t,c] = sum_{b=0}^{B-1} dout[b,t,c]
 * 
 * @param dwpe: 位置嵌入梯度输出 [T, C]
 * @param dout: 上游梯度 [B, T, C]
 * @param inp:  输入token索引 (此kernel未使用)
 * @param B, T, C: 维度参数
 * @param seed: 随机舍入种子
 */
__global__ void wpe_backward_kernel(floatX* dwpe,
                                    const floatX* dout, const int* inp,
                                    int B, int T, int C, unsigned int seed) {
    // 每个线程处理x128::size个连续通道
    // 例如 GPT-2 124M: C=768, T=1024, BF16下每warp处理256通道
    // 总共需要 T*C/x128::size 个线程
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= T * C) { return; }

    // 计算当前线程负责的 (t, c) 位置
    int t = idx / C;  // 时间步/位置索引
    int c = idx % C;  // 通道起始索引
    
    float accum[x128::size] = {0.0f};  // FP32累加器

    // 遍历所有batch，累加梯度
    for (int b = 0; b < B; b++) {
        // load128cs: cache streaming hint, 表示数据不会被重复读取
        x128 packed_dout = load128cs(dout + (b * T * C) + (t * C) + c);
        for (int k = 0; k < x128::size; k++) {
            accum[k] += (float)packed_dout[k];
        }
    }

    // 读取当前dwpe值 (支持梯度累加)
    floatX* dwpe_tc = dwpe + (t * C) + c;
    x128 packed_dwpe = load128(dwpe_tc);
    
    // 使用随机舍入写回结果
    for (unsigned int k = 0; k < x128::size; k++) {
        // 随机舍入保证期望无偏，同时保持确定性
        // seed唯一性避免SquirrelNoise5哈希函数的潜在溢出问题
        stochastic_rounding(accum[k] + (float)packed_dwpe[k], &packed_dwpe[k], seed + idx + k);
    }
    store128(dwpe_tc, packed_dwpe);
}

// ============================================================================
// Kernel Launchers - 主机端调用接口
// ============================================================================

/**
 * encoder_forward - 编码器前向传播
 * 
 * 计算: out[b,t,:] = wte[inp[b,t],:] + wpe[t,:]
 * 
 * Grid/Block配置:
 * - 每个线程处理 x128::size 个元素
 * - block_size = 256
 * - grid_size = ceil(B*T*C / (256 * x128::size))
 */
void encoder_forward(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();  // NVIDIA性能分析标记
    const int block_size = 256;
    const int N = B * T * C;
    // 每个线程处理x128::size个元素，计算需要的grid大小
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

/**
 * encoder_backward - 编码器反向传播 (完全确定性)
 * 
 * 处理流程:
 * 1. 先启动wpe_backward_kernel (与CPU预处理并行)
 * 2. CPU端将输入按(token_id, c_group)分组到bucket
 * 3. bucket按大小降序排列 (大bucket先执行，避免尾部空闲)
 * 4. 将bucket信息拷贝到GPU
 * 5. 启动wte_backward_kernel
 * 
 * 确定性保证:
 * - wpe: 每个(t,c)位置由单一线程处理，无竞争
 * - wte: bucket策略避免atomicAdd，同一dwte位置只被一个block写入
 * 
 * @param dwte:            词嵌入梯度输出 [V, C] (GPU)
 * @param dwpe:            位置嵌入梯度输出 [T, C] (GPU)
 * @param scratch:         GPU临时缓冲区 (存放bucket_info和workload_indices)
 * @param workload_indices: CPU临时缓冲区 (bucket内的bt索引)
 * @param bucket_info:     CPU临时缓冲区 (bucket元信息)
 * @param dout:            上游梯度 [B, T, C] (GPU)
 * @param inp:             输入token索引 [B, T] (GPU)
 * @param inputs_cpu:      输入token索引的CPU副本 (用于bucket分组)
 * @param seed:            随机舍入种子
 */
void encoder_backward(floatX* dwte, floatX* dwpe, floatX* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    NVTX_RANGE_FN();

    // ========== 阶段1: 先启动wpe kernel (与CPU预处理重叠) ==========
    const int block_size = 256;
    const int N = T * C / x128::size;  // wpe总线程数
    const int grid_size = CEIL_DIV(N, block_size);
    wpe_backward_kernel<<<grid_size, block_size, 0, stream>>>(dwpe, dout, inp, B, T, C, seed);
    cudaCheck(cudaGetLastError());

    // 检查scratch缓冲区大小是否足够
    int num_c_groups = CEIL_DIV(C, x128::size * WARP_SIZE);  // 通道分组数
    assert(B*T*num_c_groups * (sizeof(int4)+sizeof(int)) <= B*T*3*C * sizeof(floatX));

    // ========== 阶段2: CPU预处理 - 构建bucket ==========
    // 每个bucket = (token_id, c_group) 的组合
    // 相同token在不同位置出现时，梯度需要累加
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // 将bt、c_group、token_id打包到64位整数中
            // 低32位: bt索引
            // 32-41位: c_group (10位)
            // 42-61位: token_id (20位)
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)inputs_cpu[bt]<<42ULL);
            
            // bucket key = c_group + num_c_groups * token_id
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // ========== 阶段3: 按bucket大小降序排列 ==========
    // 大bucket先执行，避免GPU尾部空闲 (长任务先启动)
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(),
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, 
                 const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();  // 降序
              });

    // ========== 阶段4: 序列化bucket信息 ==========
    int num_buckets = buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    
    for (const auto& bucket : sortedBuckets) {
        // 填充bucket元信息 (int4)
        bucket_info[bucket_index].x = workload_index;                                    // 起始索引
        bucket_info[bucket_index].y = bucket.second.size();                              // bucket大小
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL<<20ULL)-1);   // token_id
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL<<10ULL)-1);   // c_group

        // 填充workload_indices (解包bt索引)
        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL<<31ULL)-1ULL));
        }
        bucket_index++;
    }

    // ========== 阶段5: 异步拷贝到GPU ==========
    // 使用cudaMemcpyAsync避免阻塞CPU
    int4* d_bucket_info = (int4*)scratch;
    int*  d_workload_indices = (int*)(scratch + B*T*num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, stream));

    // ========== 阶段6: 启动wte kernel ==========
    // 每个block处理一个bucket，共num_buckets个block
    wte_backward_kernel<256><<<num_buckets, 256, 0, stream>>>(dwte, d_bucket_info, d_workload_indices, dout, inp, seed, B, T, C);
    cudaCheck(cudaGetLastError());
}
