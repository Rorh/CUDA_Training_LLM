/**
 * @file cuda_utils.cuh
 * @brief CUDA 设备代码工具集
 *
 * 本文件提供了一系列用于 CUDA 设备端（__device__）代码的工具函数和数据结构，包括：
 * - Packed128: 128位打包数据结构，用于优化内存访问
 * - DType: 数据类型枚举和转换工具
 * - Warp/Block 级别的归约原语
 * - 内存管理工具
 * - 随机数生成（用于随机舍入）
 */

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda_common.h"

// ============================================================================
// Packed128 数据结构
// ============================================================================
/**
 * @brief 128位对齐的打包数据结构
 *
 * 强制编译器使用 128 位加载/存储指令（LDG.128 和 STS.128），
 * 这在支持的 GPU 上可以显著提高内存带宽利用率。
 * 类似于 float4 的用法，但支持任意精度的数据类型。
 *
 * @tparam ElementType 元素类型（如 float, half, nv_bfloat16）
 *
 * 内存布局：
 * - 总大小固定为 16 字节（128 位）
 * - 元素数量 = 16 / sizeof(ElementType)
 *   - float:       4 个元素
 *   - half/bf16:   8 个元素
 */
template<class ElementType>
struct alignas(16) Packed128 {
    // 默认构造函数
    Packed128() = default;

    /**
     * @brief 从 int4 位模式构造 Packed128
     * @param bits 原始位数据，以 int4 形式传入（128位）
     * @note explicit 关键字防止隐式转换，必须显式调用
     */
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    /**
     * @brief 创建所有元素相同值的 Packed128
     * @param value 要填充的常量值
     * @return 填充了指定值的 Packed128 对象
     */
    __device__  static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }

    /**
     * @brief 创建全零的 Packed128
     * @return 所有元素为 0.0f 的 Packed128
     */
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }

    /**
     * @brief 创建全一的 Packed128
     * @return 所有元素为 1.0f 的 Packed128
     */
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    /**
     * @brief 下标访问运算符（可修改）
     * @param index 元素索引，范围 [0, size)
     * @return 对应元素的引用
     */
    __device__ ElementType& operator[](int index) {
        return payload[index];
    }

    /**
     * @brief 下标访问运算符（只读）
     * @param index 元素索引，范围 [0, size)
     * @return 对应元素的常量引用
     */
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }

    /**
     * @brief 获取原始位表示
     * @return 以 int4 形式返回的 128 位数据，用于存储操作
     */
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }

    /** @brief 每个 Packed128 包含的元素数量（编译时常量） */
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    /** @brief 实际存储数据的数组 */
    ElementType payload[size];
};

// ============================================================================
// Packed128 加载/存储函数
// ============================================================================

/**
 * @brief 从对齐的内存地址加载 128 位数据
 * @tparam ElementType 元素类型
 * @param address 源地址（必须 16 字节对齐）
 * @return 加载的 Packed128 数据
 * @note 使用默认缓存策略（L1 + L2）
 */
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}

/**
 * @brief 从对齐的内存地址加载 128 位数据（流式缓存提示）
 * @tparam ElementType 元素类型
 * @param address 源地址（必须 16 字节对齐）
 * @return 加载的 Packed128 数据
 * @note 使用 __ldcs 指令，提示数据只使用一次，减少缓存污染
 *       适用于流式访问模式，数据不会被重复使用的场景
 */
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}

/**
 * @brief 将 Packed128 存储到对齐的内存地址
 * @tparam ElementType 元素类型
 * @param target 目标地址（必须 16 字节对齐）
 * @param value 要存储的 Packed128 数据
 * @note 使用默认缓存策略
 */
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}

/**
 * @brief 将 Packed128 存储到对齐的内存地址（流式缓存提示）
 * @tparam ElementType 元素类型
 * @param target 目标地址（必须 16 字节对齐）
 * @param value 要存储的 Packed128 数据
 * @note 使用 __stcs 指令，绕过 L1 缓存直接写入
 *       适用于写入后不会立即读取的数据
 */
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}

/**
 * @brief 将 Packed128 存储到对齐的内存地址（L2 缓存，绕过 L1）
 * @tparam ElementType 元素类型
 * @param target 目标地址（必须 16 字节对齐）
 * @param value 要存储的 Packed128 数据
 * @note 使用 __stcg 指令，数据缓存在 L2 但绕过 L1
 *       适用于可能被其他 SM 读取但本 SM 不再使用的数据
 */
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// 常用类型别名
typedef Packed128<float> f128;    ///< 128位打包的 float（4 个 float）
typedef Packed128<floatX> x128;   ///< 128位打包的 floatX（精度取决于编译配置）

// ============================================================================
// DType 数据类型支持
// ============================================================================

/**
 * @brief 张量数据类型枚举
 *
 * 用于在运行时标识张量的数据类型，支持动态类型检查和转换。
 * 使用 uint8_t 作为底层类型以节省内存。
 */
enum class DType : uint8_t {
    FP32,  ///< 32位单精度浮点（float）
    FP16,  ///< 16位半精度浮点（half）
    BF16   ///< 16位脑浮点（bfloat16），指数位更多，动态范围更大
};

/**
 * @brief 获取数据类型的字节大小
 * @param type 数据类型枚举值
 * @return 该类型单个标量的字节数
 *         - FP32: 4 字节
 *         - FP16: 2 字节
 *         - BF16: 2 字节
 * @note 遇到未知类型会打印错误并终止程序
 */
size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief 从指针类型推断数据类型（函数重载）
 * @param f 指向数据的指针（类型用于推断，值不使用）
 * @return 对应的 DType 枚举值
 */
DType dtype_of(float* f) { return DType::FP32; }
DType dtype_of(nv_bfloat16* f) { return DType::BF16; }
DType dtype_of(half* f) { return DType::FP16; }



// ============================================================================
// 数据复制与类型转换函数
// ============================================================================

/**
 * @brief 设备端类型转换函数模板（声明）
 * @tparam Td 目标类型
 * @tparam Ts 源类型
 * @param val 要转换的值
 * @return 转换后的值
 */
template<typename Td, typename Ts>
__device__ Td cast_value(Ts val);

/**
 * @brief float -> float 转换（恒等变换）
 * @param val 输入的 float 值
 * @return 原值不变
 */
template<>
__device__ float cast_value<float, float>(float val) {
    return val;
}

/**
 * @brief half -> float 转换
 * @param val 输入的 half 值
 * @return 转换后的 float 值
 * @note 使用 CUDA 内置函数 __half2float 进行硬件加速转换
 */
template<>
__device__ float cast_value<float, half>(half val) {
    return __half2float(val);
}

/**
 * @brief bfloat16 -> float 转换
 * @param val 输入的 __nv_bfloat16 值
 * @return 转换后的 float 值
 * @note 使用 CUDA 内置函数 __bfloat162float 进行硬件加速转换
 */
template<>
__device__ float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

/**
 * @brief 数据复制和类型转换内核
 *
 * 将数据从源数组复制到目标数组，同时进行类型转换。
 * 支持 2D 网格布局，可以同时处理多个批次。
 *
 * @tparam Td 目标数据类型
 * @tparam Ts 源数据类型
 * @param dst        目标数组指针
 * @param src        源数组指针
 * @param n          每个批次的元素数量
 * @param stride_dst 目标数组的批次间步长
 * @param stride_src 源数组的批次间步长
 *
 * 网格配置：
 * - gridDim.x * blockDim.x >= n（覆盖所有元素）
 * - gridDim.y = 批次数量
 *
 * @todo 实现 grid stride loop 以提高性能
 */
template<typename Td, typename Ts>
__global__ void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // need to try grid stride looping for more perf later
    if (idx < n) {
        dst[idx + stride_dst * blockIdx.y] = cast_value<Td, Ts>(src[idx + stride_src * blockIdx.y]);
    }
}

// ============================================================================
// Warp/Block 级别通信原语
// ============================================================================

/**
 * @brief Warp 级别求和归约
 *
 * 使用 shuffle 指令在单个 warp（32线程）内进行并行求和。
 * 采用蝶式归约算法，在 log2(32)=5 次迭代内完成。
 *
 * @param val 当前线程的输入值
 * @return 整个 warp 所有线程值的总和（所有线程返回相同结果）
 *
 * 算法说明：
 * - offset=16: 线程 0-15 与 16-31 交换并相加
 * - offset=8:  线程 0-7 与 8-15 交换并相加，依此类推
 * - 最终所有线程都持有完整的归约结果
 *
 * @note 要求 warp 内所有 32 个线程都处于活跃状态
 */
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Warp 级别最大值归约
 *
 * 使用 shuffle 指令在单个 warp（32线程）内查找最大值。
 * 采用蝶式归约算法，在 log2(32)=5 次迭代内完成。
 *
 * @param val 当前线程的输入值
 * @return 整个 warp 所有线程值的最大值（所有线程返回相同结果）
 *
 * @note 要求 warp 内所有 32 个线程都处于活跃状态
 */
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/** @brief 归约函数指针类型定义 */
using reduction_func_t = float (*) (float);

/**
 * @brief Block 级别归约
 *
 * 在整个线程块内进行归约操作，支持最多 1024 个线程（32 个 warp）。
 * 采用三阶段归约：
 * 1. Warp 内归约（shuffle）
 * 2. 跨 Warp 归约（shared memory）
 * 3. 最终 Warp 内归约（shuffle）
 *
 * @tparam warp_reduction 用于 warp 级别归约的函数（如 warpReduceSum, warpReduceMax）
 * @param val           当前线程的输入值
 * @param final_sync    是否在结束时执行 __syncthreads()
 *                      - false: 默认，节省一次同步开销
 *                      - true:  在循环中调用时需要设为 true，防止共享内存竞争
 * @param out_of_bounds 超出有效 warp 数量的线程使用的填充值
 *                      - 求和时应为 0.0f
 *                      - 求最大值时应为 -INFINITY
 * @return 整个 block 的归约结果（所有线程返回相同值）
 *
 * 内存使用：
 * - 静态共享内存：128 字节（32 * sizeof(float)）
 * - 每次调用该函数都会增加 128 字节的共享内存需求
 *
 * @note 要求所有线程都参与调用，即使某些线程的值无效
 */
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    // 三阶段归约，最多支持 1024 线程：
    // 1) warp 内归约（shuffle）, 2) 跨 warp（shared memory）, 3) warp 内归约（shuffle）
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;   // warp 内的线程 ID [0, 31]
    const int warp_id = threadIdx.x / WARP_SIZE;   // warp 编号 [0, num_warps-1]
    const int num_warps = blockDim.x / WARP_SIZE;  // block 中的 warp 数量

    // 第一阶段：warp 内归约
    float warp_val = warp_reduction(val);
    // 每个 warp 的第一个线程将结果写入 shared memory
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    // 第二阶段：第一个 warp 负责跨 warp 归约
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    // 第三阶段：最终 warp 内归约
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // 仅在循环中复用共享内存时需要
    }
    return block_val;
}

/**
 * @brief 确定性全局求和内核（单 block 版本）
 *
 * 通过强制使用单个 block 来保证求和的确定性（每次运行结果相同）。
 * 多 block 并行求和可能因浮点加法的非结合性导致不同的舍入误差。
 *
 * @tparam Float 输入数据类型（float, half, bfloat16）
 * @param result 输出结果指针（单个 float 值）
 * @param values 输入数组指针
 * @param count  元素数量
 *
 * 算法：
 * 1. 每个线程以 stride=blockDim.x 遍历数组，累加到本地变量
 * 2. 使用 blockReduce 对 block 内所有线程的部分和进行归约
 * 3. 线程 0 写入最终结果
 *
 * @note 只能用单个 block 启动（gridDim.x == 1）
 */
template<class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);     // 必须是单 block！
    float thread_sum = 0;
    // 每个线程以 blockDim.x 为步长遍历数组
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    // block 内归约得到最终结果
    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) {
        *result = reduction;
    }
}

/**
 * @brief 确定性全局求和的主机端接口
 *
 * 对 GPU 上的数组进行确定性求和，保证每次运行结果一致。
 *
 * @tparam Float 输入数据类型
 * @param result 输出结果指针（设备内存）
 * @param values 输入数组指针（设备内存）
 * @param count  元素数量
 * @param stream CUDA 流，用于异步执行
 *
 * @note 使用 1024 线程的单 block 配置
 * @note 适用于需要数值确定性的场景（如梯度检查、调试）
 */
template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 内存管理
// ============================================================================

/**
 * @brief 条件性内存分配（优先设备内存，OOM 时回退到统一内存）
 *
 * 尝试在 GPU 设备上分配内存。如果显存不足（OOM），则自动回退到
 * CUDA 统一内存（Unified Memory），数据优先放置在 CPU 上。
 * 这样可以避免程序崩溃，但会牺牲性能。
 *
 * @param out   输出参数，接收分配的内存指针
 * @param bytes 要分配的字节数
 * @param file  调用位置的文件名（用于错误报告）
 * @param line  调用位置的行号（用于错误报告）
 * @return 状态码：
 *         - 0: 成功分配设备内存
 *         - 1: 回退到统一内存（性能可能下降）
 *
 * @note 统一内存会按需在 CPU 和 GPU 之间迁移页面，
 *       首次访问时可能产生额外延迟
 */
int cudaMallocConditionallyManaged(void** out, size_t bytes, const char *file, int line) {
    // 尝试在设备上分配
    cudaError_t err = cudaMalloc(out, bytes);
    if(err == cudaErrorMemoryAllocation) {
        // OOM 时回退到统一内存，虽然慢但至少不会崩溃
        cudaGetLastError(); // 在下一个 API 调用前重置错误状态
        cudaCheck_(cudaMallocManaged(out, bytes), file, line);
        // 设置优先放置在 CPU 上，按需迁移到 GPU
        cudaCheck_(cudaMemAdvise(*out, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId), file, line);
        return 1;
    } else {
        cudaCheck_(err, file, line);
        return 0;
    }
}

/**
 * @brief cudaMallocConditionallyManaged 的宏包装
 *
 * 自动传递 __FILE__ 和 __LINE__ 以便错误定位。
 * 使用方式：cudaMallocConditionallyManaged(&ptr, size);
 */
#define cudaMallocConditionallyManaged(out, bytes)\
(cudaMallocConditionallyManaged((void**)out, bytes, __FILE__, __LINE__))

// ============================================================================
// 随机数生成（用于随机舍入）
// ============================================================================

/**
 * @brief SquirrelNoise5 哈希函数
 *
 * Squirrel's Raw Noise utilities (version 5) 的实现。
 * 基于位置和种子生成高质量的伪随机数。
 * 参考: http://eiserloh.net/noise/SquirrelNoise5.hpp
 *
 * @param positionX 输入位置（如线程索引）
 * @param seed      随机种子
 * @return 32位无符号伪随机数
 *
 * 算法特点：
 * - 确定性：相同输入总是产生相同输出
 * - 高质量：通过多次乘法、加法、异或操作混合位
 * - 快速：无分支，适合 GPU 执行
 *
 * @note 可能过于复杂，实际应用中可能不需要如此高质量的随机数
 */
__device__ __host__ constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
{
    // 预定义的噪声常量（精心选择的位模式）
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    
    unsigned int mangledBits = positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}

/**
 * @brief 2D 位置噪声函数
 *
 * 将 2D 坐标映射为伪随机数，常用于基于线程 ID 生成随机数。
 *
 * @param indexX 第一维索引（如 threadIdx.x）
 * @param indexY 第二维索引（如 blockIdx.x）
 * @param seed   随机种子
 * @return 32位无符号伪随机数
 *
 * 原理：使用大质数将 2D 坐标线性化后调用 SquirrelNoise5
 */
__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr unsigned int PRIME_NUMBER = 198491317u; // 大质数，位模式分布均匀
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

/**
 * @brief 随机舍入：float -> bfloat16
 *
 * 使用随机舍入（Stochastic Rounding）将 float 转换为 bfloat16。
 * 与确定性舍入不同，随机舍入可以在期望上保持梯度的无偏性，
 * 有助于低精度训练的收敛。
 *
 * @param in   输入的 float 值
 * @param out  输出的 bfloat16 指针
 * @param seed 随机种子（建议每步通过 xorshift 更新）
 *
 * 算法：
 * 1. 基于线程位置和种子生成随机阈值
 * 2. 提取 float 的低 16 位（将被截断的部分）
 * 3. 如果截断部分 > 阈值，向上舍入；否则向下舍入
 * 4. 转换为 bfloat16
 *
 * @note 每个线程获得不同的随机数，保证并行执行时的随机性
 */
__device__ __forceinline__ void stochastic_rounding(float in, __nv_bfloat16 *out, unsigned int seed) {
    // 确保每个线程获得不同的随机数
    unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed);
    unsigned int threshold = random & 0xFFFF;           // 取低 16 位作为阈值
    unsigned int float_bits = __float_as_uint(in);      // float 的位表示
    unsigned int rounded_bits = float_bits & 0x0000FFFF; // 提取将被截断的低 16 位
    // 概率性舍入：截断部分越大，向上舍入的概率越高
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = __float2bfloat16_rn(__uint_as_float(float_bits));
}

/**
 * @brief 随机舍入：float -> half（占位实现）
 *
 * @param in     输入的 float 值
 * @param out    输出的 half 指针
 * @param random 随机数（当前未使用）
 *
 * @todo 实现真正的随机舍入逻辑
 */
__device__ __forceinline__ void stochastic_rounding(float in, half *out, unsigned int random) {
    *out = (float)in; // TODO: 实现随机舍入
}

/**
 * @brief 随机舍入：float -> float（恒等变换）
 *
 * 当 floatX 为 float 时的占位函数，不进行任何转换。
 *
 * @param in     输入的 float 值
 * @param out    输出的 float 指针
 * @param random 随机数（未使用）
 */
__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
    *out = in; // FP32 模式下无需舍入
}

#endif // CUDA_UTILS_CUH