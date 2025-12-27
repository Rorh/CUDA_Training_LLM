/*
 * CUDA通用工具头文件 (Common utilities for CUDA code)
 * 
 * 本文件包含CUDA编程中常用的工具函数、宏定义和类型定义，主要包括：
 * 1. 全局设备属性定义
 * 2. CUDA错误检查工具
 * 3. 精度模式设置（FP32/FP16/BF16）
 * 4. 流式缓存加载/存储函数
 * 5. NVTX性能分析工具
 * 6. GPU与文件之间的双缓冲数据传输工具
 */
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

// ============================================================================
// 标准库和CUDA头文件引入
// ============================================================================
#include <stdlib.h>                 // 标准库：内存分配、程序退出等
#include <stdio.h>                  // 标准I/O：printf、FILE等
#include <math.h>                   // 数学函数库
#include <string>                   // C++字符串类
#include <type_traits>              // std::bool_constant - 编译时布尔常量类型
#include <cuda_runtime.h>           // CUDA运行时API（内存管理、流、事件等）
#include <nvtx3/nvToolsExt.h>       // NVIDIA Tools Extension - 性能分析标记工具
#include <nvtx3/nvToolsExtCudaRt.h> // NVTX的CUDA运行时扩展
#include <cuda_profiler_api.h>      // CUDA性能分析器API
#include <cuda_bf16.h>              // BFloat16（脑浮点）数据类型支持
#include <cuda_fp16.h>              // FP16半精度浮点数据类型支持

#include "utils.h"                  // 项目内部工具函数（如freadCheck、fwriteCheck等）

// ============================================================================
// 全局定义和设置 (Global defines and settings)
// ============================================================================

/*
 * CUDA设备属性结构体（外部声明）
 * - 声明为extern是因为各个kernel需要访问设备属性
 * - 实际的定义和初始化在主程序文件中完成
 * - 包含设备名称、计算能力、内存大小、SM数量等信息
 */
extern cudaDeviceProp deviceProp;

/*
 * Warp大小定义
 * - CUDA中warpSize是运行时常量，不是编译时常量
 * - 这里定义为宏可以让编译器更好地优化
 * - 所有NVIDIA GPU的warp大小都是32个线程
 * - U后缀表示无符号整数，避免有符号/无符号比较警告
 */
#define WARP_SIZE 32U

/*
 * 每SM最大线程块数量定义
 * - 用于__launch_bounds__指令，帮助编译器优化寄存器分配
 * - A100 (sm_80) 和 H100 (sm_90+) 架构支持每SM运行2个1024线程的block
 * - 这样可以最大化延迟容忍度（当一个block等待内存时，另一个可以执行）
 * - 必须用宏定义而非运行时查询，因为__launch_bounds__需要编译时常量
 * - __CUDA_ARCH__是编译时宏，表示目标GPU的计算能力
 */
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2   // A100/H100: 每SM 2个block
#else
#define MAX_1024_THREADS_BLOCKS 1   // 较老架构: 每SM 1个block
#endif

/*
 * 向上取整除法宏
 * - 用于计算kernel启动时的grid/block维度
 * - 例如：需要处理1000个元素，每个block处理256个
 *   CEIL_DIV(1000, 256) = (1000 + 255) / 256 = 4 个block
 * - 公式原理：(M + N - 1) / N 等价于 ceil(M / N)
 * - 注意：参数用括号包裹，防止宏展开时的运算符优先级问题
 */
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * 编译时布尔常量快捷方式
 * - 用于模板函数参数，实现编译时条件分支
 * - 与运行时bool不同，这些是类型级别的常量
 * - 允许编译器在编译时选择不同的代码路径，实现零开销抽象
 * - 例如：template<typename B> void func(B flag); 调用func(True)时
 *   编译器可以根据True类型生成优化代码
 */
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ============================================================================
// CUDA错误检查工具 (Error checking)
// ============================================================================

/*
 * CUDA错误检查函数
 * - 检查CUDA API调用是否成功
 * - 如果失败，打印错误信息（包含文件名和行号）并退出程序
 * - 函数名带下划线，因为通常通过宏调用而非直接调用
 * 
 * @param error  CUDA API返回的错误码
 * @param file   调用处的源文件名（由__FILE__宏提供）
 * @param line   调用处的行号（由__LINE__宏提供）
 */
inline void cudaCheck_(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
/* 错误检查宏 - 自动捕获文件名和行号，方便调试定位 */
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

/*
 * 安全的CUDA内存释放函数
 * - 类似cudaFree，但增加了错误检查和指针重置
 * - 释放后将指针设为nullptr，防止悬空指针和重复释放
 * - 使用模板支持任意指针类型
 * 
 * @param ptr   指向GPU指针的指针（二级指针，用于修改原指针）
 * @param file  调用处的源文件名
 * @param line  调用处的行号
 */
template<class T>
inline void cudaFreeCheck(T** ptr, const char *file, int line) {
    cudaError_t error = cudaFree(*ptr);
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;  // 重置指针，防止悬空指针
}
/* 内存释放宏 - 自动捕获文件名和行号 */
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ============================================================================
// CUDA精度设置和定义 (CUDA Precision settings and defines)
// ============================================================================

/*
 * 精度模式枚举
 * - FP32: 单精度浮点（32位），精度最高，内存占用最大
 * - FP16: 半精度浮点（16位），计算快但可能有精度问题，需要梯度缩放
 * - BF16: 脑浮点（16位），指数位与FP32相同，动态范围大，训练更稳定
 */
enum PrecisionMode {
    PRECISION_FP32,   // 单精度：适合调试和精度敏感场景
    PRECISION_FP16,   // 半精度：最快但可能溢出
    PRECISION_BF16    // 脑浮点：推荐用于训练，平衡速度和稳定性
};

/*
 * 根据编译时定义选择精度类型
 * - floatX是统一的浮点类型别名，根据编译选项映射到不同精度
 * - 通过编译时 -DENABLE_FP32 或 -DENABLE_FP16 选择精度
 * - 默认使用BF16，因为它在训练中表现最好（动态范围大，不易溢出）
 */
#if defined(ENABLE_FP32)
typedef float floatX;                    // FP32: 标准单精度浮点
#define PRECISION_MODE PRECISION_FP32
#elif defined(ENABLE_FP16)
// 警告：FP16可能需要梯度缩放器(gradient scaler)，目前未实现！
typedef half floatX;                     // FP16: CUDA半精度类型
#define PRECISION_MODE PRECISION_FP16
#else // 默认使用bfloat16
typedef __nv_bfloat16 floatX;            // BF16: NVIDIA脑浮点类型
#define PRECISION_MODE PRECISION_BF16
#endif

// ============================================================================
// 流式缓存加载/存储 (Load and store with streaming cache hints)
// ============================================================================
/*
 * __ldcs (Load with Cache Streaming): 流式加载，暗示数据只用一次，不污染缓存
 * __stcs (Store with Cache Streaming): 流式存储，绕过L1缓存直接写入L2
 * 
 * 旧版nvcc不为bfloat16提供这些函数，尽管bf16本质上就是unsigned short
 * 需要小心处理：
 * - 只在没有内置版本时定义，否则编译器会报错
 * - sm52会报"no viable overload"（没有合适的重载）
 * - sm80会报"function already exists"（函数已存在）
 * 
 * 条件编译说明：
 * - ENABLE_BF16: 使用BF16精度
 * - __CUDACC_VER_MAJOR__ < 12: CUDA版本低于12
 * - __CUDA_ARCH__ < 800: 目标架构低于Ampere
 */
#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
/* 流式加载BF16值 - 将地址重解释为unsigned short进行加载 */
__device__ floatX __ldcs(const floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}

/* 流式存储BF16值 - 将值转换为unsigned short进行存储 */
__device__ void __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif

// ============================================================================
// 性能分析工具 (Profiler utils)
// ============================================================================

/*
 * NVTX范围标记类 - 用于在Nsight Systems/Compute中标记代码区域
 * 
 * NVTX (NVIDIA Tools Extension) 允许在性能分析工具中添加自定义标记
 * 使用RAII模式：构造时开始标记，析构时结束标记
 * 这样可以自动处理作用域，即使有异常或提前返回也能正确结束标记
 * 
 * 在Nsight Systems中会显示为带标签的时间段，方便分析性能瓶颈
 */
class NvtxRange {
 public:
    // 简单字符串标记
    NvtxRange(const char* s) { nvtxRangePush(s); }
    
    // 带编号的标记（如 "Forward 5"）
    NvtxRange(const std::string& base_str, int number) {
        std::string range_string = base_str + " " + std::to_string(number);
        nvtxRangePush(range_string.c_str());
    }
    
    // 析构时自动结束标记
    ~NvtxRange() { nvtxRangePop(); }
};

/* 便捷宏：自动使用函数名作为NVTX标记名 */
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ============================================================================
// GPU内存与文件之间的数据传输工具
// (Utilities to Read & Write between CUDA memory <-> files)
// ============================================================================

/*
 * 将GPU内存数据写入文件（使用双缓冲技术）
 * 
 * 双缓冲原理：
 * - 使用两个缓冲区交替工作，实现GPU传输和磁盘写入的重叠
 * - 当缓冲区A从GPU接收数据时，缓冲区B的数据正在写入磁盘
 * - 这样可以隐藏传输延迟，提高整体吞吐量
 * 
 * 时间线示意：
 * GPU->内存: [====A====][====A====][====A====]
 * 内存->磁盘:           [====B====][====B====][====B====]
 * 
 * @param dest        目标文件指针
 * @param src         源GPU内存指针
 * @param num_bytes   要传输的总字节数
 * @param buffer_size 单个缓冲区大小（实际分配2倍）
 * @param stream      CUDA流，用于异步传输
 */
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    // 分配固定内存(pinned memory)用于更快的异步传输
    // 固定内存不会被操作系统换出，DMA可以直接访问
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size));
    
    // 将分配的空间分成两个缓冲区
    void* read_buffer = buffer_space;                // 从GPU读取数据的缓冲区
    void* write_buffer = buffer_space + buffer_size; // 写入磁盘的缓冲区

    // 预填充读取缓冲区；第一次拷贝需要同步等待
    char* gpu_read_ptr = (char*)src;  // GPU内存读取指针
    size_t copy_amount = std::min(buffer_size, num_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));  // 等待第一次传输完成
    
    size_t rest_bytes = num_bytes - copy_amount;  // 剩余待传输字节数
    size_t write_buffer_size = copy_amount;       // 写缓冲区中的有效数据量
    gpu_read_ptr += copy_amount;                  // 移动GPU读取指针

    std::swap(read_buffer, write_buffer);  // 交换缓冲区角色
    
    // 主循环：只要还有数据要传输
    while(rest_bytes > 0) {
        // 启动下一次异步GPU->内存传输
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        
        // 同时，将写缓冲区的数据写入磁盘（CPU操作，与GPU传输并行）
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        
        // 等待GPU传输完成
        cudaCheck(cudaStreamSynchronize(stream));

        // 交换缓冲区并更新状态
        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // 写入最后一个缓冲区的数据
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    
    // 释放固定内存
    cudaCheck(cudaFreeHost(buffer_space));
}

/*
 * 从文件读取数据到GPU内存（使用双缓冲技术）
 * 
 * 与device_to_file相反，实现文件->GPU的高效传输
 * 同样使用双缓冲实现磁盘读取和GPU传输的重叠
 * 
 * 特别优化：使用Write-Combined内存
 * - cudaHostAllocWriteCombined标志分配WC内存
 * - WC内存适合CPU写入、GPU读取的场景
 * - CPU写入会被合并，减少总线事务，提高传输效率
 * - 注意：WC内存的CPU读取性能很差，不要用于读取
 * 
 * 时间线示意：
 * 磁盘->内存: [====A====][====A====][====A====]
 * 内存->GPU:            [====B====][====B====][====B====]
 * 
 * @param dest        目标GPU内存指针
 * @param src         源文件指针
 * @param num_bytes   要传输的总字节数
 * @param buffer_size 单个缓冲区大小（实际分配2倍）
 * @param stream      CUDA流，用于异步传输
 */
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    // 分配Write-Combined固定内存
    // WC内存特点：CPU写入被合并，适合host->device传输场景
    // 参考文档：https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size, cudaHostAllocWriteCombined));
    
    // 将分配的空间分成两个缓冲区
    void* read_buffer = buffer_space;                // 从磁盘读取数据的缓冲区
    void* write_buffer = buffer_space + buffer_size; // 写入GPU的缓冲区

    // 预填充读取缓冲区
    char* gpu_write_ptr = (char*)dest;  // GPU内存写入指针
    size_t copy_amount = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);  // 从文件读取第一块数据

    size_t rest_bytes = num_bytes - copy_amount;  // 剩余待传输字节数
    size_t write_buffer_size = copy_amount;       // 写缓冲区中的有效数据量
    std::swap(read_buffer, write_buffer);         // 交换缓冲区角色

    // 主循环：只要还有数据要传输
    while(rest_bytes > 0) {
        // 启动异步内存->GPU传输
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_buffer_size;  // 移动GPU写入指针
        
        // 同时，从磁盘读取下一块数据（CPU操作，与GPU传输并行）
        freadCheck(read_buffer, 1, copy_amount, src);
        
        // 等待GPU传输完成
        cudaCheck(cudaStreamSynchronize(stream));

        // 交换缓冲区并更新状态
        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    // 传输最后一个缓冲区的数据到GPU
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));  // 等待最后一次传输完成
    
    // 释放固定内存
    cudaCheck(cudaFreeHost(buffer_space));
}

#endif // CUDA_COMMON_H