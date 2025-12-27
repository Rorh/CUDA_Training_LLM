/**
 * @file mfu.h
 * @brief Model Flops Utilization (MFU) 模型浮点运算利用率计算模块
 *
 * 本头文件提供了用于计算和监控 GPU 性能利用率的工具函数和数据结构。
 * 主要功能包括:
 * - 根据 GPU 型号和精度模式计算理论浮点运算性能 (TFLOPS)
 * - 通过 NVML 库获取 GPU 实时运行状态 (时钟频率、功耗、温度等)
 * - 支持 NVIDIA Volta、Ampere、Hopper、Ada 架构的多款 GPU
 *
 * MFU (Model Flops Utilization) 是衡量模型训练效率的重要指标，
 * 计算公式: MFU = 实际浮点运算数 / (理论峰值性能 × 时间)
 */

#ifndef MFU_H
#define MFU_H

/*
 * ============================================================================
 * 标准库头文件
 * ============================================================================
 */
#include <stdio.h>   // printf, fprintf 等 I/O 函数
#include <stdlib.h>  // exit, EXIT_FAILURE 等
#include <string.h>  // strcmp 字符串比较函数

/*
 * ============================================================================
 * NVML (NVIDIA Management Library) 条件编译
 * ============================================================================
 * NVML 是 NVIDIA 提供的用于监控和管理 GPU 设备的 C 语言 API。
 * 如果系统中存在 nvml.h 头文件，则启用 NVML 功能，否则禁用。
 */
#if __has_include(<nvml.h>)
#define USE_NVML 1   // 启用 NVML 支持
#include <nvml.h>
#else
#define USE_NVML 0   // 禁用 NVML 支持
#endif

/*
 * ============================================================================
 * 精度模式宏定义
 * ============================================================================
 * 定义了 Tensor Core 支持的不同浮点精度模式。
 * 这些值与 enum PrecisionMode 对应，未来可能会统一重构。
 *
 * @note 不同精度模式对应不同的 Tensor Core 计算吞吐量:
 * - FP32: 最高精度，吞吐量最低
 * - FP16: 半精度，吞吐量较高
 * - BF16: Brain Float 16，兼顾精度和吞吐量，常用于深度学习训练
 */
#define MFUH_PRECISION_FP32 0  // 32 位浮点精度 (TF32 Tensor Core 模式)
#define MFUH_PRECISION_FP16 1  // 16 位浮点精度 (FP16 + FP32 累加)
#define MFUH_PRECISION_BF16 2  // BF16 精度 (BF16 + FP32 累加)

/*
 * ============================================================================
 * NVML 错误检查工具
 * ============================================================================
 */
#if USE_NVML
/**
 * @brief NVML API 调用错误检查函数
 *
 * 检查 NVML 函数返回状态，如果发生错误则打印错误信息并终止程序。
 *
 * @param status  NVML 函数的返回值 (nvmlReturn_t 类型)
 * @param file    调用处的源文件名 (由 __FILE__ 宏提供)
 * @param line    调用处的行号 (由 __LINE__ 宏提供)
 *
 * @note 通常通过 nvmlCheck 宏调用，自动填充文件名和行号
 * @see nvmlCheck
 */
inline void nvml_check(nvmlReturn_t status, const char *file, int line) {
    if (status != NVML_SUCCESS) {
        printf("[NVML ERROR] at file %s:%d:\n%s\n", file, line, nvmlErrorString(status));
        exit(EXIT_FAILURE);
    }
};

/**
 * @brief NVML 错误检查宏
 * @param err NVML 函数返回值
 *
 * 用法示例: nvmlCheck(nvmlInit());
 */
#define nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#endif


/*
 * ============================================================================
 * GPU 性能数据结构
 * ============================================================================
 */

/**
 * @struct PerfData
 * @brief GPU 架构的 Tensor Core 理论峰值性能数据
 *
 * 存储特定 GPU 架构在不同精度模式下的理论峰值性能 (单位: TFLOPS)。
 * 数据来源于 NVIDIA 官方白皮书和技术规格。
 *
 * @note 值为 -1.0f 表示该架构不支持对应的精度模式
 */
typedef struct {
    float TF_32;       // TF32 Tensor Core 性能 (TFLOPS) - 32位精度张量计算
    float BF_16_32;    // BF16 输入 + FP32 累加器的性能 (TFLOPS)
    float FP_16_32;    // FP16 输入 + FP32 累加器的性能 (TFLOPS)
    float FP_16_16;    // FP16 输入 + FP16 累加器的性能 (TFLOPS)
    float FP_8_32;     // FP8 输入 + FP32 累加器的性能 (TFLOPS)
    float FP_8_16;     // FP8 输入 + FP16 累加器的性能 (TFLOPS)
    float CLOCK;       // 参考时钟频率 (MHz) - 来自官方规格表
    float CORES;       // Tensor Core 数量 - 来自官方规格表
} PerfData;

/*
 * ============================================================================
 * 各 GPU 架构的基准性能数据
 * ============================================================================
 * 数据来源: NVIDIA 官方白皮书
 * - https://resources.nvidia.com/en-us-tensor-core
 * - https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
 *
 * 格式: {TF_32, BF_16_32, FP_16_32, FP_16_16, FP_8_32, FP_8_16, CLOCK, CORES}
 * -1.f 表示该架构不支持对应精度模式
 */

/** @brief Volta 架构 (V100) - 首代 Tensor Core，不支持 BF16 和 FP8 */
static const PerfData VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};

/** @brief Ampere 数据中心架构 (A100) - 第二代 Tensor Core，新增 BF16 支持 */
static const PerfData AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};

/** @brief Ampere 消费级架构 (RTX 30 系列) - 精简版 Tensor Core */
static const PerfData AMPERE_CONSUMER = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};

/** @brief Hopper 架构 (H100) - 第四代 Tensor Core，支持 FP8 */
static const PerfData HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};

/** @brief Ada Lovelace 架构 (RTX 40 系列) - 消费级支持 FP8 */
static const PerfData ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

/**
 * @struct GPUEntry
 * @brief GPU 数据库条目，存储特定 GPU 型号的性能配置
 *
 * 每个条目包含 GPU 名称、所属架构的性能数据引用，
 * 以及该 GPU 实际的 Tensor Core 数量和时钟频率。
 */
typedef struct {
    const char* name;          // GPU 型号名称 (需与 NVML 查询结果匹配)
    const PerfData* perf_data; // 指向该 GPU 所属架构的基准性能数据
    float new_cores;           // 该 GPU 实际的 Tensor Core 数量
    float new_mhz;             // 该 GPU 的 Boost 时钟频率 (MHz)
} GPUEntry;

/*
 * ============================================================================
 * GPU 数据库
 * ============================================================================
 * 包含各种 NVIDIA GPU 的性能配置数据。
 * 用于根据 GPU 型号名称查找对应的性能参数。
 *
 * 性能计算原理:
 *   实际性能 = 基准性能 × (实际核心数/基准核心数) × (实际频率/基准频率)
 *
 * 示例: RTX 4080 的 BF16 性能计算
 *   97.5 TFLOPS = 165.2 × (304/512) × (2505/2520)
 */
static GPUEntry gpu_db[] = {
    /* ====================== Volta 架构 (2017) ====================== */
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},   // 数据中心 SXM2 版本
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},   // 数据中心 PCIe 版本

    /* ====================== Ampere 数据中心架构 (2020) ====================== */
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},  // PCIe 40GB
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},  // PCIe 80GB
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},  // SXM4 40GB
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},  // SXM4 80GB

    /* ====================== Ampere 专业级 (RTX A 系列) ====================== */
    {"NVIDIA RTX A2000", &AMPERE_CONSUMER, 104, 1200},   // 入门专业卡
    {"NVIDIA RTX A4000", &AMPERE_CONSUMER, 192, 1560},   // 中端专业卡
    {"NVIDIA RTX A4500", &AMPERE_CONSUMER, 224, 1650},   // 中高端专业卡
    {"NVIDIA RTX A5000", &AMPERE_CONSUMER, 256, 1695},   // 高端专业卡
    {"NVIDIA RTX A5500", &AMPERE_CONSUMER, 320, 1770},   // 高端专业卡
    {"NVIDIA RTX A6000", &AMPERE_CONSUMER, 336, 1800},   // 旗舰专业卡

    /* ====================== Ampere 消费级 (RTX 30 系列) ====================== */
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_CONSUMER, 336, 1860},  // 旗舰
    {"NVIDIA GeForce RTX 3090", &AMPERE_CONSUMER, 328, 1695},     // 次旗舰
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_CONSUMER, 320, 1665},  // 高端
    {"NVIDIA GeForce RTX 3080", &AMPERE_CONSUMER, 272, 1710},     // 高端
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_CONSUMER, 192, 1770},  // 中高端
    {"NVIDIA GeForce RTX 3070", &AMPERE_CONSUMER, 184, 1725},     // 中高端
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_CONSUMER, 152, 1665},  // 中端
    {"NVIDIA GeForce RTX 3060", &AMPERE_CONSUMER, 112, 1777},     // 入门

    /* ====================== Ada 专业级 (RTX A 系列 ADA) ====================== */
    {"NVIDIA RTX A2000 ADA", &ADA, 88, 2130},    // 入门专业卡
    {"NVIDIA RTX A4000 ADA", &ADA, 192, 2175},   // 中端专业卡
    {"NVIDIA RTX A4500 ADA", &ADA, 224, 2580},   // 中高端专业卡
    {"NVIDIA RTX A5000 ADA", &ADA, 400, 2550},   // 高端专业卡
    {"NVIDIA RTX A5880 ADA", &ADA, 440, 2460},   // 高端专业卡
    {"NVIDIA RTX A6000 ADA", &ADA, 568, 2505},   // 旗舰专业卡

    /* ====================== Ada 消费级 (RTX 40 系列) ====================== */
    {"NVIDIA GeForce RTX 4090", &ADA, 512, 2520},          // 旗舰
    {"NVIDIA GeForce RTX 4080 SUPER", &ADA, 320, 2550},    // 次旗舰升级版
    {"NVIDIA GeForce RTX 4080", &ADA, 304, 2505},          // 次旗舰
    {"NVIDIA GeForce RTX 4070 Ti SUPER", &ADA, 264, 2610}, // 高端升级版
    {"NVIDIA GeForce RTX 4070 Ti", &ADA, 240, 2610},       // 高端
    {"NVIDIA GeForce RTX 4070 SUPER", &ADA, 224, 2475},    // 中高端升级版
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},          // 中高端
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},          // 中高端 (重复条目)
    {"NVIDIA GeForce RTX 4060 Ti", &ADA, 136, 2535},       // 中端
    {"NVIDIA GeForce RTX 4060", &ADA, 96, 2460},           // 入门

    /* ====================== Hopper 架构 (2022) ====================== */
    {"NVIDIA H100 PCIe", &HOPPER, 456, 1620},              // PCIe 版本
    {"NVIDIA H100 80GB HBM3", &HOPPER, 528, 1830},         // SXM5 版本 (HBM3 内存)
};

/*
 * ============================================================================
 * 核心功能函数
 * ============================================================================
 */

/**
 * @brief 获取指定 GPU 在特定精度模式下的理论峰值浮点性能
 *
 * 本函数用于计算 Model Flops Utilization (MFU)，即模型浮点运算利用率。
 * 通过查询 GPU 数据库，获取该 GPU 型号的理论 Tensor Core 性能。
 *
 * @param device         GPU 设备名称字符串 (需与 NVML 查询返回的名称完全匹配)
 *                       例如: "NVIDIA GeForce RTX 4090", "NVIDIA A100-SXM4-80GB"
 * @param precision_mode 精度模式，可选值:
 *                       - MFUH_PRECISION_FP32 (0): TF32 精度
 *                       - MFUH_PRECISION_FP16 (1): FP16 + FP32 累加
 *                       - MFUH_PRECISION_BF16 (2): BF16 + FP32 累加
 *
 * @return 理论峰值性能 (单位: TFLOPS, 即 1e12 FLOPS)
 *         返回 -1.0f 表示:
 *         - GPU 型号未在数据库中找到
 *         - 该 GPU 不支持指定的精度模式 (如 Volta 不支持 BF16)
 *         - 精度模式参数无效
 *
 * @note 性能计算原理:
 *       对于非旗舰型号，官方通常不提供详细的 Tensor Core 性能数据。
 *       但同架构的 GPU 使用相同类型的 Tensor Core，只是数量和频率不同。
 *       因此可以通过线性缩放计算:
 *         实际性能 = 基准性能 × (实际TC数/基准TC数) × (实际频率/基准频率)
 *
 * @note 数据来源:
 *       - https://resources.nvidia.com/en-us-tensor-core
 *       - https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
 *
 * @example
 *       // 获取 RTX 4090 在 BF16 精度下的理论性能
 *       float tflops = get_flops_promised("NVIDIA GeForce RTX 4090", MFUH_PRECISION_BF16);
 *       // 预期返回约 165.2 TFLOPS
 */
float get_flops_promised(const char* device, int precision_mode) {
    /*
     * 实现说明:
     * 1. 首先验证精度模式参数的有效性
     * 2. 在 GPU 数据库中线性搜索匹配的设备名称
     * 3. 根据精度模式查找基准性能值
     * 4. 按实际核心数和频率进行线性缩放
     *
     * 验证示例 (RTX 4080):
     *   304个TC, 2505 MHz Boost 频率
     *   BF16 性能 = 165.2 × (304/512) × (2505/2520) ≈ 97.5 TFLOPS
     */

    // 步骤1: 验证精度模式参数 (必须是 FP32/FP16/BF16 之一)
    if (!(precision_mode == MFUH_PRECISION_FP32 || precision_mode == MFUH_PRECISION_FP16 || precision_mode == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_mode);
        return -1.0f;
    }

    // 步骤2: 在数据库中线性搜索目标 GPU
    int num_gpu_entries = sizeof(gpu_db) / sizeof(gpu_db[0]);
    for (int i = 0; i < num_gpu_entries; i++) {
        if (strcmp(gpu_db[i].name, device) == 0) {
            const PerfData* perf_data = gpu_db[i].perf_data;

            // 步骤3: 根据精度模式查找基准性能值
            float value = -1.0f;
            if (precision_mode == MFUH_PRECISION_BF16) { value = perf_data->BF_16_32; }
            if (precision_mode == MFUH_PRECISION_FP32) { value = perf_data->TF_32; }
            if (precision_mode == MFUH_PRECISION_FP16) { value = perf_data->FP_16_32; }

            // 检查该精度是否被支持 (例如 Volta 架构不支持 BF16)
            if (value < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_mode);
                return -1.0f;
            }

            // 步骤4: 根据实际核心数和频率进行线性缩放
            float new_cores = gpu_db[i].new_cores;  // 该 GPU 的实际 Tensor Core 数量
            float new_mhz = gpu_db[i].new_mhz;      // 该 GPU 的实际 Boost 频率
            float adjusted = value * (new_cores / perf_data->CORES) * (new_mhz / perf_data->CLOCK);
            return adjusted;
        }
    }

    // GPU 未在数据库中找到
    return -1.0f; // ¯\_(ツ)_/¯
}

/*
 * ============================================================================
 * GPU 运行状态信息结构
 * ============================================================================
 */

/**
 * @struct GPUUtilInfo
 * @brief GPU 实时运行状态信息
 *
 * 通过 NVML API 查询获得的 GPU 当前运行状态，用于监控训练过程中的硬件状态。
 * 可用于诊断性能问题，如功耗限制、温度降频等。
 */
struct GPUUtilInfo {
    unsigned int clock;          // 当前 SM 时钟频率 (MHz)
    unsigned int max_clock;      // 最大 SM 时钟频率 (MHz)
    unsigned int power;          // 当前功耗 (毫瓦, mW)
    unsigned int power_limit;    // 功耗限制 (毫瓦, mW)
    unsigned int fan;            // 风扇转速百分比 (0-100%)
    unsigned int temperature;    // 当前 GPU 核心温度 (摄氏度)
    unsigned int temp_slowdown;  // 温度降频阈值 (摄氏度)

    float gpu_utilization;       // GPU 计算单元利用率 (0.0-100.0%)
    float mem_utilization;       // 显存带宽利用率 (0.0-100.0%)
    const char* throttle_reason; // 降频原因描述字符串
};

/*
 * ============================================================================
 * NVML 工具函数 (仅在 USE_NVML=1 时编译)
 * ============================================================================
 */
#if USE_NVML

/**
 * @brief 获取 NVML 设备句柄 (懒加载单例模式)
 *
 * 首次调用时初始化 NVML 库并获取第一个 GPU 的设备句柄。
 * 后续调用直接返回缓存的句柄，避免重复初始化。
 *
 * @return nvmlDevice_t 设备句柄，用于后续 NVML API 调用
 *
 * @note 当前实现固定使用索引 0 的 GPU，不支持多 GPU 场景
 * @note 使用静态变量实现单例，线程安全性取决于编译器对静态初始化的处理
 */
nvmlDevice_t nvml_get_device() {
    static bool needs_init = true;        // 是否需要初始化的标志
    static nvmlDevice_t device;           // 缓存的设备句柄
    if(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());                              // 初始化 NVML 库
        nvmlCheck(nvmlDeviceGetHandleByIndex_v2(0, &device)); // 获取第 0 个 GPU 的句柄
    }
    return device;
}

/**
 * @brief 将降频原因位域转换为可读的文本描述
 *
 * NVML 返回的降频原因是一个位域 (bitmask)，本函数将其转换为
 * 简短的文本描述，便于日志输出和监控。
 *
 * @param bits NVML 返回的降频原因位域 (nvmlClocksThrottleReasons)
 *
 * @return 降频原因的文本描述:
 *         - "power cap"   : 功耗达到限制 (软件功耗墙或硬件功耗制动)
 *         - "thermal cap" : 温度达到限制 (软件或硬件温度保护)
 *         - "other cap"   : 其他原因导致的降频
 *         - "no cap"      : 未发生降频，运行在正常状态
 *
 * @note 这是一个有损转换，只提供大致的降频原因指示
 *       完整的降频原因需要检查 bits 中的所有标志位
 */
const char* get_throttle_reason(unsigned long long bits) {
    // 检查功耗相关的降频原因
    if(bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        return "power cap";
    }
    // 检查温度相关的降频原因
    else if (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        return "thermal cap";
    }
    // 检查其他降频原因
    else if (bits & (nvmlClocksThrottleReasonAll)) {
        return "other cap";
    }
    // 未发生降频
    else {
        return "no cap";
    }
}

/**
 * @brief 获取 GPU 实时运行状态信息
 *
 * 通过 NVML API 查询 GPU 的各项运行指标，包括:
 * - 时钟频率 (当前/最大)
 * - 功耗 (当前/限制)
 * - 温度 (当前/降频阈值)
 * - 风扇转速
 * - GPU/显存利用率
 * - 降频原因
 *
 * @return GPUUtilInfo 结构体，包含所有查询到的 GPU 状态信息
 *
 * @note GPU 利用率通过采样历史数据计算平均值，而非瞬时值
 * @note 当前实现使用固定大小的采样缓冲区 (128 个样本)，避免动态内存分配
 *
 * @warning 此函数仅在编译时启用 NVML 支持 (USE_NVML=1) 时可用
 *          否则调用将导致程序终止
 *
 * @example
 *       GPUUtilInfo info = get_gpu_utilization_info();
 *       printf("GPU 温度: %u°C, 功耗: %.1f W\n",
 *              info.temperature, info.power / 1000.0f);
 */
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();

    /*
     * ==================== 直接查询的指标 ====================
     * 这些指标通过单次 NVML API 调用即可获取
     */

    // 查询 SM (流式多处理器) 时钟频率
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));

    // 查询功耗信息 (单位: 毫瓦)
    nvmlCheck(nvmlDeviceGetPowerManagementLimit(device, &info.power_limit));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));

    // 查询温度信息 (单位: 摄氏度)
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &info.temp_slowdown));

    // 查询降频原因并转换为可读文本
    unsigned long long throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);

    // 查询风扇转速 (百分比)
    nvmlCheck(nvmlDeviceGetFanSpeed(device, &info.fan));

    /*
     * ==================== 采样计算的指标 ====================
     * GPU/显存利用率通过历史采样数据计算平均值
     *
     * 原理: NVML 驱动会周期性记录利用率采样数据
     * 我们获取最近的采样点并计算平均值，得到更稳定的利用率估计
     *
     * 注意: 理论上应该先查询可用的采样数量再分配空间，
     * 但为了避免动态内存分配，这里使用固定大小的缓冲区 (128 个样本)
     */
    constexpr const int BUFFER_LIMIT = 128;  // 采样缓冲区大小
    nvmlSample_t buffer[BUFFER_LIMIT];       // 采样数据缓冲区
    nvmlValueType_t v_type;                  // 采样值类型 (由 NVML 返回)
    unsigned int sample_count = BUFFER_LIMIT;

    // 计算 GPU 计算单元利用率 (平均值)
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer));
    float gpu_utilization = 0.f;
    for(unsigned i = 0; i < sample_count; ++i) {
        gpu_utilization += (float)buffer[i].sampleValue.uiVal;
    }
    gpu_utilization /= (float)sample_count;

    // 重置采样计数 (上一次查询可能修改了这个值)
    sample_count = BUFFER_LIMIT;

    // 计算显存带宽利用率 (平均值)
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer));
    float mem_utilization = 0.f;
    for(unsigned i = 0; i < sample_count; ++i) {
        mem_utilization += (float)buffer[i].sampleValue.uiVal;
    }
    mem_utilization /= (float)sample_count;

    // 填充利用率字段
    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;

    return info;
}

#else  // USE_NVML == 0

/**
 * @brief 无 NVML 支持时的占位函数
 *
 * 当编译时未包含 NVML 库时，调用此函数将输出错误信息并终止程序。
 *
 * @return 不返回 (调用 exit())
 */
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    exit(EXIT_FAILURE);
}

#endif  // USE_NVML

#endif  // MFU_H
