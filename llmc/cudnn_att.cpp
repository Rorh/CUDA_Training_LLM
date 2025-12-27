/*
 * ============================================================================
 * cuDNN Attention 实现
 * ============================================================================
 * 
 * 设计目标:
 * - 将所有 cuDNN 相关代码集中在此文件，避免无关代码修改时的重新编译
 * - 使用 cuDNN Frontend API 实现 Flash Attention (SDPA) 算子
 * - 支持训练和推理模式，提供前向和反向传播
 * 
 * 核心特性:
 * 1. 图缓存机制: 避免重复构建 cuDNN 计算图 (build_operation_graph 很慢)
 * 2. 动态工作空间: 根据需求分配 (H100 上约 16MB)
 * 3. 精度支持: FP16/BF16，不支持 FP32
 * 4. 因果掩码: 自回归语言模型必需
 * 
 * 性能优化:
 * - 使用 cuDNN 的 Flash Attention 实现 (内存高效)
 * - 静态缓存避免图构建开销
 * - 支持确定性算法 (1.5+版本)
 */

#define NOMINMAX  // 防止 Windows min/max 宏冲突
#include <unistd.h>
#include "cudnn_att.h"
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;  // cuDNN Frontend 命名空间

// ============================================================================
// 精度配置和全局变量
// ============================================================================

/**
 * 根据编译时宏选择 cuDNN 数据类型
 * 
 * 注意: cuDNN Frontend 不支持 FP32 模式
 * - FP16: 需要梯度缩放器 (当前未实现)
 * - BF16: 默认选择，与主程序一致
 */
#if defined(ENABLE_FP32)
static_assert(false, "cuDNN is not supported in FP32 mode.")
// 使用 FP16 (注意: 可能需要梯度缩放器，当前未实现!)
#elif defined(ENABLE_FP16)
#define CUDNN_16BIT fe::DataType_t::HALF
#else // 默认使用 bfloat16
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif

/** 全局 cuDNN 句柄和工作空间 */
static cudnnHandle_t cudnn_handle;           // cuDNN 上下文句柄
static size_t cudnn_workspace_size = 0;      // 工作空间大小 (动态分配，最大 256MiB)
static void* cudnn_workspace = NULL;         // 工作空间指针 (GPU 内存)

// ============================================================================
// 错误检查宏
// ============================================================================

/**
 * cuDNN 传统 API 错误检查
 * 
 * @param error: cuDNN 状态码
 * @param file:  文件名 (宏自动传入)
 * @param line:  行号 (宏自动传入)
 */
static void cuDNNCheck(cudnnStatus_t error, const char *file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

/**
 * cuDNN Frontend API 错误检查
 * 
 * Frontend API 使用 error_object 而非状态码
 * 
 * @param e:    Frontend 错误对象
 * @param file: 文件名 (宏自动传入)
 * @param line: 行号 (宏自动传入)
 */
static void checkCudnnFE(const fe::error_object& e, const char *file, int line) {
    if(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

// ============================================================================
// 张量唯一标识符
// ============================================================================

/**
 * cuDNN Frontend 要求为每个张量分配唯一 ID
 * 这些 ID 用于在 variant_pack 中绑定实际数据指针
 */
enum UIDs {
    Q_UID,           // Query 张量
    K_UID,           // Key 张量  
    V_UID,           // Value 张量
    Attn_scale_UID,  // 注意力缩放因子 (1/sqrt(d_k))
    O_UID,           // Output 张量
    Stats_UID,       // Softmax 统计信息 (反向传播需要)
    dO_UID,          // Output 梯度
    dQ_UID,          // Query 梯度
    dK_UID,          // Key 梯度
    dV_UID           // Value 梯度
};

// ============================================================================
// 图缓存机制
// ============================================================================

/**
 * 为什么需要缓存?
 * graph->build_operation_graph() 非常慢 (秒级)，但执行很快
 * 缓存避免每次调用都重新构建图
 * 
 * 缓存键设计:
 * - 前向: (B, H, T, HS, is_inference_only) - 包含推理模式标志
 * - 反向: (B, NH, T, HS) - 不需要推理标志
 */
using cache_type_fwd = std::map<std::tuple<int,int,int,int, int>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;

// ============================================================================
// 前向传播图构建
// ============================================================================

/**
 * lookup_cache_or_build_graph_fwd - 查找或构建前向传播计算图
 * 
 * 基于官方示例简化，适配 llmc 的内存布局
 * 
 * @param B: 批次大小
 * @param H: 注意力头数
 * @param T: 序列长度
 * @param HS: 每个头的维度 (head_size)
 * @param is_inference_only: true=推理模式 (无 stats), false=训练模式
 * @return: 共享指针指向构建好的图
 * 
 * 内存布局说明:
 * - 输入 QKV: (B, T, 3, NH, HS) -> cuDNN 直接支持，无需外部转置
 * - Q/K/V 单独: (B, NH, T, HS) 通过 stride 访问
 */
auto lookup_cache_or_build_graph_fwd(int B,int H,int T,int HS, int is_inference_only) {

    static cache_type_fwd user_maintained_cache_fwd;  // 静态缓存，跨调用持久

    auto key = std::make_tuple(B, H, T, HS, is_inference_only);

    // 缓存命中: 直接返回已构建的图
    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    // ========== 创建新的计算图 ==========
    auto graph = std::make_shared<fe::graph::Graph>();
    
    // 设置数据类型:
    // - IO: FP16/BF16 (与主程序一致)
    // - 中间/计算: FP32 (保证数值稳定性)
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // ========== 定义输入张量 ==========
    // QKV 布局: (B, T, 3, NH, HS) - cuDNN 可直接处理，无需外部转置
    // 通过 stride 分别访问 Q/K/V，避免内存拷贝
    
    // Query 张量: 从 QKV 的第 0 个通道开始
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_uid(Q_UID)
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    
    // Key 张量: 从 QKV 的第 1 个通道开始  
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_uid(K_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    
    // Value 张量: 从 QKV 的第 2 个通道开始
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_uid(V_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    
    // 注意力缩放因子: 1/sqrt(d_k)，防止 softmax 饱和
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_uid(Attn_scale_UID)
                               .set_is_pass_by_value(true)  // 按值传递，无需 GPU 存储
                               .set_data_type(fe::DataType_t::FLOAT));

    // ========== 配置 SDPA (Scaled Dot-Product Attention) 选项 ==========
    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);  // 推理模式不输出 stats
    sdpa_options.set_attn_scale(attn_scale);           // 设置缩放因子
    sdpa_options.set_causal_mask(true);                // 自回归掩码 (GPT 风格)

    // ========== 创建 SDPA 操作并获取输出张量 ==========
    // 返回: O (注意力输出), stats (softmax 统计，仅训练模式)
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // ========== 配置输出张量 ==========
    // 注意力输出: (B, T, NH, HS) - 与输入 QKV 的布局匹配
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(O_UID);

    // Softmax 统计信息: 仅训练模式需要，用于反向传播
    // 数据类型: FP32 保证数值精度
    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1})
                               .set_uid(Stats_UID);
    }

    // ========== 验证和构建图 ==========
    checkCudnnFE(graph->validate());  // 验证图结构合法性

    // 构建操作图和执行计划 (这是最慢的部分，秒级!)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});  // 启发式选择最优计划
    checkCudnnFE(graph->check_support(cudnn_handle));               // 检查硬件支持
    checkCudnnFE(graph->build_plans(cudnn_handle));                  // 构建可执行计划

    // ========== 动态分配工作空间 ==========
    // cuDNN 需要临时 GPU 内存，大小取决于配置
    // H100 上通常约 16MB，最大可达 256MB
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));  // 释放旧的工作空间
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    // 缓存构建好的图供后续使用
    user_maintained_cache_fwd.insert({key, graph});

    return graph;
}

// ============================================================================
// 反向传播图构建
// ============================================================================

/**
 * lookup_cache_or_build_graph_bwd - 查找或构建反向传播计算图
 * 
 * @param B:  批次大小
 * @param NH: 注意力头数  
 * @param T:  序列长度
 * @param HS: 每个头的维度
 * @return: 共享指针指向构建好的图
 * 
 * 反向传播需要:
 * - Q, K, V (前向输入)
 * - O (前向输出)  
 * - dO (输出梯度)
 * - stats (softmax 统计)
 */
auto lookup_cache_or_build_graph_bwd(int B, int NH, int T, int HS) {
    static cache_type_bwd user_maintained_cache_bwd;

    auto key = std::make_tuple(B, NH, T, HS);

    // 缓存命中检查
    auto it = user_maintained_cache_bwd.find(key);
    if (it != user_maintained_cache_bwd.end()) {
        return it->second;
    }

    // ========== 创建反向传播计算图 ==========
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // ========== 定义输入张量 ==========
    // QKV 输入: (B, T, 3, NH, HS) - 必须来自前向的 inp (需要转换为 FP16)
    
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, NH, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, NH, T, HS})
                            .set_uid(K_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, NH, T, HS})
                            .set_uid(V_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    
    // 前向输出和梯度
    auto O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                            .set_dim({B, NH, T, HS})
                            .set_uid(O_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                            .set_dim({B, NH, T, HS})
                            .set_uid(dO_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));

    // Softmax 统计信息 (来自前向传播)
    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                            .set_dim({B, NH, T, 1})
                            .set_uid(Stats_UID)
                            .set_stride({NH * T, T, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
    
    // 注意力缩放因子
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_uid(Attn_scale_UID)
                            .set_data_type(fe::DataType_t::FLOAT));
    
    // ========== 配置反向 SDPA 选项 ==========
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(true) // 1.5+ 版本需要此设置保证确定性
#endif
                            .set_causal_mask(true)                // 因果掩码
                            .set_attn_scale(attn_scale);           // 缩放因子

    // ========== 创建反向 SDPA 操作并获取输出梯度 ==========
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    // ========== 配置输出梯度张量 ==========
    // 梯度布局与输入 QKV 一致，便于原地更新
    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dV_UID);

    // ========== 验证和构建反向图 ==========
    checkCudnnFE(graph->validate());

    // 构建操作图 (最慢的部分)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    // ========== 动态分配工作空间 ==========
    // cuDNN 默认最多使用 256MiB，我们按需分配避免浪费
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    // 缓存反向图
    user_maintained_cache_bwd.insert({key, graph});
    return graph;
}

// ============================================================================
// 主机端接口函数
// ============================================================================

/**
 * attention_forward_cudnn - cuDNN 前向注意力计算
 * 
 * @param out:   输出张量 (B, T, NH, HS)
 * @param stats: Softmax 统计 (B, NH, T)，仅训练模式需要
 * @param inp:   输入 QKV 张量 (B, T, 3, NH, HS)
 * @param B:     批次大小
 * @param T:     序列长度
 * @param NH:    注意力头数
 * @param C:     总通道数 (NH * HS)
 * @param stream: CUDA 流
 * 
 * 执行流程:
 * 1. 检查缓存或构建计算图
 * 2. 准备张量指针映射
 * 3. 执行 cuDNN 计算图
 */
void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int HS = C / NH; // 每个头的特征维度
    bool is_inference_only = (stats == nullptr);  // 推理模式判断

    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));  // 设置 CUDA 流

    // 获取缓存的图或首次构建
    auto graph = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // ========== 准备张量指针 ==========
    // QKV 内存布局: (B, T, 3, NH, HS)，通过指针偏移访问
    void* devPtrQ = inp;                    // Q: 起始位置
    void* devPtrK = (inp + C);              // K: 偏移 C 个元素
    void* devPtrV = (inp + 2 * C);          // V: 偏移 2*C 个元素
    float attn_scale_cpu = 1.0 / sqrtf(HS); // 注意力缩放因子: 1/sqrt(d_k)
    void* devPtrO = out;                    // 输出

    // 构建 variant pack: 张量 UID 到数据指针的映射
    std::unordered_map<int64_t , void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, 
        {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // 训练模式需要 stats 张量用于反向传播
    if (is_inference_only == false) {
        variant_pack[Stats_UID] = stats;
    }

    // ========== 执行计算图 ==========
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

/**
 * attention_backward_cudnn - cuDNN 反向注意力计算
 * 
 * @param dqkvr: 输出梯度 (B, T, 3, NH, HS) - dQ, dK, dV 合并存储
 * @param dout:  输出梯度 (B, T, NH, HS)
 * @param qkvr:  前向输入 QKV (B, T, 3, NH, HS)
 * @param o:     前向输出 (B, T, NH, HS)
 * @param stats: 前向 softmax 统计 (B, NH, T)
 * @param B:     批次大小
 * @param T:     序列长度
 * @param NH:    注意力头数
 * @param C:     总通道数
 * @param stream: CUDA 流
 * 
 * 注意: qkvr 和 dqkvr 使用相同的内存布局，便于原地更新
 */
void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int HS = C / NH; // 每个头的特征维度

    // 获取缓存的反向图
    auto graph = lookup_cache_or_build_graph_bwd(B, NH, T, HS);

    // ========== 准备输入张量指针 ==========
    void* devPtrQ = qkvr;                    // 前向 Q
    void* devPtrK = (qkvr + NH * HS);        // 前向 K
    void* devPtrV = (qkvr + 2 * NH * HS);    // 前向 V
    void* devPtrO = o;                       // 前向输出
    void* devPtrdO = dout;                   // 输出梯度
    void* devPtrStats = stats;               // softmax 统计
    float attn_scale_cpu = 1.0 / sqrtf(HS);  // 缩放因子

    // ========== 准备输出梯度指针 ==========
    void* devPtrdQ = dqkvr;                  // dQ
    void* devPtrdK = (dqkvr + NH * HS);      // dK
    void* devPtrdV = (dqkvr + 2 * NH * HS);  // dV

    // 构建 variant pack: 所有张量的 UID 到指针映射
    std::unordered_map<int64_t, void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, 
        {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // ========== 执行反向计算图 ==========
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 生命周期管理
// ============================================================================

/**
 * create_cudnn - 初始化 cuDNN 环境
 * 
 * 创建 cuDNN 句柄，工作空间延迟到首次使用时分配
 */
void create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

/**
 * destroy_cudnn - 清理 cuDNN 资源
 * 
 * 释放工作空间内存和销毁 cuDNN 句柄
 */
void destroy_cudnn() {
    if (cudnn_workspace != NULL) { 
        cudaCheck(cudaFree(cudnn_workspace));  // 释放 GPU 工作空间
    }
    cuDNNCheck(cudnnDestroy(cudnn_handle));    // 销毁 cuDNN 句柄
}