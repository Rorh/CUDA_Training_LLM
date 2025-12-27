/*
 * 矩阵乘法层 (Matrix Multiplication)
 * 使用 cuBLASLt 实现高性能矩阵乘法
 * 
 * 数学公式:
 *   前向: out = inp @ weight^T + bias  (线性变换)
 *         若启用 GELU: out = GELU(inp @ weight^T + bias)
 * 
 *   反向: dinp = dout @ weight          (对输入的梯度)
 *         dweight += inp^T @ dout       (对权重的梯度，累加)
 *         dbias = sum(dout, dim=0)      (对偏置的梯度，沿 batch 维度求和)
 * 
 * cuBLASLt 特性:
 *   - 支持融合操作 (bias + GELU) 减少内存带宽
 *   - 自动选择最优算法
 *   - 支持批量矩阵乘法 (Strided Batched GEMM)
 */

#include <assert.h>
#include <type_traits>      // std::bool_constant
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"
// GELU 可以融合到 cuBLASLt 或单独计算
#include "gelu.cuh"

// ----------------------------------------------------------------------------
// CUDA 内核函数

/*
 * 偏置梯度计算内核 (高度优化版本9)
 * 
 * 数学公式: dbias[oc] = Σ dout[b,t,oc]  (对所有 batch 和 time 维度求和)
 * 
 * 优化策略:
 *   1. 向量化加载: 每个线程处理 x128::size 个连续的 OC 元素
 *   2. 分层规约: warp 内规约 → block 内规约 → (可选) 跨 block 规约
 *   3. 共享内存: 用于 warp 间通信
 * 
 * 线程块结构: (bdx=4, bdy=8, bdz=32|24)
 *   - bdx: warp 内的线程分组 (用于 shuffle 规约)
 *   - bdy: 每个 warp 处理的 OC 块数
 *   - bdz: 每个 block 中的 warp 数
 * 
 * @tparam OutFloat: 输出类型 (floatX 或 float)
 * @tparam UseAuxBuffer: 是否使用辅助缓冲区进行跨 block 规约
 * @param dbias: 偏置梯度输出 [OC] 或 [grid_size_y, OC]
 * @param dout: 输出梯度 [B, T, OC]
 * @param B: batch 大小
 * @param T: 序列长度
 * @param OC: 输出通道数
 */
template<typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    // 线程块维度常量
    constexpr const int bdx = 4;                  // warp 内 x 维度线程数
    constexpr const int bdy = WARP_SIZE / bdx;    // warp 内 y 维度线程数 (8)
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    // 线程索引
    int warp_d = (int)threadIdx.x;    // warp 内的 x 位置 [0,3]
    int warp_c = (int)threadIdx.y;    // warp 内的 y 位置 [0,7]，对应不同的 OC 块
    int block_d = (int)threadIdx.z;   // block 内的 warp 索引

    // 每个 warp 处理 64 个 OC (BF16: 8 * 8 = 64)
    const int OC_per_warp = bdy * x128::size;

    // 计算当前线程负责的 OC 范围
    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    // 计算当前线程负责的 (B*T) 范围
    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    // 累加器: 每个线程累加 x128::size 个 OC 的梯度
    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    // 阶段1: 每个线程遍历其负责的 (B*T) 范围，累加到寄存器
    if(global_oc < OC) {
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    // 共享内存: 用于 warp 间规约
    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

    // 阶段2: warp 内规约 (使用 shuffle 指令)
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        // shuffle down 规约: 4个线程 → 1个结果
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // 阶段3: block 内跨 warp 规约
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                // 直接写入最终结果 (累加到现有值)
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                // 写入辅助缓冲区 (稍后进行跨 block 规约)
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

/*
 * 跨 block 规约求和内核
 * 
 * 当 bias 梯度计算需要多个 block 时，每个 block 将部分结果写入辅助缓冲区，
 * 然后此内核将所有部分结果相加得到最终结果。
 * 
 * @param dst: 最终输出 [n]，结果累加到此
 * @param src: 辅助缓冲区 [m, n]，包含 m 个 block 的部分结果
 * @param n: 输出通道数 (OC)
 * @param m: 需要规约的 block 数量 (grid_size_y)
 */
__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    // 每个线程处理 f128::size 个连续元素 (向量化)
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    
    if (idx < n) {
        // 初始化累加器
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        // 对所有 m 个 block 的结果求和
        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        
        // 累加到最终输出
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}

// ----------------------------------------------------------------------------
// 内核启动器

/*
 * cuBLASLt 矩阵乘法封装函数
 * 
 * 计算: D = alpha * op(A) * op(B) + beta * C
 * 其中 op() 可以是转置或非转置，支持可选的 bias 加法和 GELU 激活
 * 
 * cuBLASLt 使用列主序 (column-major)，但这里的参数按行主序思维理解:
 *   - 标准 GEMM: D[m,n] = A[m,k] @ B[k,n]
 *   - 转置 A:    D[m,n] = A^T[m,k] @ B[k,n]  (A 存储为 [k,m])
 * 
 * Epilogue 融合操作:
 *   - CUBLASLT_EPILOGUE_BIAS: 加偏置
 *   - CUBLASLT_EPILOGUE_GELU_AUX: GELU 激活 (保存 pre-GELU 值用于反向)
 *   - CUBLASLT_EPILOGUE_DGELU: GELU 反向传播
 *   - CUBLASLT_EPILOGUE_BGRADB: 反向计算 bias 梯度
 * 
 * 参考: https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
 * 
 * @param d: 输出矩阵 [m, n]
 * @param a: 输入矩阵 A，若 transA=true 则 [k, m]，否则 [m, k]
 * @param b: 输入矩阵 B，若 transB=true 则 [n, k]，否则 [k, n]
 * @param bias: 可选偏置向量 [m]，为 NULL 则不加偏置
 * @param m: 输出行数 (通常是 OC)
 * @param n: 输出列数 (通常是 B*T)
 * @param k: 内部维度 (通常是 C)
 * @param stream: CUDA 流
 * @param transA: 是否转置 A (默认 true，对应 weight^T @ inp)
 * @param transB: 是否转置 B (默认 false)
 * @param batch_count: 批量 GEMM 的 batch 数 (0 表示普通 GEMM)
 * @param strideA/B/Out: 批量 GEMM 中每个 batch 间的步长
 * @param accumulate: 是否累加到输出 (true: D += result, false: D = result)
 * @param pre_gelu: 若非 NULL，启用 GELU 融合并保存 pre-GELU 值
 * @param backward: 是否为反向传播 (影响 epilogue 选择)
 */
void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false)
{
    NVTX_RANGE_FN();
    bool has_bias = (bias != NULL);
    bool has_gelu = (pre_gelu != NULL);

    // 检查内存对齐 (16字节对齐是性能最优的要求)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        printf("All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // 创建操作描述符
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    // 设置转置属性
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA)  ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // 定义矩阵布局 (列主序)
    // 对于列主序: 矩阵 [rows, cols] 的 leading dimension = rows
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        // A^T: 存储为 [k, m]，但逻辑上是 [m, k]
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }
    // FP8 模式要求 C 矩阵为 BF16 或 FP32
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

    // 批量矩阵乘法设置 (用于非 Flash Attention)
    if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        // 设置各 batch 之间的内存步长
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    // 创建算法选择偏好，指定最大工作空间
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // 设置 epilogue (融合操作): bias 加法 和/或 GELU 激活
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m;  // GELU 辅助缓冲区的 leading dimension
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            // 反向传播: 计算 GELU 的梯度
            assert(!has_bias);  // 反向不应同时有 GELU 和 bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            // 前向传播: 应用 GELU 激活，保存 pre-GELU 值用于反向
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        // 只有 bias，无 GELU
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        // 无 epilogue
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // FP8 模式要求 bias 为 BF16
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // 缩放类型始终为 FP32 (即使使用 FP8 计算)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // 使用启发式算法选择最优 GEMM 算法 (内部会缓存结果)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // 设置 alpha 和 beta: D = alpha * A * B + beta * C
    // accumulate=true 时 beta=1 实现累加
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // 执行矩阵乘法
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    // 清理资源
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

/*
 * 矩阵乘法前向传播封装
 * 
 * 计算: out = GELU(inp @ weight^T + bias)  (若启用 GELU)
 *       out = inp @ weight^T + bias         (若未启用 GELU)
 * 
 * @param out: 输出张量 [B, T, OC]
 * @param inp: 输入张量 [B, T, C]
 * @param weight: 权重矩阵 [OC, C]
 * @param bias: 偏置向量 [OC]，可为 NULL
 * @param B: batch 大小
 * @param T: 序列长度
 * @param C: 输入通道数
 * @param OC: 输出通道数
 * @param stream: CUDA 流
 * @param pre_gelu: 若非 NULL，保存 GELU 前的值用于反向传播 [B, T, OC]
 * @param gelu_fusion: GELU 融合级别
 *                     0: 不融合 (先 matmul 再单独 GELU)
 *                     1+: 融合到 cuBLASLt epilogue (H100+ 更高效)
 */
void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    // gelu_fusion < 1: 分开计算 matmul 和 GELU (Ada/Ampere 可能更高效)
    // gelu_fusion >= 1: 融合到 cuBLASLt (H100+ 推荐)
    if (gelu_fusion < 1 && pre_gelu) {
        // 分开计算: matmul → pre_gelu, 然后 GELU(pre_gelu) → out
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
        gelu_forward(out, pre_gelu, B*T*OC, stream);
    } else {
        // 融合计算: matmul + GELU 一步完成
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
    }
}

/*
 * 矩阵乘法反向传播
 * 
 * 计算三个梯度:
 *   1. dinp = dout @ weight          (对输入的梯度，若有 GELU 还需乘以 GELU')
 *   2. dweight += inp^T @ dout       (对权重的梯度，累加)
 *   3. dbias = sum(dout, axis=0)     (对偏置的梯度，沿 batch 维度求和)
 * 
 * @param dinp: 输入梯度输出 [B, T, C]
 * @param dweight: 权重梯度 [OC, C]，累加模式
 * @param dbias: 偏置梯度 [OC]，可为 NULL，累加模式
 * @param dout: 输出梯度 [B, T, OC]
 * @param inp: 前向传播的输入 [B, T, C]
 * @param weight: 权重矩阵 [OC, C]
 * @param dbias_buffer: 辅助缓冲区，用于跨 block 规约 [grid_size_y, OC]
 * @param B: batch 大小
 * @param T: 序列长度
 * @param C: 输入通道数
 * @param OC: 输出通道数
 * @param stream: CUDA 流
 * @param pre_gelu: 前向保存的 GELU 前值，用于计算 GELU 梯度
 * @param gelu_fusion: GELU 融合级别
 *                     < 2: 单独计算 GELU 反向
 *                     >= 2: 融合到 cuBLASLt
 */
void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    NVTX_RANGE_FN();

    // ========== 步骤1: 计算 bias 梯度 (如果需要) ==========
    // dbias[oc] = Σ_{b,t} dout[b,t,oc]
    if (dbias != NULL) {
        // 根据 GPU SM 数量选择 block 大小
        // H100: maxThreadsPerMultiProcessor=2048 → block_size=1024
        // 其他: maxThreadsPerMultiProcessor=1536 → block_size=768
        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        // 线程块配置: (4, 8, 32|24)
        dim3 block_dim = {4, 8, (unsigned)block_size/WARP_SIZE};
        const int OC_per_warp = block_dim.y * x128::size;  // 每个 warp 处理 64 个 OC (BF16)
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // x 方向 block 数
        // y 方向 block 数: 尽量占满整个 GPU
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x));

        if(grid_size_y == 1) {
            // 单行 block 足够: 直接写入最终结果
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, B, T, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // 多行 block: 先写入缓冲区，再规约
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            // 跨 block 规约
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        // 防止 cuBLASLt 再次计算 dbias (已经单独计算过了)
        dbias = NULL;
    }

    // ========== 步骤2: 计算输入梯度 ==========
    // dinp = dout @ weight  (若有 GELU: dinp *= GELU'(pre_gelu))
    matmul_cublaslt(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
                    gelu_fusion >= 2 ? pre_gelu : NULL, true);

    // 若 GELU 未融合，单独计算 GELU 反向
    if (gelu_fusion < 2 && pre_gelu) {
        gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream);
    }

    // ========== 步骤3: 计算权重梯度 ==========
    // dweight += inp^T @ dout (累加模式)
    matmul_cublaslt(dweight, inp, dout, NULL /*dbias*/, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
                    true /* accumulate */, NULL, true);
}
