/*
 * ============================================================================
 * ZeRO (Zero Redundancy Optimizer) 分片工具
 * ============================================================================
 * 
 * 本文件实现多 GPU 分布式训练的 ZeRO 优化策略
 * 
 * ZeRO 优化等级 (https://arxiv.org/abs/1910.02054):
 * - Stage 0: 无优化，每个 GPU 保存完整的优化器状态、梯度和模型参数
 * - Stage 1: 优化器状态分片 (OSS) - 每个 GPU 只保存部分优化器状态
 * - Stage 2: 优化器 + 梯度分片 (SDP) - 梯度也被分片
 * - Stage 3: 完全分片 (FSDP) - 模型参数也被分片
 * 
 * 显存节省分析 (N 个 GPU):
 * - Stage 1: 优化器状态减少 N 倍 (Adam 的 m/v 通常占用最多显存)
 * - Stage 2: 梯度额外减少 N 倍
 * - Stage 3: 模型参数也减少 N 倍 (当前未实现)
 * 
 * NCCL 初始化方法:
 * - MPI: 使用 MPI 广播 NCCL ID (需要 MPI 支持)
 * - TCP: 通过 TCP socket 分发 NCCL ID (跨节点通信)
 * - FS:  通过共享文件系统传递 NCCL ID (最简单)
 */

#ifndef LLMC_ZERO_CUH
#define LLMC_ZERO_CUH

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#ifdef MULTI_GPU
#include <nccl.h>  // NVIDIA Collective Communications Library
#ifdef USE_MPI
#include <mpi.h>   // Message Passing Interface
#endif
#endif

// 定义文件和 socket 操作的错误检查宏
#include "utils.h"

// ============================================================================
// 多 GPU 相关定义
// ============================================================================

#ifdef MULTI_GPU

/**
 * 根据编译时精度设置选择对应的 NCCL 数据类型
 * NCCL 通信需要指定数据类型以正确处理字节宽度
 */
#if defined(ENABLE_FP32)
const ncclDataType_t ncclFloatX = ncclFloat;    // FP32: 32位浮点
#elif defined(ENABLE_FP16)
const ncclDataType_t ncclFloatX = ncclHalf;     // FP16: 16位半精度
#else // 默认使用 bfloat16
const ncclDataType_t ncclFloatX = ncclBfloat16; // BF16: 16位脑浮点
#endif

/**
 * NCCL 错误检查函数
 * @param status: NCCL 返回状态
 * @param file:   源文件名
 * @param line:   行号
 */
void nccl_check(ncclResult_t status, const char *file, int line) {
    if (status != ncclSuccess) {
        printf("[NCCL ERROR] at file %s:%d:\n%s\n", file, line, ncclGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

#ifdef USE_MPI
/**
 * MPI 错误检查函数
 * @param status: MPI 返回状态
 * @param file:   源文件名
 * @param line:   行号
 */
void mpi_check(int status, const char *file, int line) {
    if (status != MPI_SUCCESS) {
        char mpi_error[4096];
        int mpi_error_len = 0;
        assert(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) == MPI_SUCCESS);
        printf("[MPI ERROR] at file %s:%d:\n%.*s\n", file, line, mpi_error_len, mpi_error);
        exit(EXIT_FAILURE);
    }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))
#endif

#endif // MULTI_GPU

// ============================================================================
// 多 GPU 配置结构体
// ============================================================================

/**
 * MultiGpuConfig - 多 GPU 训练配置
 * 
 * 包含分布式训练所需的所有配置信息和通信原语
 */
typedef struct {
    // ========== 进程标识 ==========
    int process_rank;      // 当前进程在所有进程中的排名 (0-based)，单 GPU 时为 0
    int num_processes;     // 总进程数，单 GPU 时为 1
    int local_device_idx;  // 当前进程在本机上使用的 GPU 索引 (多节点时每节点重新编号)

    // ========== ZeRO 优化配置 ==========
    // ZeRO 等级详解: https://fairscale.readthedocs.io/en/stable/deep_dive/oss_sdp_fsdp.html
    // 0 - 禁用: 每个 GPU 保存完整副本
    // 1 - OSS (Optimizer State Sharding): 优化器状态分片
    // 2 - SDP (Sharded Data Parallel): 优化器 + 梯度分片 (未实现)
    // 3 - FSDP (Fully Sharded Data Parallel): 完全分片 (未实现)
    int zero_stage;
    size_t shard_num_parameters;  // 本进程负责的参数数量 (= total / num_processes)

#ifdef MULTI_GPU
    // ========== NCCL 通信原语 ==========
    ncclComm_t nccl_comm;          // NCCL 通信器，用于 GPU 间集合通信
    cudaStream_t nccl_stream;      // NCCL 操作专用 CUDA 流 (与计算流并行)
    cudaEvent_t compute_nccl_sync; // 用于同步计算流和 NCCL 流的事件 (无计时以提高性能)
    float* unified_buffer;         // 统一内存缓冲区，用于 CPU 数值的跨 GPU 归约
#endif
} MultiGpuConfig;

/**
 * 全局多 GPU 配置实例
 * 使用 inline 避免多次包含头文件时的重复定义问题
 */
inline MultiGpuConfig multi_gpu_config;

#ifdef MULTI_GPU

// ============================================================================
// NCCL ID 分发函数 (TCP 方式)
// ============================================================================

#ifdef _WIN32
/**
 * Windows 版本: 向所有客户端发送 NCCL ID
 * @param nccl_id:        要发送的 NCCL 唯一标识符
 * @param client_sockets: 客户端 socket 数组
 * @param num_clients:    客户端数量
 */
void send_nccl_id_to_clients_windows(ncclUniqueId *nccl_id, SOCKET client_sockets[], int num_clients) {
    for (int i = 0; i < num_clients; ++i) {
        if (send(client_sockets[i], (const char *)nccl_id, sizeof(*nccl_id), 0) == SOCKET_ERROR) {
            printf("Failed to send nccl_id");
            WSACleanup();
            exit(EXIT_FAILURE);
        }
        closesocketCheck(client_sockets[i]);
    }
}
#else
/**
 * Linux/Unix 版本: 向所有客户端发送 NCCL ID
 * @param nccl_id:        要发送的 NCCL 唯一标识符
 * @param client_sockets: 客户端 socket 数组
 * @param num_clients:    客户端数量
 */
void send_nccl_id_to_clients(ncclUniqueId *nccl_id, int client_sockets[], int num_clients) {
    for (int i = 0; i < num_clients; ++i) {
        if (send(client_sockets[i], nccl_id, sizeof(*nccl_id), 0) == -1) {
            printf("Failed to send nccl_id");
            exit(EXIT_FAILURE);
        }
        scloseCheck(client_sockets[i]);
    }
}
#endif

#ifdef _WIN32
/**
 * get_nccl_id_via_tcp_windows - Windows 下通过 TCP 获取 NCCL ID
 * 
 * Rank 0 作为服务器等待其他进程连接，然后广播 NCCL ID
 * 其他 rank 作为客户端连接并接收 NCCL ID
 * 
 * @param result:    多 GPU 配置 (用于获取 rank 信息)
 * @param server_ip: 服务器 IP 地址 (rank 0 的 IP)
 * @return:          NCCL 唯一标识符
 */
ncclUniqueId get_nccl_id_via_tcp_windows(MultiGpuConfig* result, const char* server_ip) {
    ncclUniqueId nccl_id;
    // 使用固定端口号 (注册端口范围 1024-49151)
    int SERVER_PORT = 12345;
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        printf("WSAStartup failed");
        exit(EXIT_FAILURE);
    }

    if (result->process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));

        int MAX_CLIENTS = result->num_processes - 1;
        SOCKET client_sockets[MAX_CLIENTS];
        int num_clients = 0;
        SOCKET server_socket, new_socket;
        struct sockaddr_in address;
        int addrlen = sizeof(address);

        // Step 1) create a server TCP socket
        if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
            printf("Socket failed");
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 2) set the server address and port
        address.sin_family = AF_INET;  // IPv4
        address.sin_addr.s_addr = inet_addr(server_ip);
        address.sin_port = htons(SERVER_PORT);

        // Step 3) bind the socket to the address and port
        if (bind(server_socket, (struct sockaddr *)&address, sizeof(address)) == SOCKET_ERROR) {
            printf("Bind failed");
            closesocketCheck(server_socket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 4) MAX_CLIENTS specifies the maximum number of clients that can be queued for this server
        if (listen(server_socket, MAX_CLIENTS) == SOCKET_ERROR) {
            printf("Listen failed");
            closesocketCheck(server_socket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 5) accept connections from clients
        printf("Waiting for clients to connect...\n");
        while (num_clients < MAX_CLIENTS) {
            if ((new_socket = accept(server_socket, (struct sockaddr *)&address, &addrlen)) == INVALID_SOCKET) {
                printf("Accept failed");
                closesocketCheck(server_socket);
                WSACleanup();
                exit(EXIT_FAILURE);
            }
            client_sockets[num_clients++] = new_socket;
            printf("Client %d connected\n", num_clients);
        }

        // Step 6) send the NCCL ID to all clients
        send_nccl_id_to_clients_windows(&nccl_id, client_sockets, num_clients);
        printf("NCCL ID sent to all clients\n");

        closesocketCheck(server_socket);
    } else {
        int num_connection_attempts = 5;
        int time_to_sleep = 2;
        SOCKET client_socket;
        struct sockaddr_in serv_addr;

        // Step 1) create a client TCP socket
        if ((client_socket = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
            printf("Socket creation error");
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 2) set the server address and port
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(SERVER_PORT);
        if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
            printf("Invalid address or address not supported");
            closesocketCheck(client_socket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 3) Try to connect to the server - retry up to `num_connection_attempts` times if the connection fails
        while (connect(client_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == SOCKET_ERROR) {
            printf("%d Connection failed, retrying in %d seconds\n", result->process_rank, time_to_sleep);
            if (--num_connection_attempts == 0) {
                printf("Failed to connect to the server\n");
                closesocketCheck(client_socket);
                WSACleanup();
                exit(EXIT_FAILURE);
            }
            Sleep(time_to_sleep * 1000);
        }

        // Step 4) receive the NCCL ID from the server
        if (recv(client_socket, (char *)&nccl_id, sizeof(nccl_id), 0) <= 0) {
            printf("Failed to receive nccl_id");
            closesocketCheck(client_socket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        printf("Received NCCL ID\n");
        closesocketCheck(client_socket);
    }

    WSACleanup();
    return nccl_id;
}
#else
/**
 * get_nccl_id_via_tcp - Linux/Unix 下通过 TCP 获取 NCCL ID
 * 
 * 工作流程:
 * - Rank 0 (服务器): 创建 TCP socket -> 绑定地址 -> 监听 -> 接受连接 -> 发送 NCCL ID
 * - 其他 rank (客户端): 创建 socket -> 连接服务器 -> 接收 NCCL ID
 * 
 * @param result:    多 GPU 配置
 * @param server_ip: 服务器 IP 地址 (rank 0 的以太网 IP)
 * @return:          NCCL 唯一标识符
 */
ncclUniqueId get_nccl_id_via_tcp(MultiGpuConfig* result, const char* server_ip) {
    ncclUniqueId nccl_id;

    // 使用固定端口号 (1024-49151 范围内的注册端口)
    int SERVER_PORT = 12345;
    
    if (result->process_rank == 0) {
        // ========== Rank 0: 作为 TCP 服务器 ==========
        ncclCheck(ncclGetUniqueId(&nccl_id));

        int MAX_CLIENTS = result->num_processes - 1;
        int client_sockets[MAX_CLIENTS];
        int num_clients = 0;
        int server_socket, new_socket;
        struct sockaddr_in address;
        int addrlen = sizeof(address);
        int opt = 1;

        // Step 1) create a server TCP socket
        if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            printf("Socket failed");
            exit(EXIT_FAILURE);
        }

        // Step 2) set socket options
        // SOL_SOCKET - means that option is configured at socket level
        // SO_REUSEADDR - allows to bind to an address which is in a TIME_WAIT state (already used by another socket) - useful when restarting the server
        // SO_REUSEPORT - allows to bind to the same port multiple times
        if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
            printf("Setsockopt failed");
            exit(EXIT_FAILURE);
        }

        // Step 3) set the server address and port
        address.sin_family = AF_INET;  // IPv4
        address.sin_addr.s_addr = inet_addr(server_ip); // alternatively use INADDR_ANY to bind to all interfaces, currently we only allow ethernet
        address.sin_port = htons(SERVER_PORT);

        // Step 4) bind the socket to the address and port
        if (bind(server_socket, (struct sockaddr *)&address, sizeof(address)) < 0) {
            printf("Bind failed");
            exit(EXIT_FAILURE);
        }

        // Step 5) MAX_CLIENTS specifies the maximum number of clients that can be queued for this server
        if (listen(server_socket, MAX_CLIENTS) < 0) {
            printf("Listen failed");
            exit(EXIT_FAILURE);
        }

        // Step 6) accept connections from clients
        printf("Waiting for clients to connect...\n");
        while (num_clients < MAX_CLIENTS) {
            if ((new_socket = accept(server_socket, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
                printf("Accept failed");
                exit(EXIT_FAILURE);
            }
            client_sockets[num_clients++] = new_socket;
            printf("Client %d connected\n", num_clients);
        }

        // Step 7) send the NCCL ID to all clients
        send_nccl_id_to_clients(&nccl_id, client_sockets, num_clients);
        printf("NCCL ID sent to all clients\n");

        scloseCheck(server_socket);
    } else {
        int num_connection_attempts = 5;
        int time_to_sleep = 2;
        int client_socket;
        struct sockaddr_in serv_addr;

        // Step 1) create a client TCP socket
        if ((client_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            printf("Socket creation error");
            exit(EXIT_FAILURE);
        }

        // Step 2) set the server address and port
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(SERVER_PORT);
        if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
            printf("Invalid address or address not supported");
            exit(EXIT_FAILURE);
        }

        // Step 3) Try to connect to the server - retry up to `num_connection_attempts` times if the connection fails
        while (connect(client_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            printf("%d Connection failed, retrying in %d seconds\n", result->process_rank, time_to_sleep);
            if (--num_connection_attempts == 0) {
                printf("Failed to connect to the server\n");
                exit(EXIT_FAILURE);
            }
            sleep(time_to_sleep);
        }

        // Step 4) receive the NCCL ID from the server
        if (recv(client_socket, &nccl_id, sizeof(nccl_id), 0) <= 0) {
            printf("Failed to receive nccl_id");
            exit(EXIT_FAILURE);
        }

        printf("Received NCCL ID\n");
        scloseCheck(client_socket);
    }

    return nccl_id;
}
#endif

/**
 * get_nccl_id_via_fs - 通过共享文件系统获取 NCCL ID
 * 
 * 最简单的方法，适用于所有进程共享同一文件系统的场景 (NFS, Lustre 等)
 * 
 * 工作流程:
 * - Rank 0: 生成 NCCL ID 并写入文件
 * - 其他 rank: 等待文件出现并读取 NCCL ID
 * 
 * @param result:  多 GPU 配置
 * @param fs_path: 共享文件系统路径 (所有节点可访问)
 * @return:        NCCL 唯一标识符
 * 
 * 注意: 这种方法依赖简单的 sleep 同步，不是 100% 可靠，但通常能工作
 */
ncclUniqueId get_nccl_id_via_fs(MultiGpuConfig* result, char* fs_path) {
    ncclUniqueId nccl_id;
    FILE* idFile;
    static char filename[1024];
    snprintf(filename, sizeof(filename), "%s/ncclUniqueId.sync", fs_path);

    // 非 rank 0 进程先等待，给 rank 0 时间写文件
    if (result->process_rank != 0) {
        sleep(2);  // 简单的同步策略
    }

    if (result->process_rank == 0) {
        // Rank 0: 生成并写入 NCCL ID
        ncclCheck(ncclGetUniqueId(&nccl_id));
        idFile = fopen(filename, "wb");
        assert(idFile != NULL);
        fwriteCheck(&nccl_id, sizeof(nccl_id), 1, idFile);
        fcloseCheck(idFile);
    } else {
        // 其他 rank: 循环等待文件出现并读取
        do {
            sleep(1);  // 每秒检查一次
            idFile = fopen(filename, "rb");
            if (idFile != NULL) break;
        } while (idFile == NULL);
        freadCheck(&nccl_id, sizeof(nccl_id), 1, idFile);
        fcloseCheck(idFile);
    }

    return nccl_id;
}

#ifdef USE_MPI
/**
 * multi_gpu_get_local_device_idx - 确定当前进程应使用的本地 GPU 索引
 * 
 * 在多节点场景下，同一机器上的进程使用不同的 GPU
 * 不同机器上的进程可以使用相同索引的 GPU
 * 
 * 算法:
 * 1. 获取主机名并计算哈希
 * 2. 通过 MPI_Allgather 收集所有进程的主机名哈希
 * 3. 统计同一机器上 rank 更小的进程数量作为本地 GPU 索引
 * 
 * @param process_rank:  当前进程全局 rank
 * @param num_processes: 总进程数
 * @return:              本机上应使用的 GPU 索引
 * 
 * 参考: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
 */
int multi_gpu_get_local_device_idx(int process_rank, int num_processes) {
    char hostname[1024];
    hostname[1023] = '\0';
    // All processes on the same machine will share the same hostname.
    gethostname(hostname, 1023);
    for (int i=0; i < 1024; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            break;
        }
    }
    uint64_t hostname_hash = 5381u;
    for (int c = 0; hostname[c] != '\0'; c++){ hostname_hash = ((hostname_hash << 5u) + hostname_hash) ^ hostname[c]; }

    // Distribute all hostname hashes to all processes.
    uint64_t* all_hostsname_hashes = (uint64_t*)malloc(num_processes * sizeof(uint64_t));
    all_hostsname_hashes[process_rank] = hostname_hash;
    mpiCheck(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_hostsname_hashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    // Identify which GPU we need to use.
    int local_device_idx = 0;
    for (int current_process = 0; current_process < num_processes; ++current_process) {
        if (current_process == process_rank) {
        // Found my gpu, local_device_idx now has my target GPU index.
        break;
        }
        if (all_hostsname_hashes[current_process] == all_hostsname_hashes[process_rank]) {
        // This process ID runs on the same machine, but it's not me, skip this GPU
        local_device_idx++;
        }
    }

    free(all_hostsname_hashes);
    return local_device_idx;
}
#endif

#endif

// ============================================================================
// 多 GPU 配置初始化和清理
// ============================================================================

/**
 * multi_gpu_config_init - 初始化多 GPU 训练环境
 * 
 * 执行以下操作:
 * 1. 根据初始化方法获取 NCCL ID (MPI/TCP/FS)
 * 2. 设置 CUDA 设备
 * 3. 初始化 NCCL 通信器
 * 4. 创建 NCCL 专用流和同步事件
 * 5. 分配统一内存缓冲区
 * 
 * @param num_processes:  总进程数 (仅 TCP/FS 方式使用)
 * @param process_rank:   当前进程 rank (仅 TCP/FS 方式使用)
 * @param gpus_per_node:  每节点 GPU 数量
 * @param server_ip:      TCP 服务器 IP (仅 TCP 方式使用)
 * @param fs_path:        共享文件系统路径 (仅 FS 方式使用)
 * @param init_method:    初始化方法 ("mpi", "tcp", "fs")
 * @return:               初始化后的多 GPU 配置
 * 
 * 注意: 新版 Slurm (slurm-wlm) 禁用了 PMIx，多节点时无法使用 MPI 初始化
 */
MultiGpuConfig multi_gpu_config_init(int num_processes, int process_rank, int gpus_per_node, 
                                     char* server_ip, char* fs_path, char* init_method) {
#ifdef MULTI_GPU
    MultiGpuConfig result;
    ncclUniqueId nccl_id;
    
    // 根据初始化方法获取 NCCL ID
    if (strcmp(init_method, "mpi") == 0) {
        // ========== MPI 方式: 使用 MPI 广播 NCCL ID ==========
        #ifdef USE_MPI
        mpiCheck(MPI_Init(NULL, NULL));
        mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &result.process_rank));
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &result.num_processes));
        result.local_device_idx = multi_gpu_get_local_device_idx(result.process_rank, result.num_processes);
        if (result.process_rank == 0) {
            ncclCheck(ncclGetUniqueId(&nccl_id));
        }
        mpiCheck(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
        #else
        printf("MPI support is disabled. Please enable MPI support to use MPI-based NCCL-init method.\n");
        exit(EXIT_FAILURE);
        #endif
    } else {
        result.process_rank = process_rank;
        result.num_processes = num_processes;
        result.local_device_idx = process_rank % gpus_per_node;
        if (strcmp(init_method, "tcp") == 0) {
            #ifdef _WIN32
            nccl_id = get_nccl_id_via_tcp_windows(&result, server_ip);
            #else
            nccl_id = get_nccl_id_via_tcp(&result, server_ip);
            #endif
        } else if (strcmp(init_method, "fs") == 0) {
            nccl_id = get_nccl_id_via_fs(&result, fs_path);
        } else {
            printf("Invalid NCCL-init method\n");
            exit(EXIT_FAILURE);
        }
    }
    cudaCheck(cudaSetDevice(result.local_device_idx));
    ncclCheck(ncclCommInitRank(&result.nccl_comm, result.num_processes, nccl_id, result.process_rank));
    cudaCheck(cudaStreamCreate(&result.nccl_stream));
    // event without timing for maximum performance
    cudaCheck(cudaEventCreate(&result.compute_nccl_sync, cudaEventDisableTiming));
    nvtxNameCudaStreamA(result.nccl_stream, "nccl stream");
    nvtxNameCudaEventA(result.compute_nccl_sync, "nccl compute sync");
    cudaCheck(cudaMallocManaged(&result.unified_buffer, sizeof(float)));
    return result;
#else
    printf("Multi-GPU support is disabled. Using a single GPU.\n");
    cudaCheck(cudaSetDevice(0));
    MultiGpuConfig result;
    result.process_rank = 0;
    result.num_processes = 1;
    result.local_device_idx = 0;
    return result;
#endif
}

/**
 * multi_gpu_config_free - 释放多 GPU 配置资源
 * 
 * 释放 NCCL 通信器、CUDA 流、事件和统一内存
 * 如果使用了 MPI，也会调用 MPI_Finalize()
 * 
 * @param config: 要释放的多 GPU 配置
 */
void multi_gpu_config_free(MultiGpuConfig* config) {
#ifdef MULTI_GPU
    ncclCheck(ncclCommDestroy(config->nccl_comm));      // 销毁 NCCL 通信器
    cudaCheck(cudaStreamDestroy(config->nccl_stream));  // 销毁 NCCL 流
    cudaCheck(cudaEventDestroy(config->compute_nccl_sync)); // 销毁同步事件
    cudaCheck(cudaFree(config->unified_buffer));        // 释放统一内存
    #ifdef USE_MPI
    mpiCheck(MPI_Finalize());  // 终止 MPI 环境
    #endif
#endif
}

/**
 * multi_gpu_barrier - 多 GPU 屏障同步
 * 
 * 通过一个空的 AllReduce 操作实现所有 GPU 的同步
 * 确保所有进程在继续之前都到达此点
 * 
 * @param config: 多 GPU 配置
 */
void multi_gpu_barrier(const MultiGpuConfig* config) {
#ifdef MULTI_GPU
    if (config->num_processes > 1) {
        // 使用 AllReduce 实现屏障 (实际上就是对一个值求和)
        ncclCheck(ncclAllReduce(config->unified_buffer, config->unified_buffer, 
                               sizeof(float), ncclFloat, ncclSum, 
                               config->nccl_comm, config->nccl_stream));
    }
    cudaCheck(cudaDeviceSynchronize());  // 等待 GPU 完成
#endif
}

// ============================================================================
// 分片信息和梯度归约
// ============================================================================

/**
 * ShardInfo - 张量分片信息
 * 
 * 描述当前进程负责的分片在原张量中的位置和大小
 */
typedef struct {
    ptrdiff_t offset;  // 分片起始偏移 (元素索引)
    size_t size;       // 分片大小 (元素数量)
} ShardInfo;

/**
 * multi_gpu_get_shard_offset - 获取当前进程的分片信息
 * 
 * 根据 ZeRO stage 和进程 rank 计算当前进程负责的分片
 * 
 * @param elements:       张量总元素数
 * @param config:         多 GPU 配置
 * @param shard_at_stage: 在哪个 ZeRO stage 开始分片
 * @return:               分片的偏移和大小
 * 
 * 示例: 4 GPU, 1000 个元素, rank 1
 *   -> offset = 250, size = 250
 */
ShardInfo multi_gpu_get_shard_offset(size_t elements, const MultiGpuConfig* config, int shard_at_stage) {
    const int nproc = config->num_processes;
    if(config->zero_stage >= shard_at_stage) {
        if (elements % nproc != 0) {
            fprintf(stderr, "Number of elements %zu must be a multiple of the number of processes %d\n", elements, nproc);
            exit(EXIT_FAILURE);
        }
        return {(ptrdiff_t) (config->process_rank * (elements / nproc)), elements / nproc};
    } else {
        return {0, elements};
    }
}

/**
 * multi_gpu_async_reduce_gradient - 异步梯度归约
 * 
 * 在计算流完成后，将多个梯度张量在所有 GPU 间归约
 * 支持两种模式:
 * - ZeRO Stage 0: AllReduce (所有 GPU 得到完整梯度)
 * - ZeRO Stage 1: ReduceScatter (每个 GPU 只得到自己负责的分片)
 * 
 * @tparam N:              梯度张量数量
 * @param pointers:        梯度张量指针数组 [N]
 * @param pointers_sizes:  每个张量的元素数 [N]
 * @param config:          多 GPU 配置
 * @param compute_stream:  计算流 (归约会等待此流完成)
 * 
 * 实现细节:
 * - 使用 cudaEvent 实现计算流和 NCCL 流的同步 (避免主机端同步)
 * - 使用 ncclGroup 将多个归约操作打包成单个 GPU kernel
 * - 使用 ncclAvg 而不是 ncclSum，自动完成梯度平均
 * 
 * 注意: `(&pointers)[N]` 语法确保数组大小在编译时检查
 */
template<int N>
void multi_gpu_async_reduce_gradient(
        floatX* const (&pointers)[N], const size_t (&pointers_sizes)[N],
        MultiGpuConfig* config, cudaStream_t compute_stream) {
    if (config->num_processes == 1) {
        return; // no multi-GPU, just exit.
    }

#ifdef MULTI_GPU
    NVTX_RANGE_FN();
    // mark an event on the compute stream, and immediately wait on this in the nccl stream
    // this means that the nccl stream won't start executing before all compute kernels that
    // have been submitted before this point have finished.
    // by using an event instead of cudaSyncStream, we avoid having to synchronize the host, and
    // can enqueue new work to the GPU right away.
    cudaCheck(cudaEventRecord(config->compute_nccl_sync, compute_stream));
    cudaCheck(cudaStreamWaitEvent(config->nccl_stream, config->compute_nccl_sync));
    ncclCheck(ncclGroupStart()); // NCCL group: aggregate all pointers in a single NCCL GPU kernel.
    for (int i = 0; i < N; ++i) {
        if(config->zero_stage == 0) {
            ncclCheck(ncclAllReduce(
                    pointers[i], pointers[i],
                    pointers_sizes[i],
                    ncclFloatX, ncclAvg,
                    config->nccl_comm, config->nccl_stream
            ));
        } else if(config->zero_stage == 1) {
            assert(pointers_sizes[i] % config->num_processes == 0);
            size_t shard_size = pointers_sizes[i] / config->num_processes;
            ptrdiff_t shard_offset = (ptrdiff_t)shard_size * config->process_rank;
            ncclCheck(ncclReduceScatter(
                    pointers[i], pointers[i] + shard_offset,
                    shard_size,
                    ncclFloatX, ncclAvg,
                    config->nccl_comm, config->nccl_stream
            ));
        }
    }
    ncclCheck(ncclGroupEnd());
#endif
}

/** 
 * 便捷宏: 只在 rank 0 打印
 * 避免多 GPU 时重复输出
 */
#define printf0(...) if (::multi_gpu_config.process_rank == 0) { printf(__VA_ARGS__); }

/**
 * set_zero_configs - 配置 ZeRO 优化参数
 * 
 * 根据请求的 ZeRO stage 和参数数量设置分片配置
 * 如果参数数量不能被进程数整除，会禁用 ZeRO
 * 
 * @param config:           多 GPU 配置 (会被修改)
 * @param zero_stage:       请求的 ZeRO 等级 (0, 1, 2, 3)
 * @param total_parameters: 模型总参数数量
 */
void set_zero_configs(MultiGpuConfig* config, int zero_stage, size_t total_parameters) {
    config->zero_stage = 0;
    config->shard_num_parameters = total_parameters;
    // Check the Zero Stage and define sharding parameters
    if (zero_stage == 0) {
        printf0("| Zero Optimization is disabled                                              |\n");
    }
    else if (zero_stage == 1) {
        if (total_parameters % config->num_processes != 0) {
            printf0("| Zero Optimization is disabled, Can't equally partition parameters          |\n");
            config->zero_stage = 0;
        }
        else {
            config->zero_stage = 1;
            config->shard_num_parameters = total_parameters / config->num_processes;
        }
    }
    else{
        printf0("| Disabling Zero Optimization, Zero Stage2 and Stage3 are not yet supported  |\n");
        config->zero_stage = 0;
    }
}

/**
 * multi_gpu_cpu_float_sum - 跨 GPU 求和 CPU 值
 * 
 * 将单个 CPU 浮点数在所有 GPU 进程间求和
 * 常用于统计 loss 等标量
 * 
 * @param value:  本地值
 * @param config: 多 GPU 配置
 * @return:       所有进程值的总和
 * 
 * 实现: 使用统一内存缓冲区和 NCCL AllReduce
 */
float multi_gpu_cpu_float_sum(float value, MultiGpuConfig* config) {
#ifdef MULTI_GPU
    if (config->num_processes == 1) return value;  // 单 GPU 无需归约

    // 写入统一内存 -> AllReduce -> 读取结果
    float* unified_buffer = config->unified_buffer;
    *unified_buffer = value;
    ncclCheck(ncclAllReduce(unified_buffer, unified_buffer, sizeof(float), 
                           ncclFloat, ncclSum, config->nccl_comm, config->nccl_stream));
    cudaCheck(cudaDeviceSynchronize());  // 等待完成
    return *unified_buffer;
#else
    return value;  // 单 GPU 直接返回
#endif
}

#endif

