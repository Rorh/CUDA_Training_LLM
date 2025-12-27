/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#include "utils.h"
#include "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#include <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

typedef struct {
    // === Distributed Training Configuration ===
    // 分布式训练中当前进程的rank标识符 (0, 1, 2, ...)
    // Rank identifier for this process in distributed training (0, 1, 2, ...)
    int process_rank;
    
    // 分布式训练中的总进程数
    // Total number of processes participating in distributed training
    int num_processes;
    
    // === Batch and Token Configuration ===
    // 批次大小：同时处理的序列数量
    // Batch size: number of sequences processed simultaneously
    size_t B;
    
    // 序列长度：每个序列的token数量（上下文窗口大小）
    // Sequence length: number of tokens per sequence (context window size)
    size_t T;
    
    // 所有数据分片中的总token数量
    // Total number of tokens across all data shards
    size_t num_tokens;
    
    // 当前数据分片中每个进程的样本总数
    // Total number of samples in the current shard per process
    size_t shard_num_samples;
    
    // === Data Shards Management ===
    // glob模式匹配的结果，包含所有要迭代的数据分片文件路径
    // Result of glob pattern matching, containing all shard file paths to iterate
    glob_t glob_result;
    
    // 当前正在读取的数据分片索引
    // Index of the current shard being read from
    size_t current_shard_idx;
    
    // 当前分片中正在读取的样本索引
    // Index of the current sample being read within the current shard
    size_t current_sample_idx;
    
    // === File I/O ===
    // 当前打开的token文件句柄
    // File handle for the currently opened tokens file
    FILE* tokens_file;
    
    // === Data Buffers ===
    // 从文件读取数据的缓冲区，存储原始的uint16_t token数据
    // Buffer for reading data from file, stores raw uint16_t token data
    uint16_t* buffer;
    
    // 输入到transformer的token序列 (大小: B*T)
    // Input token sequences fed into the transformer (size: B*T)
    int* inputs;
    
    // transformer的目标token序列，通常是inputs向右偏移一位 (大小: B*T)
    // Target token sequences for the transformer, typically inputs shifted right by 1 (size: B*T)
    int* targets;
    
    // === Random Shuffling ===
    // 用于随机打乱的梅森旋转伪随机数生成器状态
    // Mersenne Twister RNG state for random shuffling
    mt19937_state shuffle_rng;
    
    // 是否启用数据打乱功能的标志
    // Flag indicating whether data shuffling is enabled
    int should_shuffle;
    
    // 数据分片的随机索引数组，用于分片级别的打乱
    // Array of randomized shard indices for shard-level shuffling
    int* shard_indices;
    
    // 分片内样本的随机索引数组，用于样本级别的打乱
    // Array of randomized sample indices within each shard for sample-level shuffling
    int* intra_shard_indices;
    
    // === Memory Layout and Offsets ===
    // 所有进程的总批次大小（字节数）：num_processes * B * T * sizeof(uint16_t)
    // Total batch size across all processes in bytes: num_processes * B * T * sizeof(uint16_t)
    size_t total_batch_size_bytes;
    
    // 当前进程在批次内的偏移量（字节数）：process_rank * B * T * sizeof(uint16_t)
    // Local batch offset for this process in bytes: process_rank * B * T * sizeof(uint16_t)
    size_t local_batch_offset_bytes;
    
    // 文件头部大小（字节数）：HEADER_SIZE * sizeof(int)
    // Header size in bytes: HEADER_SIZE * sizeof(int)
    size_t header_bytes;
    
    // 当前打开文件的总大小（字节数）
    // Total size of the currently opened file in bytes
    int64_t file_size_bytes;
} DataLoader;

/**
 * 加载指定的数据分片并验证其完整性
 * Load a specific data shard and validate its integrity
 * 
 * 此函数打开指定的数据分片文件，验证文件格式和内容的有效性，
 * 并计算该分片可以提供的样本数量。这是数据加载流程的核心函数。
 * This function opens the specified data shard file, validates the file format and content,
 * and calculates the number of samples this shard can provide. This is a core function in the data loading pipeline.
 * 
 * 文件格式规范：
 * File format specification:
 * - 头部：256个int32值，包含魔数、版本号、token数量等元信息
 * - Header: 256 int32 values containing magic number, version, token count, and other metadata
 * - 数据部分：连续的uint16_t token序列
 * - Data section: continuous sequence of uint16_t tokens
 * 
 * @param loader 已初始化的DataLoader结构体指针 / Pointer to initialized DataLoader struct
 * @param shard_index 要加载的分片索引 / Index of the shard to load
 * @return 该分片中的token总数 / Total number of tokens in this shard
 */
int64_t dataloader_load_shard_(DataLoader *loader, int shard_index) {
    // === 步骤1：处理数据打乱索引映射 ===
    // === Step 1: Handle data shuffling index mapping ===
    if (loader->should_shuffle) {
        // 如果启用打乱，将逻辑索引映射到实际的随机分片索引
        // If shuffling is enabled, map logical index to actual random shard index
        shard_index = loader->shard_indices[shard_index];
    }
    
    // === 步骤2：获取文件路径并打开文件 ===
    // === Step 2: Get file path and open file ===
    // 从glob结果中获取指定分片的文件路径
    // Get file path of specified shard from glob results
    const char* filename = loader->glob_result.gl_pathv[shard_index];
    
    // 关闭之前打开的文件（如果有），因为同时只能打开一个分片文件
    // Close previously opened file (if any), since only one shard file can be open at a time
    if (loader->tokens_file != NULL) {
        fcloseCheck(loader->tokens_file);
    }
    
    // 以二进制只读模式打开新的分片文件
    // Open new shard file in binary read-only mode
    loader->tokens_file = fopenCheck(filename, "rb");
    
    // === 步骤3：读取并验证文件头部 ===
    // === Step 3: Read and validate file header ===
    int header[HEADER_SIZE];  // HEADER_SIZE = 256
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
    
    // 验证魔数（文件格式标识符）
    // Validate magic number (file format identifier)
    if (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    
    // 验证版本号
    // Validate version number
    if (header[1] != 1) { 
        printf("Bad version in data file\n"); 
        exit(EXIT_FAILURE); 
    }
    
    // 获取文件中的token数量
    // Get number of tokens in the file
    int64_t ntok = header[2];
    assert(ntok > 0);  // 确保文件中有token数据 / Ensure there are tokens in the file
    
    // === 步骤4：验证文件大小一致性 ===
    // === Step 4: Validate file size consistency ===
    // 移动到文件末尾以获取文件大小
    // Seek to end of file to get file size
    fseekCheck(loader->tokens_file, 0, SEEK_END);
    loader->file_size_bytes = ftell(loader->tokens_file);  // 获取文件大小（字节数） / Get file size in bytes
    fseekCheck(loader->tokens_file, 0, SEEK_SET);  // 回到文件开头 / Seek back to beginning
    
    // 计算预期的文件大小：头部大小 + token数据大小
    // Calculate expected file size: header size + token data size
    int64_t expected_file_size = HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
    if (loader->file_size_bytes != expected_file_size) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }
    
    // === 步骤5：计算样本数量 ===
    // === Step 5: Calculate number of samples ===
    // 计算该分片可以提供的样本数量
    // Calculate number of samples this shard can provide
    // -1 uint16_t是因为我们读取B*T+1个token但只移动B*T个位置
    // -1 uint16_t because we read B*T+1 tokens but only advance by B*T positions
    loader->shard_num_samples = (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / loader->total_batch_size_bytes;
    
    return ntok;  // 返回该分片中的token总数 / Return total number of tokens in this shard
}

/**
 * 准备分片内样本的随机索引数组
 * Prepare random indices array for samples within a shard
 * 
 * 此函数为当前分片中的所有样本生成一个随机排列的索引数组，
 * 用于在分片内进行样本级别的数据打乱。这是两级打乱策略的第二级。
 * This function generates a randomly permuted indices array for all samples in the current shard,
 * used for sample-level data shuffling within the shard. This is the second level of the two-level shuffling strategy.
 * 
 * 两级打乱策略说明：
 * Two-level shuffling strategy explanation:
 * 1. 分片级打乱：改变分片的访问顺序 / Shard-level shuffling: change shard access order
 * 2. 分片内打乱：改变分片内样本的访问顺序 / Intra-shard shuffling: change sample access order within shard
 * 
 * @param loader 已初始化的DataLoader结构体指针 / Pointer to initialized DataLoader struct
 */
void prepare_intra_shard_indices_(DataLoader *loader) {
    // === 步骤1：清理旧的索引数组 ===
    // === Step 1: Clean up old indices array ===
    // 释放之前分配的分片内索引数组（如果存在）
    // Free previously allocated intra-shard indices array (if exists)
    // 这是必要的，因为不同分片可能有不同的样本数量
    // This is necessary because different shards may have different number of samples
    if (loader->intra_shard_indices != NULL) {
        free(loader->intra_shard_indices);
    }
    
    // === 步骤2：分配新的索引数组 ===
    // === Step 2: Allocate new indices array ===
    // 为当前分片的样本数量分配内存
    // Allocate memory for the number of samples in current shard
    loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
    
    // === 步骤3：初始化为恒等排列 ===
    // === Step 3: Initialize as identity permutation ===
    // 创建顺序索引：[0, 1, 2, ..., shard_num_samples-1]
    // Create sequential indices: [0, 1, 2, ..., shard_num_samples-1]
    init_identity_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples);
    
    // === 步骤4：随机打乱索引 ===
    // === Step 4: Randomly shuffle the indices ===
    // 使用Fisher-Yates洗牌算法随机打乱索引数组
    // Use Fisher-Yates shuffle algorithm to randomly permute the indices array
    // 使用每个进程独有的随机数生成器状态，确保不同进程有不同的打乱顺序
    // Use per-process RNG state to ensure different processes have different shuffle orders
    random_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples, &loader->shuffle_rng);
}

/**
 * 重置数据加载器到初始状态
 * Reset the data loader to initial state
 * 
 * 此函数将数据加载器重置到开始状态，重新开始一个新的epoch。
 * 它会重置所有索引，重新打乱数据顺序（如果启用），并加载第一个分片。
 * This function resets the data loader to the starting state to begin a new epoch.
 * It resets all indices, re-shuffles data order (if enabled), and loads the first shard.
 * 
 * 重置操作包括：
 * Reset operations include:
 * 1. 重置分片和样本索引 / Reset shard and sample indices
 * 2. 重新打乱分片顺序（如果启用） / Re-shuffle shard order (if enabled)
 * 3. 加载第一个分片 / Load first shard
 * 4. 重新生成分片内打乱索引（如果启用） / Regenerate intra-shard shuffle indices (if enabled)
 * 
 * @param loader 已初始化的DataLoader结构体指针 / Pointer to initialized DataLoader struct
 */
void dataloader_reset(DataLoader *loader) {
    // === 步骤1：重置索引到起始位置 ===
    // === Step 1: Reset indices to starting position ===
    loader->current_shard_idx = 0;   // 重置当前分片索引为0 / Reset current shard index to 0
    loader->current_sample_idx = 0;  // 重置当前样本索引为0 / Reset current sample index to 0

    // === 步骤2：重新打乱分片顺序（如果启用） ===
    // === Step 2: Re-shuffle shard order (if enabled) ===
    if (loader->should_shuffle) {
        // 对分片索引数组进行随机排列，改变分片的访问顺序
        // Randomly permute shard indices array to change shard access order
        // 这确保每个epoch都有不同的分片访问模式
        // This ensures each epoch has different shard access pattern
        random_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    // === 步骤3：加载第一个分片 ===
    // === Step 3: Load first shard ===
    // 根据当前分片索引加载对应的分片文件
    // Load corresponding shard file based on current shard index
    // 如果启用打乱，这将是随机排列后的第一个分片
    // If shuffling is enabled, this will be the first shard in the shuffled order
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    // === 步骤4：准备分片内随机索引（如果启用） ===
    // === Step 4: Prepare intra-shard random indices (if enabled) ===
    if (loader->should_shuffle) {
        // 为新加载的分片生成分片内样本的随机访问索引
        // Generate random access indices for samples within the newly loaded shard
        prepare_intra_shard_indices_(loader);
    }
}

/**
 * 将数据加载器推进到下一个分片
 * Advance the data loader to the next shard
 * 
 * 此函数在当前分片的数据用完时被调用，负责切换到下一个数据分片。
 * 如果已经处理完所有分片，它会自动开始新的epoch。这是数据加载器状态管理的核心函数。
 * This function is called when the current shard's data is exhausted, responsible for switching to the next data shard.
 * If all shards have been processed, it automatically starts a new epoch. This is a core function for data loader state management.
 * 
 * 推进策略：
 * Advance strategy:
 * - 如果不是最后一个分片：切换到下一个分片 / If not the last shard: switch to next shard
 * - 如果是最后一个分片：重置整个加载器，开始新epoch / If it's the last shard: reset entire loader, start new epoch
 * 
 * @param loader 已初始化的DataLoader结构体指针 / Pointer to initialized DataLoader struct
 */
void dataloader_advance_(DataLoader *loader) {
    // === 步骤1：检查是否到达最后一个分片 ===
    // === Step 1: Check if we've reached the last shard ===
    if (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
        // 如果当前是最后一个分片，说明一个完整的epoch已结束
        // If current is the last shard, it means a complete epoch has ended
        // 重置加载器到初始状态，开始新的epoch
        // Reset loader to initial state to start a new epoch
        // 这将重新打乱所有数据顺序（如果启用打乱）
        // This will re-shuffle all data order (if shuffling is enabled)
        dataloader_reset(loader);
        return;
    }

    // === 步骤2：推进到下一个分片 ===
    // === Step 2: Advance to next shard ===
    // 增加分片索引，移动到下一个分片
    // Increment shard index to move to next shard
    // 使用模运算确保索引不会越界（虽然在这里不是必需的，但保持一致性）
    // Use modulo operation to ensure index doesn't go out of bounds (not necessary here, but for consistency)
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    
    // 重置样本索引为0，从新分片的开头开始读取
    // Reset sample index to 0 to start reading from the beginning of new shard
    loader->current_sample_idx = 0;
    
    // === 步骤3：加载新的分片文件 ===
    // === Step 3: Load new shard file ===
    // 打开并加载新的分片文件，验证其格式和完整性
    // Open and load new shard file, validate its format and integrity
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    // === 步骤4：准备新分片的随机索引（如果启用） ===
    // === Step 4: Prepare random indices for new shard (if enabled) ===
    if (loader->should_shuffle) {
        // 为新加载的分片生成随机访问索引
        // Generate random access indices for the newly loaded shard
        // 这确保即使在同一个分片内，样本的访问顺序也是随机的
        // This ensures that even within the same shard, sample access order is randomized
        prepare_intra_shard_indices_(loader);
    }
}

/**
 * 初始化数据加载器，为分布式训练准备数据
 * Initialize the data loader for distributed training
 * 
 * 此函数设置分布式训练环境，发现数据分片文件，验证其有效性，
 * 并分配必要的内存空间。每个进程将处理数据的不同部分。
 * This function sets up the distributed training environment, discovers data shard files,
 * validates their integrity, and allocates necessary memory. Each process will handle different portions of the data.
 * 
 * @param loader 要初始化的DataLoader结构体指针 / Pointer to DataLoader struct to initialize
 * @param filename_pattern 数据分片文件的glob模式，如"data_*.bin" / Glob pattern for data shard files, e.g. "data_*.bin"
 * @param B 批次大小（序列数量） / Batch size (number of sequences)
 * @param T 序列长度（每个序列的token数量） / Sequence length (number of tokens per sequence)
 * @param process_rank 当前进程在分布式训练中的rank（0到num_processes-1） / Current process rank in distributed training (0 to num_processes-1)
 * @param num_processes 分布式训练的总进程数 / Total number of processes in distributed training
 * @param should_shuffle 是否启用数据打乱（1启用，0禁用） / Whether to enable data shuffling (1 to enable, 0 to disable)
 */
void dataloader_init(DataLoader *loader,
                     const char* filename_pattern,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes,
                     int should_shuffle) {
    // === 步骤1：基本配置初始化 ===
    // === Step 1: Basic configuration initialization ===
    loader->process_rank = process_rank;         // 设置当前进程rank / Set current process rank
    loader->num_processes = num_processes;       // 设置总进程数 / Set total number of processes
    loader->B = B;                               // 设置批次大小 / Set batch size
    loader->T = T;                               // 设置序列长度 / Set sequence length
    loader->tokens_file = NULL;                  // 初始化文件句柄为空 / Initialize file handle to NULL
    loader->should_shuffle = should_shuffle;     // 设置打乱标志 / Set shuffle flag
    
    // 计算内存布局相关的字节偏移量
    // Calculate memory layout related byte offsets
    loader->header_bytes = HEADER_SIZE * sizeof(int);  // 文件头部大小 / File header size
    // 所有进程的总批次大小（字节）= 进程数 × 批次大小 × 序列长度 × uint16_t大小
    // Total batch size across all processes (bytes) = num_processes × batch_size × seq_length × uint16_t_size
    loader->total_batch_size_bytes = ((loader->num_processes * (loader->B * loader->T)) * sizeof(uint16_t));
    // 当前进程的本地批次偏移量（字节）= 进程rank × 批次大小 × 序列长度 × uint16_t大小
    // Local batch offset for current process (bytes) = process_rank × batch_size × seq_length × uint16_t_size
    loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * sizeof(uint16_t);

    // === 步骤2：发现和验证数据分片文件 ===
    // === Step 2: Discover and validate data shard files ===
    // 使用glob函数匹配文件模式，获取所有数据分片文件的路径列表
    // Use glob function to match file pattern and get list of all data shard file paths
    int glob_status = glob(filename_pattern, 0, NULL, &loader->glob_result);
    if (glob_status != 0) {
        printf("Error: failed to glob pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    if (loader->glob_result.gl_pathc == 0) {
        printf("Error: no files found matching the pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }

    // === 步骤3：设置数据打乱功能（如果启用） ===
    // === Step 3: Setup data shuffling functionality (if enabled) ===
    if (should_shuffle) {
        // 初始化随机数生成器，每个进程使用不同的种子以确保不同的打乱顺序
        // Initialize RNG with different seed per process to ensure different shuffle orders
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);  // 基础种子42 + 进程rank / Base seed 42 + process rank
        loader->shuffle_rng = shuffle_rng;
        
        // 分配并初始化分片索引数组，用于分片级别的打乱
        // Allocate and initialize shard indices array for shard-level shuffling
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        init_identity_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc);
        
        // 分片内索引数组将在需要时动态分配，因为不同分片可能有不同大小
        // Intra-shard indices will be dynamically allocated as needed, since different shards may have different sizes
        loader->intra_shard_indices = NULL;
    }

    // === 步骤4：检查和验证所有数据分片 ===
    // === Step 4: Inspect and validate all data shards ===
    // 提前验证所有分片文件，避免运行时错误
    // Pre-validate all shard files to avoid runtime errors
    // 注意：如果分片太多或文件太大，这个步骤可能会很慢（将来可能需要优化）
    // Note: This step might be slow if there are too many shards or large files (may need optimization later)
    int64_t ntok_total = 0;
    for (int shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
        // 加载并验证每个分片，返回该分片中的token数量
        // Load and validate each shard, returns the number of tokens in that shard
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);
        
        // 验证每个分片至少包含一个完整的批次数据
        // Verify each shard contains at least one complete batch worth of data
        // 需要 num_processes * B * T + 1 个token（+1是因为我们读取B*T+1个token但只移动B*T个位置）
        // Need num_processes * B * T + 1 tokens (+1 because we read B*T+1 tokens but only advance B*T positions)
        assert(shard_ntok >= (int64_t) (num_processes * B * T + 1));
        ntok_total += shard_ntok;
    }
    // 调试输出（通常被注释掉） / Debug prints (usually commented out)
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // === 步骤5：分配内存缓冲区 ===
    // === Step 5: Allocate memory buffers ===
    // 分配读取缓冲区：B*T+1个uint16_t，+1是为了同时获取输入和目标token
    // Allocate read buffer: B*T+1 uint16_t tokens, +1 to get both input and target tokens simultaneously
    loader->buffer = (uint16_t*)mallocCheck((B * T + 1) * sizeof(uint16_t));
    // 分配输入token数组：B*T个int，存储输入到transformer的token序列
    // Allocate input tokens array: B*T ints, stores token sequences fed into transformer
    loader->inputs = (int*)mallocCheck(B * T * sizeof(int));
    // 分配目标token数组：B*T个int，存储transformer的预测目标（通常是输入向右偏移1位）
    // Allocate target tokens array: B*T ints, stores prediction targets for transformer (typically inputs shifted right by 1)
    loader->targets = (int*)mallocCheck(B * T * sizeof(int));
    // 记录所有分片的总token数量
    // Record total number of tokens across all shards
    loader->num_tokens = ntok_total;

    // === 步骤6：初始化加载器状态 ===
    // === Step 6: Initialize loader state ===
    // 重置加载器到初始状态，这将设置当前分片和样本索引，并加载第一个分片
    // Reset loader to initial state, this will set current shard and sample indices and load the first shard
    dataloader_reset(loader);
}

/**
 * 从当前数据分片中加载一个批次的数据
 * Load a batch of data from the current data shard
 * 
 * 此函数根据当前的样本索引，从文件中读取一个批次的token数据，
 * 并将其解码为输入和目标token序列，供transformer模型使用。
 * This function reads a batch of token data from the file based on the current sample index,
 * and decodes it into input and target token sequences for use by the transformer model.
 * 
 * 数据布局说明：
 * Data layout explanation:
 * - 文件格式：[头部][样本0][样本1]...[样本N]
 * - File format: [header][sample0][sample1]...[sampleN]
 * - 每个样本包含所有进程的数据：[进程0数据][进程1数据]...[进程N数据]
 * - Each sample contains data for all processes: [process0_data][process1_data]...[processN_data]
 * - 每个进程数据：B*T个连续的uint16_t token
 * - Each process data: B*T consecutive uint16_t tokens
 * 
 * @param loader 已初始化的DataLoader结构体指针 / Pointer to initialized DataLoader struct
 */
void dataloader_load_batch(DataLoader* loader) {
    // === 步骤1：安全性检查 ===
    // === Step 1: Safety checks ===
    // 如果启用了打乱，确保分片内索引数组已初始化
    // If shuffling is enabled, ensure intra-shard indices array is initialized
    assert(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != NULL));
    // 确保当前样本索引没有超出当前分片的样本数量
    // Ensure current sample index doesn't exceed the number of samples in current shard
    assert(loader->current_sample_idx < loader->shard_num_samples);
    
    // === 步骤2：确定要读取的样本索引 ===
    // === Step 2: Determine the sample index to read ===
    // 如果启用打乱，使用打乱后的索引；否则使用顺序索引
    // If shuffling is enabled, use shuffled index; otherwise use sequential index
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    
    // === 步骤3：计算文件中的字节偏移量 ===
    // === Step 3: Calculate byte offset in the file ===
    // 全局批次偏移量 = 样本索引 × 所有进程的总批次大小
    // Global batch offset = sample_index × total_batch_size_across_all_processes
    size_t global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
    
    // 当前读取位置 = 文件头部大小 + 全局批次偏移量 + 当前进程的本地偏移量
    // Current read position = file_header_size + global_batch_offset + local_process_offset
    // 这确保每个进程只读取属于自己的那部分数据
    // This ensures each process only reads its own portion of the data
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    // === 步骤4：从文件读取原始token数据 ===
    // === Step 4: Read raw token data from file ===
    size_t B = loader->B;  // 批次大小 / Batch size
    size_t T = loader->T;  // 序列长度 / Sequence length
    
    // 定位到计算出的文件偏移位置
    // Seek to the calculated file offset position
    fseekCheck(loader->tokens_file, (int) current_offset, SEEK_SET);
    
    // 读取B*T+1个uint16_t token到缓冲区
    // Read B*T+1 uint16_t tokens into buffer
    // +1是为了同时获得输入token和下一个token（作为目标）
    // +1 to get both input tokens and the next token (as target)
    freadCheck(loader->buffer, sizeof(uint16_t), B*T+1, loader->tokens_file);
    
    // === 步骤5：解码token数据 ===
    // === Step 5: Decode token data ===
    // 将uint16_t格式的token转换为int格式，并创建输入-目标对
    // Convert uint16_t tokens to int format and create input-target pairs
    for (int i = 0; i < B*T; i++) {
        // 输入序列：buffer[0], buffer[1], ..., buffer[B*T-1]
        // Input sequence: buffer[0], buffer[1], ..., buffer[B*T-1]
        loader->inputs[i] = (int)loader->buffer[i];
        
        // 目标序列：buffer[1], buffer[2], ..., buffer[B*T]
        // Target sequence: buffer[1], buffer[2], ..., buffer[B*T]
        // 目标序列是输入序列向右偏移一位，用于next-token预测任务
        // Target sequence is input sequence shifted right by 1, for next-token prediction task
        loader->targets[i] = (int)loader->buffer[i+1];
    }
}

/**
 * 移动到下一个批次并加载数据
 * Move to the next batch and load data
 * 
 * 此函数是数据加载器的主要接口，用于在训练循环中获取下一个批次的数据。
 * 它会自动处理分片边界，在当前分片用完时切换到下一个分片。
 * This function is the main interface of the data loader, used to get the next batch of data
 * in the training loop. It automatically handles shard boundaries and switches to the next shard
 * when the current shard is exhausted.
 * 
 * 工作流程：
 * Workflow:
 * 1. 检查是否需要切换到下一个分片
 * 1. Check if we need to switch to the next shard
 * 2. 加载当前位置的批次数据
 * 2. Load batch data at current position
 * 3. 更新样本索引为下次调用做准备
 * 3. Update sample index for next call
 * 
 * @param loader 已初始化的DataLoader结构体指针 / Pointer to initialized DataLoader struct
 */
void dataloader_next_batch(DataLoader *loader) {
    // === 步骤1：检查分片边界 ===
    // === Step 1: Check shard boundary ===
    // 如果下一个批次会超出当前分片的末尾，需要切换分片
    // If the next batch would go past the end of the current shard, need to switch shards
    if (loader->current_sample_idx >= loader->shard_num_samples) {
        // 调用dataloader_advance_()来：
        // Call dataloader_advance_() to:
        // - 切换到下一个分片（如果是最后一个分片则回到第一个分片，开始新的epoch）
        // - Switch to next shard (if last shard, go back to first shard to start new epoch)
        // - 重置样本索引为0
        // - Reset sample index to 0
        // - 如果启用打乱，重新生成分片内的随机索引
        // - If shuffling enabled, regenerate intra-shard random indices
        dataloader_advance_(loader);
    }
    
    // === 步骤2：加载当前批次数据 ===
    // === Step 2: Load current batch data ===
    // 从当前分片的当前样本位置读取一个批次的数据
    // Read one batch of data from current sample position in current shard
    dataloader_load_batch(loader);
    
    // === 步骤3：准备下次调用 ===
    // === Step 3: Prepare for next call ===
    // 增加样本索引，指向下一个要读取的样本
    // Increment sample index to point to next sample to be read
    // 注意：这里递增后，下次调用时可能会触发分片切换（在步骤1中检查）
    // Note: After incrementing, next call may trigger shard switching (checked in step 1)
    loader->current_sample_idx += 1;
}


/**
 * 恢复数据加载器到指定的训练位置
 * Resume the data loader to a specific training position
 * 
 * 此函数用于模型恢复时（-y 1 标志），将数据加载器恢复到之前保存的训练位置。
 * 这确保训练可以从上次中断的确切位置继续，而不会重复或跳过数据。
 * This function is used during model resumption (-y 1 flag) to restore the data loader
 * to a previously saved training position. This ensures training can continue from the exact
 * position where it was interrupted, without duplicating or skipping data.
 * 
 * 恢复过程：
 * Resumption process:
 * 1. 恢复分片索引和样本索引 / Restore shard index and sample index
 * 2. 加载指定的分片文件 / Load the specified shard file
 * 
 * @param loader 已初始化的DataLoader结构体指针 / Pointer to initialized DataLoader struct
 * @param current_shard_idx 要恢复到的分片索引 / Shard index to resume to
 * @param current_sample_idx 要恢复到的样本索引 / Sample index to resume to
 */
void dataloader_resume(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // === 恢复训练位置状态 ===
    // === Restore training position state ===
    // 恢复分片索引到保存的位置
    // Restore shard index to saved position
    loader->current_shard_idx = current_shard_idx;
    
    // 恢复样本索引到保存的位置
    // Restore sample index to saved position
    loader->current_sample_idx = current_sample_idx;
    
    // 加载指定的分片文件，使数据加载器准备好从恢复位置继续
    // Load the specified shard file to prepare the data loader for continuing from resume position
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);
}

/**
 * 释放数据加载器占用的所有资源
 * Free all resources occupied by the data loader
 * 
 * 此函数负责清理数据加载器使用的所有内存和文件资源。
 * 在训练完成或程序退出时必须调用此函数，以防止内存泄漏。
 * This function is responsible for cleaning up all memory and file resources used by the data loader.
 * It must be called when training is complete or the program exits to prevent memory leaks.
 * 
 * 清理的资源包括：
 * Resources to be cleaned up include:
 * - 动态分配的内存缓冲区 / Dynamically allocated memory buffers
 * - 打乱功能相关的索引数组 / Index arrays related to shuffling functionality
 * - 打开的文件句柄 / Open file handles
 * - glob匹配结果 / Glob matching results
 * 
 * @param loader 要释放资源的DataLoader结构体指针 / Pointer to DataLoader struct whose resources to free
 */
void dataloader_free(DataLoader *loader) {
    // === 释放内存缓冲区 ===
    // === Free memory buffers ===
    free(loader->buffer);    // 释放原始数据读取缓冲区 / Free raw data read buffer
    free(loader->inputs);    // 释放输入token数组 / Free input tokens array
    free(loader->targets);   // 释放目标token数组 / Free target tokens array
    
    // === 释放打乱功能相关资源（如果启用） ===
    // === Free shuffle-related resources (if enabled) ===
    if (loader->should_shuffle) {
        free(loader->shard_indices);       // 释放分片索引数组 / Free shard indices array
        free(loader->intra_shard_indices); // 释放分片内索引数组 / Free intra-shard indices array
    }
    
    // === 关闭文件和释放文件系统资源 ===
    // === Close files and free filesystem resources ===
    fcloseCheck(loader->tokens_file);  // 安全关闭当前打开的token文件 / Safely close currently opened tokens file
    globfree(&loader->glob_result);    // 释放glob模式匹配的结果 / Free glob pattern matching results
}

// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Many evals (like) HellaSwag and MMLU are multiple-choice
// where there are 4 possible continuations and a label for the correct one
// We want to load and serve these style of evals
/*
Copy pasting the section on the eval datafile format, from data_common.py:
- First comes a header with 256 int32s
- The examples follow, each example is a stream of uint16_t:
    - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
    - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
    - <EXAMPLE_INDEX>, the index of the example in the dataset
    - <LABEL>, the index of the correct completion
    - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
    - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
    - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
*/

// for now, could relax later
#define ASSUMED_NUM_COMPLETIONS 4
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

typedef struct {
    // === Distributed Evaluation Configuration ===
    // 分布式评估中当前进程的rank标识符 (0, 1, 2, ...)
    // Rank identifier for this process in distributed evaluation (0, 1, 2, ...)
    int process_rank;
    
    // 分布式评估中的总进程数
    // Total number of processes participating in distributed evaluation
    int num_processes;
    
    // === Model Hyperparameters ===
    // 批次大小：输入模型张量的（微）批次维度，使用size_t防止溢出
    // Batch size: (micro) batch size dimension of the tensor that feeds into the model, using size_t to prevent overflow
    size_t B;
    
    // 模型的最大上下文长度
    // Maximum context length of the model
    size_t T;
    
    // === File I/O ===
    // 评估数据文件句柄
    // Evaluation data file handle
    FILE* eval_file;
    
    // 从文件读取数据的缓冲区
    // Buffer for reading data from file
    uint16_t* buffer;
    
    // === Dataset Information (Public Access) ===
    // 所有进程中的总示例数量
    // Total number of examples across all processes
    int num_examples;
    
    // 处理整个数据集所需的批次数（跨所有进程）
    // Number of batches needed to process the entire dataset across all processes
    int num_batches;
    
    // 当前进程工作分配的起始示例索引（包含）
    // Starting example index for this process's work assignment (inclusive)
    int start_example_index;
    
    // 当前进程工作分配的结束示例索引（不包含）
    // Ending example index for this process's work assignment (exclusive)
    int end_example_index;
    
    // 下一个要读取的示例索引
    // Index of the next example to be read
    int current_example_index;
    
    // === Model Input/Output Tensors ===
    // 输入到transformer的token序列
    // Input token sequences fed into the transformer
    int* inputs;
    
    // transformer的目标token序列
    // Target token sequences for the transformer
    int* targets;
    
    // 掩码数组：在所有完成token位置处mask=1，用于计算损失
    // Mask array: mask=1 at all completion token locations, used for loss computation
    char* mask;
    
    // 正确完成选项的标签数组
    // Array of correct completion labels
    int* label;
    
    // 当前示例的完成选项数量（通常为4）
    // Number of completions for the current example (typically 4)
    int num_completions;
} EvalLoader;

void evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    int examples_per_process = CEIL_DIV(loader->num_examples, loader->num_processes);
    int can_fit_examples = (int) (loader->B / ASSUMED_NUM_COMPLETIONS);
    if (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        printf("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_NUM_COMPLETIONS);
        printf("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        exit(EXIT_FAILURE);
    }
    loader->num_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_index = examples_per_process * loader->process_rank;
    loader->end_example_index = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    if (loader->end_example_index > loader->num_examples) {
        loader->end_example_index = loader->num_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_SIZE * sizeof(int);
    fseekCheck(loader->eval_file, (int) header_bytes, SEEK_SET);
    for (int i = 0; i < loader->start_example_index; i++) {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], sizeof(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        assert(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        assert(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - sizeof(uint16_t) * 3;
        assert(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (int) remaining_bytes, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_index = loader->start_example_index;
}

void evalloader_init(EvalLoader *loader,
                     const char* filename,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes) {
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->eval_file);
    if (header[0] != 20240522) { printf("Bad magic in eval file\n"); exit(EXIT_FAILURE); }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    loader->num_examples = header[2]; // number of examples in the file
    assert(loader->num_examples >= num_processes); // avoid headaches for now
    size_t longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    assert(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_NUM_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
    loader->buffer = (uint16_t*)mallocCheck(longest_example_bytes);
    loader->inputs = (int*)calloc(B * T, sizeof(int));
    loader->targets = (int*)calloc(B * T, sizeof(int));
    loader->mask = (char*)mallocCheck(B * T * sizeof(char));
    loader->label = (int*)mallocCheck(can_fit_examples * sizeof(int));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

void evalloader_next_example_(EvalLoader *loader, int example_batch_index) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    int batch_dim_offset = example_batch_index * ASSUMED_NUM_COMPLETIONS;
    // read the current example header
    uint16_t example_header[3];
    freadCheck(&example_header[0], sizeof(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    assert(example_header[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    assert(example_header[2] == loader->current_example_index); // <EXAMPLE_INDEX> should match the loop index
    assert(example_header[2] >= loader->start_example_index && example_header[2] < loader->end_example_index);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_bytes = example_header[1] - sizeof(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->buffer, sizeof(char), example_bytes, loader->eval_file);
    // process the example label
    int label = (int)loader->buffer[0];
    int can_fit_examples = (int) (loader->B / ASSUMED_NUM_COMPLETIONS);
    assert(label >= 0 && label < ASSUMED_NUM_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    assert(example_batch_index >= 0 && example_batch_index < can_fit_examples);
    loader->label[example_batch_index] = label; // store for output
    // process the number of completions
    int num_completions = (int)loader->buffer[1];
    assert(num_completions == ASSUMED_NUM_COMPLETIONS); // we expect 4 completions for now
    assert(batch_dim_offset + num_completions <= B); // we expect to fit in the batch
    loader->num_completions = num_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    int context_length = (int)loader->buffer[2];
    uint16_t *context_tokens_start = &loader->buffer[3]; // where the tokens start
    assert(context_length > 0 && context_length < T); // context is non-empty and up to T
    for (int b = 0; b < num_completions; b++) {    // 遍历4个候选答案
        for (int i = 0; i < context_length; i++) {   // 遍历上下文的每个token
            int boff = batch_dim_offset + b;          // 计算当前答案在batch中的行索引
            int tok_cur = (int)context_tokens_start[i];  // 获取上下文的第i个token
            loader->inputs[boff * T + i] = tok_cur;  // 将上下文的token插入到inputs中
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->buffer + 3 + context_length; // 答案tokens起始位置
    for (int c = 0; c < num_completions; c++) {
        int coff = batch_dim_offset + c;
        int completion_length = (int)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        assert(completion_length > 0 && context_length + completion_length < T); // things fit?
        for (int i = 0; i < completion_length; i++) {
            int tok_cur = (int)completion_tokens_start[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_length + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_length + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->mask[coff * T + context_length + i - 1] = 1;
        }

//         completions_iter[0]      -> completion_length (该答案的长度)  ← 占1个uint16_t
//         completions_iter[1...]   -> completion_tokens (答案的tokens)  ← 占completion_length个uint16_t
        completions_iter += 1 + completion_length; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_index += 1;
}

void evalloader_next_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->mask, 0, B * T * sizeof(char));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS); // how many examples can we fit in the batch?
    for (int i = 0; i < can_fit_examples; i++) {
        if (loader->current_example_index >= loader->end_example_index) {
            break; // this process has exhausted its work, noop from here on
        }
        evalloader_next_example_(loader, i);
    }
}

int evalloader_stat_losses(EvalLoader *loader, float* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    int correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
    for (int i = 0; i < can_fit_examples; i++) {
        float min_loss = 0.0f;
        int min_loss_index = -1;
        char active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        for (int b = 0; b < ASSUMED_NUM_COMPLETIONS; b++) {
            int boff = i * ASSUMED_NUM_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            float average_loss = 0.0f;
            int count = 0;
            for (int t = 0; t < T; t++) {
                char mask = loader->mask[boff * T + t];
                if (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    count++;
                }
            }
            if (count > 0) { average_loss /= count; }
            if (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_index = b;
            }
        }
        if (active && (min_loss_index == loader->label[i])) {
            correct += 1;
        }
    }
    return correct;
}

void evalloader_free(EvalLoader *loader) {
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    free(loader->mask);
    free(loader->label);
    fcloseCheck(loader->eval_file);
}

#endif // DATALOADER_H