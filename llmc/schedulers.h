/*
 * 学习率调度器 (Learning Rate Schedulers)
 * =========================================
 * 
 * 功能说明:
 *   - 实现深度学习训练中常用的学习率调度策略
 *   - 支持多种调度方式: cosine, linear, constant, wsd
 * 
 * 背景知识:
 *   学习率 (Learning Rate, LR) 是深度学习中最重要的超参数之一
 *   - 太大: 训练不稳定，loss 震荡甚至发散
 *   - 太小: 收敛速度慢，容易陷入局部最优
 *   - 动态调整: 初期用较大 LR 快速收敛，后期用较小 LR 精细调整
 * 
 * 常见调度策略:
 *   1. Warmup (预热): 从小 LR 线性增加到最大 LR，避免初期梯度不稳定
 *   2. Cosine Decay: 余弦曲线衰减，平滑且效果好
 *   3. Linear Decay: 线性衰减，简单直观
 *   4. WSD (Warmup-Stable-Decay): 三阶段策略，来自最新研究
 * 
 * 典型学习率曲线示意 (以 cosine 为例):
 * 
 *   LR
 *    ^
 *    |     warmup      cosine decay
 *    |       /\
 *    |      /  \___
 *    |     /       \___
 *    |    /            \___
 *    |   /                 \___
 *    |  /                      \___
 *    +--+--------------------------> step
 *       0  warmup_iters        total_steps
 */
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

/* ==================== 标准库头文件 ==================== */
#include <assert.h>   // assert() 断言宏
#include <math.h>     // cosf(), sqrtf(), M_PI 数学函数
#include <string.h>   // strcmp() 字符串比较

/* ============================================================================
 * 学习率调度器数据结构
 * ============================================================================ */

/**
 * @struct LearningRateScheduler
 * @brief  学习率调度器结构体，封装调度策略的所有参数
 * 
 * 参数说明:
 *   - type: 调度器类型，支持 "cosine", "linear", "constant", "wsd"
 *   - learning_rate: 最大学习率 (峰值 LR)
 *   - warmup_iterations: 预热步数，从 0 线性增长到 learning_rate
 *   - train_num_batches: 总训练步数
 *   - final_learning_rate_frac: 最终 LR 占最大 LR 的比例 (如 0.1 表示衰减到 10%)
 */
typedef struct {
    const char* type;              // 调度器类型: "cosine", "linear", "constant", "wsd"
    float learning_rate;           // 最大学习率 (max_lr)
    int warmup_iterations;         // 预热步数
    int train_num_batches;         // 总训练步数
    float final_learning_rate_frac; // 最终学习率 = learning_rate * final_learning_rate_frac
} LearningRateScheduler;

/* ============================================================================
 * 初始化函数
 * ============================================================================ */

/**
 * @brief  初始化学习率调度器
 * 
 * @param scheduler            输出参数，指向要初始化的调度器结构体
 * @param scheduler_type       调度器类型字符串
 * @param learning_rate        最大学习率
 * @param warmup_iterations    预热步数
 * @param train_num_batches    总训练步数
 * @param final_learning_rate_frac  最终学习率比例
 * 
 * 使用示例:
 *   LearningRateScheduler scheduler;
 *   lr_scheduler_init(&scheduler, "cosine", 3e-4f, 700, 10000, 0.1f);
 *   // 最大 LR = 3e-4, 预热 700 步, 共 10000 步, 最终衰减到 3e-5
 */
void lr_scheduler_init(LearningRateScheduler *scheduler, const char* scheduler_type, 
                       float learning_rate, int warmup_iterations, 
                       int train_num_batches, float final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_num_batches = train_num_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

/* ============================================================================
 * 具体调度策略实现
 * ============================================================================ */

/**
 * @brief  余弦退火调度器 (Cosine Annealing)
 * 
 * @param scheduler  调度器指针
 * @param step       当前训练步数 (0-indexed)
 * @return           当前步的学习率
 * 
 * 调度曲线:
 *   LR
 *    ^
 *    |        max_lr
 *    |       /‾‾‾‾\
 *    |      /      ╲
 *    |     /        ╲
 *    |    /          ╲___min_lr
 *    +---+----------------> step
 *        0   warmup   total
 * 
 * 数学公式:
 *   阶段1 - 预热 (step < warmup_iterations):
 *     lr = max_lr * (step + 1) / warmup_iterations
 *   
 *   阶段2 - 余弦衰减 (step >= warmup_iterations):
 *     decay_ratio = (step - warmup) / (total - warmup)  ∈ [0, 1]
 *     coeff = 0.5 * (1 + cos(π * decay_ratio))          ∈ [1, 0]
 *     lr = min_lr + coeff * (max_lr - min_lr)
 * 
 * 特点:
 *   - 衰减初期慢，中期快，末期又变慢
 *   - 平滑的曲线有助于训练稳定性
 *   - 是目前最流行的调度策略之一 (GPT, LLaMA 等都使用)
 */
float get_learning_rate_cosine(LearningRateScheduler *scheduler, int step) {
    float lr = scheduler->learning_rate;
    
    if (step < scheduler->warmup_iterations) {
        // ========== 阶段1: 线性预热 ==========
        // 从 0 线性增长到 max_lr
        // 使用 (step + 1) 是为了避免 step=0 时 lr=0
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } else {
        // ========== 阶段2: 余弦衰减 ==========
        // 计算衰减进度 (0 到 1)
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / 
                           (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        
        // 余弦系数: cos(0) = 1, cos(π) = -1
        // (1 + cos(π * ratio)) / 2 将其映射到 [1, 0]
        float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio));
        assert(0.0f <= coeff && coeff <= 1.0f);
        
        // 计算最终学习率，在 min_lr 和 max_lr 之间插值
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    return lr;
}

/**
 * @brief  线性衰减调度器 (Linear Decay)
 * 
 * @param scheduler  调度器指针
 * @param step       当前训练步数
 * @return           当前步的学习率
 * 
 * 调度曲线:
 *   LR
 *    ^
 *    |       max_lr
 *    |      /\
 *    |     /  \
 *    |    /    \
 *    |   /      \___min_lr
 *    +--+-----------→ step
 *       0  warmup  total
 * 
 * 数学公式:
 *   阶段1 - 预热: lr = max_lr * (step + 1) / warmup_iterations
 *   阶段2 - 线性衰减: lr = max_lr - decay_ratio * (max_lr - min_lr)
 * 
 * 特点:
 *   - 简单直观，易于理解和调试
 *   - 衰减速度恒定
 */
float get_learning_rate_linear(LearningRateScheduler *scheduler, int step) {
    float lr = scheduler->learning_rate;
    
    if (step < scheduler->warmup_iterations) {
        // 阶段1: 线性预热
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } else {
        // 阶段2: 线性衰减
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / 
                           (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        // 线性插值: 从 max_lr 线性降到 min_lr
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    return lr;
}

/**
 * @brief  恒定学习率调度器 (Constant)
 * 
 * @param scheduler  调度器指针
 * @param step       当前训练步数 (未使用)
 * @return           恒定的学习率
 * 
 * 调度曲线:
 *   LR
 *    ^
 *    |  ─────────────────  max_lr
 *    |
 *    +-------------------→ step
 * 
 * 特点:
 *   - 最简单的策略，LR 保持不变
 *   - 适用于短期训练或调试
 *   - 通常不适合大规模训练
 */
float get_learning_rate_constant(LearningRateScheduler *scheduler, int step) {
    (void)step;  // 显式标记未使用的参数
    return scheduler->learning_rate;
}

/**
 * @brief  WSD 调度器 (Warmup-Stable-Decay)
 * 
 * @param scheduler  调度器指针
 * @param step       当前训练步数
 * @return           当前步的学习率
 * 
 * 论文来源: https://arxiv.org/abs/2405.18392
 * 
 * 调度曲线 (三阶段):
 *   LR
 *    ^
 *    |      _______________  max_lr (80% 时间保持恒定)
 *    |     /               \
 *    |    /                 \
 *    |   /                   \___ min_lr
 *    +--+--------------------+-→ step
 *      warmup           80%  total
 *       ↑        ↑            ↑
 *      阶段1    阶段2        阶段3
 * 
 * 三个阶段:
 *   阶段1 - Warmup (预热): 线性增长到 max_lr
 *   阶段2 - Stable (稳定): 保持 max_lr 不变 (约 80% 的训练时间)
 *   阶段3 - Decay (衰减): 使用 (1 - sqrt) 曲线快速衰减到 min_lr
 * 
 * 衰减公式 (阶段3):
 *   lr = min_lr + (1 - sqrt(decay_ratio)) * (max_lr - min_lr)
 *   - sqrt 衰减比 cosine 更激进，末期下降更快
 *   - 适合需要快速收敛的场景
 * 
 * 特点:
 *   - 长时间保持高学习率，加速训练
 *   - 最后 20% 快速衰减，收敛效果好
 *   - 建议 final_learning_rate_frac = 0.0
 */
float get_learning_rate_wsd(LearningRateScheduler *scheduler, int step) {
    // 衰减起始点: 80% 的总训练步数
    int decay_point = (int)(0.8f * scheduler->train_num_batches);
    float max_lr = scheduler->learning_rate;
    float lr = max_lr;
    
    if (step < scheduler->warmup_iterations) {
        // ========== 阶段1: 线性预热 ==========
        float decay_ratio = ((float)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } else if (step < decay_point) {
        // ========== 阶段2: 保持恒定 ==========
        // 什么都不做，lr 保持 max_lr
    } else {
        // ========== 阶段3: sqrt 衰减 ==========
        float decay_ratio = ((float)(step - decay_point)) / 
                           (scheduler->train_num_batches - decay_point);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        
        float min_lr = max_lr * scheduler->final_learning_rate_frac;
        // (1 - sqrt(x)) 曲线: 初期快速下降，后期趋于平缓
        return min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    return lr;
}

/* ============================================================================
 * 统一接口
 * ============================================================================ */

/**
 * @brief  获取指定步数的学习率 (统一入口)
 * 
 * @param scheduler  调度器指针
 * @param step       当前训练步数
 * @return           当前步的学习率
 * 
 * 使用示例:
 *   for (int step = 0; step < total_steps; step++) {
 *       float lr = get_learning_rate(&scheduler, step);
 *       // 使用 lr 进行参数更新...
 *   }
 * 
 * 注意:
 *   - 根据 scheduler->type 自动选择对应的调度策略
 *   - 不支持的类型会导致程序退出
 */
float get_learning_rate(LearningRateScheduler *scheduler, int step) {
    float step_learning_rate;
    
    // 根据类型字符串分发到具体实现
    if (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } else if (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } else if (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } else if (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } else {
        // 未知类型，报错退出
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    
    return step_learning_rate;
}

#endif // SCHEDULERS_H
