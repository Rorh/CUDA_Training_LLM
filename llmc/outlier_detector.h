/**
 * @file outlier_detector.h
 * @brief 异常值检测器 (Outlier Detector) - 用于监控训练过程中的 loss 和梯度范数
 *
 * 本模块实现了一个基于滑动窗口的 Z-score 异常值检测器。
 * 主要用途:
 * - 监控训练 loss 的异常波动 (如 loss spike)
 * - 检测梯度范数的异常值 (可能预示梯度爆炸)
 * - 帮助识别训练过程中的不稳定性
 *
 * 工作原理:
 * - 维护一个固定大小的滑动窗口，存储最近的测量值
 * - 每次添加新测量值时，计算该值相对于窗口内数据的 Z-score
 * - Z-score 表示新值偏离均值多少个标准差，可用于判断是否为异常值
 *
 * Z-score 解释:
 * - |Z| < 2: 正常范围内的波动
 * - |Z| ≥ 2: 可能的异常值 (约 5% 的数据会落在此范围)
 * - |Z| ≥ 3: 极端异常值 (约 0.3% 的数据会落在此范围)
 *
 * @note 使用 double 类型以减少累积误差漂移，因为均值和方差通过增量 (+= / -=) 更新
 */

#ifndef OUTLIER_DETECTOR_H
#define OUTLIER_DETECTOR_H

/*
 * ============================================================================
 * 标准库头文件
 * ============================================================================
 */
#include <stdio.h>   // 标准 I/O (未直接使用，但保留以便调试)
#include <math.h>    // sqrt(), nan() 数学函数

/*
 * ============================================================================
 * 配置常量
 * ============================================================================
 */

/**
 * @brief 滑动窗口大小 (编译时常量)
 *
 * 使用编译时常量而非动态分配，原因:
 * - 避免动态内存分配的开销和复杂性
 * - 窗口大小固定，无需运行时调整
 * - 128 个样本足以提供稳定的统计估计
 *
 * @note 窗口越大，统计估计越稳定，但对突变的响应越慢
 *       窗口越小，响应越快，但可能产生更多误报
 */
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

/*
 * ============================================================================
 * 数据结构定义
 * ============================================================================
 */

/**
 * @struct OutlierDetector
 * @brief 异常值检测器结构体
 *
 * 实现了一个环形缓冲区 (Ring Buffer) 来存储滑动窗口内的数据。
 * 使用增量更新的方式计算均值和方差，避免每次都遍历整个窗口。
 *
 * 内存布局:
 * - buffer: 环形缓冲区，存储最近 WINDOW_SIZE 个测量值
 * - count:  当前已收集的样本数 (0 ~ WINDOW_SIZE)
 * - index:  下一个写入位置 (环形索引)
 * - sum:    窗口内所有值的和 (用于计算均值)
 * - sum_sq: 窗口内所有值平方的和 (用于计算方差)
 *
 * 统计量计算:
 * - 均值 (mean) = sum / n
 * - 方差 (variance) = sum_sq / n - mean²  (利用 E[X²] - E[X]² 公式)
 * - 标准差 (std_dev) = sqrt(variance)
 * - Z-score = (x - mean) / std_dev
 */
typedef struct {
    double buffer[OUTLIER_DETECTOR_WINDOW_SIZE];  // 环形缓冲区，存储历史测量值
    int count;                                     // 已收集的样本数量
    int index;                                     // 环形缓冲区的当前写入位置
    double sum;                                    // 窗口内所有值的累加和
    double sum_sq;                                 // 窗口内所有值平方的累加和
} OutlierDetector;

/*
 * ============================================================================
 * 函数实现
 * ============================================================================
 */

/**
 * @brief 初始化异常值检测器
 *
 * 将检测器重置为初始状态，清空所有历史数据。
 * 必须在首次使用检测器前调用此函数。
 *
 * @param detector 指向要初始化的检测器结构体的指针
 *
 * @note 初始化后，检测器处于"预热"状态，
 *       需要收集 WINDOW_SIZE 个样本后才能开始正常检测
 *
 * @example
 *       OutlierDetector loss_detector;
 *       init_detector(&loss_detector);
 */
void init_detector(OutlierDetector *detector) {
    // 清零环形缓冲区
    for (int i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    // 重置状态变量
    detector->count = 0;      // 尚未收集任何样本
    detector->index = 0;      // 从位置 0 开始写入
    detector->sum = 0.0;      // 累加和清零
    detector->sum_sq = 0.0;   // 平方累加和清零
}

/**
 * @brief 更新检测器并返回新值的 Z-score
 *
 * 向检测器添加一个新的测量值，并计算该值相对于历史数据的 Z-score。
 * Z-score 表示新值偏离均值多少个标准差，是判断异常值的核心指标。
 *
 * @param detector  指向检测器结构体的指针
 * @param new_value 新的测量值 (如当前 step 的 loss 或梯度范数)
 *
 * @return Z-score 值，含义如下:
 *         - NaN: 检测器仍在预热阶段 (样本数 < WINDOW_SIZE)
 *         - 0.0: 标准差为 0 (所有样本值相同，无法计算 Z-score)
 *         - 其他: 新值的 Z-score，正值表示高于均值，负值表示低于均值
 *
 * @note 算法复杂度: O(1) - 使用增量更新，无需遍历整个窗口
 *
 * @note 预热阶段说明:
 *       在收集到 WINDOW_SIZE 个样本之前，统计量不够稳定，
 *       返回 NaN 表示"数据不足，暂不进行异常检测"
 *
 * @example
 *       double z = update_detector(&loss_detector, current_loss);
 *       if (!isnan(z) && fabs(z) > 3.0) {
 *           printf("Warning: Loss spike detected! Z-score: %.2f\n", z);
 *       }
 */
double update_detector(OutlierDetector *detector, double new_value) {

    if (detector->count < OUTLIER_DETECTOR_WINDOW_SIZE) {
        /*
         * ==================== 预热阶段 ====================
         * 窗口尚未填满，仅累积数据，不进行异常检测
         */
        detector->buffer[detector->count] = new_value;  // 存入缓冲区
        detector->sum += new_value;                      // 更新累加和
        detector->sum_sq += new_value * new_value;       // 更新平方累加和
        detector->count++;                               // 增加样本计数
        return nan("");  // 返回 NaN 表示数据不足

    } else {
        /*
         * ==================== 正常检测阶段 ====================
         * 窗口已满，使用滑动窗口进行异常检测
         * 采用 O(1) 的增量更新算法，而非 O(n) 的重新计算
         */

        // 步骤 1: 移除最旧的值 (FIFO 队列的出队操作)
        double old_value = detector->buffer[detector->index];
        detector->sum -= old_value;                      // 从累加和中减去旧值
        detector->sum_sq -= old_value * old_value;       // 从平方累加和中减去旧值的平方

        // 步骤 2: 添加新值 (FIFO 队列的入队操作)
        detector->buffer[detector->index] = new_value;   // 覆盖旧值
        detector->sum += new_value;                      // 加入新值
        detector->sum_sq += new_value * new_value;       // 加入新值的平方

        // 步骤 3: 更新环形缓冲区索引 (取模实现环形)
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;

        // 步骤 4: 计算统计量
        // 均值: μ = Σx / n
        double mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;

        // 方差: σ² = E[X²] - E[X]² = (Σx²/n) - μ²
        // 这是方差的计算公式的一种高效形式，避免两次遍历
        double variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);

        // 标准差: σ = √σ²
        double std_dev = sqrt(variance);

        // 步骤 5: 计算 Z-score
        // Z = (x - μ) / σ
        // 特殊情况: 如果标准差为 0 (所有值相同)，返回 0 避免除零错误
        if (std_dev == 0.0) {
            return 0.0;
        }
        double z = (new_value - mean) / std_dev;

        return z;
    }
}

#endif // OUTLIER_DETECTOR_H
