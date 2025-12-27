/*
 * GPT-2 Tokenizer 定义
 * =====================
 * 
 * 功能说明：
 *   - 定义了 GPT-2 模型的分词器（Tokenizer）
 *   - 仅支持解码功能，即：tokens (整数) -> strings (字符串)
 *   - 这对于无条件文本生成已经足够
 * 
 * 设计说明：
 *   - 如果后续需要支持提示词（prompt）输入模型，则需要添加编码功能
 *   - 编码功能在 C 语言中实现较为复杂，因为涉及正则表达式处理
 * 
 * 文件格式（.bin 二进制文件）：
 *   - Header: 256 个 uint32_t
 *     - header[0]: 魔数 20240328（用于验证文件格式）
 *     - header[1]: 版本号（1 或 2）
 *     - header[2]: 词汇表大小
 *     - header[3]: EOT token ID（仅版本 2）
 *   - Body: 逐个 token 的字节序列
 *     - 每个 token: 1 字节长度 + N 字节内容
 */

/* ==================== 标准库头文件 ==================== */
#include <stdint.h>   // uint32_t 等固定宽度整数类型
#include <ctype.h>    // isprint(), isspace() 字符分类函数
#include <assert.h>   // assert() 断言宏

/* ==================== 自定义工具库 ==================== */
// 提供带错误检查的文件操作和内存分配函数：
//   - fopenCheck:  带错误检查的 fopen
//   - freadCheck:  带错误检查的 fread
//   - fcloseCheck: 带错误检查的 fclose
//   - fseekCheck:  带错误检查的 fseek
//   - mallocCheck: 带错误检查的 malloc
#include "utils.h"

/* ============================================================================
 * Tokenizer 数据结构
 * ============================================================================ */

/**
 * @struct Tokenizer
 * @brief  GPT-2 分词器结构体，用于将 token ID 解码为字符串
 * 
 * 内存布局示意:
 *   Tokenizer
 *   ├── vocab_size: 50257 (GPT-2 标准词汇表大小)
 *   ├── token_table ──→ [char*] [char*] [char*] ... (vocab_size 个指针)
 *   │                      │       │       │
 *   │                      ↓       ↓       ↓
 *   │                    "!"    "\""   "#"  ... (各 token 字符串)
 *   ├── init_ok: 1 (初始化成功标志)
 *   └── eot_token: 50256 (结束符 ID)
 */
typedef struct {
    uint32_t vocab_size;   // 词汇表大小 (GPT-2: 50257)
    char **token_table;    // token 查找表，token_table[id] = token 字符串
    int init_ok;           // 初始化状态: 1=成功, 0=失败
    int eot_token;         // <|endoftext|> token id
} Tokenizer;

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

/**
 * @brief  安全打印 token 字符串，过滤不可打印字符
 * 
 * @param piece  要打印的 token 字符串 (可能包含原始字节)
 * 
 * 设计说明:
 *   - BPE tokenizer 的 token 可能是原始字节 (0x00-0xFF)
 *   - 某些字节是控制字符 (如退格、响铃等)，打印会导致终端异常
 *   - 此函数只打印可打印字符 (isprint) 或空白字符 (isspace)
 * 
 * 处理逻辑:
 *   1. NULL 或空字符串 → 直接返回
 *   2. 单字节 token → 检查是否可打印
 *   3. 多字节 token → 直接打印 (假设是有效 UTF-8 或 ASCII)
 */
void safe_printf(const char *piece) {
    // 空指针检查
    if (piece == NULL) { return; }
    // 空字符串检查
    if (piece[0] == '\0') { return; }
    
    // 处理单字节 token (piece[1] == '\0' 表示只有一个字符)
    // 注意: 每个 token 至少有一个字节，所以访问 piece[1] 是安全的
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];  // 转为无符号避免符号扩展问题
        // 检查是否为可打印字符或空白字符
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;  // 控制字符，跳过不打印
        }
    }
    
    // 打印 token 字符串
    printf("%s", piece);
}

/* ============================================================================
 * 核心函数实现
 * ============================================================================ */

/**
 * @brief  从二进制文件初始化 Tokenizer
 * 
 * @param tokenizer  指向 Tokenizer 结构体的指针 (输出参数)
 * @param filename   tokenizer 二进制文件路径 (如 "gpt2_tokenizer.bin")
 * 
 * 文件格式 (二进制):
 *   ┌─────────────────────────────────────────────────────────┐
 *   │ Header: 256 x uint32_t (1024 字节)                      │
 *   │   [0] = 20240328 (魔数，用于验证文件格式)               │
 *   │   [1] = version (1 或 2)                                │
 *   │   [2] = vocab_size (词汇表大小)                         │
 *   │   [3] = eot_token (仅 version 2，结束符 ID)             │
 *   │   [4-255] = 保留                                        │
 *   ├─────────────────────────────────────────────────────────┤
 *   │ Body: vocab_size 个 token                               │
 *   │   每个 token:                                           │
 *   │     1 字节: length (token 字节数, 1-255)                │
 *   │     N 字节: token 内容 (原始字节)                       │
 *   └─────────────────────────────────────────────────────────┘
 * 
 * 初始化流程:
 *   1. 打开文件并读取 header
 *   2. 验证魔数和版本号
 *   3. 分配 token_table 内存
 *   4. 逐个读取 token 并存储
 *   5. 设置 init_ok = 1 表示成功
 */
void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    // ========== 第一步: 打开文件 ==========
    FILE *file = fopen(filename, "rb");  // "rb" = read binary
    if (file == NULL) {
        // 文件打开失败，输出帮助信息
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;  // 标记初始化失败
        return;
    }
    
    // ========== 第二步: 读取并验证 Header ==========
    uint32_t header[256];  // 256 个 uint32_t = 1024 字节
    freadCheck(header, sizeof(uint32_t), 256, file);
    
    // 验证魔数 (magic number)，确保文件格式正确
    assert(header[0] == 20240328);  // 魔数 = 日期 2024-03-28
    
    int version = header[1];                  // 文件版本号
    tokenizer->vocab_size = header[2];        // 词汇表大小
    
    // 根据版本号处理 EOT (End Of Text) token
    if (version == 1) {
        // v1 不包含 EOT token ID，使用 GPT-2 默认值
        assert(tokenizer->vocab_size == 50257);  // GPT-2 词汇表大小
        tokenizer->eot_token = 50256;            // GPT-2 的 EOT token ID
    } else if (version == 2) {
        // v2 在 header[3] 中存储 EOT token ID
        tokenizer->eot_token = header[3];
    } else {
        // 不支持的版本号
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    
    // ========== 第三步: 分配 token 查找表内存 ==========
    // token_table 是指针数组，每个元素指向一个 token 字符串
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    
    // ========== 第四步: 逐个读取 token ==========
    unsigned char length;  // token 长度 (1-255 字节)
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        // 读取 1 字节的长度值
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0);  // 每个 token 至少有 1 个字节
        
        // 分配 token 字符串内存 (+1 用于 null 终止符)
        char *token_bytes = (char *)mallocCheck(length + 1);
        
        // 读取 token 内容
        freadCheck(token_bytes, sizeof(char), length, file);
        
        // 添加 null 终止符，使其成为有效的 C 字符串
        token_bytes[length] = '\0';
        
        // 存储到查找表
        tokenizer->token_table[i] = token_bytes;
    }
    
    // ========== 第五步: 清理并标记成功 ==========
    fcloseCheck(file);       // 关闭文件
    tokenizer->init_ok = 1;  // 标记初始化成功
}

/**
 * @brief  将 token ID 解码为字符串
 * 
 * @param tokenizer  已初始化的 Tokenizer 指针
 * @param token_id   要解码的 token ID (0 ~ vocab_size-1)
 * @return           对应的 token 字符串，失败返回 NULL
 * 
 * 使用示例:
 *   const char *token_str = tokenizer_decode(&tokenizer, 15496);
 *   // token_str = "Hello"
 * 
 * 注意事项:
 *   - 返回的字符串指针指向 Tokenizer 内部存储，不要 free
 *   - Tokenizer 释放后，返回的指针将失效
 */
const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    // 检查 Tokenizer 是否已成功初始化
    if (tokenizer->init_ok == 0) {
        return NULL;  // 未初始化，返回空
    }
    
    // 边界检查: token_id 必须在有效范围内
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];  // 查表返回
    } else {
        printf("invalid token id %u!\n", token_id);  // 错误提示
        return NULL;
    }
}

/**
 * @brief  释放 Tokenizer 占用的内存
 * 
 * @param tokenizer  要释放的 Tokenizer 指针
 * 
 * 释放流程:
 *   1. 检查是否已初始化
 *   2. 逐个释放每个 token 字符串
 *   3. 释放 token_table 指针数组
 * 
 * 注意: 此函数不释放 Tokenizer 结构体本身
 *       如果 Tokenizer 是动态分配的，调用者需要额外 free
 */
void tokenizer_free(Tokenizer *tokenizer) {
    // 只有成功初始化的 Tokenizer 才需要释放内存
    if (tokenizer->init_ok) {
        // 释放每个 token 字符串
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        // 释放指针数组本身
        free(tokenizer->token_table);
    }
}