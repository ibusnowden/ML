// Copyright 2024 mmllm contributors
// Logging utilities for mmllm

#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Log levels
typedef enum {
    MLLM_LOG_DEBUG = 0,
    MLLM_LOG_INFO,
    MLLM_LOG_WARN,
    MLLM_LOG_ERROR,
    MLLM_LOG_FATAL,
} mllm_log_level_t;

// Initialize logging (called once at startup)
void mllm_log_init(void);

// Set log level
void mllm_set_log_level(mllm_log_level_t level);

// Core log function
void mllm_log(mllm_log_level_t level,
              const char *file, int line,
              const char *func,
              const char *fmt, ...);

// Convenience macros
#define MLLM_LOG_DEBUG(fmt, ...) \
    mllm_log(MLLM_LOG_DEBUG, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)

#define MLLM_LOG_INFO(fmt, ...) \
    mllm_log(MLLM_LOG_INFO, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)

#define MLLM_LOG_WARN(fmt, ...) \
    mllm_log(MLLM_LOG_WARN, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)

#define MLLM_LOG_ERROR(fmt, ...) \
    mllm_log(MLLM_LOG_ERROR, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)

#define MLLM_LOG_FATAL(fmt, ...) \
    do { \
        mllm_log(MLLM_LOG_FATAL, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__); \
        abort(); \
    } while (0)

// Assert macro with logging
#define MLLM_CHECK(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            MLLM_LOG_FATAL("Check failed: %s - " fmt, #cond, ##__VA_ARGS__); \
        } \
    } while (0)

#ifdef __cplusplus
}
#endif
