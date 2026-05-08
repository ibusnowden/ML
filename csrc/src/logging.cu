// Copyright 2024 mmllm contributors
// Logging implementation

#include "logging.h"
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

static mllm_log_level_t current_level = MLLM_LOG_INFO;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

void mllm_log_init(void) {
    pthread_mutex_init(&log_mutex, NULL);
}

void mllm_set_log_level(mllm_log_level_t level) {
    pthread_mutex_lock(&log_mutex);
    current_level = level;
    pthread_mutex_unlock(&log_mutex);
}

void mllm_log(mllm_log_level_t level,
              const char *file, int line,
              const char *func,
              const char *fmt, ...) {
    pthread_mutex_lock(&log_mutex);
    
    if (level < current_level) {
        pthread_mutex_unlock(&log_mutex);
        return;
    }
    
    const char *level_str = "DEBUG";
    if (level == MLLM_LOG_INFO) level_str = "INFO";
    else if (level == MLLM_LOG_WARN) level_str = "WARN";
    else if (level == MLLM_LOG_ERROR) level_str = "ERROR";
    else if (level == MLLM_LOG_FATAL) level_str = "FATAL";
    
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);
    
    fprintf(stderr, "[%s] [%s] %s:%d %s: ", 
            time_buf, level_str, file, line, func);
    
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    
    fprintf(stderr, "\n");
    fflush(stderr);
    
    pthread_mutex_unlock(&log_mutex);
}
