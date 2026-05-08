// Copyright 2024 mmllm contributors
// Minimal scalar metrics logging for TensorBoard and W&B CLI workflows

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool enabled;
    bool tensorboard_enabled;
    bool wandb_enabled;
    int rank;
    char log_dir[512];
    char wandb_project[128];
    char wandb_run_name[128];
    void *jsonl_file;
    void *tb_file;
} mllm_metrics_logger_t;

int mllm_metrics_init(mllm_metrics_logger_t *logger,
                      const char *log_dir,
                      bool tensorboard_enabled,
                      bool wandb_enabled,
                      const char *wandb_project,
                      const char *wandb_run_name,
                      int rank);

int mllm_metrics_log(mllm_metrics_logger_t *logger,
                     int64_t step,
                     double wall_time,
                     float loss,
                     float lr,
                     int64_t tokens_seen,
                     double step_time_ms,
                     size_t cuda_mem_free,
                     size_t cuda_mem_total);

void mllm_metrics_close(mllm_metrics_logger_t *logger);

#ifdef __cplusplus
}
#endif
