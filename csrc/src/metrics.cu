// Copyright 2024 entropy contributors
// Minimal scalar metrics logging for TensorBoard and W&B CLI workflows

#include "metrics.h"
#include "logging.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

static int mkdir_p(const char *path) {
    if (!path || !*path) return -1;
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len == 0) return -1;
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    return mkdir(tmp, 0755) == 0 || errno == EEXIST ? 0 : -1;
}

static uint32_t crc32c_update(uint32_t crc, const unsigned char *data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int k = 0; k < 8; k++) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0x82F63B78u & mask);
        }
    }
    return ~crc;
}

static uint32_t masked_crc32c(const unsigned char *data, size_t len) {
    uint32_t x = crc32c_update(0, data, len);
    return ((x >> 15) | (x << 17)) + 0xa282ead8u;
}

static unsigned char *put_varint(unsigned char *p, uint64_t v) {
    while (v >= 0x80) {
        *p++ = (unsigned char)(v | 0x80);
        v >>= 7;
    }
    *p++ = (unsigned char)v;
    return p;
}

static unsigned char *put_fixed32(unsigned char *p, float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    for (int i = 0; i < 4; i++) *p++ = (unsigned char)((v.u >> (8 * i)) & 0xff);
    return p;
}

static unsigned char *put_fixed64(unsigned char *p, double d) {
    union { double d; uint64_t u; } v;
    v.d = d;
    for (int i = 0; i < 8; i++) *p++ = (unsigned char)((v.u >> (8 * i)) & 0xff);
    return p;
}

static unsigned char *put_string_field(unsigned char *p, int field, const char *s) {
    size_t n = strlen(s);
    p = put_varint(p, ((uint64_t)field << 3) | 2u);
    p = put_varint(p, n);
    memcpy(p, s, n);
    return p + n;
}

static size_t encode_value(unsigned char *buf, const char *tag, float value) {
    unsigned char *p = buf;
    p = put_string_field(p, 1, tag);
    p = put_varint(p, (2u << 3) | 5u);
    p = put_fixed32(p, value);
    return (size_t)(p - buf);
}

static size_t encode_summary(unsigned char *buf, const char *tag, float value) {
    unsigned char value_buf[256];
    size_t value_len = encode_value(value_buf, tag, value);
    unsigned char *p = buf;
    p = put_varint(p, (1u << 3) | 2u);
    p = put_varint(p, value_len);
    memcpy(p, value_buf, value_len);
    return (size_t)((p + value_len) - buf);
}

static int write_event_record(FILE *f, const unsigned char *data, size_t len) {
    unsigned char len_buf[8];
    uint64_t n = (uint64_t)len;
    for (int i = 0; i < 8; i++) len_buf[i] = (unsigned char)((n >> (8 * i)) & 0xff);
    uint32_t len_crc = masked_crc32c(len_buf, sizeof(len_buf));
    uint32_t data_crc = masked_crc32c(data, len);
    if (fwrite(len_buf, 1, 8, f) != 8) return -1;
    if (fwrite(&len_crc, 1, 4, f) != 4) return -1;
    if (fwrite(data, 1, len, f) != len) return -1;
    if (fwrite(&data_crc, 1, 4, f) != 4) return -1;
    return 0;
}

static int write_scalar_event(FILE *f, double wall_time, int64_t step,
                              const char *tag, float value) {
    unsigned char summary_buf[512];
    size_t summary_len = encode_summary(summary_buf, tag, value);
    unsigned char event_buf[768];
    unsigned char *p = event_buf;
    p = put_varint(p, (1u << 3) | 1u);
    p = put_fixed64(p, wall_time);
    p = put_varint(p, (2u << 3) | 0u);
    p = put_varint(p, (uint64_t)step);
    p = put_varint(p, (5u << 3) | 2u);
    p = put_varint(p, summary_len);
    memcpy(p, summary_buf, summary_len);
    p += summary_len;
    return write_event_record(f, event_buf, (size_t)(p - event_buf));
}

static int write_file_version_event(FILE *f, double wall_time) {
    unsigned char event_buf[128];
    unsigned char *p = event_buf;
    p = put_varint(p, (1u << 3) | 1u);
    p = put_fixed64(p, wall_time);
    p = put_string_field(p, 3, "brain.Event:2");
    return write_event_record(f, event_buf, (size_t)(p - event_buf));
}

int mllm_metrics_init(mllm_metrics_logger_t *logger,
                      const char *log_dir,
                      bool tensorboard_enabled,
                      bool wandb_enabled,
                      const char *wandb_project,
                      const char *wandb_run_name,
                      int rank) {
    memset(logger, 0, sizeof(*logger));
    logger->rank = rank;
    logger->tensorboard_enabled = tensorboard_enabled;
    logger->wandb_enabled = wandb_enabled;
    logger->enabled = (tensorboard_enabled || wandb_enabled) && rank == 0;
    if (!logger->enabled) return MLLM_OK;

    snprintf(logger->log_dir, sizeof(logger->log_dir), "%s", log_dir ? log_dir : "runs/entropy");
    snprintf(logger->wandb_project, sizeof(logger->wandb_project), "%s", wandb_project ? wandb_project : "entropy");
    snprintf(logger->wandb_run_name, sizeof(logger->wandb_run_name), "%s", wandb_run_name ? wandb_run_name : "entropy-run");
    if (mkdir_p(logger->log_dir) != 0) {
        MLLM_LOG_ERROR("failed to create metrics log dir: %s", logger->log_dir);
        return MLLM_ERR_INVALID_INPUT;
    }

    char path[768];
    snprintf(path, sizeof(path), "%s/metrics.jsonl", logger->log_dir);
    logger->jsonl_file = fopen(path, "a");
    if (!logger->jsonl_file) return MLLM_ERR_INVALID_INPUT;

    if (tensorboard_enabled) {
        time_t now = time(NULL);
        snprintf(path, sizeof(path), "%s/events.out.tfevents.%ld.entropy", logger->log_dir, (long)now);
        logger->tb_file = fopen(path, "ab");
        if (!logger->tb_file) return MLLM_ERR_INVALID_INPUT;
        write_file_version_event((FILE *)logger->tb_file, (double)now);
    }
    return MLLM_OK;
}

int mllm_metrics_log(mllm_metrics_logger_t *logger,
                     int64_t step,
                     double wall_time,
                     float loss,
                     float lr,
                     int64_t tokens_seen,
                     double step_time_ms,
                     size_t cuda_mem_free,
                     size_t cuda_mem_total) {
    if (!logger || !logger->enabled) return MLLM_OK;
    FILE *jsonl = (FILE *)logger->jsonl_file;
    if (jsonl) {
        fprintf(jsonl,
                "{\"step\":%ld,\"wall_time\":%.6f,\"loss\":%.8g,\"lr\":%.8g,\"tokens_seen\":%ld,\"step_time_ms\":%.3f,\"cuda_mem_free\":%zu,\"cuda_mem_total\":%zu}\n",
                (long)step, wall_time, loss, lr, (long)tokens_seen,
                step_time_ms, cuda_mem_free, cuda_mem_total);
        fflush(jsonl);
    }
    FILE *tb = (FILE *)logger->tb_file;
    if (tb) {
        write_scalar_event(tb, wall_time, step, "train/loss", loss);
        write_scalar_event(tb, wall_time, step, "train/lr", lr);
        write_scalar_event(tb, wall_time, step, "train/tokens_seen", (float)tokens_seen);
        write_scalar_event(tb, wall_time, step, "train/step_time_ms", (float)step_time_ms);
        write_scalar_event(tb, wall_time, step, "cuda/mem_free", (float)cuda_mem_free);
        write_scalar_event(tb, wall_time, step, "cuda/mem_total", (float)cuda_mem_total);
        fflush(tb);
    }
    return MLLM_OK;
}

void mllm_metrics_close(mllm_metrics_logger_t *logger) {
    if (!logger) return;
    if (logger->jsonl_file) fclose((FILE *)logger->jsonl_file);
    if (logger->tb_file) fclose((FILE *)logger->tb_file);
    if (logger->enabled && logger->wandb_enabled) {
        char cmd[1400];
        snprintf(cmd, sizeof(cmd),
                 "command -v wandb >/dev/null 2>&1 && WANDB_PROJECT='%s' WANDB_NAME='%s' wandb sync '%s' >/dev/null 2>&1",
                 logger->wandb_project, logger->wandb_run_name, logger->log_dir);
        int rc = system(cmd);
        if (rc != 0) {
            MLLM_LOG_WARN("wandb CLI sync skipped or failed for %s", logger->log_dir);
        }
    }
    memset(logger, 0, sizeof(*logger));
}
