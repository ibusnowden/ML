// Copyright 2024 entropy contributors
// Main training entry point

#include "model.h"
#include "optimizer.h"
#include "fsdp.h"
#include "checkpoint.h"
#include "data_loader.h"
#include "nccl_wrapper.h"
#include "comm.h"
#include "tensor.h"
#include "logging.h"
#include "metrics.h"
#include "error.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <signal.h>
#include <time.h>
#include <math.h>

static volatile int g_shutdown = 0;
static void signal_handler(int sig) {
    (void)sig;
    g_shutdown = 1;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  --data PATH         Training data file (required)\n"
        "  --checkpoint DIR    Checkpoint directory\n"
        "  --output DIR        Output directory for checkpoints\n"
        "  --batch-size N      Batch size per device (default: 2)\n"
        "  --seq-len N         Sequence length (default: 2048)\n"
        "  --max-steps N       Max training steps (default: 1000)\n"
        "  --lr FLOAT          Learning rate (default: 3e-4)\n"
        "  --warmup-ratio R    Warmup ratio (default: 0.05)\n"
        "  --weight-decay F    Weight decay (default: 0.1)\n"
        "  --fp16              Use FP16 training\n"
        "  --bf16              Use BF16 training\n"
        "  --save-every N      Save checkpoint every N steps\n"
        "  --eval-every N      Evaluate every N steps\n"
        "  --resume STEP       Resume from checkpoint STEP\n"
        "  --nodes N           Number of nodes (default: 1)\n"
        "  --gpus-per-node N   GPUs per node (default: 1)\n"
        "  --tp-size N         Tensor parallel size (default: 1)\n"
        "  --pp-size N         Pipeline parallel size (default: 1)\n"
        "  --host-id STR       Host ID for NCCL\n"
        "  --node-id N         Node ID (default: 0)\n"
        "  --verbose           Verbose logging\n"
        "  --help              Show this help\n",
        prog);
}

typedef struct {
    // Model config
    mllm_model_config_t config;

    // Training config
    char *data_path;
    char *checkpoint_dir;
    char *output_dir;
    int batch_size;
    int seq_len;
    int64_t max_steps;
    float lr;
    float warmup_ratio;
    float weight_decay;
    bool fp16;
    bool bf16;
    int save_every;
    int eval_every;
    int64_t resume_step;
    int nodes;
    int gpus_per_node;
    int tp_size;
    int pp_size;
    char *host_id;
    int node_id;
    bool verbose;
    char *log_dir;
    int log_every;
    bool tensorboard;
    bool wandb;
    char *wandb_project;
    char *wandb_run_name;
} mllm_train_config_t;

static void default_train_config(mllm_train_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));

    // Default model config (Qwen2.5-like)
    cfg->config.hidden_size = 3584;
    cfg->config.intermediate_size = 18944;
    cfg->config.num_hidden_layers = 28;
    cfg->config.num_attention_heads = 28;
    cfg->config.num_key_value_heads = 4;
    cfg->config.num_heads = 28;
    cfg->config.head_dim = 128;
    cfg->config.max_position_embeddings = 32768;
    cfg->config.vocab_size = 151936;
    cfg->config.image_size = 336;
    cfg->config.num_images = 4;
    cfg->config.num_experts = 0;      // 0 = dense (set >0 for MoE)
    cfg->config.num_experts_per_tok = 2;
    cfg->config.rms_norm_eps = 1e-6f;
    cfg->config.attention_dropout = 0.0f;
    cfg->config.rotary_base = 1000000.0f;

    // Vision (ViT) config
    cfg->config.vision_hidden_size = 0;        // 0 = same as hidden_size
    cfg->config.vision_num_layers = 0;         // 0 = no ViT (legacy conv only)
    cfg->config.vision_num_heads = 16;
    cfg->config.vision_intermediate_size = 0;  // 0 = vhs * 4
    cfg->config.vision_patch_size = 14;
    cfg->config.vision_image_size = 336;

    // Default training config
    cfg->batch_size = 2;
    cfg->seq_len = 2048;
    cfg->max_steps = 1000;
    cfg->lr = 3e-4f;
    cfg->warmup_ratio = 0.05f;
    cfg->weight_decay = 0.1f;
    cfg->save_every = 100;
    cfg->eval_every = 50;
    cfg->nodes = 1;
    cfg->gpus_per_node = 1;
    cfg->tp_size = 1;
    cfg->pp_size = 1;
    cfg->node_id = 0;
    cfg->verbose = false;
    cfg->log_dir = (char *)"./runs/entropy";
    cfg->log_every = 10;
    cfg->tensorboard = false;
    cfg->wandb = false;
    cfg->wandb_project = (char *)"entropy";
    cfg->wandb_run_name = (char *)"entropy-run";
}

static int parse_args(mllm_train_config_t *cfg, int argc, char *argv[]) {
    static struct option long_options[] = {
        {"data",      required_argument, 0, 'd'},
        {"checkpoint",required_argument, 0, 'k'},
        {"output",    required_argument, 0, 'o'},
        {"batch-size",required_argument, 0, 'b'},
        {"seq-len",   required_argument, 0, 's'},
        {"max-steps", required_argument, 0, 'm'},
        {"lr",        required_argument, 0, 'l'},
        {"warmup-ratio", required_argument, 0, 'w'},
        {"weight-decay", required_argument, 0, 'W'},
        {"fp16",      no_argument,       0, '1'},
        {"bf16",      no_argument,       0, '2'},
        {"save-every",required_argument, 0, 'e'},
        {"eval-every",required_argument, 0, 'v'},
        {"resume",    required_argument, 0, 'r'},
        {"nodes",     required_argument, 0, 'n'},
        {"gpus-per-node", required_argument, 0, 'g'},
        {"tp-size",   required_argument, 0, 't'},
        {"pp-size",   required_argument, 0, 'p'},
        {"host-id",   required_argument, 0, 'H'},
        {"node-id",   required_argument, 0, 'N'},
        {"log-dir",   required_argument, 0, 'L'},
        {"log-every", required_argument, 0, 'E'},
        {"tensorboard", no_argument,     0, 'T'},
        {"wandb",     no_argument,       0, 'B'},
        {"wandb-project", required_argument, 0, 'P'},
        {"wandb-run-name", required_argument, 0, 'R'},
        {"verbose",   no_argument,       0, 'V'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt_idx = 0;
    int c;
    while ((c = getopt_long(argc, argv, "d:k:o:b:s:m:l:w:W:12e:v:r:n:g:t:p:H:N:L:E:TBP:R:Vh",
                            long_options, &opt_idx)) != -1) {
        switch (c) {
            case 'd': cfg->data_path = optarg; break;
            case 'k': cfg->checkpoint_dir = optarg; break;
            case 'o': cfg->output_dir = optarg; break;
            case 'b': cfg->batch_size = atoi(optarg); break;
            case 's': cfg->seq_len = atoi(optarg); break;
            case 'm': cfg->max_steps = atoll(optarg); break;
            case 'l': cfg->lr = atof(optarg); break;
            case 'w': cfg->warmup_ratio = atof(optarg); break;
            case 'W': cfg->weight_decay = atof(optarg); break;
            case '1': cfg->fp16 = true; break;
            case '2': cfg->bf16 = true; break;
            case 'e': cfg->save_every = atoi(optarg); break;
            case 'v': cfg->eval_every = atoi(optarg); break;
            case 'r': cfg->resume_step = atoll(optarg); break;
            case 'n': cfg->nodes = atoi(optarg); break;
            case 'g': cfg->gpus_per_node = atoi(optarg); break;
            case 't': cfg->tp_size = atoi(optarg); break;
            case 'p': cfg->pp_size = atoi(optarg); break;
            case 'H': cfg->host_id = optarg; break;
            case 'N': cfg->node_id = atoi(optarg); break;
            case 'L': cfg->log_dir = optarg; break;
            case 'E': cfg->log_every = atoi(optarg); break;
            case 'T': cfg->tensorboard = true; break;
            case 'B': cfg->wandb = true; break;
            case 'P': cfg->wandb_project = optarg; break;
            case 'R': cfg->wandb_run_name = optarg; break;
            case 'V': cfg->verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return -1;
        }
    }

    if (!cfg->data_path) {
        fprintf(stderr, "Error: --data is required\n");
        print_usage(argv[0]);
        return -1;
    }

    if (!cfg->output_dir) {
        cfg->output_dir = (char *)"./checkpoints";
    }

    if (cfg->log_every <= 0) cfg->log_every = 10;
    return 0;
}

static double wall_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static void batch_release(mllm_batch_t *batch) {
    if (!batch) return;
    free(batch->position_ids);
    free(batch->labels);
    batch->position_ids = NULL;
    batch->labels = NULL;
}

// Simple linear warmup + cosine decay scheduler
static float learning_rate_schedule(float base_lr, int64_t step,
                                     int64_t max_steps, float warmup_ratio) {
    int64_t warmup_steps = (int64_t)(max_steps * warmup_ratio);
    if (warmup_steps > 0 && step < warmup_steps) {
        return base_lr * (float)(step + 1) / (float)warmup_steps;
    }
    int64_t decay_steps = max_steps - warmup_steps;
    if (decay_steps <= 0) return base_lr;
    float progress = (float)(step - warmup_steps) / (float)decay_steps;
    return base_lr * (0.5f + 0.5f * cosf(3.14159265f * progress));
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    mllm_train_config_t train_cfg;
    default_train_config(&train_cfg);

    if (parse_args(&train_cfg, argc, argv) != 0) {
        return 1;
    }

    // Initialize logging
    mllm_log_init();
    mllm_set_log_level(train_cfg.verbose ? MLLM_LOG_DEBUG : MLLM_LOG_INFO);

    // Determine global rank and world size
    int local_rank = 0;
    int node_id = train_cfg.node_id;

    if (getenv("LOCAL_RANK")) {
        local_rank = atoi(getenv("LOCAL_RANK"));
    }
    if (getenv("NODE_ID")) {
        node_id = atoi(getenv("NODE_ID"));
    }

    int local_size = train_cfg.gpus_per_node;
    int world_size = local_size * train_cfg.nodes;
    int rank = node_id * local_size + local_rank;

    char host_id[256] = "localhost";
    if (train_cfg.host_id) {
        strncpy(host_id, train_cfg.host_id, sizeof(host_id) - 1);
    } else if (getenv("MASTER_ADDR")) {
        strncpy(host_id, getenv("MASTER_ADDR"), sizeof(host_id) - 1);
    }

    MLLM_LOG_INFO("=== entropy Training ===");
    MLLM_LOG_INFO("Rank %d / %d (node %d, local %d)", rank, world_size, node_id, local_rank);
    MLLM_LOG_INFO("Host: %s, Nodes: %d, GPUs/Node: %d", host_id, train_cfg.nodes, local_size);
    MLLM_LOG_INFO("Batch size: %d, Seq len: %d, Max steps: %ld",
                  train_cfg.batch_size, train_cfg.seq_len, (long)train_cfg.max_steps);
    MLLM_LOG_INFO("LR: %.6f, Weight decay: %.4f", train_cfg.lr, train_cfg.weight_decay);

    int rc;
    mllm_metrics_logger_t metrics;
    rc = mllm_metrics_init(&metrics, train_cfg.log_dir,
                           train_cfg.tensorboard, train_cfg.wandb,
                           train_cfg.wandb_project, train_cfg.wandb_run_name,
                           rank);
    if (rc != MLLM_OK) {
        MLLM_LOG_FATAL("Metrics init failed: %s", mllm_error_str(rc));
    }

    // Set device
    MLLM_CUDA_CHECK(cudaSetDevice(local_rank));

    // Initialize NCCL
    mllm_comm_group_t dp_comm;
    memset(&dp_comm, 0, sizeof(dp_comm));
    dp_comm.world_size = world_size;
    dp_comm.rank = rank;
    dp_comm.local_rank = local_rank;
    dp_comm.local_size = local_size;
    dp_comm.node_id = node_id;

    rc = mllm_nccl_init(&dp_comm, host_id, rank);
    if (rc != MLLM_OK) {
        MLLM_LOG_FATAL("NCCL init failed: %s", mllm_error_str(rc));
    }

    // Initialize model
    mllm_model_config_t *config = &train_cfg.config;
    mllm_mp_topology_t topo;
    memset(&topo, 0, sizeof(topo));
    rc = mllm_topology_init(&topo, host_id, node_id, rank, world_size,
                            train_cfg.tp_size, train_cfg.pp_size);
    if (rc != MLLM_OK) {
        MLLM_LOG_FATAL("Topology init failed: %s", mllm_error_str(rc));
    }

    mllm_model_t model;
    rc = mllm_model_create(&model, config, &topo);
    if (rc != MLLM_OK) {
        MLLM_LOG_FATAL("Model create failed: %s", mllm_error_str(rc));
    }

    rc = mllm_model_init_weights(&model, 0);
    if (rc != MLLM_OK) {
        MLLM_LOG_FATAL("Weight init failed: %s", mllm_error_str(rc));
    }

    // Initialize optimizer
    mllm_optimizer_t optimizer;
    rc = mllm_optimizer_create(&optimizer, &model, &dp_comm,
                                train_cfg.lr, 0.9f, 0.95f, 1e-8f, train_cfg.weight_decay);
    if (rc != MLLM_OK) {
        MLLM_LOG_FATAL("Optimizer create failed: %s", mllm_error_str(rc));
    }

    // Initialize FSDP
    mllm_fsdp_t fsdp;
    rc = mllm_fsdp_create(&fsdp, &model, &optimizer, &dp_comm);
    if (rc != MLLM_OK) {
        MLLM_LOG_FATAL("FSDP create failed: %s", mllm_error_str(rc));
    }

    // Initialize data loader
    mllm_data_loader_t loader;
    rc = mllm_data_loader_create(&loader, train_cfg.data_path,
                                  train_cfg.batch_size, train_cfg.seq_len,
                                  train_cfg.config.num_images,
                                  train_cfg.config.vision_hidden_size,
                                  true, 42);
    if (rc != MLLM_OK) {
        MLLM_LOG_FATAL("Data loader create failed: %s", mllm_error_str(rc));
    }

    // Resume from checkpoint if requested
    if (train_cfg.resume_step > 0 && train_cfg.checkpoint_dir) {
        mllm_checkpoint_meta_t meta;
        rc = mllm_checkpoint_load(&fsdp, train_cfg.checkpoint_dir,
                                   train_cfg.resume_step, 0, &meta);
        if (rc == MLLM_OK) {
            MLLM_LOG_INFO("Resumed from step %ld, loss=%.4f",
                          (long)meta.step, meta.loss);
        }
    }

    // Training loop
    cudaStream_t stream = 0;
    MLLM_CUDA_CHECK(cudaStreamCreate(&stream));

    mllm_batch_t batch;
    int64_t step = 0;

    MLLM_LOG_INFO("Starting training...");

    while (step < train_cfg.max_steps && !g_shutdown) {
        rc = mllm_data_loader_next(&loader, &batch);
        if (rc != MLLM_OK) {
            // Reset loader for next epoch
            mllm_data_loader_reset(&loader);
            rc = mllm_data_loader_next(&loader, &batch);
            if (rc != MLLM_OK) break;
        }

        // Learning rate schedule
        float lr = learning_rate_schedule(train_cfg.lr, step, train_cfg.max_steps,
                                          train_cfg.warmup_ratio);

        optimizer.lr = lr;
        double step_start = wall_time_seconds();

        // FSDP step
        rc = mllm_fsdp_step(&fsdp, batch.input_ids, batch.position_ids,
                            batch.image_embeds, batch.labels,
                            batch.batch_size, batch.seq_len, stream);
        if (rc != MLLM_OK) {
            MLLM_LOG_WARN("Step %ld failed: %s", (long)step, mllm_error_str(rc));
            batch_release(&batch);
            break;
        }
        MLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
        double step_time_ms = (wall_time_seconds() - step_start) * 1000.0;

        // Update tokens seen
        fsdp.tokens_seen += (int64_t)train_cfg.batch_size * train_cfg.seq_len;

        // Print progress and metrics
        if (step % train_cfg.log_every == 0 || step == 0) {
            size_t mem_free = 0, mem_total = 0;
            cudaMemGetInfo(&mem_free, &mem_total);
            MLLM_LOG_INFO("Step %ld/%ld | loss=%.4f | lr=%.6f | tokens=%ld | %.2f ms",
                          (long)step, (long)train_cfg.max_steps,
                          fsdp.train_loss, lr, (long)fsdp.tokens_seen, step_time_ms);
            mllm_metrics_log(&metrics, step, wall_time_seconds(), fsdp.train_loss,
                             lr, fsdp.tokens_seen, step_time_ms, mem_free, mem_total);
        }

        // Save checkpoint
        if (train_cfg.save_every > 0 && step % train_cfg.save_every == 0 && step > 0) {
            mllm_checkpoint_save(&fsdp, train_cfg.output_dir, step, stream);
            MLLM_LOG_INFO("Checkpoint saved at step %ld", (long)step);
        }

        batch_release(&batch);
        step++;
    }

    MLLM_LOG_INFO("Training complete. Final step: %ld, loss: %.4f",
                  (long)step, fsdp.train_loss);

    // Cleanup
    cudaStreamDestroy(stream);
    mllm_metrics_close(&metrics);
    mllm_data_loader_destroy(&loader);
    mllm_fsdp_destroy(&fsdp);
    mllm_optimizer_destroy(&optimizer);
    mllm_model_destroy(&model);
    mllm_topology_destroy(&topo);
    mllm_nccl_destroy(&dp_comm);

    MLLM_LOG_INFO("entropy training finished.");
    return 0;
}
