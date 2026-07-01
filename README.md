# entropy — a multimodal LLM trainer in pure C/CUDA (no PyTorch)

A from-scratch transformer training stack written in C/CUDA + cuBLASLt + cuDNN +
NCCL, targeting **5 RTX Ada GPUs**. Modern llama-style architecture, mixed
precision (BF16 compute, FP32 master weights), real full forward+backward, fused
flash attention, multi-GPU data parallelism, and a multimodal (vision) path.

**Status:** the `train/` rewrite is the supported implementation. It builds a
single-TU CUDA trainer with BF16 cuBLASLt GEMMs, cuDNN flash attention, AdamW,
multimodal overfit mode, and NCCL data parallelism. Benchmark and scaling claims
should be regenerated from fresh logs before publication.

## Architecture
- Decoder: bias-free linears, **RMSNorm** (pre-norm), **RoPE** (rotate-half/HF),
  **SwiGLU** MLP, causal attention, untied LM head.
- GEMMs via **cuBLASLt** BF16 tensor-core matmuls.
- Attention via **cuDNN fused flash SDPA** (forward+backward), wired *permute-free*
  by giving cuDNN custom strides so it reads Q/K/V straight from the interleaved
  `qkv[B,T,3C]` buffer and writes `O` into `atty[B,T,C]`.
- Optimizer: **AdamW** with FP32 master weights + BF16 param copies + FP32 grads.
- Multimodal: a trainable **vision tower** (patch-embed → learned pos → per-patch
  RMSNorm/SwiGLU blocks → projector) whose image tokens are prepended to the LM
  text sequence; trained jointly end-to-end.

## Verification

Builds known from this tree:
- `scripts/build.sh train/gpt.cu build/gpt`
- `scripts/build.sh train/smoke.cu build/smoke`

Runtime checks to regenerate on a GPU node:
- `./build/gpt overfit`
- `./build/gpt mm_overfit`
- `./build/gpt mm_bench 2 50`
- `./build/gpt bench 16 20`
- `ENTROPY_ACCUM=16 srun --ntasks=5 --gres=gpu:rtx_6000:5 ./build/gpt bench_ddp 16 20`
- `python reference/bench_torch.py --B 16 --steps 20`
- `python reference/bench_torch.py --B 16 --steps 20 --compile`
- `python reference/bench_jax.py --B 16 --steps 20`
- `ENTROPY_F32_MASTER=1 python reference/bench_jax.py --B 48 --steps 20 --data <fineweb.bin>`

Publish benchmark tables only with the command, git revision, GPU model, driver,
CUDA/cuDNN/NCCL versions, and raw logs.

## Build & run
```sh
source scripts/env.sh                 # local CUDA 12.8 toolkit (.cudaenv) + cuDNN
scripts/build.sh train/gpt.cu build/gpt
# or: cmake -S . -B build-cmake && cmake --build build-cmake
# on a GPU node:
./build/gpt overfit       # correctness: fixed-batch loss -> ~0
./build/gpt bench 16 20   # single-GPU MFU
./build/gpt train <bin>   # train on a FineWeb .bin (llm.c token format)
./build/gpt plan           # pipeline/hot-spare topology estimator
./build/gpt mm_overfit    # multimodal joint overfit (vision tower + LM)
./build/gpt mm_bench 2 50 # timed multimodal train step (vision + LM + both optimizers)
# multi-GPU (5 ranks, 1 per GPU):
ENTROPY_ACCUM=16 srun --ntasks=5 --gres=gpu:rtx_6000:5 ./build/gpt bench_ddp 16 20
```
Use a different `--gres=gpu:<type>:5` only if the cluster uses another resource
name for RTX Ada GPUs.
Knobs: `ENTROPY_ATTN={cudnn|mat|flash}`, `ENTROPY_ACCUM=N`,
`ENTROPY_BF16_REDUCE=0/1`, `ENTROPY_OVERLAP_OPT=0/1`, `ENTROPY_RMS_ATOMIC=0/1`.
`ENTROPY_CE_THREADS={128|256|512|1024}` overrides the per-row cross-entropy
reduction width; the default remains 256 after H100 A/B showed larger blocks did
not improve the B=48 path.
`ENTROPY_OVERLAP_OPT` defaults on for sm89 RTX Ada builds and off for sm90 H100
builds after H100 timing showed post-backward AdamW slightly faster.
For DDP, sm90 H100 keeps the packed atomic RMSNorm backward default; sm89 RTX Ada
still defaults to `ENTROPY_RMS_ATOMIC=0` in DDP after earlier 5-GPU measurements.
With bf16 DDP reduction, gradient mean scaling is folded into the bf16-to-fp32
cast-back path unless optimizer overlap is applying the scale.
For H100 synthetic B=48 comparisons against JAX/XLA, use CUDA graph replay:
`ENTROPY_PEAK_TFLOPS=989 ENTROPY_GRAPH=1 ./build/gpt_h100 bench 48 20`.
Training also accepts `ENTROPY_B`, `ENTROPY_T`, `ENTROPY_C`, `ENTROPY_L`,
`ENTROPY_H`, `ENTROPY_I`, and `ENTROPY_LOG_EVERY`. The `train` mode uses pinned
double-buffered input staging and overlaps the next batch copy with the current
GPU step. `plan` accepts `ENTROPY_WORLD`, `ENTROPY_PP`, `ENTROPY_MICROBATCHES`, and
`ENTROPY_SPARE_GPUS` to estimate 1F1B pipeline bubbles, stage memory, activation
traffic, and restart-on-spare topology.

Hot GPU replacement is treated as restart-on-spare: reserve a GPU outside the
active NCCL communicator, checkpoint frequently from the supervisor, and relaunch
onto the spare after an Xid/ECC failure. Growing or shrinking a live NCCL
communicator in-process is not a reliable training-time primitive.

## Layout
- `train/gpt.cu` — model, forward, backward, AdamW, data loader, DDP, vision tower, modes.
- `train/gpt_kernels.cuh` — cuBLASLt matmul helpers + all CUDA kernels.
- `train/attn_cudnn.cuh` — cuDNN-frontend fused flash attention (graph cached).
- `reference/bench_torch.py`, `reference/bench_jax.py` — matched baselines.
- `scripts/{env,build}.sh` — toolchain + build.

For JAX reference runs, do not source `scripts/env.sh`; that path intentionally
prepends the C/CUDA trainer libraries and can make JAX pick incompatible cuDNN
frontend components. Use `.jaxenv/bin/python` directly on GPU nodes.

## Correctness
The supported checks are fixed-batch LM overfit, fixed-batch multimodal overfit,
single-GPU benchmark, DDP benchmark, and matched PyTorch/JAX reference scripts.
The DDP path initializes identical weights on all ranks and averages gradients over
both world size and gradient-accumulation microbatches.

## Notes
- The original `csrc/` prototype is superseded — it had naive FP32 GEMMs,
  materialized O(T²) attention, and a non-functional backward (only embeddings +
  LM head got gradients). The supported implementation lives under `train/`.
- Toolchain: a self-consistent CUDA 12.8 toolkit lives in `.cudaenv` (conda); the
  pip `nvidia-cuda-nvcc-cu12` wheel ships no `nvcc`. Runtime requires a driver
  new enough for the CUDA runtime in `.cudaenv`.
