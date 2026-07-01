# entropy status

Pure C/CUDA (+cuBLASLt/cuDNN/NCCL) multimodal LLM trainer for 5 RTX Ada GPUs.

Done:
- [x] Toolchain files (CUDA 12.8 in .cudaenv; sm_89 build path).
- [x] BF16 llama-style trainer with full fwd/bwd code path.
- [x] cuBLASLt GEMMs + cuDNN fused flash attention (permute-free).
- [x] Multi-GPU NCCL DDP: identical rank initialization, overlapped allreduce,
      gradient accumulation averaged over world*accum, optional bf16 reduce.
- [x] Multimodal: trainable vision tower + projector, image tokens fused into the
      LM, joint training path with text-token loss normalization.

Perf vs JAX-cuDNN (135M, B=16, RTX 6000 Ada) — re-investigated 2026-06-07:
- [x] cuBLASLt autotuning (top-24 timed, cached) — ENTROPY_AUTOTUNE/ENTROPY_LT_WS_MB.
- [x] gate+up fused into one GEMM (fwd 2->1, bwd 4->2).
- [x] vision-tower gate+up fused into one GEMM and timed with `mm_bench`.
- [x] Q/K RoPE fused into one forward launch and one backward launch.
- [x] RMSNorm backward atomic-partial variant benchmarked and enabled for single-GPU.
- [x] Overlapped AdamW enabled by default for single-GPU `bench`.
- [x] division-free rope/swiglu (rope fwd 3.9->0.9ms).
- [x] copy_k passthrough elimination (folded into rmsnorm_dx add-source).
- [x] Event-region profiler (ENTROPY_PROF=1) + JAX fwd-only/F32_MASTER split.
- Result before the latest pass: forward tied with JAX (36.5 vs 35.2ms), but full
  step was ~130 vs ~123ms. After Q/K RoPE fusion, single-GPU RMSNorm-bwd atomic
  partials, and overlapped AdamW, the single-GPU CUDA trainer now measures
  120.59ms / 28.9% MFU on the same 135M B=16 benchmark.
- Remaining gap is multi-GPU efficiency: 5-GPU DDP reaches 25.1% MFU, below the
  single-GPU 28.9%, because allreduce/synchronization pressure is now visible.

Verified on RTX 6000 Ada via Slurm after Q/K RoPE + RMSNorm-bwd + overlap updates:
- `./build/gpt overfit`: loss 5.58087 -> 0.00133.
- `./build/gpt mm_overfit`: loss 5.56500 -> 0.00115.
- `./build/norm_bench2 768`: current RMSNorm-bwd 2.80ms/25 calls vs atomic-partial
  2.24ms/25 calls.
- `ENTROPY_PROF=1 ./build/gpt bench 16 10`: 120.59ms, 135,866 tok/s,
  105.1 TFLOP/s, 28.9% MFU; profile total 118.90ms.
- `ENTROPY_ACCUM=16 ./build/gpt bench_ddp 16 20` on 5 RTX Ada GPUs: opt-step
  2220.02ms, 590,409 global tok/s, 456.9 TFLOP/s aggregate, 25.1% MFU.

Frontier to actually beat JAX-cuDNN (each ~ a few ms, all reduce backward HBM traffic):
- [ ] Fuse the backward elementwise chain (swiglu-bwd / rmsnorm-bwd / rope-bwd) to
      cut activation round-trips — the XLA edge. Largest remaining lever.
- [ ] bf16 optimizer state (m,v[,grad]) keeping fp32 master (~-2ms, declined: quality).
- [ ] Fused linear+cross-entropy to avoid materializing the 1GB logits (~-2ms).
- [ ] Full ViT attention in the vision tower (drop-in: cuDNN non-causal SDPA).
- [ ] Direct gradient-check vs PyTorch autograd; checkpoint save/load; real MM dataset.

H100 pass, 2026-06-11:
- [x] JAX-cuDNN reference restored in `.jaxenv`; H100 B=48 reference is 87.74ms,
      433.5 TFLOP/s, 43.8% MFU with the 989 TFLOP/s dense BF16 denominator.
- [x] Current best C/CUDA H100 B=48 profile is 88.36ms, 430.5 TFLOP/s, 43.5% MFU.
      That is effectively tied with JAX-cuDNN but not cleanly beaten yet.
- [x] CE online path is enabled by default and improves B=48 from 90.89ms to 89.84ms.
- [x] Chunked LM-head CE is correct but slower: useful as a memory fallback, not the
      speed path. A true fused CE+dW+dX kernel is still open.
- [x] Packed/contiguous cuDNN attention is not worth explicit permutes at B=48:
      benchmarked 12 layers at fwd 2.89ms/bwd 9.90ms strided vs fwd 2.79ms/bwd
      9.67ms contiguous. The 0.33ms gross win cannot pay for pack/unpack.
- [x] Native H100 FP8 E4M3 GEMMs are promising in isolation at B=48:
      qkv 1123 TFLOP/s, gate 1150, down 1429, lm_head 1204, big 1583.
      Standalone BF16->FP8 casts are only ~1.1 TB/s, so naive per-GEMM activation
      quantization erases the small-GEMM gains. The viable trainer route is cached
      FP8 weights plus fused producer quantization (RMSNorm/SwiGLU/CE), not a simple
      wrapper around every existing GEMM.
- [x] Whole-step CUDA graph replay is not a default H100 win for this workload:
      `ENTROPY_GRAPH=1 ./build/gpt_h100 bench 48 8` measured 89.16ms / 43.1% MFU,
      while graph disabled measured in the same 88-90ms band. Keep `ENTROPY_GRAPH`
      as an opt-in A/B knob.
- [x] Rebuilt default H100 binary with graph disabled: short verification
      `./build/gpt_h100 bench 48 4` measured 87.83ms, 433.1 TFLOP/s,
      43.8% MFU.
- [x] H100 knob sweep, B=48, 8-step bench: default 89.94ms; beta0 grads
      90.04ms; RMS atomic off 92.25ms; CE old 92.00ms; MLP dual 90.09ms;
      autotune64+1GiB workspace 90.01ms; graph replay 89.78ms. The only
      useful signal was optimizer overlap off at 89.41ms.
- [x] H100 train default changed to post-backward AdamW for sm90 builds while
      preserving `ENTROPY_OVERLAP_OPT` override and sm89 overlap-on default. Real
      FineWeb B=48 no-overlap timing improved the best steady window to 86.27ms,
      569,767 tok/s, 440.9 TFLOP/s, 44.6% MFU.
- [x] Real-data H100 training timing, B=48 on
      `/project/inniang/vibe/autoresearch/chal/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin`:
      after the first compile/autotune/load window, steady windows were 88.74ms,
      86.68ms, 89.97ms, and 86.88ms per step; best window 567k tok/s,
      438.8 TFLOP/s, 44.4% MFU.
- [x] Apples-to-apples real-data JAX-cuDNN comparison is now measured. Added
      `reference/bench_jax.py --data` for the same llm.c `.bin` loader semantics
      as CUDA train mode. JAX must run from `.jaxenv` without sourcing
      `scripts/env.sh`; the C trainer LD path makes JAX pick incompatible cuDNN
      components.
- [x] On the same H100, same FineWeb shard, B=48, T=1024, fp32-master AdamW:
      JAX-cuDNN best steady window was 91.34ms, 538,119 tok/s, 416.4 TFLOP/s,
      42.1% MFU (`logs/h100_jax_realdata_cleanenv_20260612_150852.log`).
      Current CUDA default best steady window was 86.24ms, 569,926 tok/s,
      441.0 TFLOP/s, 44.6% MFU
      (`logs/h100_cuda_realdata_default_20260612_151013.log`). This is a clean
      real-data training-time win for CUDA, while the synthetic bench remains
      effectively tied with JAX-cuDNN.
- [x] H100 CE cleanup, 2026-06-13: removed the online-CE `target_logit` shared
      reduction and switched CE exp/log calls to explicit CUDA fast intrinsics
      under the existing `--use_fast_math` build. Added `ENTROPY_CE_THREADS` for
      A/B. Correctness still passes (`./build/gpt_h100 overfit`: loss 5.58089 ->
      0.00134). B=48 short profiled synthetic run reached 87.55ms / 434.5 TFLOP/s
      / 43.9% MFU, but the 20-step synthetic average is still ~89.9ms, so this
      is not yet a robust synthetic JAX-cuDNN beat.
- [x] H100 optimizer/memset pass, 2026-06-13: added a 4-wide AdamW kernel for
      aligned non-FP8 optimizer updates, with packed BF16x2 param stores, and
      switched the full-gradient zero from `zero_f32_k` to `cudaMemsetAsync`.
      Correctness still passes (`./build/gpt_h100 overfit`: loss 5.58089 ->
      0.00133). The AdamW profile region improved from ~1.67ms to ~1.50ms.
      Best short profiled B=48 synthetic run reached 86.81ms / 438.2 TFLOP/s /
      44.3% MFU. Same-day fp32-master JAX-cuDNN B=48 rerun measured 89.03ms;
      CUDA graph replay measured 88.65ms, so current CUDA beats same-day JAX.
      The older 87.74ms JAX log remains the long-wall synthetic target to beat
      robustly; the next lever is still fused LM-head CE or backward elementwise
      traffic.
- [x] H100 online-CE BF16x2 pass, 2026-06-13: added pairwise BF16x2 online CE
      stats/grad kernels and then fused the even-vocab stats+grad path into one
      row kernel (`ce_online_fwd_bwd2_k`). This keeps the same materialized
      logits/dlogits GEMM contract but cuts CE loop/store overhead. Correctness
      still passes (`./build/gpt_h100 overfit`: loss 5.58089 -> 0.00133).
      B=48 profile improved `B_ce` from ~6.4ms before pairwise CE to 4.37ms,
      with short synthetic timing 85.82ms / 443.2 TFLOP/s / 44.8% MFU.
      Long H100 B=48 timings now beat the older JAX-cuDNN synthetic log:
      `ENTROPY_GRAPH=1 ./build/gpt_h100 bench 48 20` measured 87.55ms, then
      86.64ms / 439.0 TFLOP/s / 44.4% MFU on the current fused-CE binary,
      versus the saved JAX-cuDNN B=48 reference 87.74ms
      (`logs/h100_jax_refs_fixed_20260611_175045.log`). Non-graph CUDA also
      measured 87.74ms in the same pass, effectively tied with the old log.
- [x] H100 backward/elementwise pass 1, 2026-06-13: added BF16x2 packed
      SwiGLU forward/backward kernels for even intermediate sizes
      (`swiglu_forward_gu2_k`, `swiglu_backward_gu2_k`) and wired both LM and
      vision-tower BF16 paths with scalar fallbacks. This is the first concrete
      item-2 step; RMSNorm/RoPE backward fusion remains open. Correctness still
      passes (`./build/gpt_h100 overfit`: loss 5.58089 -> 0.00133). B=48 H100
      profile improved `F_mlp+norm` to 14.99ms and `B_mlp_gemm` to 19.05ms,
      with short synthetic timing 83.42ms / 455.9 TFLOP/s / 46.1% MFU.
      Long graph timings improved to 84.68ms and 85.24ms
      (`ENTROPY_GRAPH=1 ./build/gpt_h100 bench 48 20`), or 45.4% and 45.1% MFU.
- [x] H100 backward/elementwise pass 2, 2026-06-13: added BF16x2 packed RMSNorm
      backward for the default atomic-partial path
      (`rmsnorm_dx_dweight_atomic_partial2_k`) with scalar fallback for odd
      channel counts and for `ENTROPY_RMS_ATOMIC=0`. Correctness still passes
      (`./build/gpt_h100 overfit`: loss 5.58089 -> 0.00134). B=48 profile
      improved `B_norm` from ~5.78ms to 4.99ms and short synthetic timing to
      82.73ms / 459.7 TFLOP/s / 46.5% MFU. Long graph timing measured 84.64ms
      / 449.4 TFLOP/s / 45.4% MFU; wall-time graph remains run-noise-limited,
      but the event profile confirms reduced norm traffic.
- [x] H100 DDP/item-4 baseline pass, 2026-06-13: measured 2-H100 DDP after the
      CE/SwiGLU/RMSNorm updates and changed the DDP RMSNorm default so sm90 H100
      keeps the packed atomic RMSNorm path while sm89 RTX Ada keeps the older
      atomic-free default unless overridden. With per-GPU B=16, world=2:
      accum=1 measured 37.9% MFU, `ENTROPY_DDP_OVERLAP_OPT=1` measured 38.2%;
      accum=4 measured 41.0%, and accum=8 measured 41.5% before the RMS default
      change. With packed atomic RMSNorm enabled/default on H100, accum=4 measured
      42.7% and accum=8 measured 43.3% MFU
      (`ENTROPY_ACCUM=8 ./build/gpt_h100 bench_ddp 16 4`: opt-step 236.65ms,
      857.2 aggregate TFLOP/s, 1.108M global tok/s). This is a useful distributed
      baseline, but still below the single-GPU graph MFU; next item-4 work is
      overlap/reduce-scatter style communication rather than optimizer overlap.
- [x] H100 DDP scale-fold pass, 2026-06-13: folded DDP mean scaling into the
      bf16 allreduce cast-back path (`cast_b2f_scale`) so bf16-reduce DDP avoids
      the post-communication full-gradient `scale_f32_k` pass when optimizer
      overlap is not already applying the scale. Correctness still passes
      (`./build/gpt_h100 overfit`: loss 5.58089 -> 0.00134). 2-H100 DDP with
      per-GPU B=16, accum=8 improved to opt-step 235.84ms, 860.1 aggregate
      TFLOP/s, 1.112M global tok/s, 43.5% MFU. A packed BF16x2 staging-cast
      experiment did not improve timing and was not kept.
- [ ] Real-data train graph replay, experimental: added `ENTROPY_TRAIN_GRAPH=1`
      for `train` mode. It captures two forward+backward CUDA graphs, one per
      existing double-buffered device input slot, and leaves AdamW outside replay
      so per-step bias correction remains correct. After graph instantiation the
      two static device input slots are refreshed, so the training loop does not
      reuse the warmup batch. This is the first item-5 path toward graph-friendly
      real-data training; it builds for sm90, but still needs correctness/timing
      on a compatible H100 runtime.
- [ ] FP8 scaling path, experimental: `ENTROPY_FP8_FWD=1` now supports scalar
      activation and weight quantization scales (`ENTROPY_FP8_ACT_SCALE`,
      `ENTROPY_FP8_W_SCALE`). Producer-fused RMSNorm/SwiGLU kernels write scaled
      E4M3 activation caches, cached FP8 weights are updated with the weight
      scale, and cuBLASLt FP8 GEMMs receive matching scalar dequant pointers.
      Defaults are 1.0, so the previous unscaled behavior remains the default.
      This is still not Transformer-Engine-grade dynamic scaling, but it removes
      the hard-coded naive cast assumption and gives the H100 FP8 path a real
      scaling control surface.
- [ ] LMCE fused-dX experiment: `ENTROPY_LMCE_CHUNK=<n> ENTROPY_LMCE_FUSED_DX=1`
      now computes LM-head input gradient directly from each chunk's logits,
      row softmax stats, and LM weights instead of feeding chunk dlogits through
      the dX GEMM. The dW path still uses the chunk dlogits Tensor Core GEMM, so
      this is not the final fused CE+dW+dX kernel, but it derisks the CE-derived
      dX formula without touching the default fast path.
- [ ] LMCE fused-dW experiment: `ENTROPY_LMCE_CHUNK=<n> ENTROPY_LMCE_FUSED_DW=1`
      adds a direct CE-derived chunk dW kernel. When combined with
      `ENTROPY_LMCE_FUSED_DX=1`, the chunked path computes loss, dW, and dX from
      chunk logits and row stats without allocating or materializing chunk
      dlogits. This is expected to be a derisk/reference path rather than a fast
      replacement for the current Tensor Core dW GEMM. Added CPU-only
      `./build/gpt_h100 lmce_check` before CUDA init to verify the direct dW/dX
      formulas against a materialized-dlogits reference across beta=0/nonzero,
      single-chunk, non-divisible vocab chunks, ignored rows, all-valid rows,
      and single-valid-row edge cases. The verifier also compares chunked online
      stats/loss against an independent full-vocab softmax reference. Added
      per-region `ENTROPY_PROF=1` marks for chunked LMCE stats, logits, dlogits/loss,
      dW, and dX so the next H100 run can isolate the direct-kernel bottleneck.
      Local container shells do not expose `/dev/nvidia*`; GPU timing should be run
      through Slurm. Verified the path on `itiger04` (RTX 6000 Ada, driver 560.28.03):
      `./build/gpt bench 2 2` ran at 18.19ms/step, and a tiny
      fused-DX/fused-DW LMCE profile smoke emitted the new `B_lmce_*` regions.
- [x] Chunked LMCE final-norm cleanup: the `ENTROPY_LMCE_CHUNK` path now keeps the
      LM-head input gradient in FP32 scratch through final RMSNorm backward,
      avoiding the previous `dln_accum -> bf16 dln` cast and immediate BF16
      reread. Added f32-dout RMSNorm backward kernels for this handoff and
      removed the now-redundant full `dln_accum` zero before chunk accumulation;
      default non-chunked LMCE remains unchanged.
