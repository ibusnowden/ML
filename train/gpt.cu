// entropy — modern (llama-style) GPT decoder trained in pure C/CUDA with cuBLASLt
// BF16 tensor-core GEMMs. Bias-free linears, RMSNorm (pre-norm), RoPE, SwiGLU,
// causal attention, untied LM head, AdamW (FP32 master weights). Real full
// forward + backward through every layer.
#include "gpt_kernels.cuh"
#include "attn_cudnn.cuh"
#include <nccl.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <unistd.h>
#include <algorithm>

// ---------------------------------------------------------------- config
struct Config {
  int V;      // vocab
  int C;      // d_model
  int L;      // layers
  int H;      // heads
  int I;      // ffn intermediate
  int T;      // seq len
  float rope_theta = 10000.0f;
  float eps = 1e-5f;
};

static int hd_of(const Config&c){ return c.C / c.H; }

// ---------------------------------------------------------------- profiler
// Event-based region timing on the compute stream (no host sync until report).
// mark(name) = "time from previous mark to here is attributed to `name`".
struct Prof {
  cudaStream_t s=0; bool on=false;
  std::vector<cudaEvent_t> evs; std::vector<std::string> names;
  void mark(const char* n){ if(!on) return; cudaEvent_t e; cudaEventCreate(&e);
    cudaEventRecord(e,s); evs.push_back(e); names.push_back(n); }
  void report(){ if(!on||evs.size()<2) return; cudaEventSynchronize(evs.back());
    std::vector<std::pair<std::string,float>> order; std::unordered_map<std::string,int> idx;
    for(size_t i=1;i<evs.size();i++){ float ms; cudaEventElapsedTime(&ms,evs[i-1],evs[i]);
      auto it=idx.find(names[i]); if(it==idx.end()){ idx[names[i]]=order.size(); order.push_back({names[i],ms}); }
      else order[it->second].second+=ms; }
    float tot=0; for(auto&kv:order) tot+=kv.second;
    printf("  -- profile (1 step) --\n");
    for(auto&kv:order) printf("    %-16s %6.2f ms  (%4.1f%%)\n",kv.first.c_str(),kv.second,100*kv.second/tot);
    printf("    %-16s %6.2f ms\n","TOTAL",tot);
    for(auto e:evs) cudaEventDestroy(e); evs.clear(); names.clear();
  }
};
static Prof* g_prof=nullptr;
#define PMARK(n) do{ if(g_prof) g_prof->mark(n); }while(0)

// ---------------------------------------------------------------- DDP (NCCL)
#define NCCL_CHECK(x) do{ ncclResult_t r=(x); if(r!=ncclSuccess){ printf("NCCL error %s @%s:%d\n", ncclGetErrorString(r),__FILE__,__LINE__); exit(1);} }while(0)
struct DDP { int rank=0, world=1, local=0; ncclComm_t comm; };

// per-layer linear param sizes
static long layer_stride(const Config&c){
  long C=c.C, I=c.I;
  return C + 3L*C*C + C*C + C + (long)I*C + (long)I*C + (long)C*I; // ln1,qkv,o,ln2,gate,up,down
}
struct ParamOff { long ln1,qkv,o,ln2,gate,up,down; };
static ParamOff layer_offsets(const Config&c){
  long C=c.C,I=c.I; ParamOff p;
  p.ln1=0; p.qkv=C; p.o=p.qkv+3L*C*C; p.ln2=p.o+(long)C*C;
  p.gate=p.ln2+C; p.up=p.gate+(long)I*C; p.down=p.up+(long)I*C;
  return p;
}
static long num_params(const Config&c){
  long C=c.C,V=c.V;
  return (long)V*C + (long)c.L*layer_stride(c) + C + (long)V*C; // wte + layers + lnf + lm
}
static long off_layers(const Config&c){ return (long)c.V*c.C; }
static long off_lnf(const Config&c){ return off_layers(c) + (long)c.L*layer_stride(c); }
static long off_lm(const Config&c){ return off_lnf(c) + c.C; }

// ---------------------------------------------------------------- model
struct GPT {
  Config cfg;
  LtCtx lt, lt_aux;
  cudaStream_t stream;
  // parameters (contiguous, identical layout across the 5 buffers)
  float* master;   // fp32 master weights
  bf16*  params;    // bf16 copy for GEMMs
  __nv_fp8_e4m3* params8=nullptr; // cached FP8 E4M3 weights for opt-in forward GEMMs
  float* grads;    // fp32 grads
  float* m; float* v; // adam state
  long np;
  // rope tables
  float* cosb; float* sinb;
  // activations
  bf16 *encoded;
  std::vector<bf16*> ln1_out,qkv,atty,resid1,ln2_out,gu,swiglu,resid2;
  std::vector<__nv_fp8_e4m3*> ln1_out8,ln2_out8,swiglu8;
  std::vector<bf16*> probs;
  std::vector<float*> rstd1,rstd2;
  bf16 *lnf_out; float* rstd_f;
  __nv_fp8_e4m3* lnf_out8=nullptr;
  bf16 *logits; float* losses;
  bf16 *lm_logits_chunk=nullptr, *lm_dlogits_chunk=nullptr;
  float *ce_rowmax=nullptr, *ce_rowsum=nullptr, *ce_target_logit=nullptr, *dln_accum=nullptr;
  int lmce_chunk=0;
  // backward scratch
  bf16 *dresid,*dresid1,*dln,*dln2,*datty,*dqkv,*dgu,*dswiglu,*dlogits;
  float* ds;
  // attention scratch ([B,H,T,hd] / [B,H,T,T])
  bf16 *qb,*kb,*vb,*ob,*dqb,*dkb,*dvb,*dPb,*dob;
  std::vector<float*> lse;   // per-layer logsumexp / cuDNN stats [B,H,T]
  float* Dbuf;               // dO·O reduction [B,H,T]
#ifdef USE_CUDNN
  CudnnAttn cattn;           // cuDNN fused flash attention, permute-free (strided)
  CudnnAttn cattn_packed;    // cuDNN with dense [B,H,T,hd] Q/K/V/O (caller permutes)
#endif
  // DDP: overlapped gradient allreduce on a comm stream
  DDP* ddp=nullptr; cudaStream_t cstream=0; cudaEvent_t ev=0, cev=0;
  bf16* gbf16=nullptr;   // bf16 grad staging for halved-bandwidth allreduce
  int bf16_reduce=1;
  float ddp_cast_scale=1.0f;
  int fp8_fwd=0;              // ENTROPY_FP8_FWD=1: producer-fused FP8 forward GEMMs
  float fp8_act_scale=1.0f, fp8_w_scale=1.0f;
  float *fp8_act_dequant=nullptr, *fp8_w_dequant=nullptr;
  float* norm_partial=nullptr;  // [R,C] scratch for atomic-free rmsnorm dweight
  int B; // max batch for allocated activations
  // Overlapped optimizer: AdamW on each param region runs on a side stream as
  // soon as that region's grads are final (a region's params aren't read again
  // after its own backward), hiding the memory-bound optimizer behind backward
  // compute — what XLA does by fusing the optimizer into the backward.
  cudaStream_t ostream=0, mstream=0; cudaEvent_t oev=0, mev=0;
  int overlap_opt=0;
  float o_lr=1e-4f,o_b1=0.9f,o_b2=0.95f,o_eps=1e-8f,o_wd=0.1f,o_bc1=1,o_bc2=1,o_grad_scale=1.0f;
};
#ifndef ENTROPY_NORM_R
#define ENTROPY_NORM_R 128
#endif
#define NORM_R ENTROPY_NORM_R

static bf16* dmalloc_bf16(long n){ bf16* p; CUDA_CHECK(cudaMalloc(&p,(size_t)n*sizeof(bf16))); return p; }
static float* dmalloc_f32(long n){ float* p; CUDA_CHECK(cudaMalloc(&p,(size_t)n*sizeof(float))); return p; }
static __nv_fp8_e4m3* dmalloc_fp8(long n){ __nv_fp8_e4m3* p; CUDA_CHECK(cudaMalloc(&p,(size_t)n)); return p; }
static int attn_mode();
static int env_int_raw(const char*n,int dflt){ const char*v=getenv(n); return v? atoi(v):dflt; }
static float env_float_raw(const char*n,float dflt){ const char*v=getenv(n); return v? (float)atof(v):dflt; }

static void gpt_build(GPT* g, Config cfg, int B){
  g->cfg=cfg; g->B=B;
  g->fp8_fwd=env_int_raw("ENTROPY_FP8_FWD",0);
  g->fp8_act_scale=env_float_raw("ENTROPY_FP8_ACT_SCALE",1.0f);
  g->fp8_w_scale=env_float_raw("ENTROPY_FP8_W_SCALE",1.0f);
  if(g->fp8_act_scale<=0.0f) g->fp8_act_scale=1.0f;
  if(g->fp8_w_scale<=0.0f) g->fp8_w_scale=1.0f;
  // Guard fixed-size stack/shared arrays in elementwise kernels.
  // RMS_VPT caps C at blockDim(256)*RMS_VPT=4096; FA_HD_MAX caps hd at 128 for
  // the flash path. Fail fast with a clear message rather than silently overflow.
  {
    int hd = hd_of(cfg);
    if(cfg.C > 256*RMS_VPT){
      printf("ERROR: C=%d exceeds 256*RMS_VPT=%d (RMSNorm forward kernels use a "
             "fixed vals[RMS_VPT] per-thread stash). Reduce C or raise RMS_VPT.\n",
             cfg.C, 256*RMS_VPT);
      exit(1);
    }
    int attn=attn_mode();
    if(attn==2 && hd > FA_HD_MAX){
      printf("ERROR: hd=C/H=%d exceeds FA_HD_MAX=%d (flash attention kernels use "
             "fixed-size qreg/acc/dq arrays). Use cuDNN (default) or mat attention, "
             "or raise FA_HD_MAX in gpt_kernels.cuh.\n", hd, FA_HD_MAX);
      exit(1);
    }
  }
  CUDA_CHECK(cudaStreamCreate(&g->stream));
  CUDA_CHECK(cudaStreamCreate(&g->cstream));
  CUDA_CHECK(cudaStreamCreate(&g->ostream));
  CUDA_CHECK(cudaStreamCreate(&g->mstream));
  CUDA_CHECK(cudaEventCreateWithFlags(&g->ev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&g->cev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&g->oev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&g->mev, cudaEventDisableTiming));
  lt_init(&g->lt);
  lt_init(&g->lt_aux);
  g->np = num_params(cfg);
  g->master=dmalloc_f32(g->np); g->params=dmalloc_bf16(g->np);
  if(g->fp8_fwd){
    g->params8=dmalloc_fp8(g->np);
    g->fp8_act_dequant=dmalloc_f32(1);
    g->fp8_w_dequant=dmalloc_f32(1);
    float h_act=1.0f/g->fp8_act_scale, h_w=1.0f/g->fp8_w_scale;
    CUDA_CHECK(cudaMemcpy(g->fp8_act_dequant,&h_act,sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g->fp8_w_dequant,&h_w,sizeof(float),cudaMemcpyHostToDevice));
  }
  g->grads=dmalloc_f32(g->np); g->m=dmalloc_f32(g->np); g->v=dmalloc_f32(g->np);
  CUDA_CHECK(cudaMemset(g->m,0,(size_t)g->np*4));
  CUDA_CHECK(cudaMemset(g->v,0,(size_t)g->np*4));
  int hd=hd_of(cfg), half=hd/2;
  g->cosb=dmalloc_f32((long)cfg.T*half); g->sinb=dmalloc_f32((long)cfg.T*half);
  rope_precompute_k<<<cfg.T, half, 0, g->stream>>>(g->cosb,g->sinb,cfg.T,hd,cfg.rope_theta);
  long BT=(long)B*cfg.T, C=cfg.C, I=cfg.I, V=cfg.V, H=cfg.H, T=cfg.T;
  long probs_sz=(long)B*H*T*T;
  int ATTN=attn_mode();
  bool need_materialized_attn = (ATTN==1);
  bool need_packed_attn = (ATTN==1 || ATTN==2 || ATTN==3);
  bool need_flash_scratch = (ATTN==2);
  g->encoded=dmalloc_bf16(BT*C);
  for(int l=0;l<cfg.L;l++){
    g->ln1_out.push_back(dmalloc_bf16(BT*C));
    g->ln1_out8.push_back(g->fp8_fwd ? dmalloc_fp8(BT*C) : nullptr);
    g->qkv.push_back(dmalloc_bf16(BT*3*C));
    g->probs.push_back(need_materialized_attn ? dmalloc_bf16(probs_sz) : nullptr);
    g->atty.push_back(dmalloc_bf16(BT*C));
    g->resid1.push_back(dmalloc_bf16(BT*C));
    g->ln2_out.push_back(dmalloc_bf16(BT*C));
    g->ln2_out8.push_back(g->fp8_fwd ? dmalloc_fp8(BT*C) : nullptr);
    g->gu.push_back(dmalloc_bf16(BT*2*I));   // fused gate|up: [BT,2I]
    g->swiglu.push_back(dmalloc_bf16(BT*I));
    g->swiglu8.push_back(g->fp8_fwd ? dmalloc_fp8(BT*I) : nullptr);
    g->resid2.push_back(dmalloc_bf16(BT*C));
    g->rstd1.push_back(dmalloc_f32(BT));
    g->rstd2.push_back(dmalloc_f32(BT));
    g->lse.push_back(dmalloc_f32((long)B*H*T));
  }
  g->lnf_out=dmalloc_bf16(BT*C); g->lnf_out8=g->fp8_fwd ? dmalloc_fp8(BT*C) : nullptr; g->rstd_f=dmalloc_f32(BT);
  bool use_lmce_chunk = env_int_raw("ENTROPY_LMCE_CHUNK",0)>0;
  bool use_lmce_no_dlogits = use_lmce_chunk &&
      env_int_raw("ENTROPY_LMCE_FUSED_DX",0) && env_int_raw("ENTROPY_LMCE_FUSED_DW",0);
  bool use_ce_online = !use_lmce_chunk && env_int_raw("ENTROPY_CE_ONLINE",1);
  g->logits=use_lmce_chunk ? nullptr : dmalloc_bf16(BT*V);
  g->losses=dmalloc_f32(BT);
  if(use_lmce_chunk || use_ce_online){
    g->ce_rowmax=dmalloc_f32(BT);
    g->ce_rowsum=dmalloc_f32(BT);
    g->ce_target_logit=dmalloc_f32(BT);
  }
  if(use_lmce_chunk){
    g->lmce_chunk=env_int_raw("ENTROPY_LMCE_CHUNK",4096);
    if(g->lmce_chunk<1) g->lmce_chunk=4096;
    if(g->lmce_chunk>V) g->lmce_chunk=V;
    g->lm_logits_chunk=dmalloc_bf16(BT*g->lmce_chunk);
    g->lm_dlogits_chunk=use_lmce_no_dlogits ? nullptr : dmalloc_bf16(BT*g->lmce_chunk);
    g->dln_accum=dmalloc_f32(BT*C);
  }
  g->dresid=dmalloc_bf16(BT*C); g->dresid1=dmalloc_bf16(BT*C);
  g->dln=dmalloc_bf16(BT*C); g->dln2=dmalloc_bf16(BT*C);
  g->datty=dmalloc_bf16(BT*C); g->dqkv=dmalloc_bf16(BT*3*C);
  g->dgu=dmalloc_bf16(BT*2*I); g->dswiglu=dmalloc_bf16(BT*I);
  g->dlogits=use_lmce_chunk ? nullptr : dmalloc_bf16(BT*V); g->ds=nullptr;
  g->qb=need_packed_attn ? dmalloc_bf16(BT*C) : nullptr;
  g->kb=need_packed_attn ? dmalloc_bf16(BT*C) : nullptr;
  g->vb=need_packed_attn ? dmalloc_bf16(BT*C) : nullptr;
  g->ob=need_packed_attn ? dmalloc_bf16(BT*C) : nullptr;
  g->dqb=need_packed_attn ? dmalloc_bf16(BT*C) : nullptr;
  g->dkb=need_packed_attn ? dmalloc_bf16(BT*C) : nullptr;
  g->dvb=need_packed_attn ? dmalloc_bf16(BT*C) : nullptr;
  g->dob=need_packed_attn ? dmalloc_bf16(BT*C) : nullptr;
  g->dPb=need_materialized_attn ? dmalloc_bf16(probs_sz) : nullptr;
  g->Dbuf=need_flash_scratch ? dmalloc_f32((long)B*H*T) : nullptr;
#ifdef USE_CUDNN
  g->cattn.init();
  g->cattn.build(B,H,T,hd_of(cfg),C,1.0f/sqrtf((float)hd_of(cfg)));
  if(getenv("ENTROPY_ATTN")&&!strcmp(getenv("ENTROPY_ATTN"),"packed")){
    g->cattn_packed.init();
    g->cattn_packed.build(B,H,T,hd_of(cfg),C,1.0f/sqrtf((float)hd_of(cfg)),/*packed=*/true);
  }
#endif
  g->gbf16=dmalloc_bf16(g->np);   // bf16 grad staging for DDP
  g->norm_partial=dmalloc_f32((long)NORM_R*C);
}
// RMSNorm backward: default uses row-wise dx plus atomic dweight partials after
// benchmarking faster on RTX 6000 Ada; ENTROPY_RMS_ATOMIC=0 restores the older
// atomic-free dx + coalesced dweight partial reduction.
// dx = add + d(rmsnorm) (add=nullptr -> overwrite, add==dx -> accumulate in place)
static void rmsnorm_bwd(cudaStream_t s, float* partial, bf16* dx, const bf16* add, float* dweight,
                        const bf16* dout, const bf16* x, const bf16* w, const float* rstd,
                        int N, int C){
  static int use_atomic = [](){
    const char* v=getenv("ENTROPY_RMS_ATOMIC");
    return v ? (v[0] != '0') : 1;
  }();
  if(use_atomic){
    zero_f32_k<<<ceil_div((long)NORM_R*C,256),256,0,s>>>(partial,(long)NORM_R*C);
    if((C & 1) == 0)
      rmsnorm_dx_dweight_atomic_partial2_k<<<N,256,0,s>>>(partial,dx,add,dout,x,w,rstd,N,C,NORM_R);
    else
      rmsnorm_dx_dweight_atomic_partial_k<<<N,256,0,s>>>(partial,dx,add,dout,x,w,rstd,N,C,NORM_R);
  } else {
    rmsnorm_dx_k<<<N,256,0,s>>>(dx,add,dout,x,w,rstd,N,C);
    dim3 gp(ceil_div(C,256), NORM_R);
    rmsnorm_dweight_partial_k<<<gp,256,0,s>>>(partial,dout,x,rstd,N,C,NORM_R);
  }
  reduce_cols_add_k<<<ceil_div(C,256),256,0,s>>>(dweight,partial,NORM_R,C);
}
static void rmsnorm_bwd_f32dout(cudaStream_t s, float* partial, bf16* dx, float* dweight,
                                const float* dout, const bf16* x, const bf16* w,
                                const float* rstd, int N, int C){
  zero_f32_k<<<ceil_div((long)NORM_R*C,256),256,0,s>>>(partial,(long)NORM_R*C);
  if((C & 1) == 0)
    rmsnorm_dx_dweight_f32dout_atomic_partial2_k<<<N,256,0,s>>>(partial,dx,dout,x,w,rstd,N,C,NORM_R);
  else
    rmsnorm_dx_dweight_f32dout_atomic_partial_k<<<N,256,0,s>>>(partial,dx,dout,x,w,rstd,N,C,NORM_R);
  reduce_cols_add_k<<<ceil_div(C,256),256,0,s>>>(dweight,partial,NORM_R,C);
}

// parameter accessors (bf16 + grad fp32)
static bf16* P(GPT*g,long off){ return g->params+off; }
static __nv_fp8_e4m3* P8(GPT*g,long off){ return g->params8+off; }
static float* G(GPT*g,long off){ return g->grads+off; }
static long LP(const Config&c,int l,long intra){ return off_layers(c)+ (long)l*layer_stride(c)+intra; }

// Overlapped allreduce of a contiguous grad slice on the comm stream: comm waits
// for the compute that just produced this slice, then runs concurrently with the
// next layers' backward compute.
static void areduce_slice(GPT* g, long off, long n){
  if(!g->ddp || g->ddp->world<=1) return;
  CUDA_CHECK(cudaEventRecord(g->ev, g->stream));
  CUDA_CHECK(cudaStreamWaitEvent(g->cstream, g->ev, 0));
  if(g->bf16_reduce && g->gbf16){
    // halve PCIe transfer: cast fp32->bf16, allreduce in bf16, cast back
    cast_f2b<<<ceil_div(n,256),256,0,g->cstream>>>(g->gbf16+off, g->grads+off, n);
    NCCL_CHECK(ncclAllReduce(g->gbf16+off, g->gbf16+off, (size_t)n, ncclBfloat16, ncclSum,
                             g->ddp->comm, g->cstream));
    if(g->ddp_cast_scale != 1.0f){
      cast_b2f_scale<<<ceil_div(n,256),256,0,g->cstream>>>(g->grads+off, g->gbf16+off, n, g->ddp_cast_scale);
    } else {
      cast_b2f<<<ceil_div(n,256),256,0,g->cstream>>>(g->grads+off, g->gbf16+off, n);
    }
  } else {
    NCCL_CHECK(ncclAllReduce(g->grads+off, g->grads+off, (size_t)n, ncclFloat, ncclSum,
                             g->ddp->comm, g->cstream));
  }
}
// After backward: compute stream waits for all comm, then mean-scale grads across
// data-parallel ranks and gradient-accumulation microbatches.
static void ddp_finish(GPT* g, int accum=1){
  int world = (g->ddp && g->ddp->world > 1) ? g->ddp->world : 1;
  if(world > 1){
    CUDA_CHECK(cudaEventRecord(g->cev, g->cstream));
    CUDA_CHECK(cudaStreamWaitEvent(g->stream, g->cev, 0));
  }
  int a = accum > 0 ? accum : 1;
  float denom = (float)(world * a);
  bool scaled_in_bf16_cast = g->bf16_reduce && g->gbf16 && g->ddp_cast_scale != 1.0f;
  if(denom != 1.0f && !scaled_in_bf16_cast && !(g->overlap_opt && g->o_grad_scale != 1.0f))
    scale_f32_k<<<ceil_div(g->np,256),256,0,g->stream>>>(g->grads, g->np, 1.0f/denom);
}

// Overlapped AdamW on one contiguous param region, on the optimizer side stream.
// Records that the region's grads are final (compute stream), the optimizer stream
// waits on it, then runs the update concurrently with the rest of backward. Safe
// because a region's params are not read again after its own backward completes.
static void opt_slice(GPT* g, long off, long n, bool allow=true){
  if(!g->overlap_opt || !allow) return;
  int world = (g->ddp && g->ddp->world > 1) ? g->ddp->world : 1;
  if(world > 1){
    CUDA_CHECK(cudaEventRecord(g->oev, g->cstream));
    CUDA_CHECK(cudaStreamWaitEvent(g->ostream, g->oev, 0));
  } else {
    CUDA_CHECK(cudaEventRecord(g->oev, g->stream));
    CUDA_CHECK(cudaStreamWaitEvent(g->ostream, g->oev, 0));
  }
  if(g->fp8_fwd)
    adamw_fp8_k<<<ceil_div(n,256),256,0,g->ostream>>>(g->master+off,g->params+off,g->params8+off,g->grads+off,
        g->m+off,g->v+off,n, g->o_lr,g->o_b1,g->o_b2,g->o_eps,g->o_wd,g->o_bc1,g->o_bc2,g->o_grad_scale,g->fp8_w_scale);
  else if(((off | n) & 3L) == 0)
    adamw4_k<<<ceil_div(n/4,256),256,0,g->ostream>>>(g->master+off,g->params+off,g->grads+off,
        g->m+off,g->v+off,n/4, g->o_lr,g->o_b1,g->o_b2,g->o_eps,g->o_wd,g->o_bc1,g->o_bc2,g->o_grad_scale);
  else
    adamw_k<<<ceil_div(n,256),256,0,g->ostream>>>(g->master+off,g->params+off,g->grads+off,
        g->m+off,g->v+off,n, g->o_lr,g->o_b1,g->o_b2,g->o_eps,g->o_wd,g->o_bc1,g->o_bc2,g->o_grad_scale);
}

static int env_flag(const char*n){ const char*v=getenv(n); return v&&v[0]&&v[0]!='0'; }
static int ce_threads(){
  static int t=[](){
    int v=env_int_raw("ENTROPY_CE_THREADS",0);
    if(v<=0){
      v = 256;
    }
    if(v>=1024) return 1024;
    if(v>=512) return 512;
    if(v>=256) return 256;
    return 128;
  }();
  return t;
}
// attention backend: 0=cuDNN flash (default), 1=materialized tensor-core, 2=scalar flash
static int attn_mode(){
  const char*v=getenv("ENTROPY_ATTN");
  if(v&&!strcmp(v,"mat")) return 1;
  if(v&&!strcmp(v,"flash")) return 2;
  if(v&&!strcmp(v,"packed")) return 3;
#ifdef USE_CUDNN
  return 0;
#else
  return 1;
#endif
}

static void lmhead_ce_backward_chunked(GPT* g, const int* d_tgt, int B, int T, int valid_tokens,
                                       float wgrad_beta){
  Config&c=g->cfg; int C=c.C,V=c.V; long BT=(long)B*T; cudaStream_t s=g->stream;
  int cet=ce_threads();
  static int LMCE_FUSED_DX=env_flag("ENTROPY_LMCE_FUSED_DX");
  static int LMCE_FUSED_DW=env_flag("ENTROPY_LMCE_FUSED_DW");
  int valid = valid_tokens > 0 ? valid_tokens : (int)BT;
  if(valid <= 0){ printf("invalid valid token count %d\n", valid); exit(1); }
  float dscale = 1.0f/(float)valid;
  int base_chunk = g->lmce_chunk;
  ce_stats_init_k<<<ceil_div(BT,256),256,0,s>>>(g->ce_rowmax,g->ce_rowsum,g->ce_target_logit,(int)BT);
  for(int off=0; off<V; off+=base_chunk){
    int n = std::min(base_chunk,V-off);
    linear_forward(&g->lt,s,g->lnf_out,P(g,off_lm(c)+(long)off*C),g->lm_logits_chunk,(int)BT,n,C);
    ce_chunk_online_stats_k<<<BT,cet,0,s>>>(g->ce_rowmax,g->ce_rowsum,g->ce_target_logit,
                                            g->lm_logits_chunk,d_tgt,(int)BT,n,off);
  }
  PMARK("B_lmce_stats");
  bool skip_dlogits = LMCE_FUSED_DX && LMCE_FUSED_DW;
  if(skip_dlogits){
    ce_loss_from_stats_k<<<ceil_div(BT,256),256,0,s>>>(g->losses,d_tgt,g->ce_rowmax,g->ce_rowsum,g->ce_target_logit,(int)BT);
    PMARK("B_lmce_loss");
  }
  for(int off=0; off<V; off+=base_chunk){
    int n = std::min(base_chunk,V-off);
    linear_forward(&g->lt,s,g->lnf_out,P(g,off_lm(c)+(long)off*C),g->lm_logits_chunk,(int)BT,n,C);
    PMARK("B_lmce_logits");
    if(!skip_dlogits){
      ce_chunk_grad_loss_k<<<BT,cet,0,s>>>(g->losses,g->lm_dlogits_chunk,g->lm_logits_chunk,d_tgt,
                                          g->ce_rowmax,g->ce_rowsum,g->ce_target_logit,
                                          (int)BT,n,off,dscale);
      PMARK("B_lmce_grad");
    }
    if(LMCE_FUSED_DW)
      ce_chunk_dweight_accum_k<<<dim3(n,ceil_div(C,128)),128,0,s>>>(G(g,off_lm(c)+(long)off*C),
                                                                    g->lm_logits_chunk,g->lnf_out,d_tgt,
                                                                    g->ce_rowmax,g->ce_rowsum,
                                                                    (int)BT,n,C,off,dscale,wgrad_beta);
    else
      linear_backward_weight(&g->lt,s,g->lm_dlogits_chunk,g->lnf_out,
                             G(g,off_lm(c)+(long)off*C),(int)BT,n,C,wgrad_beta);
    PMARK("B_lmce_dw");
    if(LMCE_FUSED_DX)
      ce_chunk_dx_accum_k<<<BT,256,0,s>>>(g->dln_accum,g->lm_logits_chunk,P(g,off_lm(c)+(long)off*C),
                                          d_tgt,g->ce_rowmax,g->ce_rowsum,(int)BT,n,C,off,dscale,
                                          off==0 ? 1 : 0);
    else
      linear_backward_inp_accum_f32(&g->lt,s,g->lm_dlogits_chunk,P(g,off_lm(c)+(long)off*C),
                                    g->dln_accum,(int)BT,n,C, off==0 ? 0.0f : 1.0f);
    PMARK("B_lmce_dx");
  }
}
// ---------------------------------------------------------------- forward
// returns mean loss over valid tokens (if targets!=null)
static void gpt_forward(GPT* g, const int* d_ids, int B, int T,
                        const bf16* prefix=nullptr, int n_pre=0){
  Config&c=g->cfg; int C=c.C,I=c.I,H=c.H,V=c.V,hd=hd_of(c),half=hd/2;
  long BT=(long)B*T; cudaStream_t s=g->stream;
  ParamOff po=layer_offsets(c);
  int blk=256;
  static int SKIP_ATTN=env_flag("ENTROPY_SKIP_ATTN");
  static int SKIP_MLP =env_flag("ENTROPY_SKIP_MLP");
  static int SKIP_HEAD=env_flag("ENTROPY_SKIP_HEAD");
  static int SKIP_ROPE=env_flag("ENTROPY_SKIP_ROPE");
  static int SKIP_NORM=env_flag("ENTROPY_SKIP_NORM");
  static int SKIP_SWIGLU=env_flag("ENTROPY_SKIP_SWIGLU");
  static int ATTN=attn_mode();
  bool use_fp8 = g->fp8_fwd && !SKIP_NORM && !SKIP_SWIGLU;
  // embedding (+ optional image-token prefix for multimodal)
  encoder_forward_k<<<BT,256,0,s>>>(g->encoded,d_ids,P(g,0),BT,C);
  if(prefix && n_pre>0)
    set_prefix_embeds_k<<<ceil_div((long)B*n_pre*C,256),256,0,s>>>(g->encoded,prefix,B,T,n_pre,C);
  bf16* x=g->encoded;
  float scale=1.0f/sqrtf((float)hd);
  // precompute first layer's input RMSNorm; subsequent ones are produced fused
  // with the previous layer's MLP residual add (cross-layer fusion).
  if(!SKIP_NORM){
    if(use_fp8)
      rmsnorm_forward_fp8_k<<<BT,256,0,s>>>(g->ln1_out[0],g->ln1_out8[0],g->rstd1[0],x,P(g,LP(c,0,po.ln1)),BT,C,c.eps,g->fp8_act_scale);
    else
      rmsnorm_forward_k<<<BT,256,0,s>>>(g->ln1_out[0],g->rstd1[0],x,P(g,LP(c,0,po.ln1)),BT,C,c.eps);
  }
  for(int l=0;l<c.L;l++){
    bf16* ln1=g->ln1_out[l];
    // qkv
    if(use_fp8) linear_forward_fp8(&g->lt,s,g->ln1_out8[l],P8(g,LP(c,l,po.qkv)),g->qkv[l],BT,3*C,C,g->fp8_act_dequant,g->fp8_w_dequant);
    else linear_forward(&g->lt,s,ln1,P(g,LP(c,l,po.qkv)),g->qkv[l],BT,3*C,C);
    // rope on q,k slices of qkv[B,T,3C]: q at offset 0, k at offset C; stride 3C.
    // division-free, ~256 threads/block via 2D block(hd/2, HPB).
    int HPB=(256/half)>0?(256/half):1; if(HPB>H)HPB=H;
    dim3 rope_block(half,HPB), rope_grid(T,ceil_div(H,HPB),B);
    if(!SKIP_ROPE)
      rope_apply_qk_k<<<rope_grid,rope_block,0,s>>>(g->qkv[l],g->cosb,g->sinb,H,hd,3*C,T);
    PMARK("F_qkv+rope");
    // attention
    long ne=(long)B*T*H*hd;
    if(SKIP_ATTN){
      copy_k<<<ceil_div(BT*C,blk),blk,0,s>>>(g->atty[l],g->ln1_out[l],BT*C);
    }
#ifdef USE_CUDNN
    else if(ATTN==0){
      // cuDNN fused flash, permute-free: reads q,k,v from interleaved qkv, writes O->atty
      g->cattn.forward(s, g->qkv[l], g->qkv[l]+C, g->qkv[l]+2*C, g->atty[l], g->lse[l]);
    }
    else if(ATTN==3){
      // cuDNN with dense Q/K/V: permute -> cuDNN -> permute O back
      permute_qkv_k<<<ceil_div(ne,256),256,0,s>>>(g->qb,g->kb,g->vb,g->qkv[l],B,T,H,hd);
      g->cattn_packed.forward(s, g->qb, g->kb, g->vb, g->ob, g->lse[l]);
      permute_o_k<<<ceil_div(ne,256),256,0,s>>>(g->atty[l],g->ob,B,T,H,hd);
    }
#endif
    else {
      permute_qkv_k<<<ceil_div(ne,256),256,0,s>>>(g->qb,g->kb,g->vb,g->qkv[l],B,T,H,hd);
      if(ATTN==2){
        const int BR=64,BC=64; dim3 fg(B*H,(T+BR-1)/BR); size_t shm=(size_t)2*BC*hd*sizeof(float);
        flash_fwd_k<BR,BC><<<fg,BR,shm,s>>>(g->ob,g->lse[l],g->qb,g->kb,g->vb,B,H,T,hd,scale);
      } else {
        bmm(&g->lt,s,CUBLAS_OP_T,CUBLAS_OP_N,T,T,hd,scale, g->kb,hd,(long)T*hd, g->qb,hd,(long)T*hd,
            0.0f, g->probs[l],T,(long)T*T, B*H);
        softmax_causal_fwd_k<<<(long)B*H*T,256,(size_t)T*sizeof(float),s>>>(g->probs[l],g->probs[l],B*H,T);
        bmm(&g->lt,s,CUBLAS_OP_N,CUBLAS_OP_N,hd,T,T,1.0f, g->vb,hd,(long)T*hd, g->probs[l],T,(long)T*T,
            0.0f, g->ob,hd,(long)T*hd, B*H);
      }
      permute_o_k<<<ceil_div(ne,256),256,0,s>>>(g->atty[l],g->ob,B,T,H,hd);
    }
    PMARK("F_attn");
    // o-proj + fused (resid1 = x + attproj ; ln2 = rmsnorm(resid1))
    linear_forward(&g->lt,s,g->atty[l],P(g,LP(c,l,po.o)),g->dln,BT,C,C); // attproj into dln
    if(SKIP_NORM) residual_forward_k<<<ceil_div(BT*C,blk),blk,0,s>>>(g->resid1[l],x,g->dln,BT*C);
    else if(use_fp8) add_rmsnorm_fwd_fp8_k<<<BT,256,0,s>>>(g->resid1[l],g->ln2_out[l],g->ln2_out8[l],g->rstd2[l],x,g->dln,P(g,LP(c,l,po.ln2)),BT,C,c.eps,g->fp8_act_scale);
    else add_rmsnorm_fwd_k<<<BT,256,0,s>>>(g->resid1[l],g->ln2_out[l],g->rstd2[l],x,g->dln,P(g,LP(c,l,po.ln2)),BT,C,c.eps);
    // next layer's input norm (or final norm) is produced fused with this MLP's residual
    bf16*  nnorm = (l<c.L-1)? g->ln1_out[l+1] : g->lnf_out;
    __nv_fp8_e4m3* nnorm8 = (l<c.L-1)? g->ln1_out8[l+1] : g->lnf_out8;
    float* nrstd = (l<c.L-1)? g->rstd1[l+1]   : g->rstd_f;
    bf16*  nw    = (l<c.L-1)? P(g,LP(c,l+1,po.ln1)) : P(g,off_lnf(c));
    if(!SKIP_MLP){
      // fused gate|up: single GEMM ln2 @ [W_gate;W_up]^T -> gu[BT,2I] (gate|up adjacent in params)
      if(use_fp8) linear_forward_fp8(&g->lt,s,g->ln2_out8[l],P8(g,LP(c,l,po.gate)),g->gu[l],BT,2*I,C,g->fp8_act_dequant,g->fp8_w_dequant);
      else linear_forward(&g->lt,s,g->ln2_out[l],P(g,LP(c,l,po.gate)),g->gu[l],BT,2*I,C);
      if(!SKIP_SWIGLU){
        if(use_fp8) swiglu_forward_gu_fp8_k<<<dim3(BT,ceil_div(I,blk)),blk,0,s>>>(g->swiglu[l],g->swiglu8[l],g->gu[l],BT,I,g->fp8_act_scale);
        else if((I & 1) == 0) swiglu_forward_gu2_k<<<dim3(BT,ceil_div(I/2,blk)),blk,0,s>>>(g->swiglu[l],g->gu[l],BT,I);
        else swiglu_forward_gu_k<<<dim3(BT,ceil_div(I,blk)),blk,0,s>>>(g->swiglu[l],g->gu[l],BT,I);
      }
      if(use_fp8) linear_forward_fp8(&g->lt,s,g->swiglu8[l],P8(g,LP(c,l,po.down)),g->dln,BT,C,I,g->fp8_act_dequant,g->fp8_w_dequant);
      else linear_forward(&g->lt,s,g->swiglu[l],P(g,LP(c,l,po.down)),g->dln,BT,C,I); // mlp out into dln
      // fused: resid2 = resid1 + mlp_out ; next_norm = rmsnorm(resid2)
      if(SKIP_NORM) residual_forward_k<<<ceil_div(BT*C,blk),blk,0,s>>>(g->resid2[l],g->resid1[l],g->dln,BT*C);
      else if(use_fp8) add_rmsnorm_fwd_fp8_k<<<BT,256,0,s>>>(g->resid2[l],nnorm,nnorm8,nrstd,g->resid1[l],g->dln,nw,BT,C,c.eps,g->fp8_act_scale);
      else add_rmsnorm_fwd_k<<<BT,256,0,s>>>(g->resid2[l],nnorm,nrstd,g->resid1[l],g->dln,nw,BT,C,c.eps);
    } else {
      copy_k<<<ceil_div(BT*C,blk),blk,0,s>>>(g->resid2[l],g->resid1[l],BT*C);
      if(use_fp8) rmsnorm_forward_fp8_k<<<BT,256,0,s>>>(nnorm,nnorm8,nrstd,g->resid2[l],nw,BT,C,c.eps,g->fp8_act_scale);
      else rmsnorm_forward_k<<<BT,256,0,s>>>(nnorm,nrstd,g->resid2[l],nw,BT,C,c.eps);
    }
    x=g->resid2[l];
    PMARK("F_mlp+norm");
  }
  if(!SKIP_HEAD && !g->lmce_chunk){
    if(use_fp8) linear_forward_fp8(&g->lt,s,g->lnf_out8,P8(g,off_lm(c)),g->logits,BT,V,C,g->fp8_act_dequant,g->fp8_w_dequant);
    else linear_forward(&g->lt,s,g->lnf_out,P(g,off_lm(c)),g->logits,BT,V,C);
  }
  PMARK("F_lmhead");
}

// ---------------------------------------------------------------- backward
static void gpt_backward(GPT* g, const int* d_ids, const int* d_tgt, int B, int T,
                         bool zero_grads=true, bool do_reduce=true, int n_pre=0,
                         int valid_tokens=0){
  Config&c=g->cfg; int C=c.C,I=c.I,H=c.H,V=c.V,hd=hd_of(c);
  long BT=(long)B*T; cudaStream_t s=g->stream;
  ParamOff po=layer_offsets(c);
  int blk=256; float scale=1.0f/sqrtf((float)hd);
  static int ATTN=attn_mode();
  static int SKIP_NORM_BWD=env_flag("ENTROPY_SKIP_NORM_BWD");
  static int SKIP_GEMM_BWD=env_flag("ENTROPY_SKIP_GEMM_BWD");
  static int MLP_DUAL=env_flag("ENTROPY_MLP_DUAL");
  static int BETA0_GRADS=env_flag("ENTROPY_BETA0_GRADS");
  long lstride=layer_stride(c);
  bool opt_now = (!g->ddp || g->ddp->world <= 1 || do_reduce);
  // zero all grads (skip between gradient-accumulation microbatches)
  float wgrad_beta = (BETA0_GRADS && zero_grads) ? 0.0f : 1.0f;
  if(zero_grads){
    if(BETA0_GRADS){
      zero_f32_k<<<ceil_div((long)V*C,256),256,0,s>>>(G(g,0),(long)V*C); // sparse embedding grad
      zero_f32_k<<<ceil_div(C,256),256,0,s>>>(G(g,off_lnf(c)),C);
      for(int zl=0; zl<c.L; zl++){
        zero_f32_k<<<ceil_div(C,256),256,0,s>>>(G(g,LP(c,zl,po.ln1)),C);
        zero_f32_k<<<ceil_div(C,256),256,0,s>>>(G(g,LP(c,zl,po.ln2)),C);
      }
    } else {
      CUDA_CHECK(cudaMemsetAsync(g->grads,0,(size_t)g->np*sizeof(float),s));
    }
  }
  // fused CE -> losses + dlogits
  int valid = valid_tokens > 0 ? valid_tokens : (n_pre > 0 ? (int)(BT - (long)B*n_pre) : (int)BT);
  if(valid <= 0){ printf("invalid valid token count %d\n", valid); exit(1); }
  float dscale = 1.0f/(float)valid;
  int cet=ce_threads();
  if(g->lmce_chunk){
    (void)dscale;
    lmhead_ce_backward_chunked(g,d_tgt,B,T,valid,wgrad_beta);
  } else {
    if(g->ce_rowmax){
      if((V & 1) == 0){
        ce_online_fwd_bwd2_k<<<BT,cet,0,s>>>(g->losses,g->dlogits,g->logits,d_tgt,
                                             BT,V,dscale);
      } else {
        ce_online_stats_k<<<BT,cet,0,s>>>(g->ce_rowmax,g->ce_rowsum,g->ce_target_logit,
                                          g->logits,d_tgt,BT,V);
        ce_online_grad_k<<<BT,cet,0,s>>>(g->losses,g->dlogits,g->logits,d_tgt,
                                         g->ce_rowmax,g->ce_rowsum,g->ce_target_logit,
                                         BT,V,dscale);
      }
    } else {
      crossentropy_forward_backward_k<<<BT,cet,0,s>>>(g->losses,g->dlogits,g->logits,d_tgt,BT,V,dscale);
    }
    PMARK("B_ce");
    // lm head backward
    linear_backward_weight(&g->lt,s,g->dlogits,g->lnf_out,G(g,off_lm(c)),BT,V,C,wgrad_beta);
    linear_backward_inp(&g->lt,s,g->dlogits,P(g,off_lm(c)),g->dln,BT,V,C,0.0f); // dln = d_lnf_out
    PMARK("B_lmhead");
  }
  // final rmsnorm backward into dresid (add=nullptr -> overwrite, no memset needed)
  bf16* xlast = (c.L>0)? g->resid2[c.L-1] : g->encoded;
  if(!SKIP_NORM_BWD && g->lmce_chunk) rmsnorm_bwd_f32dout(s,g->norm_partial,g->dresid,G(g,off_lnf(c)),g->dln_accum,xlast,P(g,off_lnf(c)),g->rstd_f,BT,C);
  else if(!SKIP_NORM_BWD) rmsnorm_bwd(s,g->norm_partial,g->dresid,nullptr,G(g,off_lnf(c)),g->dln,xlast,P(g,off_lnf(c)),g->rstd_f,BT,C);
  else CUDA_CHECK(cudaMemsetAsync(g->dresid,0,(size_t)BT*C*sizeof(bf16),s));
  // lm_head + lnf grads ready -> allreduce now, overlapping the whole layer loop
  if(do_reduce) areduce_slice(g, off_lm(c), (long)V*C);
  opt_slice(g, off_lm(c), (long)V*C, opt_now);
  if(do_reduce) areduce_slice(g, off_lnf(c), C);
  opt_slice(g, off_lnf(c), C, opt_now);
  for(int l=c.L-1;l>=0;l--){
    bf16* x_in = (l==0)? g->encoded : g->resid2[l-1];
    // --- MLP branch --- (dresid holds grad wrt resid2 = d_mlpout; passthrough to
    // resid1 is folded into rmsnorm2's dx write via add=dresid below, no copy_k)
    // down backward (dresid == d_mlpout)
    if(MLP_DUAL){
      CUDA_CHECK(cudaEventRecord(g->mev,s));
      CUDA_CHECK(cudaStreamWaitEvent(g->mstream,g->mev,0));
      linear_backward_weight(&g->lt,s,g->dresid,g->swiglu[l],G(g,LP(c,l,po.down)),BT,C,I,wgrad_beta);
      linear_backward_inp(&g->lt_aux,g->mstream,g->dresid,P(g,LP(c,l,po.down)),g->dswiglu,BT,C,I,0.0f);
      CUDA_CHECK(cudaEventRecord(g->mev,g->mstream));
      CUDA_CHECK(cudaStreamWaitEvent(s,g->mev,0));
    } else {
      linear_backward_weight(&g->lt,s,g->dresid,g->swiglu[l],G(g,LP(c,l,po.down)),BT,C,I,wgrad_beta);
      linear_backward_inp(&g->lt,s,g->dresid,P(g,LP(c,l,po.down)),g->dswiglu,BT,C,I,0.0f);
    }
    if((I & 1) == 0) swiglu_backward_gu2_k<<<dim3(BT,ceil_div(I/2,blk)),blk,0,s>>>(g->dgu,g->dswiglu,g->gu[l],BT,I);
    else swiglu_backward_gu_k<<<dim3(BT,ceil_div(I,blk)),blk,0,s>>>(g->dgu,g->dswiglu,g->gu[l],BT,I);
    // fused gate|up backward: one dW GEMM (-> contiguous gate|up grads) + one dInp
    // GEMM (dln2 = dgu @ [W_gate;W_up]), reading ln2 once instead of twice.
    if(MLP_DUAL){
      CUDA_CHECK(cudaEventRecord(g->mev,s));
      CUDA_CHECK(cudaStreamWaitEvent(g->mstream,g->mev,0));
      linear_backward_weight(&g->lt,s,g->dgu,g->ln2_out[l],G(g,LP(c,l,po.gate)),BT,2*I,C,wgrad_beta);
      linear_backward_inp(&g->lt_aux,g->mstream,g->dgu,P(g,LP(c,l,po.gate)),g->dln2,BT,2*I,C,0.0f);
      CUDA_CHECK(cudaEventRecord(g->mev,g->mstream));
      CUDA_CHECK(cudaStreamWaitEvent(s,g->mev,0));
    } else {
      linear_backward_weight(&g->lt,s,g->dgu,g->ln2_out[l],G(g,LP(c,l,po.gate)),BT,2*I,C,wgrad_beta);
      linear_backward_inp(&g->lt,s,g->dgu,P(g,LP(c,l,po.gate)),g->dln2,BT,2*I,C,0.0f);
    }
    PMARK("B_mlp_gemm");
    // rmsnorm2 backward: dresid1 = dresid(passthrough) + d(norm), no copy_k
    if(!SKIP_NORM_BWD) rmsnorm_bwd(s,g->norm_partial,g->dresid1,g->dresid,G(g,LP(c,l,po.ln2)),g->dln2,g->resid1[l],P(g,LP(c,l,po.ln2)),g->rstd2[l],BT,C);
    else copy_k<<<ceil_div(BT*C,blk),blk,0,s>>>(g->dresid1,g->dresid,BT*C);
    PMARK("B_norm");
    // --- attention branch --- (dresid1 holds grad wrt resid1 = d_attproj;
    // passthrough to x_in folded into rmsnorm1's dx write via add=dresid1 below)
    // o-proj backward (dresid1 == d_attproj)
    linear_backward_weight(&g->lt,s,g->dresid1,g->atty[l],G(g,LP(c,l,po.o)),BT,C,C,wgrad_beta);
    linear_backward_inp(&g->lt,s,g->dresid1,P(g,LP(c,l,po.o)),g->datty,BT,C,C,0.0f);
    // attention backward
    long ne=(long)B*T*H*hd;
#ifdef USE_CUDNN
    if(ATTN==0){
      // permute-free: cuDNN reads Q,K,V from qkv[l], O from atty[l], dO from datty,
      // and writes dQ,dK,dV directly into the interleaved dqkv slices.
      g->cattn.backward(s, g->qkv[l],g->qkv[l]+C,g->qkv[l]+2*C, g->atty[l], g->datty, g->lse[l],
                        g->dqkv, g->dqkv+C, g->dqkv+2*C);
    } else if(ATTN==3){
      // dense cuDNN: permute Q,K,V and O,dO -> cuDNN bwd -> unpermute dQ,dK,dV
      permute_qkv_k<<<ceil_div(ne,256),256,0,s>>>(g->qb,g->kb,g->vb,g->qkv[l],B,T,H,hd);
      permute_o_bwd_k<<<ceil_div(ne,256),256,0,s>>>(g->dob,g->datty,B,T,H,hd);
      permute_o_bwd_k<<<ceil_div(ne,256),256,0,s>>>(g->ob,g->atty[l],B,T,H,hd);
      g->cattn_packed.backward(s, g->qb,g->kb,g->vb, g->ob, g->dob, g->lse[l],
                               g->dqb, g->dkb, g->dvb);
      unpermute_dqkv_k<<<ceil_div(ne,256),256,0,s>>>(g->dqkv,g->dqb,g->dkb,g->dvb,B,T,H,hd);
    } else
#endif
    {
    permute_qkv_k<<<ceil_div(ne,256),256,0,s>>>(g->qb,g->kb,g->vb,g->qkv[l],B,T,H,hd); // recover Q,K,V
    permute_o_bwd_k<<<ceil_div(ne,256),256,0,s>>>(g->dob,g->datty,B,T,H,hd);            // dO into dob
    if(ATTN==2){
      permute_o_bwd_k<<<ceil_div(ne,256),256,0,s>>>(g->ob,g->atty[l],B,T,H,hd);         // O into ob
      flash_dO_O_k<<<(long)B*H*T,128,0,s>>>(g->Dbuf,g->dob,g->ob,B*H,T,hd);
      dim3 fg(B*H,T);
      flash_bwd_dq_k<<<fg,128,0,s>>>(g->dqb,g->qb,g->kb,g->vb,g->dob,g->lse[l],g->Dbuf,B,H,T,hd,scale);
      flash_bwd_dkv_k<<<fg,128,0,s>>>(g->dkb,g->dvb,g->qb,g->kb,g->vb,g->dob,g->lse[l],g->Dbuf,B,H,T,hd,scale);
    } else {
      // dV = P^T @ dO ; dP = dO @ V^T ; dS = softmax_bwd ; dQ = scale*dS@K ; dK = scale*dS^T@Q
      bmm(&g->lt,s,CUBLAS_OP_N,CUBLAS_OP_T,hd,T,T,1.0f, g->dob,hd,(long)T*hd, g->probs[l],T,(long)T*T,
          0.0f, g->dvb,hd,(long)T*hd, B*H);
      bmm(&g->lt,s,CUBLAS_OP_T,CUBLAS_OP_N,T,T,hd,1.0f, g->vb,hd,(long)T*hd, g->dob,hd,(long)T*hd,
          0.0f, g->dPb,T,(long)T*T, B*H);
      softmax_causal_bwd_k<<<(long)B*H*T,256,0,s>>>(g->dPb,g->dPb,g->probs[l],B*H,T);
      bmm(&g->lt,s,CUBLAS_OP_N,CUBLAS_OP_N,hd,T,T,scale, g->kb,hd,(long)T*hd, g->dPb,T,(long)T*T,
          0.0f, g->dqb,hd,(long)T*hd, B*H);
      bmm(&g->lt,s,CUBLAS_OP_N,CUBLAS_OP_T,hd,T,T,scale, g->qb,hd,(long)T*hd, g->dPb,T,(long)T*T,
          0.0f, g->dkb,hd,(long)T*hd, B*H);
    }
    unpermute_dqkv_k<<<ceil_div(ne,256),256,0,s>>>(g->dqkv,g->dqb,g->dkb,g->dvb,B,T,H,hd);
    }
    PMARK("B_attn");
    // rope backward on dq,dk slices (division-free, 2D block for occupancy)
    int half=hd/2, HPB=(256/half)>0?(256/half):1; if(HPB>H)HPB=H;
    dim3 rope_block(half,HPB), rope_grid(T,ceil_div(H,HPB),B);
    rope_backward_qk_k<<<rope_grid,rope_block,0,s>>>(g->dqkv,g->cosb,g->sinb,H,hd,3*C,T);
    // qkv-proj backward
    linear_backward_weight(&g->lt,s,g->dqkv,g->ln1_out[l],G(g,LP(c,l,po.qkv)),BT,3*C,C,wgrad_beta);
    linear_backward_inp(&g->lt,s,g->dqkv,P(g,LP(c,l,po.qkv)),g->dln,BT,3*C,C,0.0f); // dln = d_ln1
    PMARK("B_qkv_gemm");
    // rmsnorm1 backward: dresid = dresid1(passthrough) + d(norm) = grad wrt x_in
    if(!SKIP_NORM_BWD) rmsnorm_bwd(s,g->norm_partial,g->dresid,g->dresid1,G(g,LP(c,l,po.ln1)),g->dln,x_in,P(g,LP(c,l,po.ln1)),g->rstd1[l],BT,C);
    else copy_k<<<ceil_div(BT*C,blk),blk,0,s>>>(g->dresid,g->dresid1,BT*C);
    PMARK("B_norm");
    // layer l grads complete -> overlapped allreduce of its contiguous slice
    if(do_reduce) areduce_slice(g, off_layers(c)+(long)l*lstride, lstride);
    opt_slice(g, off_layers(c)+(long)l*lstride, lstride, opt_now);   // overlapped AdamW
  }
  // embedding backward (skips image-token rows; their grad stays in g->dresid)
  encoder_backward_k<<<BT,256,0,s>>>(G(g,0),d_ids,g->dresid,BT,C,T,n_pre);
  if(do_reduce) areduce_slice(g, 0, (long)V*C);   // wte grads
  opt_slice(g, 0, (long)V*C, opt_now);             // overlapped AdamW on wte
  // compute stream must see all optimizer updates before the next forward
  if(g->overlap_opt){ CUDA_CHECK(cudaEventRecord(g->oev,g->ostream));
                      CUDA_CHECK(cudaStreamWaitEvent(g->stream,g->oev,0)); }
}

// ---------------------------------------------------------------- optimizer
static void gpt_adamw(GPT* g, float lr, float b1, float b2, float eps, float wd, int t, float grad_scale=1.0f){
  float bc1=1.0f-powf(b1,(float)t), bc2=1.0f-powf(b2,(float)t);
  if(g->fp8_fwd)
    adamw_fp8_k<<<ceil_div(g->np,256),256,0,g->stream>>>(g->master,g->params,g->params8,g->grads,g->m,g->v,g->np,
                                                        lr,b1,b2,eps,wd,bc1,bc2,grad_scale,g->fp8_w_scale);
  else if((g->np & 3L) == 0)
    adamw4_k<<<ceil_div(g->np/4,256),256,0,g->stream>>>(g->master,g->params,g->grads,g->m,g->v,g->np/4,
                                                        lr,b1,b2,eps,wd,bc1,bc2,grad_scale);
  else
    adamw_k<<<ceil_div(g->np,256),256,0,g->stream>>>(g->master,g->params,g->grads,g->m,g->v,g->np,
                                                     lr,b1,b2,eps,wd,bc1,bc2,grad_scale);
}

static float gpt_mean_loss(GPT* g, int B, int T, int valid_tokens=0){
  long BT=(long)B*T;
  int valid = valid_tokens > 0 ? valid_tokens : (int)BT;
  if(valid <= 0){ printf("invalid valid token count %d\n", valid); exit(1); }
  std::vector<float> h(BT);
  CUDA_CHECK(cudaMemcpyAsync(h.data(),g->losses,BT*sizeof(float),cudaMemcpyDeviceToHost,g->stream));
  CUDA_CHECK(cudaStreamSynchronize(g->stream));
  double s=0; for(long i=0;i<BT;i++) s+=h[i];
  return (float)(s/valid);
}

// ---------------------------------------------------------------- init
struct RNG{ unsigned long st; RNG(unsigned long s):st(s){} 
  float nf(){ st^=st<<13; st^=st>>7; st^=st<<17; unsigned int x=(unsigned int)(st>>16);
    return (float)(x&0xFFFFFF)/(float)0x1000000; } // [0,1)
  float normal(){ float u1=nf()+1e-7f,u2=nf(); return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); }
};
static void gpt_init_weights(GPT* g, unsigned long seed){
  Config&c=g->cfg; long np=g->np; std::vector<float> h(np);
  RNG r(seed);
  float std=0.02f; float rs=1.0f/sqrtf(2.0f*c.L);
  ParamOff po=layer_offsets(c);
  long C=c.C,I=c.I,V=c.V;
  // wte
  for(long i=0;i<(long)V*C;i++) h[i]=std*r.normal();
  for(int l=0;l<c.L;l++){
    long base=off_layers(c)+(long)l*layer_stride(c);
    for(long i=0;i<C;i++) h[base+po.ln1+i]=1.0f;
    for(long i=0;i<3L*C*C;i++) h[base+po.qkv+i]=std*r.normal();
    for(long i=0;i<(long)C*C;i++) h[base+po.o+i]=std*r.normal()*rs;
    for(long i=0;i<C;i++) h[base+po.ln2+i]=1.0f;
    for(long i=0;i<(long)I*C;i++) h[base+po.gate+i]=std*r.normal();
    for(long i=0;i<(long)I*C;i++) h[base+po.up+i]=std*r.normal();
    for(long i=0;i<(long)C*I;i++) h[base+po.down+i]=std*r.normal()*rs;
  }
  for(long i=0;i<C;i++) h[off_lnf(c)+i]=1.0f;
  for(long i=0;i<(long)V*C;i++) h[off_lm(c)+i]=std*r.normal();
  CUDA_CHECK(cudaMemcpy(g->master,h.data(),(size_t)np*4,cudaMemcpyHostToDevice));
  if(g->fp8_fwd)
    cast_f2b_fp8_e4m3<<<ceil_div(np,256),256,0,g->stream>>>(g->params,g->params8,g->master,np,g->fp8_w_scale);
  else
    cast_f2b<<<ceil_div(np,256),256,0,g->stream>>>(g->params,g->master,np);
  CUDA_CHECK(cudaStreamSynchronize(g->stream));
}

// ================================================================ Vision tower
// LLaVA-style trainable image adapter: patch-embed -> learned pos -> per-patch
// (RMSNorm + SwiGLU MLP) x vL -> final norm -> projector into LM embedding space.
// Image tokens (one per patch) are prepended to the LM text sequence and the
// whole stack is trained jointly (gradients flow LM -> projector -> tower).
struct VT {
  int vC, vL, vI, patch_dim, n_patch, C;   // C = LM d_model
  long np; float* master; bf16* params; float* grads; float* m; float* v;
  long o_pw, o_pos, o_layers, vstride, o_fn, o_proj;
  // activations (NP = B*n_patch rows)
  bf16 *patch_out; std::vector<bf16*> ln,gu,sw,resid; std::vector<float*> rstd;
  bf16 *fn_out; float* fn_rstd; bf16 *img; // img = projected image tokens [NP,C]
  // backward scratch
  bf16 *dx,*dln,*dgu,*dsw,*dfn,*dimg; float* norm_partial;
  GPT* lm; int B;
};
static long vt_layer_stride(VT*t){ return t->vC + 3L*t->vI*t->vC; } // ln+gate+up+down
static void vt_build(VT* t, GPT* lm, int vC,int vL,int vI,int patch_dim,int n_patch,int B){
  t->vC=vC;t->vL=vL;t->vI=vI;t->patch_dim=patch_dim;t->n_patch=n_patch;t->C=lm->cfg.C;t->lm=lm;t->B=B;
  t->o_pw=0; t->o_pos=(long)vC*patch_dim; t->o_layers=t->o_pos+(long)n_patch*vC;
  t->vstride=vt_layer_stride(t); t->o_fn=t->o_layers+(long)vL*t->vstride;
  t->o_proj=t->o_fn+vC; t->np=t->o_proj+(long)t->C*vC;
  t->master=dmalloc_f32(t->np); t->params=dmalloc_bf16(t->np); t->grads=dmalloc_f32(t->np);
  t->m=dmalloc_f32(t->np); t->v=dmalloc_f32(t->np);
  CUDA_CHECK(cudaMemset(t->m,0,(size_t)t->np*4)); CUDA_CHECK(cudaMemset(t->v,0,(size_t)t->np*4));
  long NP=(long)B*n_patch;
  t->patch_out=dmalloc_bf16(NP*vC);
  for(int l=0;l<vL;l++){ t->ln.push_back(dmalloc_bf16(NP*vC)); t->gu.push_back(dmalloc_bf16(NP*2*vI));
    t->sw.push_back(dmalloc_bf16(NP*vI));
    t->resid.push_back(dmalloc_bf16(NP*vC)); t->rstd.push_back(dmalloc_f32(NP)); }
  t->fn_out=dmalloc_bf16(NP*vC); t->fn_rstd=dmalloc_f32(NP); t->img=dmalloc_bf16(NP*t->C);
  t->dx=dmalloc_bf16(NP*vC); t->dln=dmalloc_bf16(NP*vC); t->dgu=dmalloc_bf16(NP*2*vI);
  t->dsw=dmalloc_bf16(NP*vI); t->dfn=dmalloc_bf16(NP*vC);
  t->dimg=dmalloc_bf16(NP*t->C);
  t->norm_partial=dmalloc_f32((long)NORM_R*vC);
}
static bf16* VP(VT*t,long o){ return t->params+o; }
static float* VG(VT*t,long o){ return t->grads+o; }
static long VL_(VT*t,int l,long intra){ return t->o_layers+(long)l*t->vstride+intra; }
static void vt_init(VT* t, unsigned long seed){
  std::vector<float> h(t->np); RNG r(seed); float s=0.02f;
  for(long i=0;i<t->np;i++) h[i]=s*r.normal();
  // norm weights = 1
  long off; // ln per layer at intra 0
  for(int l=0;l<t->vL;l++){ off=VL_(t,l,0); for(int i=0;i<t->vC;i++) h[off+i]=1.0f; }
  for(int i=0;i<t->vC;i++) h[t->o_fn+i]=1.0f;
  CUDA_CHECK(cudaMemcpy(t->master,h.data(),(size_t)t->np*4,cudaMemcpyHostToDevice));
  cast_f2b<<<ceil_div(t->np,256),256,0,t->lm->stream>>>(t->params,t->master,t->np);
  CUDA_CHECK(cudaStreamSynchronize(t->lm->stream));
}
// forward: patches[NP,patch_dim] -> img[NP,C]
static void vt_forward(VT* t, const bf16* patches, int B){
  GPT* g=t->lm; cudaStream_t s=g->stream; int vC=t->vC,vI=t->vI,blk=256; long NP=(long)B*t->n_patch;
  // patch embed
  linear_forward(&g->lt,s,patches,VP(t,t->o_pw),t->patch_out,NP,vC,t->patch_dim);
  add_posembed_k<<<ceil_div(NP*vC,256),256,0,s>>>(t->patch_out,VP(t,t->o_pos),NP,t->n_patch,vC);
  bf16* x=t->patch_out;
  for(int l=0;l<t->vL;l++){
    rmsnorm_forward_k<<<NP,256,0,s>>>(t->ln[l],t->rstd[l],x,VP(t,VL_(t,l,0)),NP,vC,1e-5f);
    long og=t->vC, ou=og+(long)vI*vC, od=ou+(long)vI*vC; // gate,up,down intra offsets
    linear_forward(&g->lt,s,t->ln[l],VP(t,VL_(t,l,og)),t->gu[l],NP,2*vI,vC);
    if((vI & 1) == 0) swiglu_forward_gu2_k<<<dim3(NP,ceil_div(vI/2,blk)),blk,0,s>>>(t->sw[l],t->gu[l],NP,vI);
    else swiglu_forward_gu_k<<<dim3(NP,ceil_div(vI,blk)),blk,0,s>>>(t->sw[l],t->gu[l],NP,vI);
    linear_forward(&g->lt,s,t->sw[l],VP(t,VL_(t,l,od)),t->dx,NP,vC,vI); // mlp out into dx (temp)
    residual_forward_k<<<ceil_div(NP*vC,blk),blk,0,s>>>(t->resid[l],x,t->dx,NP*vC);
    x=t->resid[l];
  }
  rmsnorm_forward_k<<<NP,256,0,s>>>(t->fn_out,t->fn_rstd,x,VP(t,t->o_fn),NP,vC,1e-5f);
  linear_forward(&g->lt,s,t->fn_out,VP(t,t->o_proj),t->img,NP,t->C,vC); // -> [NP,C]
}
// backward: consumes t->dimg (grad of img tokens) -> grads; needs patches
static void vt_backward(VT* t, const bf16* patches, int B){
  GPT* g=t->lm; cudaStream_t s=g->stream; int vC=t->vC,vI=t->vI,blk=256; long NP=(long)B*t->n_patch; int C=t->C;
  zero_f32_k<<<ceil_div(t->np,256),256,0,s>>>(t->grads,t->np);
  // projector backward: dfn = dimg @ proj ; dproj += dimg^T @ fn_out
  linear_backward_weight(&g->lt,s,t->dimg,t->fn_out,VG(t,t->o_proj),NP,C,vC);
  linear_backward_inp(&g->lt,s,t->dimg,VP(t,t->o_proj),t->dfn,NP,C,vC,0.0f);
  // final norm backward -> dx (add=nullptr overwrites, no memset)
  bf16* xlast = (t->vL>0)? t->resid[t->vL-1] : t->patch_out;
  rmsnorm_bwd(s,t->norm_partial,t->dx,nullptr,VG(t,t->o_fn),t->dfn,xlast,VP(t,t->o_fn),t->fn_rstd,NP,vC);
  for(int l=t->vL-1;l>=0;l--){
    bf16* x_in = (l==0)? t->patch_out : t->resid[l-1];
    long og=t->vC, ou=og+(long)vI*vC, od=ou+(long)vI*vC;
    // dx currently = grad wrt resid[l]; mlp out grad = dx (residual passthrough kept in dx too)
    linear_backward_weight(&g->lt,s,t->dx,t->sw[l],VG(t,VL_(t,l,od)),NP,vC,vI);
    linear_backward_inp(&g->lt,s,t->dx,VP(t,VL_(t,l,od)),t->dsw,NP,vC,vI,0.0f);
    if((vI & 1) == 0) swiglu_backward_gu2_k<<<dim3(NP,ceil_div(vI/2,blk)),blk,0,s>>>(t->dgu,t->dsw,t->gu[l],NP,vI);
    else swiglu_backward_gu_k<<<dim3(NP,ceil_div(vI,blk)),blk,0,s>>>(t->dgu,t->dsw,t->gu[l],NP,vI);
    linear_backward_weight(&g->lt,s,t->dgu,t->ln[l],VG(t,VL_(t,l,og)),NP,2*vI,vC);
    linear_backward_inp(&g->lt,s,t->dgu,VP(t,VL_(t,l,og)),t->dln,NP,2*vI,vC,0.0f);
    // rmsnorm backward accumulates dinp into dx (add=dx keeps the residual passthrough)
    rmsnorm_bwd(s,t->norm_partial,t->dx,t->dx,VG(t,VL_(t,l,0)),t->dln,x_in,VP(t,VL_(t,l,0)),t->rstd[l],NP,vC);
  }
  // pos embed grad + patch-embed weight grad
  posembed_grad_k<<<ceil_div(NP*vC,256),256,0,s>>>(VG(t,t->o_pos),t->dx,NP,t->n_patch,vC);
  linear_backward_weight(&g->lt,s,t->dx,patches,VG(t,t->o_pw),NP,vC,t->patch_dim);
}
static void vt_adamw(VT* t, float lr,float b1,float b2,float eps,float wd,int step){
  float bc1=1.0f-powf(b1,(float)step), bc2=1.0f-powf(b2,(float)step);
  adamw_k<<<ceil_div(t->np,256),256,0,t->lm->stream>>>(t->master,t->params,t->grads,t->m,t->v,t->np,
                                                       lr,b1,b2,eps,wd,bc1,bc2,1.0f);
}

// ---------------------------------------------------------------- flops / mfu
static double flops_per_step(const Config&c,int B){
  long C=c.C,I=c.I,V=c.V,T=c.T,L=c.L;
  long Wlayer = 3L*C*C + (long)C*C + (long)I*C + (long)I*C + (long)C*I;
  long Ngemm = L*Wlayer + (long)V*C; // matmul params (no wte gather)
  double tokens=(double)B*T;
  double gemm = 6.0*Ngemm*tokens;
  double attn = 12.0*L*B*(double)T*T*C;
  return gemm+attn;
}

// ---------------------------------------------------------------- data loader
struct Loader{
  FILE* f; long ntok; long pos; int B,T; std::vector<uint16_t> buf; std::vector<int> ids,tgt;
};
static void loader_open(Loader*ld,const char*path,int B,int T){
  ld->f=fopen(path,"rb"); if(!ld->f){printf("cannot open %s\n",path);exit(1);}
  int header[64]; fread(header,sizeof(int),64,ld->f);
  if(header[0]!=20240520){printf("bad magic %d\n",header[0]);exit(1);}
  ld->ntok=header[2]; ld->B=B; ld->T=T; ld->pos=0;
  ld->buf.resize((long)B*T+1); ld->ids.resize((long)B*T); ld->tgt.resize((long)B*T);
  printf("[loader] %s : %ld tokens\n",path,ld->ntok);
}
static void loader_next_into(Loader*ld, int* ids, int* tgt){
  long need=(long)ld->B*ld->T+1;
  if(ld->pos+need>ld->ntok){ ld->pos=0; }
  fseek(ld->f, 64*sizeof(int)+ld->pos*sizeof(uint16_t), SEEK_SET);
  size_t got=fread(ld->buf.data(),sizeof(uint16_t),need,ld->f);
  if(got!=(size_t)need){ printf("short read from training data\n"); exit(1); }
  long BT=(long)ld->B*ld->T;
  for(long i=0;i<BT;i++){ ids[i]=ld->buf[i]; tgt[i]=ld->buf[i+1]; }
  ld->pos += BT;
}

// Two-slot pinned input staging. Refill of a slot is ordered after the compute
// stream has consumed that slot, so CPU fread/fill and H2D copies overlap the
// current GPU step without risking an input overwrite.
struct BatchPipe{
  Loader* ld=nullptr; int B=0,T=0; long BT=0;
  int *h_ids[2]{}, *h_tgt[2]{}, *d_ids[2]{}, *d_tgt[2]{};
  cudaStream_t copy_stream=0;
  cudaEvent_t ready[2]{}, consumed[2]{};
};
static void batch_pipe_init(BatchPipe* p, Loader* ld){
  p->ld=ld; p->B=ld->B; p->T=ld->T; p->BT=(long)ld->B*ld->T;
  CUDA_CHECK(cudaStreamCreateWithFlags(&p->copy_stream, cudaStreamNonBlocking));
  for(int s=0;s<2;s++){
    CUDA_CHECK(cudaHostAlloc(&p->h_ids[s], p->BT*sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&p->h_tgt[s], p->BT*sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&p->d_ids[s], p->BT*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&p->d_tgt[s], p->BT*sizeof(int)));
    CUDA_CHECK(cudaEventCreateWithFlags(&p->ready[s], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&p->consumed[s], cudaEventDisableTiming));
  }
}
static void batch_pipe_enqueue_copy(BatchPipe* p, int slot){
  CUDA_CHECK(cudaMemcpyAsync(p->d_ids[slot],p->h_ids[slot],p->BT*sizeof(int),
                             cudaMemcpyHostToDevice,p->copy_stream));
  CUDA_CHECK(cudaMemcpyAsync(p->d_tgt[slot],p->h_tgt[slot],p->BT*sizeof(int),
                             cudaMemcpyHostToDevice,p->copy_stream));
  CUDA_CHECK(cudaEventRecord(p->ready[slot],p->copy_stream));
}
static void batch_pipe_prefetch_initial(BatchPipe* p, int slot){
  loader_next_into(p->ld,p->h_ids[slot],p->h_tgt[slot]);
  batch_pipe_enqueue_copy(p,slot);
}
static void batch_pipe_refill_after_compute(BatchPipe* p, int slot, cudaStream_t compute){
  loader_next_into(p->ld,p->h_ids[slot],p->h_tgt[slot]);
  CUDA_CHECK(cudaEventRecord(p->consumed[slot],compute));
  CUDA_CHECK(cudaStreamWaitEvent(p->copy_stream,p->consumed[slot],0));
  batch_pipe_enqueue_copy(p,slot);
}
static void batch_pipe_refill_after_consumed_event(BatchPipe* p, int slot){
  loader_next_into(p->ld,p->h_ids[slot],p->h_tgt[slot]);
  CUDA_CHECK(cudaStreamWaitEvent(p->copy_stream,p->consumed[slot],0));
  batch_pipe_enqueue_copy(p,slot);
}
static void batch_pipe_wait(BatchPipe* p, int slot, cudaStream_t compute){
  CUDA_CHECK(cudaStreamWaitEvent(compute,p->ready[slot],0));
}

// ---------------------------------------------------------------- DDP (NCCL)
// File-based ncclUniqueId exchange over the shared filesystem (no MPI needed).
static void ddp_init(DDP* d, int rank, int world, int local, const char* idfile){
  d->rank=rank; d->world=world; d->local=local;
  ncclUniqueId id;
  if(world>1){
    if(rank==0){
      NCCL_CHECK(ncclGetUniqueId(&id));
      char tmp[512]; snprintf(tmp,sizeof tmp,"%s.tmp",idfile);
      FILE* f=fopen(tmp,"wb"); fwrite(&id,sizeof id,1,f); fclose(f);
      rename(tmp, idfile); // atomic publish
    } else {
      FILE* f=nullptr; for(int t=0;t<6000;t++){ f=fopen(idfile,"rb"); if(f) break; usleep(10000); }
      if(!f){ printf("rank %d: timed out waiting for %s\n",rank,idfile); exit(1); }
      size_t got=fread(&id,sizeof id,1,f); fclose(f); (void)got;
    }
    NCCL_CHECK(ncclCommInitRank(&d->comm, world, id, rank));
  }
}
// ---------------------------------------------------------------- modes
static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec/1e9; }
static int env_int(const char*n,int dflt){ const char*v=getenv(n); return v? atoi(v):dflt; }
static int default_overlap_opt(){
  const char* v=getenv("ENTROPY_OVERLAP_OPT");
  if(v) return atoi(v)!=0;
#ifdef ENTROPY_BUILD_SM
  if(ENTROPY_BUILD_SM >= 90) return 0; // H100 measured faster without side-stream AdamW overlap.
#endif
  return 1;
}
static double peak_flops_per_gpu(){
  const char* v=getenv("ENTROPY_PEAK_TFLOPS");
  double tflops = (v && atof(v)>0.0) ? atof(v) : 364.0; // RTX 6000 Ada BF16 dense default
  return tflops*1.0e12;
}

static void check_device_arch_or_die(const cudaDeviceProp& prop){
#ifdef ENTROPY_BUILD_SM
  int build_sm = ENTROPY_BUILD_SM;
  int device_sm = prop.major * 10 + prop.minor;
  if(build_sm > 0 && build_sm != device_sm){
    printf("ERROR: this binary was built for sm_%d but the active GPU is sm_%d (%s).\n",
           build_sm, device_sm, prop.name);
    printf("       Rebuild with GPU_ARCH=%d scripts/build.sh train/gpt.cu <out>.\n", device_sm);
    exit(1);
  }
#endif
}

static void print_topology_plan(){
  Config c; c.V=env_int("ENTROPY_V",32768); c.C=env_int("ENTROPY_C",768);
  c.L=env_int("ENTROPY_L",12); c.H=env_int("ENTROPY_H",12);
  c.I=env_int("ENTROPY_I",2048); c.T=env_int("ENTROPY_T",1024);
  int world=env_int("ENTROPY_WORLD", env_int("SLURM_NTASKS", env_int("WORLD",5)));
  int spare=env_int("ENTROPY_SPARE_GPUS",0);
  int pp=env_int("ENTROPY_PP",1);
  int mb=env_int("ENTROPY_MICROBATCHES", env_int("ENTROPY_ACCUM",8));
  int B=env_int("ENTROPY_B",8);
  if(spare<0) spare=0; if(spare>=world) spare=world-1;
  int active=world-spare;
  if(pp<1) pp=1; if(pp>active) pp=active;
  int dp=std::max(1, active/pp);
  int rem=active%pp;
  double bubble = pp>1 ? (double)(pp-1)/(double)(mb+pp-1) : 0.0;
  double pipe_eff = 1.0 - bubble;
  long lstride=layer_stride(c);
  double param_bytes_per_weight = 18.0; // bf16 param + fp32 master/grads/m/v
  double act_boundary_mb = (double)B*c.T*c.C*2.0/1048576.0;
  printf("Topology plan: world=%d active=%d spare=%d pp=%d dp=%d remainder=%d microbatches=%d\n",
         world,active,spare,pp,dp,rem,mb);
  printf("  model: V=%d C=%d L=%d H=%d I=%d T=%d B/microbatch=%d params=%.1fM\n",
         c.V,c.C,c.L,c.H,c.I,c.T,B,num_params(c)/1e6);
  printf("  1F1B pipeline bubble: %.1f%%  ideal pipe utilization: %.1f%%\n",
         100.0*bubble,100.0*pipe_eff);
  printf("  activation boundary traffic: %.1f MiB per boundary per fwd tensor (bf16)\n",
         act_boundary_mb);
  for(int s=0;s<pp;s++){
    int l0=(int)((long)c.L*s/pp), l1=(int)((long)c.L*(s+1)/pp);
    long stage_params = (long)(l1-l0)*lstride;
    if(s==0) stage_params += (long)c.V*c.C;
    if(s==pp-1) stage_params += c.C + (long)c.V*c.C;
    printf("  stage %d: layers [%d,%d) params %.1fM optimizer+grads %.1f GiB\n",
           s,l0,l1,stage_params/1e6,stage_params*param_bytes_per_weight/1073741824.0);
  }
  if(spare>0){
    printf("  hot spare: keep %d GPU(s) outside NCCL. On Xid/ECC failure, checkpoint/relaunch onto a spare;\n",spare);
    printf("             active NCCL communicators cannot safely grow/shrink in-process.\n");
  } else {
    printf("  hot spare: none reserved. Set ENTROPY_SPARE_GPUS=1 for restart-on-spare scheduling.\n");
  }
  printf("  recommendation: use PP only when per-GPU batch is memory-bound; otherwise DP+accum usually has higher MFU.\n");
}

static int lmce_formula_check_case_cpu(int BT, int V, int C, int chunk, float beta,
                                       unsigned long seed, int ignore_mod, int verbose){
  if(BT<1) BT=7; if(V<2) V=19; if(C<1) C=11; if(chunk<1) chunk=5; if(chunk>V) chunk=V;
  RNG r(seed);
  std::vector<float> x((long)BT*C), W((long)V*C), logits((long)BT*V), dlogits((long)BT*V);
  std::vector<float> dW_ref((long)V*C), dW_dir((long)V*C), dx_ref((long)BT*C), dx_dir((long)BT*C);
  std::vector<float> rowmax(BT,-1.0e30f), rowsum(BT,0.0f), target_logit(BT,0.0f), loss_ref(BT), loss_dir(BT);
  std::vector<float> full_rowmax(BT,-1.0e30f), full_rowsum(BT,0.0f), full_target_logit(BT,0.0f);
  std::vector<int> tgt(BT);
  for(float&v:x) v=0.2f*r.normal();
  for(float&v:W) v=0.2f*r.normal();
  for(long i=0;i<(long)V*C;i++){ float base=0.1f*r.normal(); dW_ref[i]=beta*base; dW_dir[i]=beta*base; }
  for(int row=0; row<BT; row++){
    bool ignored = false;
    if(ignore_mod == 1) ignored = row != 0;          // single valid row
    else if(ignore_mod > 1) ignored = (row%ignore_mod)==0;
    tgt[row] = ignored ? -1 : (int)(r.nf()*V);       // ignore_mod<=0 means all valid
  }
  int valid=0; for(int t:tgt) if(t>=0) valid++;
  if(valid<=0){ printf("lmce_check: no valid targets\n"); return 0; }
  float dscale=1.0f/(float)valid;
  for(int row=0; row<BT; row++){
    for(int v=0; v<V; v++){
      float acc=0.0f;
      for(int c=0; c<C; c++) acc += x[(long)row*C+c]*W[(long)v*C+c];
      logits[(long)row*V+v]=acc;
    }
  }
  for(int row=0; row<BT; row++){
    if(tgt[row]<0) continue;
    float m=-1.0e30f, sum=0.0f;
    for(int v=0; v<V; v++){
      float z=logits[(long)row*V+v];
      if(z>m){ sum=sum*expf(m-z)+1.0f; m=z; }
      else   { sum+=expf(z-m); }
    }
    full_rowmax[row]=m;
    full_rowsum[row]=sum;
    full_target_logit[row]=logits[(long)row*V+tgt[row]];
    loss_ref[row]=logf(sum)+m-full_target_logit[row];
  }
  for(int off=0; off<V; off+=chunk){
    int n=std::min(chunk,V-off);
    for(int row=0; row<BT; row++){
      if(tgt[row]<0) continue;
      float m=-1.0e30f, sum=0.0f;
      for(int j=0; j<n; j++){
        float z=logits[(long)row*V+off+j];
        if(z>m){ sum=sum*expf(m-z)+1.0f; m=z; }
        else   { sum+=expf(z-m); }
      }
      float old_m=rowmax[row], old_s=rowsum[row], mm=fmaxf(old_m,m);
      rowmax[row]=mm;
      rowsum[row]=old_s*expf(old_m-mm)+sum*expf(m-mm);
      if(tgt[row]>=off && tgt[row]<off+n) target_logit[row]=logits[(long)row*V+tgt[row]];
    }
  }
  for(int row=0; row<BT; row++){
    if(tgt[row]<0){ loss_ref[row]=loss_dir[row]=0.0f; continue; }
    loss_dir[row]=logf(rowsum[row])+rowmax[row]-target_logit[row];
    for(int v=0; v<V; v++){
      float p=expf(logits[(long)row*V+v]-full_rowmax[row])/full_rowsum[row];
      dlogits[(long)row*V+v]=(p-(v==tgt[row]?1.0f:0.0f))*dscale;
    }
  }
  for(int row=0; row<BT; row++){
    for(int c=0; c<C; c++){
      float acc=0.0f;
      for(int v=0; v<V; v++) acc += dlogits[(long)row*V+v]*W[(long)v*C+c];
      dx_ref[(long)row*C+c]=acc;
    }
  }
  for(int v=0; v<V; v++){
    for(int c=0; c<C; c++){
      float acc=dW_ref[(long)v*C+c];
      for(int row=0; row<BT; row++) acc += dlogits[(long)row*V+v]*x[(long)row*C+c];
      dW_ref[(long)v*C+c]=acc;
    }
  }
  for(int off=0; off<V; off+=chunk){
    int n=std::min(chunk,V-off);
    bool first=(off==0);
    for(int row=0; row<BT; row++){
      for(int c=0; c<C; c++){
        float acc=first ? 0.0f : dx_dir[(long)row*C+c];
        if(tgt[row]>=0){
          for(int j=0; j<n; j++){
            int v=off+j;
            float p=expf(logits[(long)row*V+v]-rowmax[row])/rowsum[row];
            float g=(p-(v==tgt[row]?1.0f:0.0f))*dscale;
            acc += g*W[(long)v*C+c];
          }
        }
        dx_dir[(long)row*C+c]=acc;
      }
    }
    for(int j=0; j<n; j++){
      int v=off+j;
      for(int c=0; c<C; c++){
        float acc=dW_dir[(long)v*C+c];
        for(int row=0; row<BT; row++){
          if(tgt[row]<0) continue;
          float p=expf(logits[(long)row*V+v]-rowmax[row])/rowsum[row];
          float g=(p-(v==tgt[row]?1.0f:0.0f))*dscale;
          acc += g*x[(long)row*C+c];
        }
        dW_dir[(long)v*C+c]=acc;
      }
    }
  }
  float max_dx=0.0f, max_dw=0.0f, max_loss=0.0f, max_stats=0.0f;
  for(size_t i=0;i<dx_ref.size();i++) max_dx=fmaxf(max_dx,fabsf(dx_ref[i]-dx_dir[i]));
  for(size_t i=0;i<dW_ref.size();i++) max_dw=fmaxf(max_dw,fabsf(dW_ref[i]-dW_dir[i]));
  for(size_t i=0;i<loss_ref.size();i++) max_loss=fmaxf(max_loss,fabsf(loss_ref[i]-loss_dir[i]));
  for(int row=0; row<BT; row++){
    if(tgt[row]<0) continue;
    max_stats=fmaxf(max_stats,fabsf(full_rowmax[row]-rowmax[row]));
    max_stats=fmaxf(max_stats,fabsf(full_rowsum[row]-rowsum[row]));
    max_stats=fmaxf(max_stats,fabsf(full_target_logit[row]-target_logit[row]));
  }
  if(verbose)
    printf("lmce_check CPU: BT=%d V=%d C=%d chunk=%d beta=%.2g ignore_mod=%d valid=%d  max|dx| %.3g  max|dW| %.3g  max|loss| %.3g  max|stats| %.3g\n",
           BT,V,C,chunk,beta,ignore_mod,valid,max_dx,max_dw,max_loss,max_stats);
  float tol=2.0e-5f;
  if(max_dx>tol || max_dw>tol || max_loss>tol || max_stats>tol){
    printf("lmce_check FAILED (tol %.1e)\n",tol);
    return 0;
  }
  if(verbose) printf("lmce_check OK\n");
  return 1;
}

static int lmce_formula_check_cpu(){
  bool custom = getenv("ENTROPY_CHECK_BT") || getenv("ENTROPY_CHECK_V") ||
                getenv("ENTROPY_CHECK_C") || getenv("ENTROPY_CHECK_CHUNK") ||
                getenv("ENTROPY_CHECK_BETA") || getenv("ENTROPY_CHECK_IGNORE_MOD");
  if(custom){
    int BT=env_int("ENTROPY_CHECK_BT",7), V=env_int("ENTROPY_CHECK_V",19);
    int C=env_int("ENTROPY_CHECK_C",11), chunk=env_int("ENTROPY_CHECK_CHUNK",5);
    float beta=env_float_raw("ENTROPY_CHECK_BETA",0.37f);
    int ignore_mod=env_int("ENTROPY_CHECK_IGNORE_MOD",5);
    int ok=lmce_formula_check_case_cpu(BT,V,C,chunk,beta,123,ignore_mod,1);
    return ok ? 0 : 1;
  }
  struct Case { int BT,V,C,chunk,ignore_mod; float beta; unsigned long seed; };
  Case cases[] = {
    {7,  19, 11, 5, 5, 0.37f, 123},
    {9,  23, 13, 7, 4, 0.00f, 321},
    {4,  16,  8,16, 3, 1.00f, 555},
    {10, 31, 17, 6, 6, 0.50f, 777},
    {3,   5,  3, 2, 2, 0.25f, 999},
    {8,  29,  7, 4, 0, 0.75f, 2026},
    {6,  17,  5, 3, 1, 0.10f, 4242},
  };
  int pass=0, total=(int)(sizeof(cases)/sizeof(cases[0]));
  for(int i=0;i<total;i++){
    Case c=cases[i];
    pass += lmce_formula_check_case_cpu(c.BT,c.V,c.C,c.chunk,c.beta,c.seed,c.ignore_mod,1);
  }
  printf("lmce_check sweep: %d/%d cases passed\n",pass,total);
  if(pass!=total) return 1;
  return 0;
}

int main(int argc,char**argv){
  std::string mode = argc>1? argv[1] : "overfit";
  if(mode=="lmce_check") return lmce_formula_check_cpu();
  if(mode=="plan" || mode=="plan_pp" || mode=="topology"){
    print_topology_plan();
    return 0;
  }
  if(mode=="bench_ddp" || mode=="train_ddp"){
    int local=env_int("SLURM_LOCALID", env_int("LOCAL_RANK",0));
    CUDA_CHECK(cudaSetDevice(local));
  }
  int devcount; CUDA_CHECK(cudaGetDeviceCount(&devcount));
  int active_dev=0; CUDA_CHECK(cudaGetDevice(&active_dev));
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,active_dev));
  check_device_arch_or_die(prop);
  printf("== entropy gpt :: %s on %s (%d SMs) ==\n",mode.c_str(),prop.name,prop.multiProcessorCount);

  if(mode=="overfit"){
    Config c; c.V=256; c.C=128; c.L=4; c.H=4; c.I=256; c.T=64;
    int B=4;
    GPT g; gpt_build(&g,c,B);
    gpt_init_weights(&g,1234);
    long BT=(long)B*c.T;
    std::vector<int> ids(BT),tgt(BT); RNG r(99);
    for(long i=0;i<BT;i++){ ids[i]=(int)(r.nf()*c.V); tgt[i]=(int)(r.nf()*c.V); }
    int *d_ids,*d_tgt; CUDA_CHECK(cudaMalloc(&d_ids,BT*4)); CUDA_CHECK(cudaMalloc(&d_tgt,BT*4));
    CUDA_CHECK(cudaMemcpy(d_ids,ids.data(),BT*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt,tgt.data(),BT*4,cudaMemcpyHostToDevice));
    printf("overfitting a fixed random batch (BT=%ld, %ld params)...\n",BT,g.np);
    for(int step=0;step<200;step++){
      gpt_forward(&g,d_ids,B,c.T);
      gpt_backward(&g,d_ids,d_tgt,B,c.T);
      gpt_adamw(&g,1e-3f,0.9f,0.95f,1e-8f,0.0f,step+1);
      if(step%20==0||step==199){ float ls=gpt_mean_loss(&g,B,c.T);
        printf("step %3d  loss %.5f\n",step,ls); }
    }
    printf("(expect loss -> near 0 if fwd/bwd/optimizer are correct)\n");
    return 0;
  }

  if(mode=="mm_overfit"){
    // Joint multimodal overfit: vision tower + LM on a fixed image+text batch.
    int B=2, n_patch=16, patch_dim=3*16*16, T_text=48;
    int T=n_patch+T_text;
    Config c; c.V=256; c.C=128; c.L=4; c.H=4; c.I=256; c.T=T;
    GPT g; gpt_build(&g,c,B); gpt_init_weights(&g,1234);
    VT t; vt_build(&t,&g, /*vC*/96,/*vL*/3,/*vI*/192, patch_dim, n_patch, B); vt_init(&t,777);
    long NP=(long)B*n_patch, BT=(long)B*T;
    // fixed random image patches, text ids, targets
    std::vector<bf16> hp(NP*patch_dim); RNG r(5);
    for(long i=0;i<NP*patch_dim;i++) hp[i]=__float2bfloat16(r.normal()*0.5f);
    bf16* d_patch; CUDA_CHECK(cudaMalloc(&d_patch,NP*patch_dim*sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(d_patch,hp.data(),NP*patch_dim*sizeof(bf16),cudaMemcpyHostToDevice));
    std::vector<int> ids(BT),tgt(BT);
    for(int b=0;b<B;b++) for(int p=0;p<T;p++){ long i=(long)b*T+p;
      if(p<n_patch){ ids[i]=0; tgt[i]=-1; }                         // image slots
      else { ids[i]=(int)(r.nf()*c.V); tgt[i]=(int)(r.nf()*c.V); } } // text
    int *d_ids,*d_tgt; CUDA_CHECK(cudaMalloc(&d_ids,BT*4)); CUDA_CHECK(cudaMalloc(&d_tgt,BT*4));
    CUDA_CHECK(cudaMemcpy(d_ids,ids.data(),BT*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt,tgt.data(),BT*4,cudaMemcpyHostToDevice));
    printf("multimodal overfit: %d img tokens + %d text, LM %.0fK params, vision %.0fK params\n",
           n_patch,T_text,g.np/1e3,t.np/1e3);
    for(int step=0;step<200;step++){
      vt_forward(&t,d_patch,B);
      gpt_forward(&g,d_ids,B,T,t.img,n_patch);
      gpt_backward(&g,d_ids,d_tgt,B,T,/*zero=*/true,/*reduce=*/false,/*n_pre=*/n_patch,
                   /*valid_tokens=*/B*T_text);
      gather_prefix_grad_k<<<ceil_div(NP*c.C,256),256,0,g.stream>>>(t.dimg,g.dresid,B,T,n_patch,c.C);
      vt_backward(&t,d_patch,B);
      gpt_adamw(&g,1e-3f,0.9f,0.95f,1e-8f,0.0f,step+1);
      vt_adamw(&t,1e-3f,0.9f,0.95f,1e-8f,0.0f,step+1);
      if(step%20==0||step==199){ float ls=gpt_mean_loss(&g,B,T,B*T_text);
        printf("step %3d  loss %.5f\n",step,ls); }
    }
    printf("(loss -> ~0 means gradients flow LM -> projector -> vision tower correctly)\n");
    return 0;
  }

  if(mode=="mm_bench"){
    // Timed joint multimodal training path: vision forward -> LM forward/backward
    // -> vision backward -> both optimizers. Same graph as mm_overfit, measured.
    int B=argc>2? atoi(argv[2]) : 2;
    int steps=argc>3? atoi(argv[3]) : 50;
    int n_patch=env_int("ENTROPY_MM_PATCHES",16), patch_dim=3*16*16;
    int T_text=env_int("ENTROPY_MM_TEXT",48);
    int T=n_patch+T_text;
    Config c; c.V=env_int("ENTROPY_V",256); c.C=env_int("ENTROPY_C",128);
    c.L=env_int("ENTROPY_L",4); c.H=env_int("ENTROPY_H",4);
    c.I=env_int("ENTROPY_I",256); c.T=T;
    GPT g; gpt_build(&g,c,B); gpt_init_weights(&g,1234);
    VT t; vt_build(&t,&g, env_int("ENTROPY_VC",96),env_int("ENTROPY_VL",3),
                   env_int("ENTROPY_VI",192), patch_dim, n_patch, B);
    vt_init(&t,777);
    long NP=(long)B*n_patch, BT=(long)B*T;
    std::vector<bf16> hp(NP*patch_dim); RNG r(5);
    for(long i=0;i<NP*patch_dim;i++) hp[i]=__float2bfloat16(r.normal()*0.5f);
    bf16* d_patch; CUDA_CHECK(cudaMalloc(&d_patch,NP*patch_dim*sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(d_patch,hp.data(),NP*patch_dim*sizeof(bf16),cudaMemcpyHostToDevice));
    std::vector<int> ids(BT),tgt(BT);
    for(int b=0;b<B;b++) for(int p=0;p<T;p++){ long i=(long)b*T+p;
      if(p<n_patch){ ids[i]=0; tgt[i]=-1; }
      else { ids[i]=(int)(r.nf()*c.V); tgt[i]=(int)(r.nf()*c.V); } }
    int *d_ids,*d_tgt; CUDA_CHECK(cudaMalloc(&d_ids,BT*4)); CUDA_CHECK(cudaMalloc(&d_tgt,BT*4));
    CUDA_CHECK(cudaMemcpy(d_ids,ids.data(),BT*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt,tgt.data(),BT*4,cudaMemcpyHostToDevice));
    auto train_step=[&](int step){
      vt_forward(&t,d_patch,B);
      gpt_forward(&g,d_ids,B,T,t.img,n_patch);
      gpt_backward(&g,d_ids,d_tgt,B,T,/*zero=*/true,/*reduce=*/false,/*n_pre=*/n_patch,
                   /*valid_tokens=*/B*T_text);
      gather_prefix_grad_k<<<ceil_div(NP*c.C,256),256,0,g.stream>>>(t.dimg,g.dresid,B,T,n_patch,c.C);
      vt_backward(&t,d_patch,B);
      gpt_adamw(&g,1e-4f,0.9f,0.95f,1e-8f,0.1f,step);
      vt_adamw(&t,1e-4f,0.9f,0.95f,1e-8f,0.1f,step);
    };
    printf("MM bench: B=%d patches=%d text=%d LM %.1fM params vision %.1fM params steps=%d\n",
           B,n_patch,T_text,g.np/1e6,t.np/1e6,steps);
    for(int i=0;i<3;i++) train_step(i+1);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0=now_s();
    for(int i=0;i<steps;i++) train_step(i+10);
    CUDA_CHECK(cudaDeviceSynchronize());
    double dt=(now_s()-t0)/steps;
    float ls=gpt_mean_loss(&g,B,T,B*T_text);
    printf("mm step: %.2f ms | text tok/s %.0f | image tok/s %.0f | loss %.4f\n",
           dt*1e3, (double)B*T_text/dt, (double)B*n_patch/dt, ls);
    return 0;
  }

  if(mode=="bench"){
    // default ~124M-ish llama config; override via env for larger models
    Config c; c.V=env_int("ENTROPY_V",32768); c.C=env_int("ENTROPY_C",768);
    c.L=env_int("ENTROPY_L",12); c.H=env_int("ENTROPY_H",12);
    c.I=env_int("ENTROPY_I",2048); c.T=env_int("ENTROPY_T",1024);
    int B = argc>2? atoi(argv[2]) : 8;
    int steps = argc>3? atoi(argv[3]) : 20;
    GPT g; gpt_build(&g,c,B);
    gpt_init_weights(&g,1234);
    long BT=(long)B*c.T;
    std::vector<int> ids(BT),tgt(BT); RNG r(7);
    for(long i=0;i<BT;i++){ ids[i]=(int)(r.nf()*c.V); tgt[i]=(int)(r.nf()*c.V); }
    int *d_ids,*d_tgt; CUDA_CHECK(cudaMalloc(&d_ids,BT*4)); CUDA_CHECK(cudaMalloc(&d_tgt,BT*4));
    CUDA_CHECK(cudaMemcpy(d_ids,ids.data(),BT*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt,tgt.data(),BT*4,cudaMemcpyHostToDevice));
    printf("config: V=%d C=%d L=%d H=%d I=%d T=%d B=%d  params=%.1fM\n",
           c.V,c.C,c.L,c.H,c.I,c.T,B,g.np/1e6);
    // warmup
    int skip_adamw=env_int("ENTROPY_SKIP_ADAMW",0);
    int use_graph=env_int("ENTROPY_GRAPH",0);  // optional whole-step CUDA graph replay
    if(default_overlap_opt()){                // fuse AdamW into backward (side stream)
      g.overlap_opt=1; skip_adamw=1;
      g.o_lr=1e-4f; g.o_b1=0.9f; g.o_b2=0.95f; g.o_eps=1e-8f; g.o_wd=0.1f;
      g.o_bc1=1.0f-powf(0.9f,10); g.o_bc2=1.0f-powf(0.95f,10);
    }
  for(int i=0;i<3;i++){ gpt_forward(&g,d_ids,B,c.T); gpt_backward(&g,d_ids,d_tgt,B,c.T); if(!skip_adamw) gpt_adamw(&g,1e-4f,0.9f,0.95f,1e-8f,0.1f,i+1);}
    CUDA_CHECK(cudaDeviceSynchronize());
    if(env_int("ENTROPY_PROF",0)){   // one event-instrumented step -> per-region breakdown
      Prof prof; prof.s=g.stream; prof.on=true; g_prof=&prof; prof.mark("step_start");
      gpt_forward(&g,d_ids,B,c.T); gpt_backward(&g,d_ids,d_tgt,B,c.T);
      if(!skip_adamw){ gpt_adamw(&g,1e-4f,0.9f,0.95f,1e-8f,0.1f,10); prof.mark("adamw"); }
      prof.report(); g_prof=nullptr;
    }
    // breakdown: forward-only
    double tf0=now_s();
    for(int i=0;i<steps;i++){ gpt_forward(&g,d_ids,B,c.T); }
    CUDA_CHECK(cudaDeviceSynchronize());
    double dtf=(now_s()-tf0)/steps;
    double dt;
    if(use_graph){
      // Capture forward+backward(+adamw) into a CUDA graph, then replay. Removes
      // per-kernel CPU launch latency / GPU bubbles (what XLA whole-program buys).
      cudaGraph_t graph; cudaGraphExec_t exec;
      CUDA_CHECK(cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal));
      gpt_forward(&g,d_ids,B,c.T);
      gpt_backward(&g,d_ids,d_tgt,B,c.T);
      if(!skip_adamw) gpt_adamw(&g,1e-4f,0.9f,0.95f,1e-8f,0.1f,10);
      CUDA_CHECK(cudaStreamEndCapture(g.stream,&graph));
      CUDA_CHECK(cudaGraphInstantiate(&exec,graph,0));
      CUDA_CHECK(cudaGraphLaunch(exec,g.stream));     // extra warmup replay
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
      double t0=now_s();
      for(int i=0;i<steps;i++) CUDA_CHECK(cudaGraphLaunch(exec,g.stream));
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
      dt=(now_s()-t0)/steps;
      cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    } else {
      double t0=now_s();
      for(int i=0;i<steps;i++){ gpt_forward(&g,d_ids,B,c.T); gpt_backward(&g,d_ids,d_tgt,B,c.T); if(!skip_adamw) gpt_adamw(&g,1e-4f,0.9f,0.95f,1e-8f,0.1f,i+10);}
      CUDA_CHECK(cudaDeviceSynchronize());
      dt=(now_s()-t0)/steps;
    }
    printf("  [fwd-only %.2f ms | fwd+bwd+opt %.2f ms%s]\n", dtf*1e3, dt*1e3, use_graph?" (graph)":"");
    double tok=(double)B*c.T;
    double fl=flops_per_step(c,B);
    double peak=peak_flops_per_gpu();
    printf("step time: %.2f ms   throughput: %.0f tok/s   TFLOP/s: %.1f   MFU: %.1f%%\n",
           dt*1e3, tok/dt, fl/dt/1e12, 100.0*fl/dt/peak);
    return 0;
  }

  if(mode=="train"){
    const char* data = argc>2? argv[2] : "/project/inniang/jaxchat/data/fineweb32k_real_29/fineweb_train_000000.bin";
    int steps = argc>3? atoi(argv[3]) : 200;
    Config c; c.V=env_int("ENTROPY_V",32768); c.C=env_int("ENTROPY_C",768);
    c.L=env_int("ENTROPY_L",12); c.H=env_int("ENTROPY_H",12);
    c.I=env_int("ENTROPY_I",2048); c.T=env_int("ENTROPY_T",1024);
    int B=env_int("ENTROPY_B",8);
    GPT g; gpt_build(&g,c,B); gpt_init_weights(&g,1234);
    Loader ld; loader_open(&ld,data,B,c.T);
    long BT=(long)B*c.T;
    BatchPipe pipe; batch_pipe_init(&pipe,&ld);
    int train_graph=env_int("ENTROPY_TRAIN_GRAPH",0);
    batch_pipe_prefetch_initial(&pipe,0);
    if(steps>1 || train_graph) batch_pipe_prefetch_initial(&pipe,1);
    int log_every=env_int("ENTROPY_LOG_EVERY",10);
    int skip_adamw=0;
    if(default_overlap_opt() && !train_graph){
      g.overlap_opt=1; skip_adamw=1;
      g.o_lr=3e-4f; g.o_b1=0.9f; g.o_b2=0.95f; g.o_eps=1e-8f; g.o_wd=0.1f;
    }
    printf("training %d steps: V=%d C=%d L=%d H=%d I=%d T=%d B=%d params=%.1fM async_input=on overlap_opt=%d train_graph=%d\n",
           steps,c.V,c.C,c.L,c.H,c.I,c.T,B,g.np/1e6,g.overlap_opt,train_graph);
    cudaGraph_t train_graphs[2]{};
    cudaGraphExec_t train_execs[2]{};
    if(train_graph){
      // Warm cuBLASLt/cuDNN plans before capture. AdamW is intentionally outside
      // the graph because its bias-correction scalars change each optimizer step.
      batch_pipe_wait(&pipe,0,g.stream);
      gpt_forward(&g,pipe.d_ids[0],B,c.T);
      gpt_backward(&g,pipe.d_ids[0],pipe.d_tgt[0],B,c.T);
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
      for(int slot=0; slot<2; slot++){
        CUDA_CHECK(cudaEventSynchronize(pipe.ready[slot]));
        CUDA_CHECK(cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal));
        gpt_forward(&g,pipe.d_ids[slot],B,c.T);
        gpt_backward(&g,pipe.d_ids[slot],pipe.d_tgt[slot],B,c.T);
        CUDA_CHECK(cudaStreamEndCapture(g.stream,&train_graphs[slot]));
        CUDA_CHECK(cudaGraphInstantiate(&train_execs[slot],train_graphs[slot],0));
      }
      // Capture records the device pointers, not the batch contents. Refresh both
      // static slots so the timed/train loop does not reuse the warmup batch.
      batch_pipe_prefetch_initial(&pipe,0);
      batch_pipe_prefetch_initial(&pipe,1);
    }
    double t0=now_s(), w0=t0; int wsteps=0;
    for(int step=0;step<steps;step++){
      int slot=step&1;
      batch_pipe_wait(&pipe,slot,g.stream);
      int *d_ids=pipe.d_ids[slot], *d_tgt=pipe.d_tgt[slot];
      float lr=3e-4f;
      if(train_graph){
        CUDA_CHECK(cudaGraphLaunch(train_execs[slot],g.stream));
        CUDA_CHECK(cudaEventRecord(pipe.consumed[slot],g.stream));
        if(step+2<steps) batch_pipe_refill_after_consumed_event(&pipe,slot);
        gpt_adamw(&g,lr,0.9f,0.95f,1e-8f,0.1f,step+1);
      } else {
        gpt_forward(&g,d_ids,B,c.T);
        if(g.overlap_opt){
          g.o_lr=lr; g.o_bc1=1.0f-powf(g.o_b1,(float)(step+1)); g.o_bc2=1.0f-powf(g.o_b2,(float)(step+1));
        }
        gpt_backward(&g,d_ids,d_tgt,B,c.T);
        if(!skip_adamw) gpt_adamw(&g,lr,0.9f,0.95f,1e-8f,0.1f,step+1);
        if(step+2<steps) batch_pipe_refill_after_compute(&pipe,slot,g.stream);
      }
      wsteps++;
      if(step%log_every==0 || step==steps-1){ float ls=gpt_mean_loss(&g,B,c.T);
        double t=now_s(), dt=t-w0;
        double tok=(double)BT*wsteps, fl=flops_per_step(c,B)*wsteps;
        double peak=peak_flops_per_gpu();
        printf("step %4d  loss %.4f  %.2f ms/step  %.0f tok/s  %.1f TFLOP/s  MFU %.1f%%  elapsed %.2f s\n",
               step,ls,dt*1e3/wsteps,tok/dt,fl/dt/1e12,100.0*fl/dt/peak,t-t0);
        w0=t; wsteps=0;
      }
    }
    if(train_graph){
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
      for(int slot=0; slot<2; slot++){
        cudaGraphExecDestroy(train_execs[slot]);
        cudaGraphDestroy(train_graphs[slot]);
      }
    }
    return 0;
  }

  if(mode=="bench_ddp" || mode=="train_ddp"){
    // Multi-GPU data parallel. Rank/world/local from SLURM (or RANK/WORLD/LOCAL_RANK env).
    int rank=env_int("SLURM_PROCID", env_int("RANK",0));
    int world=env_int("SLURM_NTASKS", env_int("WORLD",1));
    int local=env_int("SLURM_LOCALID", env_int("LOCAL_RANK",0));
    CUDA_CHECK(cudaSetDevice(local));
    // The atomic RMSNorm-bwd partial path wins single-GPU latency. Older 5-GPU
    // RTX Ada DDP runs preferred the atomic-free column reduction, but the packed
    // atomic path is faster on H100 DDP. Keep an env override so A/B runs stay easy.
#ifdef ENTROPY_BUILD_SM
    if(ENTROPY_BUILD_SM < 90)
      setenv("ENTROPY_RMS_ATOMIC","0",/*overwrite=*/0);
#else
    setenv("ENTROPY_RMS_ATOMIC","0",/*overwrite=*/0);
#endif
    const char* idfile = getenv("ENTROPY_NCCL_ID");
    std::string idf;
    if(idfile) idf = idfile;
    else if(const char* job=getenv("SLURM_JOB_ID")){
      const char* step=getenv("SLURM_STEP_ID");
      idf = std::string("/project/inniang/entropy/.nccl_id_") + job + "_" + (step?step:"0");
    } else {
      idf = "/project/inniang/entropy/.nccl_id";
    }
    DDP ddp; ddp_init(&ddp, rank, world, local, idf.c_str());
    Config c; c.V=32768; c.C=768; c.L=12; c.H=12; c.I=2048; c.T=1024;
    int B = argc>2? atoi(argv[2]) : 16;
    int steps = argc>3? atoi(argv[3]) : 30;
    int accum = env_int("ENTROPY_ACCUM", 1);   // gradient accumulation microbatches
    GPT g; gpt_build(&g,c,B); gpt_init_weights(&g,1234); g.ddp=&ddp;
    g.bf16_reduce = env_int("ENTROPY_BF16_REDUCE", 1);
    int ddp_overlap_opt = env_int("ENTROPY_DDP_OVERLAP_OPT", 0);
    if(ddp_overlap_opt){
      g.overlap_opt=1;
      g.o_lr=1e-4f; g.o_b1=0.9f; g.o_b2=0.95f; g.o_eps=1e-8f; g.o_wd=0.1f;
      g.o_grad_scale=1.0f/(float)(world*accum);
    }
    long BT=(long)B*c.T;
    std::vector<int> ids(BT),tgt(BT); RNG r(7+rank*131);
    for(long i=0;i<BT;i++){ ids[i]=(int)(r.nf()*c.V); tgt[i]=(int)(r.nf()*c.V); }
    int *d_ids,*d_tgt; CUDA_CHECK(cudaMalloc(&d_ids,BT*4)); CUDA_CHECK(cudaMalloc(&d_tgt,BT*4));
    CUDA_CHECK(cudaMemcpy(d_ids,ids.data(),BT*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt,tgt.data(),BT*4,cudaMemcpyHostToDevice));
    if(rank==0) printf("DDP world=%d  per-GPU B=%d  accum=%d  global tokens/opt-step=%d  params=%.1fM overlap_opt=%d\n",
                       world,B,accum,B*c.T*world*accum,g.np/1e6,ddp_overlap_opt);
    auto opt_step=[&](int t){
      g.ddp_cast_scale = (g.bf16_reduce && !ddp_overlap_opt) ? 1.0f/(float)(world*accum) : 1.0f;
      if(ddp_overlap_opt){
        g.o_bc1=1.0f-powf(g.o_b1,(float)t);
        g.o_bc2=1.0f-powf(g.o_b2,(float)t);
        g.o_grad_scale=1.0f/(float)(world*accum);
      }
      for(int k=0;k<accum;k++){
        gpt_forward(&g,d_ids,B,c.T);
        gpt_backward(&g,d_ids,d_tgt,B,c.T, /*zero=*/k==0, /*reduce=*/k==accum-1);
      }
      ddp_finish(&g, accum);
      if(!ddp_overlap_opt) gpt_adamw(&g,1e-4f,0.9f,0.95f,1e-8f,0.1f,t);
    };
    for(int i=0;i<3;i++) opt_step(i+1);             // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0=now_s();
    for(int i=0;i<steps;i++) opt_step(i+10);
    CUDA_CHECK(cudaDeviceSynchronize());
    double dt=(now_s()-t0)/steps;
    if(rank==0){
      double gtok=(double)B*c.T*world*accum;          // global tokens / optimizer step
      double fl=flops_per_step(c,B)*world*accum;       // global flops / optimizer step
      double peak=peak_flops_per_gpu()*world;
      printf("opt-step %.2f ms | global %.0f tok/s | %.1f TFLOP/s agg | MFU %.1f%% | per-GPU %.0f tok/s\n",
             dt*1e3, gtok/dt, fl/dt/1e12, 100.0*fl/dt/peak, (double)B*c.T*accum/dt);
    }
    if(world>1) ncclCommDestroy(ddp.comm);
    return 0;
  }

  printf("unknown mode %s (use overfit|mm_overfit|mm_bench|bench|train|bench_ddp|train_ddp|plan)\n",mode.c_str());
  return 1;
}
