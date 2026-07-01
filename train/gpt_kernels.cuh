// entropy — CUDA kernels + cuBLASLt matmul helpers for a modern (llama-style)
// GPT decoder trained in mixed precision (BF16 activations/params, FP32 master
// weights + FP32 weight grads). All linear layers are bias-free.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <unordered_map>

typedef __nv_bfloat16 bf16;

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)
#define CUBLAS_CHECK(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1);} }while(0)

// ----------------------------------------------------------------------------
// cuBLASLt matmul: D = alpha*op(A)@op(B) + beta*C  (column-major, C aliases D)
// ----------------------------------------------------------------------------
// A cached matmul plan (descriptor + layouts + chosen algo) for one GEMM shape.
struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t LA,LB,LD;
  cublasLtMatmulAlgo_t algo; bool have_algo;
};
struct LtCtx {
  cublasLtHandle_t handle;
  cublasHandle_t   blas;   // legacy handle for batched GEMM
  void*  workspace;
  size_t wssize;
  std::unordered_map<uint64_t,LtPlan> cache;  // shape -> plan (avoids per-call heuristic)
  void* tune_scratch=nullptr; size_t tune_sz=0;  // scratch D for autotuning (beta=0)
  int autotune=1;          // time top-K algos and cache the fastest
  int autotune_candidates=24;
};

static void lt_init(LtCtx* c, size_t wssize = (size_t)512*1024*1024) {
  CUBLAS_CHECK(cublasLtCreate(&c->handle));
  CUBLAS_CHECK(cublasCreate(&c->blas));
  CUBLAS_CHECK(cublasSetMathMode(c->blas, CUBLAS_TENSOR_OP_MATH));
  const char* ws=getenv("ENTROPY_LT_WS_MB"); if(ws&&atoi(ws)>0) wssize=(size_t)atoi(ws)*1024*1024;
  c->wssize = wssize;
  CUDA_CHECK(cudaMalloc(&c->workspace, wssize));
  const char* at=getenv("ENTROPY_AUTOTUNE"); c->autotune = at? (at[0]!='0') : 1;
  const char* kc=getenv("ENTROPY_AUTOTUNE_K"); c->autotune_candidates = kc? atoi(kc) : 24;
  if(c->autotune_candidates < 1) c->autotune_candidates = 1;
  if(c->autotune_candidates > 128) c->autotune_candidates = 128;
}

// Batched strided BF16 tensor-core GEMM (compute FP32): per batch,
// D = alpha*op(A)@op(B) + beta*D, column-major, used for attention.
static void bmm(LtCtx* c, cudaStream_t stream,
                cublasOperation_t ta, cublasOperation_t tb,
                int m,int n,int k,float alpha,
                const bf16* A,int lda,long sA,
                const bf16* B,int ldb,long sB,float beta,
                bf16* D,int ldd,long sD,int batch){
  CUBLAS_CHECK(cublasSetStream(c->blas,stream));
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(c->blas, ta, tb, m,n,k,&alpha,
      A,CUDA_R_16BF,lda,sA, B,CUDA_R_16BF,ldb,sB, &beta,
      D,CUDA_R_16BF,ldd,sD, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static uint64_t lt_key(int m,int n,int k,cublasOperation_t ta,cublasOperation_t tb,
                       int lda,int ldb,int ldd,cudaDataType_t dA,cudaDataType_t dB,cudaDataType_t dD,
                       const void* a_scale=nullptr, const void* b_scale=nullptr){
  uint64_t h=1469598103934665603ULL;
  int vals[]={m,n,k,(int)ta,(int)tb,lda,ldb,ldd,(int)dA,(int)dB,(int)dD};
  for(int v: vals){ h=(h^(uint32_t)v)*1099511628211ULL; }
  uintptr_t ptrs[]={reinterpret_cast<uintptr_t>(a_scale), reinterpret_cast<uintptr_t>(b_scale)};
  for(uintptr_t p: ptrs){
    h=(h^(uint32_t)p)*1099511628211ULL;
    h=(h^(uint32_t)(p>>32))*1099511628211ULL;
  }
  return h;
}
static size_t lt_dtsize(cudaDataType_t t){
  return t==CUDA_R_32F ? 4 : (t==CUDA_R_16BF || t==CUDA_R_16F ? 2 :
         (t==CUDA_R_8F_E4M3 || t==CUDA_R_8F_E5M2 ? 1 : 4));
}

static void lt_matmul(LtCtx* c, cudaStream_t stream,
                      const void* A, const void* B, void* D,
                      int m, int n, int k,
                      cublasOperation_t ta, cublasOperation_t tb,
                      int lda, int ldb, int ldd,
                      cudaDataType_t dA, cudaDataType_t dB, cudaDataType_t dD,
                      float alpha, float beta,
                      const float* a_scale=nullptr, const float* b_scale=nullptr) {
  uint64_t key=lt_key(m,n,k,ta,tb,lda,ldb,ldd,dA,dB,dD,a_scale,b_scale);
  auto it=c->cache.find(key);
  if(it==c->cache.end()){
    LtPlan p; p.have_algo=false;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    if(a_scale){
      int smode=CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode)));
    }
    if(b_scale){
      int smode=CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode)));
    }
    if (ta == CUBLAS_OP_N) CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.LA, dA, m, k, lda));
    else                   CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.LA, dA, k, m, lda));
    if (tb == CUBLAS_OP_N) CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.LB, dB, k, n, ldb));
    else                   CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.LB, dB, n, k, ldb));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.LD, dD, m, n, ldd));
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c->wssize, sizeof(c->wssize)));
    cublasLtMatmulHeuristicResult_t heur[128]; int nres=0;
    cublasLtMatmulAlgoGetHeuristic(c->handle, p.op, p.LA, p.LB, p.LD, p.LD, pref,
                                   c->autotune_candidates, heur, &nres);
    int chosen = nres>0 ? 0 : -1;
    if(c->autotune && nres>1){
      // Time each candidate algo into a scratch D (beta=0, no read of C) and keep
      // the fastest. cuBLASLt heuristics order != actual fastest; this is XLA-style
      // autotuning. Runs once per unique GEMM shape (during warmup).
      size_t need=(size_t)m*n*lt_dtsize(dD);
      if(need>c->tune_sz){ if(c->tune_scratch) cudaFree(c->tune_scratch);
        if(cudaMalloc(&c->tune_scratch,need)!=cudaSuccess){ c->tune_scratch=nullptr; c->tune_sz=0; }
        else c->tune_sz=need; }
      if(c->tune_scratch){
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float a1=1.0f,b0=0.0f; float best_ms=1e30f;
        for(int i=0;i<nres;i++){
          if(heur[i].state!=CUBLAS_STATUS_SUCCESS) continue;
          // 2 warmup + 5 timed iters
          bool ok=true;
          for(int w=0;w<2;w++){ if(cublasLtMatmul(c->handle,p.op,&a1,A,p.LA,B,p.LB,&b0,
                c->tune_scratch,p.LD,c->tune_scratch,p.LD,&heur[i].algo,c->workspace,c->wssize,stream)!=CUBLAS_STATUS_SUCCESS){ok=false;break;} }
          if(!ok) continue;
          cudaEventRecord(e0,stream);
          for(int t=0;t<5;t++) cublasLtMatmul(c->handle,p.op,&a1,A,p.LA,B,p.LB,&b0,
                c->tune_scratch,p.LD,c->tune_scratch,p.LD,&heur[i].algo,c->workspace,c->wssize,stream);
          cudaEventRecord(e1,stream); cudaEventSynchronize(e1);
          float ms; cudaEventElapsedTime(&ms,e0,e1);
          if(ms<best_ms){ best_ms=ms; chosen=i; }
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
      }
    }
    if(chosen>=0){ p.algo=heur[chosen].algo; p.have_algo=true; }
    cublasLtMatmulPreferenceDestroy(pref);
    c->cache[key]=p; it=c->cache.find(key);
  }
  LtPlan& p=it->second;
  CUBLAS_CHECK(cublasLtMatmul(c->handle, p.op, &alpha, A, p.LA, B, p.LB, &beta,
                              D, p.LD, D, p.LD, p.have_algo?&p.algo:nullptr,
                              c->workspace, c->wssize, stream));
}

// Linear forward:  out[M,N] = inp[M,K] @ W[N,K]^T   (W stored row-major [N,K])
static void linear_forward(LtCtx* c, cudaStream_t s, const bf16* inp, const bf16* W,
                           bf16* out, int M, int N, int K) {
  lt_matmul(c, s, W, inp, out, N, M, K, CUBLAS_OP_T, CUBLAS_OP_N, K, K, N,
            CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 1.0f, 0.0f);
}
// FP8 E4M3 forward: out[M,N] = inp8[M,K] @ W8[N,K]^T, BF16 output.
static void linear_forward_fp8(LtCtx* c, cudaStream_t s, const __nv_fp8_e4m3* inp,
                               const __nv_fp8_e4m3* W, bf16* out, int M, int N, int K,
                               const float* inp_dequant=nullptr, const float* w_dequant=nullptr) {
  lt_matmul(c, s, W, inp, out, N, M, K, CUBLAS_OP_T, CUBLAS_OP_N, K, K, N,
            CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, 1.0f, 0.0f, w_dequant, inp_dequant);
}
// dInp[M,K] = dOut[M,N] @ W[N,K]   (writes bf16 dInp; beta to accumulate)
static void linear_backward_inp(LtCtx* c, cudaStream_t s, const bf16* dOut, const bf16* W,
                                bf16* dInp, int M, int N, int K, float beta) {
  lt_matmul(c, s, W, dOut, dInp, K, M, N, CUBLAS_OP_N, CUBLAS_OP_N, K, N, K,
            CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 1.0f, beta);
}
// dInp[M,K] += dOut[M,N] @ W[N,K], accumulating in FP32 scratch.
static void linear_backward_inp_accum_f32(LtCtx* c, cudaStream_t s, const bf16* dOut, const bf16* W,
                                          float* dInp, int M, int N, int K, float beta) {
  lt_matmul(c, s, W, dOut, dInp, K, M, N, CUBLAS_OP_N, CUBLAS_OP_N, K, N, K,
            CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, 1.0f, beta);
}
// dW[N,K] += dOut[M,N]^T @ inp[M,K]   (FP32 dW accumulate)
static void linear_backward_weight(LtCtx* c, cudaStream_t s, const bf16* dOut, const bf16* inp,
                                   float* dW, int M, int N, int K, float beta=1.0f) {
  lt_matmul(c, s, inp, dOut, dW, K, N, M, CUBLAS_OP_N, CUBLAS_OP_T, K, N, K,
            CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, 1.0f, beta);
}

// ----------------------------------------------------------------------------
// elementwise / norm / activation kernels
// ----------------------------------------------------------------------------
__global__ void cast_f2b(bf16* dst, const float* src, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) dst[i] = __float2bfloat16(src[i]);
}
__global__ void cast_f2b_fp8_e4m3(bf16* dst, __nv_fp8_e4m3* dst8, const float* src, size_t n, float fp8_scale){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n){
    float v=src[i];
    dst[i]=__float2bfloat16(v);
    dst8[i]=__nv_fp8_e4m3(v*fp8_scale);
  }
}
__global__ void cast_bf16_to_fp8_e4m3_k(__nv_fp8_e4m3* dst, const bf16* src, size_t n, float fp8_scale){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) dst[i]=__nv_fp8_e4m3(__bfloat162float(src[i])*fp8_scale);
}
__global__ void cast_b2f(float* dst, const bf16* src, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) dst[i] = __bfloat162float(src[i]);
}
__global__ void cast_b2f_scale(float* dst, const bf16* src, size_t n, float scale){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) dst[i] = __bfloat162float(src[i]) * scale;
}

// encoder: out[BT,C] = wte[ids[BT], :]
__global__ void encoder_forward_k(bf16* out, const int* ids, const bf16* wte, int BT, int C){
  int row = blockIdx.x; if(row>=BT) return;
  int id = ids[row];
  const bf16* src = wte + (size_t)id*C;
  bf16* dst = out + (size_t)row*C;
  for(int j=threadIdx.x;j<C;j+=blockDim.x) dst[j]=src[j];
}
// dwte[ids[row],:] += dout[row,:]   (FP32 accumulate). Skips the first `n_pre`
// positions of each T-length sample (those are image tokens, not text).
__global__ void encoder_backward_k(float* dwte, const int* ids, const bf16* dout,
                                   int BT, int C, int T, int n_pre){
  int row = blockIdx.x; if(row>=BT) return;
  if(n_pre>0 && (row % T) < n_pre) return;   // image-token row: handled by vision bwd
  int id = ids[row];
  float* dst = dwte + (size_t)id*C;
  const bf16* src = dout + (size_t)row*C;
  for(int j=threadIdx.x;j<C;j+=blockDim.x) atomicAdd(&dst[j], __bfloat162float(src[j]));
}
// Overwrite the first n_pre rows of each T-length sample with prefix embeds.
// encoded[(b*T+p)*C+c] = prefix[(b*n_pre+p)*C+c]  for p<n_pre
__global__ void set_prefix_embeds_k(bf16* encoded, const bf16* prefix,
                                    int B,int T,int n_pre,int C){
  long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; long tot=(long)B*n_pre*C;
  if(idx>=tot) return;
  int c=idx%C; long r=idx/C; int p=r%n_pre; int b=r/n_pre;
  encoded[((long)b*T+p)*C+c]=prefix[((long)b*n_pre+p)*C+c];
}
// Gather grad of the prefix rows: d_prefix[(b*n_pre+p)*C+c] = dresid[(b*T+p)*C+c]
__global__ void gather_prefix_grad_k(bf16* d_prefix, const bf16* dresid,
                                     int B,int T,int n_pre,int C){
  long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; long tot=(long)B*n_pre*C;
  if(idx>=tot) return;
  int c=idx%C; long r=idx/C; int p=r%n_pre; int b=r/n_pre;
  d_prefix[((long)b*n_pre+p)*C+c]=dresid[((long)b*T+p)*C+c];
}
// add bias-free broadcast: x[BT,C] += pos[ (row%n) , :]  (learned pos embed over n rows)
__global__ void add_posembed_k(bf16* x, const bf16* pos, int rows, int n, int C){
  long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; long tot=(long)rows*C;
  if(idx>=tot) return; int c=idx%C; long r=idx/C; int pr=r%n;
  x[idx]=__float2bfloat16(__bfloat162float(x[idx])+__bfloat162float(pos[(long)pr*C+c]));
}
__global__ void posembed_grad_k(float* dpos, const bf16* dx, int rows, int n, int C){
  // dpos[p,c] += sum over rows with row%n==p of dx
  long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; long tot=(long)rows*C;
  if(idx>=tot) return; int c=idx%C; long r=idx/C; int pr=r%n;
  atomicAdd(&dpos[(long)pr*C+c], __bfloat162float(dx[idx]));
}

#define RMS_VPT 16   // max elements/thread (blockDim 256 -> C up to 4096)
// RMSNorm forward (one block per row), single global read cached in registers.
__global__ void rmsnorm_forward_k(bf16* out, float* rstd, const bf16* x, const bf16* w,
                                  int N, int C, float eps){
  int row = blockIdx.x; if(row>=N) return;
  const bf16* xr = x + (size_t)row*C;
  bf16* orow = out + (size_t)row*C;
  __shared__ float sh[256];
  float vals[RMS_VPT]; int cnt=0; float acc=0.f;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){ float v=__bfloat162float(xr[j]); vals[cnt++]=v; acc+=v*v; }
  sh[threadIdx.x]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float rs = rsqrtf(sh[0]/C + eps);
  if(threadIdx.x==0) rstd[row]=rs;
  cnt=0;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){ orow[j]=__float2bfloat16(vals[cnt++]*rs*__bfloat162float(w[j])); }
}
// RMSNorm producer with fused FP8 activation cache for downstream FP8 GEMMs.
__global__ void rmsnorm_forward_fp8_k(bf16* out, __nv_fp8_e4m3* out8, float* rstd,
                                      const bf16* x, const bf16* w,
                                      int N, int C, float eps, float fp8_scale){
  int row = blockIdx.x; if(row>=N) return;
  const bf16* xr = x + (size_t)row*C;
  bf16* orow = out + (size_t)row*C;
  __nv_fp8_e4m3* o8 = out8 + (size_t)row*C;
  __shared__ float sh[256];
  float vals[RMS_VPT]; int cnt=0; float acc=0.f;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){ float v=__bfloat162float(xr[j]); vals[cnt++]=v; acc+=v*v; }
  sh[threadIdx.x]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float rs = rsqrtf(sh[0]/C + eps);
  if(threadIdx.x==0) rstd[row]=rs;
  cnt=0;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){
    float v=vals[cnt++]*rs*__bfloat162float(w[j]);
    orow[j]=__float2bfloat16(v);
    o8[j]=__nv_fp8_e4m3(v*fp8_scale);
  }
}
// RMSNorm backward, atomic-free. dx kernel (one block/row) + dweight via partial reduction.
// `add` folds the residual passthrough into the dx write: dx = add + dnorm (add=nullptr
// -> overwrite; add==dx -> accumulate). This removes the separate copy_k passthrough
// kernels that previously seeded the residual gradient.
__global__ void rmsnorm_dx_k(bf16* dx, const bf16* add, const bf16* dout, const bf16* x, const bf16* w,
                             const float* rstd, int N, int C){
  int row = blockIdx.x; if(row>=N) return;
  const bf16* xr=x+(size_t)row*C; const bf16* dr=dout+(size_t)row*C; bf16* dxr=dx+(size_t)row*C;
  const bf16* ar=add? add+(size_t)row*C : nullptr;
  float rs=rstd[row]; __shared__ float sh[256];
  float dot=0.f;
  for(int j=threadIdx.x;j<C;j+=blockDim.x) dot += __bfloat162float(dr[j])*__bfloat162float(w[j])*__bfloat162float(xr[j]);
  sh[threadIdx.x]=dot; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float dotv=sh[0]; float invC=1.0f/C;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){
    float xj=__bfloat162float(xr[j]), wj=__bfloat162float(w[j]), dj=__bfloat162float(dr[j]);
    float dval = rs*wj*dj - xj*rs*rs*rs*invC*dotv;
    dxr[j]=__float2bfloat16((ar? __bfloat162float(ar[j]):0.0f)+dval);
  }
}
// RMSNorm backward variant: compute dx and dweight partials in the same row-wise
// pass. This trades a second coalesced activation read for atomic adds into
// partial[row % R, c]; the GPT wrapper enables it for single-GPU runs after
// benchmarking faster on RTX 6000 Ada.
__global__ void rmsnorm_dx_dweight_atomic_partial_k(float* partial, bf16* dx, const bf16* add,
                                                    const bf16* dout, const bf16* x, const bf16* w,
                                                    const float* rstd, int N, int C, int R){
  int row = blockIdx.x; if(row>=N) return;
  const bf16* xr=x+(size_t)row*C; const bf16* dr=dout+(size_t)row*C; bf16* dxr=dx+(size_t)row*C;
  const bf16* ar=add? add+(size_t)row*C : nullptr;
  float rs=rstd[row]; __shared__ float sh[256];
  float dot=0.f;
  for(int j=threadIdx.x;j<C;j+=blockDim.x) dot += __bfloat162float(dr[j])*__bfloat162float(w[j])*__bfloat162float(xr[j]);
  sh[threadIdx.x]=dot; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float dotv=sh[0]; float invC=1.0f/C; int pr=row%R;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){
    float xj=__bfloat162float(xr[j]), wj=__bfloat162float(w[j]), dj=__bfloat162float(dr[j]);
    float dval = rs*wj*dj - xj*rs*rs*rs*invC*dotv;
    dxr[j]=__float2bfloat16((ar? __bfloat162float(ar[j]):0.0f)+dval);
    atomicAdd(&partial[(long)pr*C+j], dj*xj*rs);
  }
}
__global__ void rmsnorm_dx_dweight_atomic_partial2_k(float* partial, bf16* dx, const bf16* add,
                                                     const bf16* dout, const bf16* x, const bf16* w,
                                                     const float* rstd, int N, int C, int R){
  int row = blockIdx.x; if(row>=N) return;
  int C2=C>>1;
  const __nv_bfloat162* xr2=reinterpret_cast<const __nv_bfloat162*>(x+(size_t)row*C);
  const __nv_bfloat162* dr2=reinterpret_cast<const __nv_bfloat162*>(dout+(size_t)row*C);
  const __nv_bfloat162* w2=reinterpret_cast<const __nv_bfloat162*>(w);
  const __nv_bfloat162* ar2=add ? reinterpret_cast<const __nv_bfloat162*>(add+(size_t)row*C) : nullptr;
  __nv_bfloat162* dx2=reinterpret_cast<__nv_bfloat162*>(dx+(size_t)row*C);
  float rs=rstd[row]; __shared__ float sh[256];
  float dot=0.f;
  for(int j2=threadIdx.x;j2<C2;j2+=blockDim.x){
    float2 xv=__bfloat1622float2(xr2[j2]);
    float2 dv=__bfloat1622float2(dr2[j2]);
    float2 wv=__bfloat1622float2(w2[j2]);
    dot += dv.x*wv.x*xv.x + dv.y*wv.y*xv.y;
  }
  sh[threadIdx.x]=dot; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float dotv=sh[0]; float invC=1.0f/C; int pr=row%R;
  float rs3=rs*rs*rs;
  for(int j2=threadIdx.x;j2<C2;j2+=blockDim.x){
    float2 xv=__bfloat1622float2(xr2[j2]);
    float2 dv=__bfloat1622float2(dr2[j2]);
    float2 wv=__bfloat1622float2(w2[j2]);
    float2 av=ar2 ? __bfloat1622float2(ar2[j2]) : make_float2(0.0f,0.0f);
    float d0 = rs*wv.x*dv.x - xv.x*rs3*invC*dotv;
    float d1 = rs*wv.y*dv.y - xv.y*rs3*invC*dotv;
    dx2[j2]=__floats2bfloat162_rn(av.x+d0,av.y+d1);
    int j=j2<<1;
    atomicAdd(&partial[(long)pr*C+j],   dv.x*xv.x*rs);
    atomicAdd(&partial[(long)pr*C+j+1], dv.y*xv.y*rs);
  }
}
__global__ void rmsnorm_dx_dweight_f32dout_atomic_partial_k(float* partial, bf16* dx,
                                                            const float* dout, const bf16* x,
                                                            const bf16* w, const float* rstd,
                                                            int N, int C, int R){
  int row = blockIdx.x; if(row>=N) return;
  const bf16* xr=x+(size_t)row*C; const float* dr=dout+(size_t)row*C; bf16* dxr=dx+(size_t)row*C;
  float rs=rstd[row]; __shared__ float sh[256];
  float dot=0.f;
  for(int j=threadIdx.x;j<C;j+=blockDim.x) dot += dr[j]*__bfloat162float(w[j])*__bfloat162float(xr[j]);
  sh[threadIdx.x]=dot; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float dotv=sh[0]; float invC=1.0f/C; int pr=row%R; float rs3=rs*rs*rs;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){
    float xj=__bfloat162float(xr[j]), wj=__bfloat162float(w[j]), dj=dr[j];
    float dval = rs*wj*dj - xj*rs3*invC*dotv;
    dxr[j]=__float2bfloat16(dval);
    atomicAdd(&partial[(long)pr*C+j], dj*xj*rs);
  }
}
__global__ void rmsnorm_dx_dweight_f32dout_atomic_partial2_k(float* partial, bf16* dx,
                                                             const float* dout, const bf16* x,
                                                             const bf16* w, const float* rstd,
                                                             int N, int C, int R){
  int row = blockIdx.x; if(row>=N) return;
  int C2=C>>1;
  const __nv_bfloat162* xr2=reinterpret_cast<const __nv_bfloat162*>(x+(size_t)row*C);
  const __nv_bfloat162* w2=reinterpret_cast<const __nv_bfloat162*>(w);
  __nv_bfloat162* dx2=reinterpret_cast<__nv_bfloat162*>(dx+(size_t)row*C);
  const float* dr=dout+(size_t)row*C;
  float rs=rstd[row]; __shared__ float sh[256];
  float dot=0.f;
  for(int j2=threadIdx.x;j2<C2;j2+=blockDim.x){
    float2 xv=__bfloat1622float2(xr2[j2]);
    float2 wv=__bfloat1622float2(w2[j2]);
    int j=j2<<1;
    dot += dr[j]*wv.x*xv.x + dr[j+1]*wv.y*xv.y;
  }
  sh[threadIdx.x]=dot; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float dotv=sh[0]; float invC=1.0f/C; int pr=row%R; float rs3=rs*rs*rs;
  for(int j2=threadIdx.x;j2<C2;j2+=blockDim.x){
    float2 xv=__bfloat1622float2(xr2[j2]);
    float2 wv=__bfloat1622float2(w2[j2]);
    int j=j2<<1;
    float d0 = rs*wv.x*dr[j]   - xv.x*rs3*invC*dotv;
    float d1 = rs*wv.y*dr[j+1] - xv.y*rs3*invC*dotv;
    dx2[j2]=__floats2bfloat162_rn(d0,d1);
    atomicAdd(&partial[(long)pr*C+j],   dr[j]*xv.x*rs);
    atomicAdd(&partial[(long)pr*C+j+1], dr[j+1]*xv.y*rs);
  }
}
// partial[r,c] = sum over rows≡r (mod R) of dout*x*rstd ; coalesced (thread=column)
__global__ void rmsnorm_dweight_partial_k(float* partial, const bf16* dout, const bf16* x,
                                          const float* rstd, int N, int C, int R){
  int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=C) return; int r=blockIdx.y;
  float acc=0.f;
  for(int row=r; row<N; row+=R)
    acc += __bfloat162float(dout[(long)row*C+c])*__bfloat162float(x[(long)row*C+c])*rstd[row];
  partial[(long)r*C+c]=acc;
}
// dst[c] += sum_r partial[r,c]
__global__ void reduce_cols_add_k(float* dst, const float* partial, int R, int C){
  int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=C) return;
  float acc=0.f; for(int r=0;r<R;r++) acc+=partial[(long)r*C+c];
  dst[c]+=acc;
}

// fused: resid = a + b ; norm = rmsnorm(resid, w)   (one block per row)
__global__ void add_rmsnorm_fwd_k(bf16* resid, bf16* norm, float* rstd,
                                  const bf16* a, const bf16* b, const bf16* w,
                                  int N, int C, float eps){
  int row=blockIdx.x; if(row>=N) return;
  const bf16* ar=a+(size_t)row*C; const bf16* br=b+(size_t)row*C;
  bf16* rr=resid+(size_t)row*C; bf16* nr=norm+(size_t)row*C;
  __shared__ float sh[256];
  float vals[RMS_VPT]; int cnt=0; float acc=0.f;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){ float v=__bfloat162float(ar[j])+__bfloat162float(br[j]); rr[j]=__float2bfloat16(v); vals[cnt++]=v; acc+=v*v; }
  sh[threadIdx.x]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float rs=rsqrtf(sh[0]/C+eps); if(threadIdx.x==0) rstd[row]=rs;
  cnt=0;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){ nr[j]=__float2bfloat16(vals[cnt++]*rs*__bfloat162float(w[j])); }
}
__global__ void add_rmsnorm_fwd_fp8_k(bf16* resid, bf16* norm, __nv_fp8_e4m3* norm8, float* rstd,
                                      const bf16* a, const bf16* b, const bf16* w,
                                      int N, int C, float eps, float fp8_scale){
  int row=blockIdx.x; if(row>=N) return;
  const bf16* ar=a+(size_t)row*C; const bf16* br=b+(size_t)row*C;
  bf16* rr=resid+(size_t)row*C; bf16* nr=norm+(size_t)row*C; __nv_fp8_e4m3* n8=norm8+(size_t)row*C;
  __shared__ float sh[256];
  float vals[RMS_VPT]; int cnt=0; float acc=0.f;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){ float v=__bfloat162float(ar[j])+__bfloat162float(br[j]); rr[j]=__float2bfloat16(v); vals[cnt++]=v; acc+=v*v; }
  sh[threadIdx.x]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float rs=rsqrtf(sh[0]/C+eps); if(threadIdx.x==0) rstd[row]=rs;
  cnt=0;
  for(int j=threadIdx.x;j<C;j+=blockDim.x){
    float v=vals[cnt++]*rs*__bfloat162float(w[j]);
    nr[j]=__float2bfloat16(v);
    n8[j]=__nv_fp8_e4m3(v*fp8_scale);
  }
}
// residual: out = a + b
__global__ void residual_forward_k(bf16* out, const bf16* a, const bf16* b, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i<n) out[i]=__float2bfloat16(__bfloat162float(a[i])+__bfloat162float(b[i]));
}
// copy
__global__ void copy_k(bf16* dst, const bf16* src, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i<n) dst[i]=src[i];
}
// add into dst (dst += src)
__global__ void add_k(bf16* dst, const bf16* src, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i<n) dst[i]=__float2bfloat16(__bfloat162float(dst[i])+__bfloat162float(src[i]));
}

// SwiGLU forward: out = silu(gate)*up   (gate,up,out [N,I])
__device__ __forceinline__ float siluf(float x){ return x/(1.0f+expf(-x)); }
__global__ void swiglu_forward_k(bf16* out, const bf16* gate, const bf16* up, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i>=n) return;
  float g=__bfloat162float(gate[i]); float u=__bfloat162float(up[i]);
  out[i]=__float2bfloat16(siluf(g)*u);
}
// SwiGLU backward. dout[N,I] -> dgate,dup. silu'(g)=sig*(1+g*(1-sig))
__global__ void swiglu_backward_k(bf16* dgate, bf16* dup, const bf16* dout,
                                  const bf16* gate, const bf16* up, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i>=n) return;
  float g=__bfloat162float(gate[i]); float u=__bfloat162float(up[i]); float d=__bfloat162float(dout[i]);
  float sig=1.0f/(1.0f+expf(-g));
  float silu=g*sig;
  float dsilu=sig*(1.0f+g*(1.0f-sig));
  dgate[i]=__float2bfloat16(d*u*dsilu);
  dup[i]=__float2bfloat16(d*silu);
}
// Fused gate+up variants: a single GEMM produces gu[BT,2I] with row layout
// [gate(I) | up(I)]. These read/write that interleaved layout directly, so the
// MLP up-projection is 1 GEMM (fwd) / 2 GEMMs (bwd) instead of 2 / 4.
// Launched with grid(BT, ceil(I/blk)), block(blk): row=blockIdx.x, column from
// the grid -> no per-thread division/modulo by I.
__global__ void swiglu_forward_gu_k(bf16* out, const bf16* gu, int BT, int I){
  int i=blockIdx.y*blockDim.x+threadIdx.x; if(i>=I) return; long r=blockIdx.x;
  const bf16* row=gu+r*2*(long)I;
  float g=__bfloat162float(row[i]); float u=__bfloat162float(row[(long)I+i]);
  out[r*(long)I+i]=__float2bfloat16(siluf(g)*u);
}
__global__ void swiglu_forward_gu2_k(bf16* out, const bf16* gu, int BT, int I){
  int i2=blockIdx.y*blockDim.x+threadIdx.x; int I2=I>>1; if(i2>=I2) return; long r=blockIdx.x;
  const bf16* row=gu+r*2*(long)I;
  const __nv_bfloat162* gate2=reinterpret_cast<const __nv_bfloat162*>(row);
  const __nv_bfloat162* up2=reinterpret_cast<const __nv_bfloat162*>(row+I);
  __nv_bfloat162* out2=reinterpret_cast<__nv_bfloat162*>(out+r*(long)I);
  float2 g=__bfloat1622float2(gate2[i2]);
  float2 u=__bfloat1622float2(up2[i2]);
  out2[i2]=__floats2bfloat162_rn(siluf(g.x)*u.x, siluf(g.y)*u.y);
}
__global__ void swiglu_forward_gu_fp8_k(bf16* out, __nv_fp8_e4m3* out8, const bf16* gu, int BT, int I, float fp8_scale){
  int i=blockIdx.y*blockDim.x+threadIdx.x; if(i>=I) return; long r=blockIdx.x;
  const bf16* row=gu+r*2*(long)I;
  float g=__bfloat162float(row[i]); float u=__bfloat162float(row[(long)I+i]);
  float v=siluf(g)*u;
  long o=r*(long)I+i;
  out[o]=__float2bfloat16(v);
  out8[o]=__nv_fp8_e4m3(v*fp8_scale);
}
__global__ void swiglu_backward_gu_k(bf16* dgu, const bf16* dout, const bf16* gu, int BT, int I){
  int i=blockIdx.y*blockDim.x+threadIdx.x; if(i>=I) return; long r=blockIdx.x;
  const bf16* row=gu+r*2*(long)I; bf16* drow=dgu+r*2*(long)I;
  float g=__bfloat162float(row[i]); float u=__bfloat162float(row[(long)I+i]); float d=__bfloat162float(dout[r*(long)I+i]);
  float sig=1.0f/(1.0f+expf(-g));
  float silu=g*sig;
  float dsilu=sig*(1.0f+g*(1.0f-sig));
  drow[i]=__float2bfloat16(d*u*dsilu);
  drow[(long)I+i]=__float2bfloat16(d*silu);
}
__global__ void swiglu_backward_gu2_k(bf16* dgu, const bf16* dout, const bf16* gu, int BT, int I){
  int i2=blockIdx.y*blockDim.x+threadIdx.x; int I2=I>>1; if(i2>=I2) return; long r=blockIdx.x;
  const bf16* row=gu+r*2*(long)I; bf16* drow=dgu+r*2*(long)I;
  const __nv_bfloat162* gate2=reinterpret_cast<const __nv_bfloat162*>(row);
  const __nv_bfloat162* up2=reinterpret_cast<const __nv_bfloat162*>(row+I);
  const __nv_bfloat162* dout2=reinterpret_cast<const __nv_bfloat162*>(dout+r*(long)I);
  __nv_bfloat162* dgate2=reinterpret_cast<__nv_bfloat162*>(drow);
  __nv_bfloat162* dup2=reinterpret_cast<__nv_bfloat162*>(drow+I);
  float2 g=__bfloat1622float2(gate2[i2]);
  float2 u=__bfloat1622float2(up2[i2]);
  float2 d=__bfloat1622float2(dout2[i2]);
  float sig0=1.0f/(1.0f+expf(-g.x));
  float sig1=1.0f/(1.0f+expf(-g.y));
  float silu0=g.x*sig0, silu1=g.y*sig1;
  float dsilu0=sig0*(1.0f+g.x*(1.0f-sig0));
  float dsilu1=sig1*(1.0f+g.y*(1.0f-sig1));
  dgate2[i2]=__floats2bfloat162_rn(d.x*u.x*dsilu0, d.y*u.y*dsilu1);
  dup2[i2]=__floats2bfloat162_rn(d.x*silu0, d.y*silu1);
}

// RoPE precompute cos/sin: [T, hd/2]
__global__ void rope_precompute_k(float* cosb, float* sinb, int T, int hd, float theta){
  int t=blockIdx.x; int i=threadIdx.x; int half=hd/2;
  if(t>=T||i>=half) return;
  float freq = powf(theta, -2.0f*i/(float)hd);
  float ang = t*freq;
  cosb[t*half+i]=cosf(ang); sinb[t*half+i]=sinf(ang);
}
// Apply RoPE in place (rotate-half, llama/HF convention). `base` points to the
// first element of the target slice (q or k) for (b=0,t=0,h=0); tokens are
// `row_stride` apart (=3C when operating on a q/k slice of qkv[B,T,3C]).
// Launched grid(T, ceil(H/HPB), B), block(hd/2, HPB): division-free indexing AND
// ~256 threads/block (good occupancy). threadIdx.x=i, threadIdx.y=head-in-block.
__global__ void rope_apply_k(bf16* base, const float* cosb, const float* sinb,
                             int H,int hd,int row_stride,int T){
  int half=hd/2; int i=threadIdx.x;
  int h=blockIdx.y*blockDim.y+threadIdx.y; if(h>=H) return;
  int t=blockIdx.x, b=blockIdx.z;
  bf16* p = base + ((long)b*T+t)*row_stride + (long)h*hd;
  float c=cosb[t*half+i], s=sinb[t*half+i];
  float x0=__bfloat162float(p[i]); float x1=__bfloat162float(p[i+half]);
  p[i]      = __float2bfloat16(x0*c - x1*s);
  p[i+half] = __float2bfloat16(x1*c + x0*s);
}
// Apply RoPE to both Q and K slices of interleaved qkv[B,T,3C] in one launch.
__global__ void rope_apply_qk_k(bf16* qkv, const float* cosb, const float* sinb,
                                int H,int hd,int row_stride,int T){
  int half=hd/2; int i=threadIdx.x;
  int h=blockIdx.y*blockDim.y+threadIdx.y; if(h>=H) return;
  int t=blockIdx.x, b=blockIdx.z;
  long C=(long)H*hd;
  bf16* q = qkv + ((long)b*T+t)*row_stride + (long)h*hd;
  bf16* k = q + C;
  float c=cosb[t*half+i], s=sinb[t*half+i];
  float q0=__bfloat162float(q[i]); float q1=__bfloat162float(q[i+half]);
  float k0=__bfloat162float(k[i]); float k1=__bfloat162float(k[i+half]);
  q[i]      = __float2bfloat16(q0*c - q1*s);
  q[i+half] = __float2bfloat16(q1*c + q0*s);
  k[i]      = __float2bfloat16(k0*c - k1*s);
  k[i+half] = __float2bfloat16(k1*c + k0*s);
}
// RoPE backward (transpose rotation): given dy in place -> dx in place.
__global__ void rope_backward_k(bf16* base, const float* cosb, const float* sinb,
                                int H,int hd,int row_stride,int T){
  int half=hd/2; int i=threadIdx.x;
  int h=blockIdx.y*blockDim.y+threadIdx.y; if(h>=H) return;
  int t=blockIdx.x, b=blockIdx.z;
  bf16* p = base + ((long)b*T+t)*row_stride + (long)h*hd;
  float c=cosb[t*half+i], s=sinb[t*half+i];
  float d0=__bfloat162float(p[i]); float d1=__bfloat162float(p[i+half]);
  p[i]      = __float2bfloat16(d0*c + d1*s);
  p[i+half] = __float2bfloat16(d1*c - d0*s);
}
// Backward RoPE for both Q and K slices of interleaved dqkv[B,T,3C].
__global__ void rope_backward_qk_k(bf16* dqkv, const float* cosb, const float* sinb,
                                   int H,int hd,int row_stride,int T){
  int half=hd/2; int i=threadIdx.x;
  int h=blockIdx.y*blockDim.y+threadIdx.y; if(h>=H) return;
  int t=blockIdx.x, b=blockIdx.z;
  long C=(long)H*hd;
  bf16* q = dqkv + ((long)b*T+t)*row_stride + (long)h*hd;
  bf16* k = q + C;
  float c=cosb[t*half+i], s=sinb[t*half+i];
  float q0=__bfloat162float(q[i]); float q1=__bfloat162float(q[i+half]);
  float k0=__bfloat162float(k[i]); float k1=__bfloat162float(k[i+half]);
  q[i]      = __float2bfloat16(q0*c + q1*s);
  q[i+half] = __float2bfloat16(q1*c - q0*s);
  k[i]      = __float2bfloat16(k0*c + k1*s);
  k[i+half] = __float2bfloat16(k1*c - k0*s);
}

// ----------------------------------------------------------------------------
// Attention (correctness-first; q,k,v are slices of qkv[B,T,3C] -> [B,T,H,hd]).
// qkv layout per token: [q(C) | k(C) | v(C)], each viewed as [H,hd].
// probs stored [B,H,T,T] (bf16). scale = 1/sqrt(hd). Causal.
// ----------------------------------------------------------------------------
__global__ void attention_forward_k(bf16* atty, bf16* probs, const bf16* qkv,
                                    int B,int T,int H,int hd, float scale){
  // one block per (b,h,i); threads cooperate over keys/dims
  int bh = blockIdx.x; int i = blockIdx.y;
  int b = bh / H, h = bh % H;
  if(b>=B||i>=T) return;
  int C = H*hd; int C3=3*C;
  const bf16* qrow = qkv + ((size_t)b*T + i)*C3 + h*hd;            // q for (b,i,h)
  bf16* prow = probs + (((size_t)b*H + h)*T + i)*T;
  extern __shared__ float sm[];                                    // size T
  // scores
  for(int j=threadIdx.x;j<=i;j+=blockDim.x){
    const bf16* krow = qkv + ((size_t)b*T + j)*C3 + C + h*hd;
    float s=0.f;
    for(int d=0; d<hd; d++) s += __bfloat162float(qrow[d])*__bfloat162float(krow[d]);
    sm[j]=s*scale;
  }
  __syncthreads();
  // softmax over j in [0,i] (single thread for numerical simplicity)
  __shared__ float ssum;
  if(threadIdx.x==0){
    float mx=-1e30f; for(int j=0;j<=i;j++) mx=fmaxf(mx,sm[j]);
    float sum=0.f; for(int j=0;j<=i;j++){ float e=expf(sm[j]-mx); sm[j]=e; sum+=e; }
    ssum=sum;
  }
  __syncthreads();
  float inv=1.0f/ssum;
  for(int j=threadIdx.x;j<=i;j+=blockDim.x) prow[j]=__float2bfloat16(sm[j]*inv);
  for(int j=i+1+threadIdx.x;j<T;j+=blockDim.x) prow[j]=__float2bfloat16(0.f);
  __syncthreads();
  // output: atty[b,i,h,:] = sum_j p[j]*v[b,j,h,:]
  bf16* orow = atty + ((size_t)b*T + i)*C + h*hd;
  for(int d=threadIdx.x; d<hd; d+=blockDim.x){
    float acc=0.f;
    for(int j=0;j<=i;j++){
      const bf16* vrow = qkv + ((size_t)b*T + j)*C3 + 2*C + h*hd;
      acc += __bfloat162float(prow[j])*__bfloat162float(vrow[d]);
    }
    orow[d]=__float2bfloat16(acc);
  }
}

// backward step 1+2: dp[i,j]=dot(datty_i, v_j); then ds[i,j]=p[i,j]*(dp[i,j]-rowsum)
__global__ void attention_bwd_ds_k(float* ds, const bf16* datty, const bf16* qkv,
                                   const bf16* probs, int B,int T,int H,int hd){
  int bh=blockIdx.x; int i=blockIdx.y; int b=bh/H,h=bh%H;
  if(b>=B||i>=T) return;
  int C=H*hd,C3=3*C;
  const bf16* drow = datty + ((size_t)b*T+i)*C + h*hd;
  const bf16* prow = probs + (((size_t)b*H+h)*T+i)*T;
  float* dsrow = ds + (((size_t)b*H+h)*T+i)*T;
  __shared__ float sh[1024];
  float rs=0.f;
  for(int j=threadIdx.x;j<=i;j+=blockDim.x){
    const bf16* vrow=qkv+((size_t)b*T+j)*C3+2*C+h*hd;
    float dp=0.f; for(int d=0;d<hd;d++) dp+=__bfloat162float(drow[d])*__bfloat162float(vrow[d]);
    float p=__bfloat162float(prow[j]);
    dsrow[j]=p*dp;        // temp store p*dp; subtract rowsum*p below
    rs += p*dp;
  }
  sh[threadIdx.x]=rs; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float rowsum=sh[0];
  for(int j=threadIdx.x;j<=i;j+=blockDim.x){
    float p=__bfloat162float(prow[j]);
    dsrow[j] = dsrow[j] - p*rowsum;  // = p*(dp - rowsum)
  }
  for(int j=i+1+threadIdx.x;j<T;j+=blockDim.x) dsrow[j]=0.f;
}
// dq[b,i,h,d] = scale*sum_{j<=i} ds[i,j]*k[j,d]   (write into dqkv q-slice)
__global__ void attention_bwd_dq_k(bf16* dqkv, const float* ds, const bf16* qkv,
                                   int B,int T,int H,int hd,float scale){
  int bh=blockIdx.x; int i=blockIdx.y; int b=bh/H,h=bh%H;
  if(b>=B||i>=T) return;
  int C=H*hd,C3=3*C;
  const float* dsrow=ds+(((size_t)b*H+h)*T+i)*T;
  bf16* dq=dqkv+((size_t)b*T+i)*C3+h*hd;
  for(int d=threadIdx.x;d<hd;d+=blockDim.x){
    float acc=0.f;
    for(int j=0;j<=i;j++){
      const bf16* krow=qkv+((size_t)b*T+j)*C3+C+h*hd;
      acc+=dsrow[j]*__bfloat162float(krow[d]);
    }
    dq[d]=__float2bfloat16(acc*scale);
  }
}
// dk[b,j,h,d] = scale*sum_{i>=j} ds[i,j]*q[i,d]   (write into dqkv k-slice)
__global__ void attention_bwd_dk_k(bf16* dqkv, const float* ds, const bf16* qkv,
                                   int B,int T,int H,int hd,float scale){
  int bh=blockIdx.x; int j=blockIdx.y; int b=bh/H,h=bh%H;
  if(b>=B||j>=T) return;
  int C=H*hd,C3=3*C;
  bf16* dk=dqkv+((size_t)b*T+j)*C3+C+h*hd;
  for(int d=threadIdx.x;d<hd;d+=blockDim.x){
    float acc=0.f;
    for(int i=j;i<T;i++){
      float dsv=ds[(((size_t)b*H+h)*T+i)*T + j];
      const bf16* qrow=qkv+((size_t)b*T+i)*C3+h*hd;
      acc+=dsv*__bfloat162float(qrow[d]);
    }
    dk[d]=__float2bfloat16(acc*scale);
  }
}
// dv[b,j,h,d] = sum_{i>=j} probs[i,j]*datty[i,d]   (write into dqkv v-slice)
__global__ void attention_bwd_dv_k(bf16* dqkv, const bf16* probs, const bf16* datty,
                                   int B,int T,int H,int hd){
  int bh=blockIdx.x; int j=blockIdx.y; int b=bh/H,h=bh%H;
  if(b>=B||j>=T) return;
  int C=H*hd,C3=3*C;
  bf16* dv=dqkv+((size_t)b*T+j)*C3+2*C+h*hd;
  for(int d=threadIdx.x;d<hd;d+=blockDim.x){
    float acc=0.f;
    for(int i=j;i<T;i++){
      float p=__bfloat162float(probs[(((size_t)b*H+h)*T+i)*T + j]);
      const bf16* drow=datty+((size_t)b*T+i)*C+h*hd;
      acc+=p*__bfloat162float(drow[d]);
    }
    dv[d]=__float2bfloat16(acc);
  }
}

// ----------------------------------------------------------------------------
// Tensor-core attention via cuBLAS batched GEMM. Split/permute qkv[B,T,3C] into
// q,k,v [B,H,T,hd]; QK^T and P@V are batched (batch=B*H) BF16 tensor-core GEMMs;
// causal softmax is a fast parallel kernel.
// ----------------------------------------------------------------------------
// qkv[B,T,3C] (q|k|v each [H,hd]) -> q,k,v each [B,H,T,hd]
__global__ void permute_qkv_k(bf16* q, bf16* k, bf16* v, const bf16* qkv,
                              int B,int T,int H,int hd){
  long C=(long)H*hd, C3=3*C;
  long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; long tot=(long)B*T*H*hd;
  if(idx>=tot) return;
  int d=idx%hd; long r=idx/hd; int h=r%H; r/=H; int t=r%T; int b=r/T;
  long dst=(((long)b*H+h)*T+t)*hd+d;
  const bf16* base=qkv+((long)b*T+t)*C3+(long)h*hd+d;
  q[dst]=base[0]; k[dst]=base[C]; v[dst]=base[2*C];
}
// dq,dk,dv [B,H,T,hd] -> dqkv[B,T,3C]
__global__ void unpermute_dqkv_k(bf16* dqkv, const bf16* dq, const bf16* dk, const bf16* dv,
                                 int B,int T,int H,int hd){
  long C=(long)H*hd, C3=3*C;
  long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; long tot=(long)B*T*H*hd;
  if(idx>=tot) return;
  int d=idx%hd; long r=idx/hd; int h=r%H; r/=H; int t=r%T; int b=r/T;
  long src=(((long)b*H+h)*T+t)*hd+d;
  bf16* base=dqkv+((long)b*T+t)*C3+(long)h*hd+d;
  base[0]=dq[src]; base[C]=dk[src]; base[2*C]=dv[src];
}
// o[B,H,T,hd] -> atty[B,T,C]    (and reverse for grads)
__global__ void permute_o_k(bf16* atty, const bf16* o, int B,int T,int H,int hd){
  long C=(long)H*hd; long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; long tot=(long)B*T*H*hd;
  if(idx>=tot) return;
  int d=idx%hd; long r=idx/hd; int h=r%H; r/=H; int t=r%T; int b=r/T;
  atty[((long)b*T+t)*C+(long)h*hd+d]=o[(((long)b*H+h)*T+t)*hd+d];
}
__global__ void permute_o_bwd_k(bf16* dO, const bf16* datty, int B,int T,int H,int hd){
  long C=(long)H*hd; long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; long tot=(long)B*T*H*hd;
  if(idx>=tot) return;
  int d=idx%hd; long r=idx/hd; int h=r%H; r/=H; int t=r%T; int b=r/T;
  dO[(((long)b*H+h)*T+t)*hd+d]=datty[((long)b*T+t)*C+(long)h*hd+d];
}
// causal softmax over last dim of S[B*H*T, T] (row i within its (b,h): mask j>i).
// Loads the active row into shared once (1 global read + 1 write).
__global__ void softmax_causal_fwd_k(bf16* P, const bf16* S, int BH, int T){
  long row=blockIdx.x; if(row>=(long)BH*T) return;
  int i = row % T;
  const bf16* sr=S+row*T; bf16* pr=P+row*T;
  extern __shared__ float rowbuf[];   // T floats
  __shared__ float red[256];
  float mx=-1e30f;
  for(int j=threadIdx.x;j<=i;j+=blockDim.x){ float v=__bfloat162float(sr[j]); rowbuf[j]=v; mx=fmaxf(mx,v); }
  red[threadIdx.x]=mx; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+s]); __syncthreads(); }
  mx=red[0]; __syncthreads();
  float sum=0.f;
  for(int j=threadIdx.x;j<=i;j+=blockDim.x){ float e=__expf(rowbuf[j]-mx); rowbuf[j]=e; sum+=e; }
  red[threadIdx.x]=sum; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
  float inv=1.0f/red[0];
  for(int j=threadIdx.x;j<T;j+=blockDim.x) pr[j]= (j<=i)? __float2bfloat16(rowbuf[j]*inv) : __float2bfloat16(0.f);
}
// causal softmax backward: dS[i,j]=P[i,j]*(dP[i,j]-sum_j' P[i,j']dP[i,j'])
__global__ void softmax_causal_bwd_k(bf16* dS, const bf16* dP, const bf16* P, int BH, int T){
  long row=blockIdx.x; if(row>=(long)BH*T) return;
  int i=row%T;
  const bf16* dpr=dP+row*T; const bf16* pr=P+row*T; bf16* dsr=dS+row*T;
  __shared__ float sh[1024];
  float acc=0.f;
  for(int j=threadIdx.x;j<=i;j+=blockDim.x) acc+=__bfloat162float(pr[j])*__bfloat162float(dpr[j]);
  sh[threadIdx.x]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float rs=sh[0];
  for(int j=threadIdx.x;j<T;j+=blockDim.x){
    if(j<=i){ float p=__bfloat162float(pr[j]); dsr[j]=__float2bfloat16(p*(__bfloat162float(dpr[j])-rs)); }
    else dsr[j]=__float2bfloat16(0.f);
  }
}

// ----------------------------------------------------------------------------
// FlashAttention (causal) — fused, no T^2 materialization. q,k,v,o: [B,H,T,hd].
// lse[B,H,T] = m + log(l) saved for backward. One block per (b,h,q-tile).
// ----------------------------------------------------------------------------
#define FA_HD_MAX 128
template<int BR,int BC>
__global__ void flash_fwd_k(bf16* O, float* lse, const bf16* Q, const bf16* K, const bf16* V,
                            int B,int H,int T,int hd,float scale){
  extern __shared__ float sh[]; // [BC*hd] K then [BC*hd] V
  float* shK=sh; float* shV=sh+BC*hd;
  int bh=blockIdx.x; int qtile=blockIdx.y; int tid=threadIdx.x;
  long base=(long)bh*T*hd;
  int qi=qtile*BR+tid;
  float qreg[FA_HD_MAX], acc[FA_HD_MAX];
  float m=-1e30f, l=0.f;
  bool valid = qi<T;
  if(valid){ const bf16* qr=Q+base+(long)qi*hd; for(int d=0;d<hd;d++){ qreg[d]=__bfloat162float(qr[d]); acc[d]=0.f; } }
  int kt_max = ((qtile+1)*BR-1)/BC;
  for(int kt=0;kt<=kt_max;kt++){
    int kstart=kt*BC;
    for(int idx=tid; idx<BC*hd; idx+=BR){
      int kk=kstart+idx/hd;
      if(kk<T){ shK[idx]=__bfloat162float(K[base+(long)kstart*hd+idx]); shV[idx]=__bfloat162float(V[base+(long)kstart*hd+idx]); }
      else { shK[idx]=0.f; shV[idx]=0.f; }
    }
    __syncthreads();
    if(valid){
      for(int jj=0;jj<BC;jj++){
        int kj=kstart+jj; if(kj>qi) break; if(kj>=T) break;
        const float* kp=shK+jj*hd; const float* vp=shV+jj*hd;
        float s=0.f; for(int d=0;d<hd;d++) s+=qreg[d]*kp[d]; s*=scale;
        float mn=fmaxf(m,s); float corr=expf(m-mn); float p=expf(s-mn);
        l=l*corr+p; for(int d=0;d<hd;d++) acc[d]=acc[d]*corr+p*vp[d];
        m=mn;
      }
    }
    __syncthreads();
  }
  if(valid){ bf16* orow=O+base+(long)qi*hd; float inv=1.0f/l;
    for(int d=0;d<hd;d++) orow[d]=__float2bfloat16(acc[d]*inv);
    lse[(long)bh*T+qi]=m+logf(l); }
}
// D[b,h,i] = sum_d dO[i,d]*O[i,d]
__global__ void flash_dO_O_k(float* D, const bf16* dO, const bf16* O, int BH,int T,int hd){
  long row=blockIdx.x; if(row>=(long)BH*T) return;
  const bf16* d=dO+row*hd; const bf16* o=O+row*hd;
  float acc=0.f; for(int x=threadIdx.x;x<hd;x+=blockDim.x) acc+=__bfloat162float(d[x])*__bfloat162float(o[x]);
  __shared__ float sh[256]; sh[threadIdx.x]=acc; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  if(threadIdx.x==0) D[row]=sh[0];
}
// dQ: one block per (b,h,i); recompute P from lse. dQ_i = scale*sum_{j<=i} dS_ij k_j
__global__ void flash_bwd_dq_k(bf16* dQ, const bf16* Q,const bf16* K,const bf16* V,
                               const bf16* dO,const float* lse,const float* D,
                               int B,int H,int T,int hd,float scale){
  int bh=blockIdx.x; int i=blockIdx.y; if(i>=T) return; int tid=threadIdx.x;
  long base=(long)bh*T*hd;
  __shared__ float qreg[FA_HD_MAX], doreg[FA_HD_MAX]; __shared__ float Li, Di;
  for(int d=tid; d<hd; d+=blockDim.x){ qreg[d]=__bfloat162float(Q[base+(long)i*hd+d]); doreg[d]=__bfloat162float(dO[base+(long)i*hd+d]); }
  if(tid==0){ Li=lse[(long)bh*T+i]; Di=D[(long)bh*T+i]; }
  __syncthreads();
  float dq[FA_HD_MAX]; for(int d=0;d<hd;d++) dq[d]=0.f; // each thread partial over key-subset
  for(int j=tid;j<=i;j+=blockDim.x){
    const bf16* kp=K+base+(long)j*hd; const bf16* vp=V+base+(long)j*hd;
    float s=0.f,dp=0.f;
    for(int d=0;d<hd;d++){ float kd=__bfloat162float(kp[d]); s+=qreg[d]*kd; dp+=doreg[d]*__bfloat162float(vp[d]); }
    float p=expf(s*scale-Li); float ds=p*(dp-Di);
    for(int d=0;d<hd;d++) dq[d]+=ds*__bfloat162float(kp[d]);
  }
  // reduce dq across threads via shared (reuse shK as scratch [blockDim*?]) -> use atomic in shared
  __shared__ float red[FA_HD_MAX];
  for(int d=0;d<hd;d++){ if(tid==0) red[d]=0.f; }
  __syncthreads();
  for(int d=0;d<hd;d++) atomicAdd(&red[d], dq[d]);
  __syncthreads();
  bf16* dqr=dQ+base+(long)i*hd;
  for(int d=tid;d<hd;d+=blockDim.x) dqr[d]=__float2bfloat16(red[d]*scale);
}
// dK,dV: one block per (b,h,j). dV_j=sum_{i>=j} P_ij dO_i ; dK_j=scale*sum_{i>=j} dS_ij q_i
__global__ void flash_bwd_dkv_k(bf16* dK, bf16* dV, const bf16* Q,const bf16* K,const bf16* V,
                                const bf16* dO,const float* lse,const float* D,
                                int B,int H,int T,int hd,float scale){
  int bh=blockIdx.x; int j=blockIdx.y; if(j>=T) return; int tid=threadIdx.x;
  long base=(long)bh*T*hd;
  __shared__ float kreg[FA_HD_MAX], vreg[FA_HD_MAX];
  for(int d=tid; d<hd; d+=blockDim.x){ kreg[d]=__bfloat162float(K[base+(long)j*hd+d]); vreg[d]=__bfloat162float(V[base+(long)j*hd+d]); }
  __syncthreads();
  float dk[FA_HD_MAX], dv[FA_HD_MAX]; for(int d=0;d<hd;d++){ dk[d]=0.f; dv[d]=0.f; }
  for(int i=j+tid;i<T;i+=blockDim.x){
    const bf16* qp=Q+base+(long)i*hd; const bf16* dop=dO+base+(long)i*hd;
    float s=0.f,dp=0.f;
    for(int d=0;d<hd;d++){ s+=__bfloat162float(qp[d])*kreg[d]; dp+=__bfloat162float(dop[d])*vreg[d]; }
    float Li=lse[(long)bh*T+i], Di=D[(long)bh*T+i];
    float p=expf(s*scale-Li); float ds=p*(dp-Di);
    for(int d=0;d<hd;d++){ dv[d]+=p*__bfloat162float(dop[d]); dk[d]+=ds*__bfloat162float(qp[d]); }
  }
  __shared__ float rk[FA_HD_MAX], rv[FA_HD_MAX];
  for(int d=0;d<hd;d++){ if(tid==0){ rk[d]=0.f; rv[d]=0.f; } }
  __syncthreads();
  for(int d=0;d<hd;d++){ atomicAdd(&rk[d],dk[d]); atomicAdd(&rv[d],dv[d]); }
  __syncthreads();
  bf16* dkr=dK+base+(long)j*hd; bf16* dvr=dV+base+(long)j*hd;
  for(int d=tid;d<hd;d+=blockDim.x){ dkr[d]=__float2bfloat16(rk[d]*scale); dvr[d]=__float2bfloat16(rv[d]); }
}

// ----------------------------------------------------------------------------
// Fused softmax cross-entropy. logits[BT,V] bf16. targets[BT] (-1 = ignore).
// Writes per-row loss (float) and dlogits[BT,V] bf16 = (softmax - onehot)*dscale.
// dscale = 1/num_valid.
// ----------------------------------------------------------------------------
__global__ void crossentropy_forward_backward_k(float* losses, bf16* dlogits,
                                                const bf16* logits, const int* targets,
                                                int BT, int V, float dscale){
  int row=blockIdx.x; if(row>=BT) return;
  const bf16* lr = logits + (size_t)row*V;
  int tgt = targets[row];
  __shared__ float sh[1024];
  // max
  float mx=-1e30f;
  for(int j=threadIdx.x;j<V;j+=blockDim.x) mx=fmaxf(mx,__bfloat162float(lr[j]));
  sh[threadIdx.x]=mx; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads(); }
  mx=sh[0]; __syncthreads();
  float sum=0.f;
  for(int j=threadIdx.x;j<V;j+=blockDim.x) sum+=__expf(__bfloat162float(lr[j])-mx);
  sh[threadIdx.x]=sum; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  sum=sh[0];
  float invsum=1.0f/sum;
  if(threadIdx.x==0){
    if(tgt>=0){ float lt=__bfloat162float(lr[tgt]); losses[row] = -(lt-mx-__logf(sum)); }
    else losses[row]=0.f;
  }
  bf16* dr = dlogits + (size_t)row*V;
  if(tgt<0){ for(int j=threadIdx.x;j<V;j+=blockDim.x) dr[j]=__float2bfloat16(0.f); return; }
  for(int j=threadIdx.x;j<V;j+=blockDim.x){
    float p=__expf(__bfloat162float(lr[j])-mx)*invsum;
    float g=(p - (j==tgt?1.0f:0.0f))*dscale;
    dr[j]=__float2bfloat16(g);
  }
}

// Two-pass CE over materialized logits. This computes row max, denominator, and
// target logit in one online-softmax scan, then the grad kernel reads logits once.
__global__ void ce_online_stats_k(float* rowmax, float* rowsum, float* target_logit,
                                  const bf16* logits, const int* targets, int BT, int V){
  int row=blockIdx.x; if(row>=BT) return;
  int tid=threadIdx.x, tgt=targets[row];
  if(tgt<0){
    if(tid==0){ rowmax[row]=-1.0e30f; rowsum[row]=0.0f; target_logit[row]=0.0f; }
    return;
  }
  const bf16* lr=logits+(size_t)row*V;
  __shared__ float sh_m[1024], sh_s[1024];
  float m=-1.0e30f, sum=0.0f;
  for(int j=tid;j<V;j+=blockDim.x){
    float x=__bfloat162float(lr[j]);
    if(x>m){ sum=sum*__expf(m-x)+1.0f; m=x; }
    else   { sum+=__expf(x-m); }
  }
  sh_m[tid]=m; sh_s[tid]=sum; __syncthreads();
  for(int stride=blockDim.x/2; stride>0; stride>>=1){
    if(tid<stride){
      float m1=sh_m[tid], s1=sh_s[tid], m2=sh_m[tid+stride], s2=sh_s[tid+stride];
      float mm=fmaxf(m1,m2);
      sh_s[tid]=s1*__expf(m1-mm)+s2*__expf(m2-mm);
      sh_m[tid]=mm;
    }
    __syncthreads();
  }
  if(tid==0){ rowmax[row]=sh_m[0]; rowsum[row]=sh_s[0]; target_logit[row]=__bfloat162float(lr[tgt]); }
}
__global__ void ce_online_stats2_k(float* rowmax, float* rowsum, float* target_logit,
                                   const bf16* logits, const int* targets, int BT, int V){
  int row=blockIdx.x; if(row>=BT) return;
  int tid=threadIdx.x, tgt=targets[row], V2=V>>1;
  if(tgt<0){
    if(tid==0){ rowmax[row]=-1.0e30f; rowsum[row]=0.0f; target_logit[row]=0.0f; }
    return;
  }
  const __nv_bfloat162* lr2=reinterpret_cast<const __nv_bfloat162*>(logits+(size_t)row*V);
  __shared__ float sh_m[1024], sh_s[1024];
  float m=-1.0e30f, sum=0.0f;
  for(int j=tid;j<V2;j+=blockDim.x){
    float2 x=__bfloat1622float2(lr2[j]);
    if(x.x>m){ sum=sum*__expf(m-x.x)+1.0f; m=x.x; }
    else     { sum+=__expf(x.x-m); }
    if(x.y>m){ sum=sum*__expf(m-x.y)+1.0f; m=x.y; }
    else     { sum+=__expf(x.y-m); }
  }
  sh_m[tid]=m; sh_s[tid]=sum; __syncthreads();
  for(int stride=blockDim.x/2; stride>0; stride>>=1){
    if(tid<stride){
      float m1=sh_m[tid], s1=sh_s[tid], m2=sh_m[tid+stride], s2=sh_s[tid+stride];
      float mm=fmaxf(m1,m2);
      sh_s[tid]=s1*__expf(m1-mm)+s2*__expf(m2-mm);
      sh_m[tid]=mm;
    }
    __syncthreads();
  }
  if(tid==0){
    rowmax[row]=sh_m[0];
    rowsum[row]=sh_s[0];
    target_logit[row]=__bfloat162float(logits[(size_t)row*V+tgt]);
  }
}

__global__ void ce_online_grad_k(float* losses, bf16* dlogits, const bf16* logits,
                                 const int* targets, const float* rowmax,
                                 const float* rowsum, const float* target_logit,
                                 int BT, int V, float dscale){
  int row=blockIdx.x; if(row>=BT) return;
  int tgt=targets[row];
  bf16* dr=dlogits+(size_t)row*V;
  if(tgt<0){
    for(int j=threadIdx.x;j<V;j+=blockDim.x) dr[j]=__float2bfloat16(0.0f);
    if(threadIdx.x==0) losses[row]=0.0f;
    return;
  }
  const bf16* lr=logits+(size_t)row*V;
  float mx=rowmax[row], invsum=1.0f/rowsum[row];
  for(int j=threadIdx.x;j<V;j+=blockDim.x){
    float p=__expf(__bfloat162float(lr[j])-mx)*invsum;
    float g=(p - (j==tgt?1.0f:0.0f))*dscale;
    dr[j]=__float2bfloat16(g);
  }
  if(threadIdx.x==0) losses[row]=__logf(rowsum[row])+mx-target_logit[row];
}
__global__ void ce_online_grad2_k(float* losses, bf16* dlogits, const bf16* logits,
                                  const int* targets, const float* rowmax,
                                  const float* rowsum, const float* target_logit,
                                  int BT, int V, float dscale){
  int row=blockIdx.x; if(row>=BT) return;
  int tgt=targets[row], V2=V>>1;
  __nv_bfloat162* dr2=reinterpret_cast<__nv_bfloat162*>(dlogits+(size_t)row*V);
  if(tgt<0){
    for(int j=threadIdx.x;j<V2;j+=blockDim.x) dr2[j]=__floats2bfloat162_rn(0.0f,0.0f);
    if(threadIdx.x==0) losses[row]=0.0f;
    return;
  }
  const __nv_bfloat162* lr2=reinterpret_cast<const __nv_bfloat162*>(logits+(size_t)row*V);
  float mx=rowmax[row], invsum=1.0f/rowsum[row];
  for(int j=threadIdx.x;j<V2;j+=blockDim.x){
    float2 x=__bfloat1622float2(lr2[j]);
    int j0=j<<1, j1=j0+1;
    float g0=(__expf(x.x-mx)*invsum - (j0==tgt?1.0f:0.0f))*dscale;
    float g1=(__expf(x.y-mx)*invsum - (j1==tgt?1.0f:0.0f))*dscale;
    dr2[j]=__floats2bfloat162_rn(g0,g1);
  }
  if(threadIdx.x==0) losses[row]=__logf(rowsum[row])+mx-target_logit[row];
}
__global__ void ce_online_fwd_bwd2_k(float* losses, bf16* dlogits,
                                     const bf16* logits, const int* targets,
                                     int BT, int V, float dscale){
  int row=blockIdx.x; if(row>=BT) return;
  int tid=threadIdx.x, tgt=targets[row], V2=V>>1;
  __nv_bfloat162* dr2=reinterpret_cast<__nv_bfloat162*>(dlogits+(size_t)row*V);
  if(tgt<0){
    for(int j=tid;j<V2;j+=blockDim.x) dr2[j]=__floats2bfloat162_rn(0.0f,0.0f);
    if(tid==0) losses[row]=0.0f;
    return;
  }
  const bf16* lr=logits+(size_t)row*V;
  const __nv_bfloat162* lr2=reinterpret_cast<const __nv_bfloat162*>(lr);
  __shared__ float sh_m[1024], sh_s[1024];
  float m=-1.0e30f, sum=0.0f;
  for(int j=tid;j<V2;j+=blockDim.x){
    float2 x=__bfloat1622float2(lr2[j]);
    if(x.x>m){ sum=sum*__expf(m-x.x)+1.0f; m=x.x; }
    else     { sum+=__expf(x.x-m); }
    if(x.y>m){ sum=sum*__expf(m-x.y)+1.0f; m=x.y; }
    else     { sum+=__expf(x.y-m); }
  }
  sh_m[tid]=m; sh_s[tid]=sum; __syncthreads();
  for(int stride=blockDim.x/2; stride>0; stride>>=1){
    if(tid<stride){
      float m1=sh_m[tid], s1=sh_s[tid], m2=sh_m[tid+stride], s2=sh_s[tid+stride];
      float mm=fmaxf(m1,m2);
      sh_s[tid]=s1*__expf(m1-mm)+s2*__expf(m2-mm);
      sh_m[tid]=mm;
    }
    __syncthreads();
  }
  float mx=sh_m[0], invsum=1.0f/sh_s[0];
  for(int j=tid;j<V2;j+=blockDim.x){
    float2 x=__bfloat1622float2(lr2[j]);
    int j0=j<<1, j1=j0+1;
    float g0=(__expf(x.x-mx)*invsum - (j0==tgt?1.0f:0.0f))*dscale;
    float g1=(__expf(x.y-mx)*invsum - (j1==tgt?1.0f:0.0f))*dscale;
    dr2[j]=__floats2bfloat162_rn(g0,g1);
  }
  if(tid==0) losses[row]=__logf(sh_s[0])+mx-__bfloat162float(lr[tgt]);
}

__global__ void ce_stats_init_k(float* rowmax, float* rowsum, float* target_logit, int BT){
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row<BT){ rowmax[row]=-1.0e30f; rowsum[row]=0.0f; target_logit[row]=0.0f; }
}
__global__ void ce_chunk_max_k(float* rowmax, const bf16* logits, const int* targets,
                               int BT, int chunk){
  int row=blockIdx.x; if(row>=BT) return;
  const bf16* lr=logits+(size_t)row*chunk;
  __shared__ float sh[1024];
  float mx=-1.0e30f;
  if(targets[row]>=0)
    for(int j=threadIdx.x;j<chunk;j+=blockDim.x) mx=fmaxf(mx,__bfloat162float(lr[j]));
  sh[threadIdx.x]=mx; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads(); }
  if(threadIdx.x==0) rowmax[row]=fmaxf(rowmax[row],sh[0]);
}
__global__ void ce_chunk_sum_tgt_k(float* rowsum, float* target_logit, const bf16* logits,
                                   const int* targets, const float* rowmax,
                                   int BT, int chunk, int vocab_off){
  int row=blockIdx.x; if(row>=BT) return;
  int tgt=targets[row];
  if(tgt<0) return;
  const bf16* lr=logits+(size_t)row*chunk;
  __shared__ float sh[1024];
  float sum=0.0f, mx=rowmax[row];
  for(int j=threadIdx.x;j<chunk;j+=blockDim.x) sum+=__expf(__bfloat162float(lr[j])-mx);
  sh[threadIdx.x]=sum; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  if(threadIdx.x==0) atomicAdd(&rowsum[row],sh[0]);
  int local=tgt-vocab_off;
  if(local>=0 && local<chunk && threadIdx.x==0) target_logit[row]=__bfloat162float(lr[local]);
}
__global__ void ce_chunk_online_stats_k(float* rowmax, float* rowsum, float* target_logit,
                                        const bf16* logits, const int* targets,
                                        int BT, int chunk, int vocab_off){
  int row=blockIdx.x; if(row>=BT) return;
  int tgt=targets[row];
  if(tgt<0) return;
  const bf16* lr=logits+(size_t)row*chunk;
  __shared__ float sh_m[1024], sh_s[1024];
  float m=-1.0e30f, sum=0.0f;
  for(int j=threadIdx.x;j<chunk;j+=blockDim.x){
    float x=__bfloat162float(lr[j]);
    if(x>m){ sum=sum*__expf(m-x)+1.0f; m=x; }
    else   { sum+=__expf(x-m); }
  }
  sh_m[threadIdx.x]=m; sh_s[threadIdx.x]=sum; __syncthreads();
  for(int stride=blockDim.x/2; stride>0; stride>>=1){
    if(threadIdx.x<stride){
      float m1=sh_m[threadIdx.x], s1=sh_s[threadIdx.x];
      float m2=sh_m[threadIdx.x+stride], s2=sh_s[threadIdx.x+stride];
      float mm=fmaxf(m1,m2);
      sh_s[threadIdx.x]=s1*__expf(m1-mm)+s2*__expf(m2-mm);
      sh_m[threadIdx.x]=mm;
    }
    __syncthreads();
  }
  if(threadIdx.x==0){
    float old_m=rowmax[row], old_s=rowsum[row];
    float new_m=sh_m[0], new_s=sh_s[0];
    float mm=fmaxf(old_m,new_m);
    rowmax[row]=mm;
    rowsum[row]=old_s*__expf(old_m-mm)+new_s*__expf(new_m-mm);
    int local=tgt-vocab_off;
    if(local>=0 && local<chunk) target_logit[row]=__bfloat162float(lr[local]);
  }
}
__global__ void ce_chunk_grad_loss_k(float* losses, bf16* dlogits, const bf16* logits,
                                     const int* targets, const float* rowmax,
                                     const float* rowsum, const float* target_logit,
                                     int BT, int chunk, int vocab_off, float dscale){
  int row=blockIdx.x; if(row>=BT) return;
  int tgt=targets[row];
  bf16* dr=dlogits+(size_t)row*chunk;
  if(tgt<0){
    for(int j=threadIdx.x;j<chunk;j+=blockDim.x) dr[j]=__float2bfloat16(0.0f);
    if(threadIdx.x==0) losses[row]=0.0f;
    return;
  }
  const bf16* lr=logits+(size_t)row*chunk;
  float mx=rowmax[row], invsum=1.0f/rowsum[row];
  int local=tgt-vocab_off;
  for(int j=threadIdx.x;j<chunk;j+=blockDim.x){
    float p=__expf(__bfloat162float(lr[j])-mx)*invsum;
    float g=(p - (j==local?1.0f:0.0f))*dscale;
    dr[j]=__float2bfloat16(g);
  }
  if(threadIdx.x==0) losses[row]=__logf(rowsum[row])+mx-target_logit[row];
}
__global__ void ce_loss_from_stats_k(float* losses, const int* targets, const float* rowmax,
                                     const float* rowsum, const float* target_logit, int BT){
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=BT) return;
  losses[row] = targets[row] < 0 ? 0.0f : (__logf(rowsum[row])+rowmax[row]-target_logit[row]);
}
__global__ void ce_chunk_dx_accum_k(float* dln_accum, const bf16* logits, const bf16* W,
                                    const int* targets, const float* rowmax, const float* rowsum,
                                    int BT, int chunk, int C, int vocab_off, float dscale,
                                    int first_chunk){
  int row=blockIdx.x; if(row>=BT) return;
  int tgt=targets[row];
  if(tgt<0){
    if(first_chunk){
      float* dx=dln_accum+(size_t)row*C;
      for(int c=threadIdx.x; c<C; c+=blockDim.x) dx[c]=0.0f;
    }
    return;
  }
  const bf16* lr=logits+(size_t)row*chunk;
  float* dx=dln_accum+(size_t)row*C;
  float mx=rowmax[row], invsum=1.0f/rowsum[row];
  for(int c=threadIdx.x; c<C; c+=blockDim.x){
    float acc = first_chunk ? 0.0f : dx[c];
    for(int j=0; j<chunk; j++){
      int v=vocab_off+j;
      float p=__expf(__bfloat162float(lr[j])-mx)*invsum;
      float g=(p - (v==tgt?1.0f:0.0f))*dscale;
      acc += g * __bfloat162float(W[(size_t)j*C+c]);
    }
    dx[c]=acc;
  }
}
__global__ void ce_chunk_dweight_accum_k(float* dW, const bf16* logits, const bf16* x,
                                         const int* targets, const float* rowmax, const float* rowsum,
                                         int BT, int chunk, int C, int vocab_off, float dscale,
                                         float beta){
  int vlocal=blockIdx.x;
  int c=blockIdx.y*blockDim.x+threadIdx.x;
  if(vlocal>=chunk || c>=C) return;
  int v=vocab_off+vlocal;
  float acc = beta==0.0f ? 0.0f : beta*dW[(size_t)vlocal*C+c];
  for(int row=0; row<BT; row++){
    int tgt=targets[row];
    if(tgt<0) continue;
    const bf16* lr=logits+(size_t)row*chunk;
    float p=__expf(__bfloat162float(lr[vlocal])-rowmax[row])*(1.0f/rowsum[row]);
    float g=(p - (v==tgt?1.0f:0.0f))*dscale;
    acc += g*__bfloat162float(x[(size_t)row*C+c]);
  }
  dW[(size_t)vlocal*C+c]=acc;
}

// ----------------------------------------------------------------------------
// AdamW: master(fp32) update with grad(fp32), write param(bf16). Decoupled WD.
// ----------------------------------------------------------------------------
__global__ void adamw_k(float* master, bf16* param, const float* grad,
                        float* m, float* v, size_t n,
                        float lr, float b1, float b2, float eps, float wd,
                        float bc1, float bc2, float grad_scale){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
  float g=grad[i]*grad_scale;
  float mi=b1*m[i]+(1-b1)*g;       m[i]=mi;
  float vi=b2*v[i]+(1-b2)*g*g;     v[i]=vi;
  float mh=mi/bc1, vh=vi/bc2;
  float p=master[i];
  p -= lr*(mh/(sqrtf(vh)+eps) + wd*p);
  master[i]=p;
  param[i]=__float2bfloat16(p);
}
__device__ __forceinline__ float adamw_update_one(float p, float g, float& mi, float& vi,
                                                  float lr, float b1, float b2, float eps,
                                                  float wd, float bc1, float bc2,
                                                  float grad_scale){
  g *= grad_scale;
  mi=b1*mi+(1.0f-b1)*g;
  vi=b2*vi+(1.0f-b2)*g*g;
  float mh=mi/bc1, vh=vi/bc2;
  return p - lr*(mh/(sqrtf(vh)+eps) + wd*p);
}
__global__ void adamw4_k(float* master, bf16* param, const float* grad,
                         float* m, float* v, size_t n4,
                         float lr, float b1, float b2, float eps, float wd,
                         float bc1, float bc2, float grad_scale){
  size_t i4=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i4>=n4) return;
  float4 p4=reinterpret_cast<float4*>(master)[i4];
  float4 g4=reinterpret_cast<const float4*>(grad)[i4];
  float4 m4=reinterpret_cast<float4*>(m)[i4];
  float4 v4=reinterpret_cast<float4*>(v)[i4];
  p4.x=adamw_update_one(p4.x,g4.x,m4.x,v4.x,lr,b1,b2,eps,wd,bc1,bc2,grad_scale);
  p4.y=adamw_update_one(p4.y,g4.y,m4.y,v4.y,lr,b1,b2,eps,wd,bc1,bc2,grad_scale);
  p4.z=adamw_update_one(p4.z,g4.z,m4.z,v4.z,lr,b1,b2,eps,wd,bc1,bc2,grad_scale);
  p4.w=adamw_update_one(p4.w,g4.w,m4.w,v4.w,lr,b1,b2,eps,wd,bc1,bc2,grad_scale);
  reinterpret_cast<float4*>(master)[i4]=p4;
  reinterpret_cast<float4*>(m)[i4]=m4;
  reinterpret_cast<float4*>(v)[i4]=v4;
  __nv_bfloat162 q0=__floats2bfloat162_rn(p4.x,p4.y);
  __nv_bfloat162 q1=__floats2bfloat162_rn(p4.z,p4.w);
  reinterpret_cast<__nv_bfloat162*>(param)[i4*2+0]=q0;
  reinterpret_cast<__nv_bfloat162*>(param)[i4*2+1]=q1;
}
__global__ void adamw_fp8_k(float* master, bf16* param, __nv_fp8_e4m3* param8, const float* grad,
                            float* m, float* v, size_t n,
                            float lr, float b1, float b2, float eps, float wd,
                            float bc1, float bc2, float grad_scale, float fp8_scale){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=n) return;
  float g=grad[i]*grad_scale;
  float mi=b1*m[i]+(1-b1)*g;       m[i]=mi;
  float vi=b2*v[i]+(1-b2)*g*g;     v[i]=vi;
  float mh=mi/bc1, vh=vi/bc2;
  float p=master[i];
  p -= lr*(mh/(sqrtf(vh)+eps) + wd*p);
  master[i]=p;
  param[i]=__float2bfloat16(p);
  param8[i]=__nv_fp8_e4m3(p*fp8_scale);
}
__global__ void zero_f32_k(float* p, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) p[i]=0.f; }
__global__ void scale_f32_k(float* p, size_t n, float a){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) p[i]*=a; }

// small reductions
static inline int ceil_div(long a, long b){ return (int)((a+b-1)/b); }
