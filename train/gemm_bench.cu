// Microbench: peak achievable TFLOP/s of our linear_forward (cuBLASLt BF16)
// and native FP8 tensor-core GEMMs on the transformer's main shapes.
#include "gpt_kernels.cuh"
#include <cuda_fp8.h>
#include <ctime>
#include <cstring>

static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec/1e9; }

static float peak_bf16_tflops(){
  const char* e=getenv("ENTROPY_PEAK_TFLOPS");
  return (e&&atof(e)>0) ? (float)atof(e) : 989.0f; // H100 SXM dense BF16/FP8 default
}

__global__ void fill_fp8_e4m3_k(__nv_fp8_e4m3* p, size_t n, float v){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i<n) p[i]=__nv_fp8_e4m3(v);
}
__global__ void fill_bf16_k(bf16* p, size_t n, float v){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i<n) p[i]=__float2bfloat16(v);
}
__global__ void bf16_to_fp8_e4m3_k(__nv_fp8_e4m3* dst, const bf16* src, size_t n, float scale){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i<n) dst[i]=__nv_fp8_e4m3(__bfloat162float(src[i])*scale);
}

static void fill_fp8(__nv_fp8_e4m3* p, size_t n, float v){
  fill_fp8_e4m3_k<<<(n+255)/256,256>>>(p,n,v);
  CUDA_CHECK(cudaGetLastError());
}
static void fill_bf16(bf16* p, size_t n, float v){
  fill_bf16_k<<<(n+255)/256,256>>>(p,n,v);
  CUDA_CHECK(cudaGetLastError());
}

static const char* cublas_status(cublasStatus_t s){
  switch(s){
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "not_initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "alloc_failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "invalid_value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "arch_mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR: return "mapping_error";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "execution_failed";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "internal_error";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "not_supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "license_error";
    default: return "unknown";
  }
}

static void run_bf16(LtCtx*lt,const char*name,int M,int N,int K,int iters){
  bf16 *a,*w,*o; CUDA_CHECK(cudaMalloc(&a,(size_t)M*K*2)); CUDA_CHECK(cudaMalloc(&w,(size_t)N*K*2)); CUDA_CHECK(cudaMalloc(&o,(size_t)M*N*2));
  for(int i=0;i<5;i++) linear_forward(lt,0,a,w,o,M,N,K);
  CUDA_CHECK(cudaDeviceSynchronize());
  double t0=now_s();
  for(int i=0;i<iters;i++) linear_forward(lt,0,a,w,o,M,N,K);
  CUDA_CHECK(cudaDeviceSynchronize());
  double dt=(now_s()-t0)/iters;
  double fl=2.0*M*N*K;
  float peak=peak_bf16_tflops();
  printf("bf16 %-12s M=%-6d N=%-6d K=%-5d  %.3f ms  %.1f TFLOP/s (%.1f%% of %.0f)\n",
         name,M,N,K,dt*1e3,fl/dt/1e12,100*fl/dt/(peak*1e12),peak);
  cudaFree(a);cudaFree(w);cudaFree(o);
}

static void run_fp8(cublasLtHandle_t lt, void* ws, size_t wssz,
                    const char*name,int M,int N,int K,int iters){
  __nv_fp8_e4m3 *a,*w; bf16* o;
  CUDA_CHECK(cudaMalloc(&a,(size_t)M*K));
  CUDA_CHECK(cudaMalloc(&w,(size_t)N*K));
  CUDA_CHECK(cudaMalloc(&o,(size_t)M*N*2));
  fill_fp8(a,(size_t)M*K,0.125f);
  fill_fp8(w,(size_t)N*K,0.125f);
  CUDA_CHECK(cudaMemset(o,0,(size_t)M*N*2));

  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t LA,LB,LD;
  cublasLtMatmulPreference_t pref;
  cublasOperation_t ta=CUBLAS_OP_T, tb=CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&LA, CUDA_R_8F_E4M3, K, N, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&LB, CUDA_R_8F_E4M3, K, M, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&LD, CUDA_R_16BF,   N, M, N));
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wssz, sizeof(wssz)));

  cublasLtMatmulHeuristicResult_t heur[128]; int nres=0;
  cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt, op, LA, LB, LD, LD, pref, 128, heur, &nres);
  if(hs!=CUBLAS_STATUS_SUCCESS || nres==0){
    printf("fp8  %-12s M=%-6d N=%-6d K=%-5d  unsupported (%s, nres=%d)\n",
           name,M,N,K,cublas_status(hs),nres);
    goto done;
  }

  {
    float alpha=1.0f, beta=0.0f;
    int chosen=-1; float best_ms=1e30f;
    cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    for(int i=0;i<nres;i++){
      if(heur[i].state!=CUBLAS_STATUS_SUCCESS) continue;
      bool ok=true;
      for(int wup=0;wup<2;wup++){
        cublasStatus_t s=cublasLtMatmul(lt,op,&alpha,w,LA,a,LB,&beta,o,LD,o,LD,
                                        &heur[i].algo,ws,wssz,0);
        if(s!=CUBLAS_STATUS_SUCCESS){ ok=false; break; }
      }
      if(!ok) continue;
      CUDA_CHECK(cudaEventRecord(e0,0));
      for(int t=0;t<5;t++) cublasLtMatmul(lt,op,&alpha,w,LA,a,LB,&beta,o,LD,o,LD,
                                          &heur[i].algo,ws,wssz,0);
      CUDA_CHECK(cudaEventRecord(e1,0));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
      if(ms<best_ms){ best_ms=ms; chosen=i; }
    }
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    if(chosen<0){
      printf("fp8  %-12s M=%-6d N=%-6d K=%-5d  unsupported (all algos failed)\n",name,M,N,K);
      goto done;
    }
    for(int i=0;i<5;i++) CUBLAS_CHECK(cublasLtMatmul(lt,op,&alpha,w,LA,a,LB,&beta,o,LD,o,LD,
                                                      &heur[chosen].algo,ws,wssz,0));
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0=now_s();
    for(int i=0;i<iters;i++) CUBLAS_CHECK(cublasLtMatmul(lt,op,&alpha,w,LA,a,LB,&beta,o,LD,o,LD,
                                                         &heur[chosen].algo,ws,wssz,0));
    CUDA_CHECK(cudaDeviceSynchronize());
    double dt=(now_s()-t0)/iters;
    double fl=2.0*M*N*K;
    float peak=peak_bf16_tflops()*2.0f;
    printf("fp8  %-12s M=%-6d N=%-6d K=%-5d  %.3f ms  %.1f TFLOP/s (%.1f%% of %.0f)\n",
           name,M,N,K,dt*1e3,fl/dt/1e12,100*fl/dt/(peak*1e12),peak);
  }

done:
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(LA); cublasLtMatrixLayoutDestroy(LB); cublasLtMatrixLayoutDestroy(LD);
  cublasLtMatmulDescDestroy(op);
  cudaFree(a); cudaFree(w); cudaFree(o);
}

static void run_cast(const char* name, size_t n, int iters){
  bf16* src; __nv_fp8_e4m3* dst;
  CUDA_CHECK(cudaMalloc(&src,n*2));
  CUDA_CHECK(cudaMalloc(&dst,n));
  fill_bf16(src,n,0.125f);
  for(int i=0;i<10;i++) bf16_to_fp8_e4m3_k<<<(n+255)/256,256>>>(dst,src,n,1.0f);
  CUDA_CHECK(cudaDeviceSynchronize());
  double t0=now_s();
  for(int i=0;i<iters;i++) bf16_to_fp8_e4m3_k<<<(n+255)/256,256>>>(dst,src,n,1.0f);
  CUDA_CHECK(cudaDeviceSynchronize());
  double dt=(now_s()-t0)/iters;
  double gb=(double)n*3.0/1e9; // read bf16 + write fp8
  printf("cast %-12s elems=%-12zu %.3f ms  %.1f GB/s\n", name,n,dt*1e3,gb/dt);
  cudaFree(src); cudaFree(dst);
}

int main(){
  LtCtx lt; lt_init(&lt);
  int BT=16384, C=768, I=2048, V=32768;
  const char* b=getenv("ENTROPY_BENCH_BT"); if(b&&atoi(b)>0) BT=atoi(b);
  printf("== cuBLASLt GEMM efficiency, BT=%d ==\n", BT);
  run_bf16(&lt,"qkv",   BT,3*C,C, 50);
  run_bf16(&lt,"o",     BT,C,C,   50);
  run_bf16(&lt,"gate",  BT,I,C,   50);
  run_bf16(&lt,"down",  BT,C,I,   50);
  run_bf16(&lt,"lm_head",BT,V,C,  30);
  run_bf16(&lt,"big",   16384,16384,16384, 20);
  printf("== Native FP8 E4M3 inputs, BF16 output ==\n");
  run_fp8(lt.handle,lt.workspace,lt.wssize,"qkv",   BT,3*C,C, 80);
  run_fp8(lt.handle,lt.workspace,lt.wssize,"o",     BT,C,C,   80);
  run_fp8(lt.handle,lt.workspace,lt.wssize,"gate",  BT,I,C,   80);
  run_fp8(lt.handle,lt.workspace,lt.wssize,"down",  BT,C,I,   80);
  run_fp8(lt.handle,lt.workspace,lt.wssize,"lm_head",BT,V,C,  40);
  run_fp8(lt.handle,lt.workspace,lt.wssize,"big",   16384,16384,16384, 20);
  printf("== BF16 -> FP8 E4M3 cast cost ==\n");
  run_cast("act_C",    (size_t)BT*C,   200);
  run_cast("act_3C",   (size_t)BT*3*C, 100);
  run_cast("act_2I",   (size_t)BT*2*I,  60);
  run_cast("lm_weight",(size_t)V*C,    200);
  run_cast("logits",   (size_t)BT*V,    20);
  return 0;
}
