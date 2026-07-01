// Realistic RMSNorm cost: cycle through DISTINCT buffers each call so reads come
// from HBM (as in the real model, where each layer has its own activations), not
// from an artificially-hot L2 (the flaw in norm_bench.cu reusing one buffer).
#include "gpt_kernels.cuh"
#include <ctime>
#include <vector>
static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec/1e9; }
#define NORM_R 128
static void rmsnorm_bwd(cudaStream_t s, float* partial, bf16* dx, float* dweight,
                        const bf16* dout, const bf16* x, const bf16* w, const float* rstd,
                        int N, int C){
  rmsnorm_dx_k<<<N,256,0,s>>>(dx,nullptr,dout,x,w,rstd,N,C);
  dim3 gp(ceil_div(C,256), NORM_R);
  rmsnorm_dweight_partial_k<<<gp,256,0,s>>>(partial,dout,x,rstd,N,C,NORM_R);
  reduce_cols_add_k<<<ceil_div(C,256),256,0,s>>>(dweight,partial,NORM_R,C);
}
static void rmsnorm_bwd_atomic(cudaStream_t s, float* partial, bf16* dx, float* dweight,
                               const bf16* dout, const bf16* x, const bf16* w, const float* rstd,
                               int N, int C){
  zero_f32_k<<<ceil_div((long)NORM_R*C,256),256,0,s>>>(partial,(long)NORM_R*C);
  rmsnorm_dx_dweight_atomic_partial_k<<<N,256,0,s>>>(partial,dx,nullptr,dout,x,w,rstd,N,C,NORM_R);
  reduce_cols_add_k<<<ceil_div(C,256),256,0,s>>>(dweight,partial,NORM_R,C);
}
int main(int argc,char**argv){
  int N=16384, C=argc>1?atoi(argv[1]):768, R=128, calls=25, SETS=25;
  // SETS distinct buffer sets so total footprint (>> L2) forces HBM reads
  std::vector<bf16*> X(SETS),O(SETS),DOUT(SETS),DX(SETS); std::vector<float*> RSTD(SETS);
  bf16 *w; float *dw,*partial;
  CUDA_CHECK(cudaMalloc(&w,(size_t)C*2)); CUDA_CHECK(cudaMalloc(&dw,(size_t)C*4));
  CUDA_CHECK(cudaMalloc(&partial,(size_t)R*C*4));
  std::vector<bf16> bx(N*C); for(int i=0;i<N*C;i++) bx[i]=__float2bfloat16(((i*7)%101-50)*0.02f);
  std::vector<bf16> bw(C,__float2bfloat16(1.0f)); CUDA_CHECK(cudaMemcpy(w,bw.data(),(size_t)C*2,cudaMemcpyHostToDevice));
  for(int s=0;s<SETS;s++){
    CUDA_CHECK(cudaMalloc(&X[s],(size_t)N*C*2)); CUDA_CHECK(cudaMalloc(&O[s],(size_t)N*C*2));
    CUDA_CHECK(cudaMalloc(&DOUT[s],(size_t)N*C*2)); CUDA_CHECK(cudaMalloc(&DX[s],(size_t)N*C*2));
    CUDA_CHECK(cudaMalloc(&RSTD[s],(size_t)N*4));
    CUDA_CHECK(cudaMemcpy(X[s],bx.data(),(size_t)N*C*2,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(DOUT[s],bx.data(),(size_t)N*C*2,cudaMemcpyHostToDevice));
  }
  double mb_each = (double)N*C*2/1e6;
  printf("N=%d C=%d  footprint/set=%.1fMB  sets=%d (total %.0fMB)\n",N,C,mb_each,SETS,mb_each*SETS*4);
  // fwd
  for(int i=0;i<5;i++) rmsnorm_forward_k<<<N,256>>>(O[i%SETS],RSTD[i%SETS],X[i%SETS],w,N,C,1e-5f);
  CUDA_CHECK(cudaDeviceSynchronize());
  double t0=now_s();
  for(int it=0; it<calls; it++) rmsnorm_forward_k<<<N,256>>>(O[it%SETS],RSTD[it%SETS],X[it%SETS],w,N,C,1e-5f);
  CUDA_CHECK(cudaDeviceSynchronize());
  double fwd=(now_s()-t0)*1e3;
  // backward: current atomic-free dx + coalesced dweight reduction
  for(int i=0;i<5;i++) rmsnorm_bwd(0,partial,DX[i%SETS],dw,DOUT[i%SETS],X[i%SETS],w,RSTD[i%SETS],N,C);
  CUDA_CHECK(cudaDeviceSynchronize());
  t0=now_s();
  for(int it=0; it<calls; it++) rmsnorm_bwd(0,partial,DX[it%SETS],dw,DOUT[it%SETS],X[it%SETS],w,RSTD[it%SETS],N,C);
  CUDA_CHECK(cudaDeviceSynchronize());
  double bwd=(now_s()-t0)*1e3;
  // backward: experimental row-wise dx + atomic dweight partials
  for(int i=0;i<5;i++) rmsnorm_bwd_atomic(0,partial,DX[i%SETS],dw,DOUT[i%SETS],X[i%SETS],w,RSTD[i%SETS],N,C);
  CUDA_CHECK(cudaDeviceSynchronize());
  t0=now_s();
  for(int it=0; it<calls; it++) rmsnorm_bwd_atomic(0,partial,DX[it%SETS],dw,DOUT[it%SETS],X[it%SETS],w,RSTD[it%SETS],N,C);
  CUDA_CHECK(cudaDeviceSynchronize());
  double bwd_atomic=(now_s()-t0)*1e3;
  printf("rmsnorm fwd x%d: %.2f ms (%.3f each)\n", calls,fwd,fwd/calls);
  printf("  bwd current x%d: %.2f ms (%.3f each) | fwd+bwd=%.2f ms\n",
         calls,bwd,bwd/calls,fwd+bwd);
  printf("  bwd atomic  x%d: %.2f ms (%.3f each) | fwd+bwd=%.2f ms | delta %.2f ms\n",
         calls,bwd_atomic,bwd_atomic/calls,fwd+bwd_atomic,bwd_atomic-bwd);
  return 0;
}
