// Microbench the RMSNorm kernels in isolation with valid (normal-magnitude) data,
// to get true cost free of the activation-explosion confound in the skip-ablation.
#include "gpt_kernels.cuh"
#include <ctime>
#include <vector>
static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec/1e9; }
int main(){
  int N=16384, C=768, R=128, calls=25;
  bf16 *x,*o,*w,*dout,*dx; float *rstd,*dw,*partial;
  CUDA_CHECK(cudaMalloc(&x,(size_t)N*C*2)); CUDA_CHECK(cudaMalloc(&o,(size_t)N*C*2));
  CUDA_CHECK(cudaMalloc(&w,(size_t)C*2)); CUDA_CHECK(cudaMalloc(&dout,(size_t)N*C*2));
  CUDA_CHECK(cudaMalloc(&dx,(size_t)N*C*2)); CUDA_CHECK(cudaMalloc(&rstd,(size_t)N*4));
  CUDA_CHECK(cudaMalloc(&dw,(size_t)C*4)); CUDA_CHECK(cudaMalloc(&partial,(size_t)R*C*4));
  std::vector<float> hx(N*C); for(int i=0;i<N*C;i++) hx[i]=((i*7)%101-50)*0.02f;
  std::vector<bf16> bx(N*C); for(int i=0;i<N*C;i++) bx[i]=__float2bfloat16(hx[i]);
  CUDA_CHECK(cudaMemcpy(x,bx.data(),(size_t)N*C*2,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dout,bx.data(),(size_t)N*C*2,cudaMemcpyHostToDevice));
  std::vector<bf16> bw(C,__float2bfloat16(1.0f)); CUDA_CHECK(cudaMemcpy(w,bw.data(),(size_t)C*2,cudaMemcpyHostToDevice));
  // warmup
  for(int i=0;i<5;i++) rmsnorm_forward_k<<<N,256>>>(o,rstd,x,w,N,C,1e-5f);
  CUDA_CHECK(cudaDeviceSynchronize());
  double t0=now_s();
  for(int it=0; it<calls; it++) rmsnorm_forward_k<<<N,256>>>(o,rstd,x,w,N,C,1e-5f);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("rmsnorm_forward x%d : %.2f ms total (%.3f ms each)\n", calls,(now_s()-t0)*1e3,(now_s()-t0)*1e3/calls);
  // backward (dx + dweight partial-reduce)
  for(int i=0;i<5;i++){ rmsnorm_dx_k<<<N,256>>>(dx,dout,x,w,rstd,N,C);
    dim3 gp(ceil_div(C,256),R); rmsnorm_dweight_partial_k<<<gp,256>>>(partial,dout,x,rstd,N,C,R);
    reduce_cols_add_k<<<ceil_div(C,256),256>>>(dw,partial,R,C); }
  CUDA_CHECK(cudaDeviceSynchronize());
  t0=now_s();
  for(int it=0; it<calls; it++){ rmsnorm_dx_k<<<N,256>>>(dx,dout,x,w,rstd,N,C);
    dim3 gp(ceil_div(C,256),R); rmsnorm_dweight_partial_k<<<gp,256>>>(partial,dout,x,rstd,N,C,R);
    reduce_cols_add_k<<<ceil_div(C,256),256>>>(dw,partial,R,C); }
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("rmsnorm_backward x%d : %.2f ms total (%.3f ms each)\n", calls,(now_s()-t0)*1e3,(now_s()-t0)*1e3/calls);
  // dx-only and dweight-only split
  t0=now_s(); for(int it=0;it<calls;it++) rmsnorm_dx_k<<<N,256>>>(dx,dout,x,w,rstd,N,C);
  CUDA_CHECK(cudaDeviceSynchronize()); printf("  dx-only x%d: %.2f ms\n",calls,(now_s()-t0)*1e3);
  t0=now_s(); for(int it=0;it<calls;it++){ dim3 gp(ceil_div(C,256),R); rmsnorm_dweight_partial_k<<<gp,256>>>(partial,dout,x,rstd,N,C,R); reduce_cols_add_k<<<ceil_div(C,256),256>>>(dw,partial,R,C);}
  CUDA_CHECK(cudaDeviceSynchronize()); printf("  dweight-only x%d: %.2f ms\n",calls,(now_s()-t0)*1e3);
  return 0;
}
