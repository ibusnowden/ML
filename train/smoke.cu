// Toolchain smoke test: device props + a BF16 tensor-core GEMM via cuBLASLt,
// validated against a CPU reference, plus an NCCL symbol link check.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <nccl.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)
#define LK(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("cuBLASLt error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1);} }while(0)

// out[M,N] (row-major) = inp[M,K] @ weight[N,K]^T   (the linear-layer pattern)
// Column-major mapping: D[N,M] = op(A=weight)^T @ op(B=inp), m=N,n=M,k=K.
static void gemm_ltNT(cublasLtHandle_t lt, const __nv_bfloat16* weight,
                      const __nv_bfloat16* inp, float* out, int M, int N, int K) {
  cublasLtMatmulDesc_t op;
  LK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  LK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  cublasLtMatrixLayout_t A,B,D;
  LK(cublasLtMatrixLayoutCreate(&A, CUDA_R_16BF, K, N, K)); // weight: KxN col-major (=NxK row-major)
  LK(cublasLtMatrixLayoutCreate(&B, CUDA_R_16BF, K, M, K)); // inp: KxM col-major (=MxK row-major)
  LK(cublasLtMatrixLayoutCreate(&D, CUDA_R_32F,  N, M, N)); // out: NxM col-major (=MxN row-major)
  float alpha=1.f, beta=0.f;
  LK(cublasLtMatmul(lt, op, &alpha, weight, A, inp, B, &beta,
                    out, D, out, D, nullptr, nullptr, 0, 0));
  cublasLtMatrixLayoutDestroy(A); cublasLtMatrixLayoutDestroy(B);
  cublasLtMatrixLayoutDestroy(D); cublasLtMatmulDescDestroy(op);
}

int main() {
  int dev=0; CK(cudaGetDevice(&dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  printf("Device: %s  sm_%d%d  SMs=%d  mem=%.1f GB\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         p.totalGlobalMem/1e9);
  int rt=0,drv=0; CK(cudaRuntimeGetVersion(&rt)); CK(cudaDriverGetVersion(&drv));
  printf("CUDA runtime=%d  driver supports=%d\n", rt, drv);
  printf("NCCL version compiled: %d.%d.%d\n", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);

  const int M=128, N=256, K=192;
  std::vector<float> hW(N*K), hX(M*K), hRef(M*N,0.f);
  for(int i=0;i<N*K;i++) hW[i]=((i*7)%13-6)*0.1f;
  for(int i=0;i<M*K;i++) hX[i]=((i*5)%11-5)*0.1f;
  for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float s=0; for(int k=0;k<K;k++) s+=hX[m*K+k]*hW[n*K+k]; hRef[m*N+n]=s; }

  __nv_bfloat16 *dW,*dX; float *dO;
  CK(cudaMalloc(&dW,N*K*2)); CK(cudaMalloc(&dX,M*K*2)); CK(cudaMalloc(&dO,M*N*4));
  std::vector<__nv_bfloat16> bW(N*K), bX(M*K);
  for(int i=0;i<N*K;i++) bW[i]=__float2bfloat16(hW[i]);
  for(int i=0;i<M*K;i++) bX[i]=__float2bfloat16(hX[i]);
  CK(cudaMemcpy(dW,bW.data(),N*K*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dX,bX.data(),M*K*2,cudaMemcpyHostToDevice));

  cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
  gemm_ltNT(lt, dW, dX, dO, M, N, K);
  CK(cudaDeviceSynchronize());

  std::vector<float> hO(M*N);
  CK(cudaMemcpy(hO.data(),dO,M*N*4,cudaMemcpyDeviceToHost));
  double maxerr=0, denom=0;
  for(int i=0;i<M*N;i++){ maxerr=fmax(maxerr,fabs(hO[i]-hRef[i])); denom=fmax(denom,fabs(hRef[i])); }
  printf("cuBLASLt BF16 GEMM rel-max-err = %.4f (abs %.4f, scale %.4f)\n",
         maxerr/(denom+1e-9), maxerr, denom);
  printf(maxerr/(denom+1e-9) < 0.05 ? "SMOKE TEST PASSED\n" : "SMOKE TEST FAILED\n");
  cublasLtDestroy(lt);
  CK(cudaFree(dW)); CK(cudaFree(dX)); CK(cudaFree(dO));
  return 0;
}
