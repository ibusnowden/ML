// Standalone derisk: cuDNN frontend fused SDPA (causal) forward, validated
// against a CPU reference on a small shape.
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>
namespace fe = cudnn_frontend;
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

int main(){
  int B=2,H=4,T=64,hd=32; float scale=1.0f/sqrtf((float)hd);
  long n=(long)B*H*T*hd;
  std::vector<float> q(n),k(n),v(n),ref(n,0.f);
  for(long i=0;i<n;i++){ q[i]=((i*7)%13-6)*0.1f; k[i]=((i*5)%11-5)*0.1f; v[i]=((i*3)%9-4)*0.1f; }
  // CPU reference causal SDPA
  for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int i=0;i<T;i++){
    std::vector<float> s(i+1); float mx=-1e30f;
    for(int j=0;j<=i;j++){ float d=0; for(int x=0;x<hd;x++) d+=q[((( (long)b*H+h)*T+i)*hd)+x]*k[((((long)b*H+h)*T+j)*hd)+x]; s[j]=d*scale; mx=fmaxf(mx,s[j]); }
    float sum=0; for(int j=0;j<=i;j++){ s[j]=expf(s[j]-mx); sum+=s[j]; }
    for(int x=0;x<hd;x++){ float a=0; for(int j=0;j<=i;j++) a+=s[j]/sum*v[((((long)b*H+h)*T+j)*hd)+x]; ref[((((long)b*H+h)*T+i)*hd)+x]=a; }
  }
  std::vector<__nv_bfloat16> qb(n),kb(n),vb(n);
  for(long i=0;i<n;i++){ qb[i]=__float2bfloat16(q[i]); kb[i]=__float2bfloat16(k[i]); vb[i]=__float2bfloat16(v[i]); }
  __nv_bfloat16 *dq,*dk,*dv,*dout; float* dstats;
  CK(cudaMalloc(&dq,n*2)); CK(cudaMalloc(&dk,n*2)); CK(cudaMalloc(&dv,n*2)); CK(cudaMalloc(&dout,n*2));
  CK(cudaMalloc(&dstats,(long)B*H*T*sizeof(float)));
  CK(cudaMemcpy(dq,qb.data(),n*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dk,kb.data(),n*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv,vb.data(),n*2,cudaMemcpyHostToDevice));

  cudnnHandle_t h; cudnnCreate(&h);
  auto g=std::make_shared<fe::graph::Graph>();
  g->set_io_data_type(fe::DataType_t::BFLOAT16)
   .set_intermediate_data_type(fe::DataType_t::FLOAT)
   .set_compute_data_type(fe::DataType_t::FLOAT);
  std::vector<int64_t> dim={B,H,T,hd}, stride={(long)H*T*hd,(long)T*hd,hd,1};
  auto Q=g->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(dim).set_stride(stride));
  auto K=g->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(dim).set_stride(stride));
  auto V=g->tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(dim).set_stride(stride));
  auto opts=fe::graph::SDPA_attributes().set_name("sdpa")
            .set_causal_mask(true).set_attn_scale(scale);
  auto [O,Stats]=g->sdpa(Q,K,V,opts);
  O->set_output(true).set_dim(dim).set_stride(stride);
  Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({B,H,T,1}).set_stride({(long)H*T,(long)T,1,1});
  if(g->validate().is_bad()){ printf("validate failed\n"); return 1; }
  if(g->build_operation_graph(h).is_bad()){ printf("build_opgraph failed\n"); return 1; }
  if(g->create_execution_plans({fe::HeurMode_t::A}).is_bad()){ printf("plans failed\n"); return 1; }
  if(g->check_support(h).is_bad()){ printf("check_support failed\n"); return 1; }
  if(g->build_plans(h, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_bad()){ printf("build_plans failed\n"); return 1; }
  int64_t ws=g->get_workspace_size(); void* wsp=nullptr; if(ws) CK(cudaMalloc(&wsp,ws));
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,void*> pack={
    {Q,dq},{K,dk},{V,dv},{O,dout},{Stats,dstats}};
  if(g->execute(h,pack,wsp).is_bad()){ printf("execute failed\n"); return 1; }
  CK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> ho(n); CK(cudaMemcpy(ho.data(),dout,n*2,cudaMemcpyDeviceToHost));
  double me=0,sc=0; for(long i=0;i<n;i++){ double o=__bfloat162float(ho[i]); me=fmax(me,fabs(o-ref[i])); sc=fmax(sc,fabs(ref[i])); }
  printf("cuDNN SDPA fwd rel-max-err=%.4f (abs %.4f scale %.4f)\n", me/(sc+1e-9), me, sc);
  printf(me/(sc+1e-9)<0.05?"CUDNN SDPA OK\n":"CUDNN SDPA FAIL\n");
  return 0;
}
