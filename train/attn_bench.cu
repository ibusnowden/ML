// Isolate cuDNN SDPA fwd+bwd cost for two memory layouts at the real shape:
//  (A) interleaved/strided qkv (our permute-free path): Q/K/V stride {T*3C,hd,3C,1}
//  (B) contiguous packed [B,H,T,hd]:                    stride {H*T*hd,T*hd,hd,1}
// If (B) is much faster for backward, the permute-free layout is the culprit and
// permuting (cheap elementwise) before cuDNN is a net win.
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <ctime>
#include <memory>
#include <unordered_map>
namespace fe = cudnn_frontend;
typedef __nv_bfloat16 bf16;
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec/1e9; }

struct G {
  std::shared_ptr<fe::graph::Graph> gf,gb;
  std::shared_ptr<fe::graph::Tensor_attributes> Qf,Kf,Vf,Of,Sf;
  std::shared_ptr<fe::graph::Tensor_attributes> Qb,Kb,Vb,Ob,dOb,Sb,dQ,dK,dV;
  void* ws=nullptr; size_t wssz=0;
};
static void build(cudnnHandle_t h, G& g, int B,int H,int T,int hd,
                  std::vector<int64_t> stqkv, std::vector<int64_t> sto, float scale){
  std::vector<int64_t> dim={B,H,T,hd}, sdim={B,H,T,1}, sst={(long)H*T,(long)T,1,1};
  g.gf=std::make_shared<fe::graph::Graph>();
  g.gf->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
  g.Qf=g.gf->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(dim).set_stride(stqkv));
  g.Kf=g.gf->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(dim).set_stride(stqkv));
  g.Vf=g.gf->tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(dim).set_stride(stqkv));
  auto o=fe::graph::SDPA_attributes().set_name("sdpa").set_is_inference(false).set_causal_mask(true).set_attn_scale(scale);
  auto [O,S]=g.gf->sdpa(g.Qf,g.Kf,g.Vf,o); g.Of=O; g.Sf=S;
  g.Of->set_output(true).set_dim(dim).set_stride(sto);
  g.Sf->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim(sdim).set_stride(sst);
  g.gf->validate(); g.gf->build_operation_graph(h); g.gf->create_execution_plans({fe::HeurMode_t::A});
  g.gf->check_support(h); g.gf->build_plans(h, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
  g.gb=std::make_shared<fe::graph::Graph>();
  g.gb->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
  g.Qb=g.gb->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(dim).set_stride(stqkv));
  g.Kb=g.gb->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(dim).set_stride(stqkv));
  g.Vb=g.gb->tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(dim).set_stride(stqkv));
  g.Ob=g.gb->tensor(fe::graph::Tensor_attributes().set_name("O").set_dim(dim).set_stride(sto));
  g.dOb=g.gb->tensor(fe::graph::Tensor_attributes().set_name("dO").set_dim(dim).set_stride(sto));
  g.Sb=g.gb->tensor(fe::graph::Tensor_attributes().set_name("S").set_data_type(fe::DataType_t::FLOAT).set_dim(sdim).set_stride(sst));
  auto ob=fe::graph::SDPA_backward_attributes().set_name("sdpa_bwd").set_causal_mask(true).set_attn_scale(scale);
  auto [dq,dk,dv]=g.gb->sdpa_backward(g.Qb,g.Kb,g.Vb,g.Ob,g.dOb,g.Sb,ob); g.dQ=dq;g.dK=dk;g.dV=dv;
  g.dQ->set_output(true).set_dim(dim).set_stride(stqkv);
  g.dK->set_output(true).set_dim(dim).set_stride(stqkv);
  g.dV->set_output(true).set_dim(dim).set_stride(stqkv);
  g.gb->validate(); g.gb->build_operation_graph(h); g.gb->create_execution_plans({fe::HeurMode_t::A});
  g.gb->check_support(h); g.gb->build_plans(h, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
  size_t a=g.gf->get_workspace_size(), b=g.gb->get_workspace_size(); g.wssz=a>b?a:b;
  if(g.wssz) CK(cudaMalloc(&g.ws,g.wssz));
}

int main(int argc,char**argv){
  int B=16,H=12,T=1024,hd=64,C=H*hd,L=12,steps=20; float scale=1.0f/sqrtf((float)hd);
  const char* eb=getenv("ENTROPY_BENCH_B"); if(eb&&atoi(eb)>0) B=atoi(eb);
  const char* es=getenv("ENTROPY_BENCH_STEPS"); if(es&&atoi(es)>0) steps=atoi(es);
  cudnnHandle_t h; cudnnCreate(&h);
  // Buffers: interleaved qkv[B,T,3C] and contiguous packed[B,H,T,hd]
  long nqkv=(long)B*T*3*C, npack=(long)B*H*T*hd, no=(long)B*T*C, nstat=(long)B*H*T;
  bf16 *qkv,*atty,*datty,*dqkv, *pk_q,*pk_k,*pk_v,*pk_o,*pk_do,*pk_dq,*pk_dk,*pk_dv; float *stats;
  CK(cudaMalloc(&qkv,nqkv*2)); CK(cudaMalloc(&atty,no*2)); CK(cudaMalloc(&datty,no*2)); CK(cudaMalloc(&dqkv,nqkv*2));
  CK(cudaMalloc(&pk_q,npack*2)); CK(cudaMalloc(&pk_k,npack*2)); CK(cudaMalloc(&pk_v,npack*2));
  CK(cudaMalloc(&pk_o,npack*2)); CK(cudaMalloc(&pk_do,npack*2));
  CK(cudaMalloc(&pk_dq,npack*2)); CK(cudaMalloc(&pk_dk,npack*2)); CK(cudaMalloc(&pk_dv,npack*2));
  CK(cudaMalloc(&stats,nstat*4));
  // layout A: strided interleaved
  G ga; build(h,ga,B,H,T,hd, {(long)T*3*C,hd,3*C,1}, {(long)T*C,hd,C,1}, scale);
  // layout B: contiguous packed
  G gb; build(h,gb,B,H,T,hd, {(long)H*T*hd,(long)T*hd,hd,1}, {(long)H*T*hd,(long)T*hd,hd,1}, scale);
  cudnnSetStream(h,0);
  auto run=[&](G&g, const char* nm, bf16*Q,bf16*K,bf16*V,bf16*O,bf16*dO,bf16*dQ,bf16*dK,bf16*dV){
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,void*> pf={{g.Qf,Q},{g.Kf,K},{g.Vf,V},{g.Of,O},{g.Sf,stats}};
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,void*> pb={{g.Qb,Q},{g.Kb,K},{g.Vb,V},{g.Ob,O},{g.dOb,dO},{g.Sb,stats},{g.dQ,dQ},{g.dK,dK},{g.dV,dV}};
    for(int i=0;i<5;i++){ g.gf->execute(h,pf,g.ws); g.gb->execute(h,pb,g.ws);} CK(cudaDeviceSynchronize());
    double t0=now_s(); for(int s=0;s<steps;s++) for(int l=0;l<L;l++) g.gf->execute(h,pf,g.ws); CK(cudaDeviceSynchronize());
    double fwd=(now_s()-t0)/steps*1e3;
    t0=now_s(); for(int s=0;s<steps;s++) for(int l=0;l<L;l++) g.gb->execute(h,pb,g.ws); CK(cudaDeviceSynchronize());
    double bwd=(now_s()-t0)/steps*1e3;
    printf("%-12s  fwd %.2f ms  bwd %.2f ms  (bwd/fwd=%.1fx)  [%d layers]\n",nm,fwd,bwd,bwd/fwd,L);
  };
  printf("B=%d H=%d T=%d hd=%d  L=%d\n",B,H,T,hd,L);
  run(ga,"strided",  qkv, qkv+C, qkv+2*C, atty, datty, dqkv, dqkv+C, dqkv+2*C);
  run(gb,"contiguous",pk_q,pk_k,pk_v, pk_o, pk_do, pk_dq,pk_dk,pk_dv);
  return 0;
}
