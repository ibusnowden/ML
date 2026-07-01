// cuDNN fused flash attention (causal) — forward + backward, graph cached for a
// fixed (B,H,T,hd). q,k,v,o,dq,dk,dv: [B,H,T,hd] BF16; stats: [B,H,T,1] FP32.
#pragma once
#ifdef USE_CUDNN
#include <cudnn_frontend.h>
#include <memory>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
namespace fe = cudnn_frontend;

static void cudnn_status_check(cudnnStatus_t status, const char* what){
  if(status != CUDNN_STATUS_SUCCESS){
    std::fprintf(stderr, "cuDNN error at %s: %s\n", what, cudnnGetErrorString(status));
    std::exit(1);
  }
}
static void cudnn_fe_check(fe::error_t err, const char* what){
  if(err.is_bad()){
    std::fprintf(stderr, "cuDNN frontend error at %s: %s\n", what, err.get_message().c_str());
    std::exit(1);
  }
}
static void cudnn_cuda_check(cudaError_t err, const char* what){
  if(err != cudaSuccess){
    std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

struct CudnnAttn {
  cudnnHandle_t handle=nullptr;
  bool built=false;
  int B,H,T,hd; float scale;
  void* ws=nullptr; size_t wssize=0;
  // forward graph + tensor handles
  std::shared_ptr<fe::graph::Graph> gf;
  std::shared_ptr<fe::graph::Tensor_attributes> Qf,Kf,Vf,Of,Sf;
  // backward graph + handles
  std::shared_ptr<fe::graph::Graph> gb;
  std::shared_ptr<fe::graph::Tensor_attributes> Qb,Kb,Vb,Ob,dOb,Sb,dQb,dKb,dVb;

  void init(){ if(!handle) cudnn_status_check(cudnnCreate(&handle), "cudnnCreate"); }
  void set_stream(cudaStream_t s){ cudnn_status_check(cudnnSetStream(handle,s), "cudnnSetStream"); }

  void grow_ws(size_t need){
    if(need>wssize){
      if(ws) cudnn_cuda_check(cudaFree(ws), "cudaFree(cuDNN workspace)");
      cudnn_cuda_check(cudaMalloc(&ws,need), "cudaMalloc(cuDNN workspace)");
      wssize=need;
    }
  }

  // Strides chosen so cuDNN reads/writes directly from the interleaved buffers,
  // avoiding all permute kernels:
  //   qkv[B,T,3C]: Q/K/V (and dQ/dK/dV) slices -> stride {T*3C, hd, 3C, 1}
  //   atty/datty [B,T,C]: O/dO               -> stride {T*C,  hd, C,  1}
  // packed=false: Q/K/V read straight from interleaved qkv[B,T,3C] (permute-free).
  // packed=true:  Q/K/V/O are dense [B,H,T,hd] (caller permutes) — full cache-line
  //               utilization, which helps the bandwidth-bound backward.
  void build(int B_,int H_,int T_,int hd_,int C_,float scale_,bool packed=false){
    B=B_;H=H_;T=T_;hd=hd_;scale=scale_; long C=C_;
    std::vector<int64_t> dim={B,H,T,hd};
    std::vector<int64_t> stq = packed ? std::vector<int64_t>{(long)H*T*hd,(long)T*hd,(long)hd,1}
                                      : std::vector<int64_t>{T*3*C, (long)hd, 3*C, 1};   // qkv-interleaved
    std::vector<int64_t> sto = packed ? std::vector<int64_t>{(long)H*T*hd,(long)T*hd,(long)hd,1}
                                      : std::vector<int64_t>{T*C,   (long)hd, C,   1};   // o-contiguous-by-head
    std::vector<int64_t> sdim={B,H,T,1}, sst={(long)H*T,(long)T,1,1};
    auto st=stq; (void)st;
    // ---- forward ----
    gf=std::make_shared<fe::graph::Graph>();
    gf->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);
    Qf=gf->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(dim).set_stride(stq));
    Kf=gf->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(dim).set_stride(stq));
    Vf=gf->tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(dim).set_stride(stq));
    auto o=fe::graph::SDPA_attributes().set_name("sdpa").set_is_inference(false)
            .set_causal_mask(true).set_attn_scale(scale);
    auto [O,S]=gf->sdpa(Qf,Kf,Vf,o);
    Of=O; Sf=S;
    Of->set_output(true).set_dim(dim).set_stride(sto);
    Sf->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim(sdim).set_stride(sst);
    cudnn_fe_check(gf->validate(), "sdpa forward validate");
    cudnn_fe_check(gf->build_operation_graph(handle), "sdpa forward build_operation_graph");
    cudnn_fe_check(gf->create_execution_plans({fe::HeurMode_t::A}), "sdpa forward create_execution_plans");
    cudnn_fe_check(gf->check_support(handle), "sdpa forward check_support");
    cudnn_fe_check(gf->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "sdpa forward build_plans");
    // ---- backward ----
    gb=std::make_shared<fe::graph::Graph>();
    gb->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);
    Qb=gb->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(dim).set_stride(stq));
    Kb=gb->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(dim).set_stride(stq));
    Vb=gb->tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(dim).set_stride(stq));
    Ob=gb->tensor(fe::graph::Tensor_attributes().set_name("O").set_dim(dim).set_stride(sto));
    dOb=gb->tensor(fe::graph::Tensor_attributes().set_name("dO").set_dim(dim).set_stride(sto));
    Sb=gb->tensor(fe::graph::Tensor_attributes().set_name("S").set_data_type(fe::DataType_t::FLOAT).set_dim(sdim).set_stride(sst));
    auto ob=fe::graph::SDPA_backward_attributes().set_name("sdpa_bwd").set_causal_mask(true).set_attn_scale(scale);
    auto [dQ,dK,dV]=gb->sdpa_backward(Qb,Kb,Vb,Ob,dOb,Sb,ob);
    dQb=dQ;dKb=dK;dVb=dV;
    dQb->set_output(true).set_dim(dim).set_stride(stq);
    dKb->set_output(true).set_dim(dim).set_stride(stq);
    dVb->set_output(true).set_dim(dim).set_stride(stq);
    cudnn_fe_check(gb->validate(), "sdpa backward validate");
    cudnn_fe_check(gb->build_operation_graph(handle), "sdpa backward build_operation_graph");
    cudnn_fe_check(gb->create_execution_plans({fe::HeurMode_t::A}), "sdpa backward create_execution_plans");
    cudnn_fe_check(gb->check_support(handle), "sdpa backward check_support");
    cudnn_fe_check(gb->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "sdpa backward build_plans");
    size_t need = gf->get_workspace_size(); size_t nb=gb->get_workspace_size();
    grow_ws(need>nb?need:nb);
    built=true;
  }

  void forward(cudaStream_t s, const void* Q,const void* K,const void* V, void* O, void* stats){
    set_stream(s);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,void*> p={
      {Qf,(void*)Q},{Kf,(void*)K},{Vf,(void*)V},{Of,O},{Sf,stats}};
    cudnn_fe_check(gf->execute(handle,p,ws), "sdpa forward execute");
  }
  void backward(cudaStream_t s, const void* Q,const void* K,const void* V,const void* O,
                const void* dO, const void* stats, void* dQ,void* dK,void* dV){
    set_stream(s);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,void*> p={
      {Qb,(void*)Q},{Kb,(void*)K},{Vb,(void*)V},{Ob,(void*)O},{dOb,(void*)dO},{Sb,(void*)stats},
      {dQb,dQ},{dKb,dK},{dVb,dV}};
    cudnn_fe_check(gb->execute(handle,p,ws), "sdpa backward execute");
  }
};
#endif
