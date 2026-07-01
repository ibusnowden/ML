"""Matched PyTorch reference for the entropy C/CUDA trainer.
Same llama-style arch (RMSNorm, RoPE rotate-half, SwiGLU, causal MHA, bias-free,
untied LM head) and the SAME FLOP formula, so MFU is directly comparable.
"""
import os, sys, time, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F

def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

def rope_cos_sin(T, hd, theta, device, dtype):
    i = torch.arange(0, hd, 2, device=device).float()
    freq = theta ** (-i / hd)                      # [hd/2]
    t = torch.arange(T, device=device).float()
    ang = torch.outer(t, freq)                     # [T, hd/2]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)

def apply_rope(x, cos, sin):
    # x: [B,H,T,hd]; rotate-half (HF/llama): pair (i, i+hd/2)
    B,H,T,hd = x.shape
    x0, x1 = x[..., :hd//2], x[..., hd//2:]
    c = cos[None,None]; s = sin[None,None]
    return torch.cat([x0*c - x1*s, x1*c + x0*s], dim=-1)

class Block(nn.Module):
    def __init__(s, C, H, I):
        super().__init__()
        s.H=H; s.hd=C//H
        s.ln1=nn.Parameter(torch.ones(C)); s.ln2=nn.Parameter(torch.ones(C))
        s.qkv=nn.Linear(C,3*C,bias=False); s.o=nn.Linear(C,C,bias=False)
        s.gate=nn.Linear(C,I,bias=False); s.up=nn.Linear(C,I,bias=False); s.down=nn.Linear(I,C,bias=False)
    def forward(s, x, cos, sin):
        B,T,C=x.shape
        h=rmsnorm(x,s.ln1)
        qkv=s.qkv(h).view(B,T,3,s.H,s.hd).permute(2,0,3,1,4)  # [3,B,H,T,hd]
        q,k,v=qkv[0],qkv[1],qkv[2]
        q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)  # flash
        a=a.transpose(1,2).reshape(B,T,C)
        x=x+s.o(a)
        h=rmsnorm(x,s.ln2)
        x=x+s.down(F.silu(s.gate(h))*s.up(h))
        return x

class GPT(nn.Module):
    def __init__(s,V,C,L,H,I,T,theta=10000.0):
        super().__init__()
        s.V,s.C,s.L,s.H,s.I,s.T,s.theta=V,C,L,H,I,T,theta
        s.wte=nn.Embedding(V,C)
        s.blocks=nn.ModuleList([Block(C,H,I) for _ in range(L)])
        s.lnf=nn.Parameter(torch.ones(C))
        s.lm=nn.Linear(C,V,bias=False)
    def forward(s, idx, targets):
        B,T=idx.shape
        cos,sin=rope_cos_sin(T,s.C//s.H,s.theta,idx.device,torch.bfloat16)
        x=s.wte(idx)
        for b in s.blocks: x=b(x,cos,sin)
        x=rmsnorm(x,s.lnf)
        logits=s.lm(x)
        loss=F.cross_entropy(logits.float().view(-1,s.V), targets.view(-1))
        return loss

def flops_per_step(V,C,L,H,I,T,B):
    Wlayer = 3*C*C + C*C + I*C + I*C + C*I
    Ngemm  = L*Wlayer + V*C
    tokens = B*T
    return 6.0*Ngemm*tokens + 12.0*L*B*T*T*C

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--B",type=int,default=8); ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--compile",action="store_true")
    a=ap.parse_args()
    ge=lambda n,d:int(os.environ.get(n,d))
    V,C,L,H,I,T=ge("ENTROPY_V",32768),ge("ENTROPY_C",768),ge("ENTROPY_L",12),ge("ENTROPY_H",12),ge("ENTROPY_I",2048),ge("ENTROPY_T",1024)
    dev="cuda"
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision("high")
    m=GPT(V,C,L,H,I,T).to(dev).bfloat16()
    nparam=sum(p.numel() for p in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    fwd=m
    if a.compile: fwd=torch.compile(m)
    g=torch.Generator(device=dev).manual_seed(7)
    idx=torch.randint(0,V,(a.B,T),device=dev,generator=g)
    tgt=torch.randint(0,V,(a.B,T),device=dev,generator=g)
    prop=torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} on {prop.name}  params={nparam/1e6:.1f}M  compile={a.compile}")
    for i in range(4):  # warmup
        opt.zero_grad(set_to_none=True); loss=fwd(idx,tgt); loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0=time.time()
    for i in range(a.steps):
        opt.zero_grad(set_to_none=True); loss=fwd(idx,tgt); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt=(time.time()-t0)/a.steps
    peak_tflops=float(os.environ.get("ENTROPY_PEAK_TFLOPS","364"))
    fl=flops_per_step(V,C,L,H,I,T,a.B); peak=peak_tflops*1e12
    print(f"B={a.B}  step {dt*1e3:.2f} ms   {a.B*T/dt:.0f} tok/s   {fl/dt/1e12:.1f} TFLOP/s   MFU {100*fl/dt/peak:.1f}%")

if __name__=="__main__": main()
