"""Matched JAX reference (same llama arch + flop formula) for MFU comparison."""
import os, time, argparse, math
import numpy as np
import jax, jax.numpy as jnp
from functools import partial

def rms(x,w,eps=1e-5): return x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+eps)*w
def rope_tab(T,hd,theta,dt):
    i=jnp.arange(0,hd,2); freq=theta**(-i/hd); ang=jnp.outer(jnp.arange(T),freq)
    return jnp.cos(ang).astype(dt), jnp.sin(ang).astype(dt)
def rope(x,cos,sin):
    B,T,H,hd=x.shape; x0=x[...,:hd//2]; x1=x[...,hd//2:]
    c=cos[None,:,None]; s=sin[None,:,None]
    return jnp.concatenate([x0*c-x1*s, x1*c+x0*s],-1)

def init(key,V,C,L,H,I):
    k=jax.random.split(key,4+L*5); ki=iter(k); n=lambda *s: 0.02*jax.random.normal(next(ki),s,jnp.bfloat16)
    p={'wte':n(V,C),'lnf':jnp.ones(C,jnp.bfloat16),'lm':n(C,V),'blocks':[]}
    for _ in range(L):
        p['blocks'].append({'ln1':jnp.ones(C,jnp.bfloat16),'ln2':jnp.ones(C,jnp.bfloat16),
            'qkv':n(C,3*C),'o':n(C,C),'gate':n(C,I),'up':n(C,I),'down':n(I,C)})
    return p

def fwd(p,idx,tgt,C,H,I,theta=10000.0):
    B,T=idx.shape; hd=C//H; cos,sin=rope_tab(T,hd,theta,jnp.bfloat16)
    x=p['wte'][idx]
    for b in p['blocks']:
        h=rms(x,b['ln1']); qkv=(h@b['qkv']).reshape(B,T,3,H,hd)
        q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]; q=rope(q,cos,sin); k=rope(k,cos,sin)
        impl=os.environ.get("JAX_ATTN_IMPL","cudnn"); bf=jnp.bfloat16
        a=jax.nn.dot_product_attention(q.astype(bf),k.astype(bf),v.astype(bf),
                                       is_causal=True, implementation=impl)
        a=a.reshape(B,T,C); x=x+a@b['o']
        h=rms(x,b['ln2']); x=x+ (jax.nn.silu(h@b['gate'])*(h@b['up']))@b['down']
    x=rms(x,p['lnf']); logits=x@p['lm']
    ls=logits.astype(jnp.float32)
    lse=jax.scipy.special.logsumexp(ls,-1)
    tok=jnp.take_along_axis(ls,tgt[...,None],-1)[...,0]
    return jnp.mean(lse-tok)

def flops(V,C,L,H,I,T,B):
    Wl=3*C*C+C*C+I*C+I*C+C*I; Ng=L*Wl+V*C; tk=B*T
    return 6.0*Ng*tk+12.0*L*B*T*T*C

class TokenBinLoader:
    def __init__(self,path,B,T):
        with open(path,"rb") as f:
            header=np.fromfile(f,dtype=np.int32,count=64)
        if header.size != 64 or int(header[0]) != 20240520:
            got=int(header[0]) if header.size else -1
            raise ValueError(f"bad token bin header magic {got} in {path}")
        self.path=path; self.ntok=int(header[2]); self.B=B; self.T=T; self.pos=0
        self.data=np.memmap(path,dtype=np.uint16,mode="r",offset=64*4)
        print(f"[loader] {path} : {self.ntok} tokens")
    def next(self):
        need=self.B*self.T+1
        if self.pos+need>self.ntok: self.pos=0
        x=np.asarray(self.data[self.pos:self.pos+need],dtype=np.int32)
        self.pos += self.B*self.T
        return x[:-1].reshape(self.B,self.T), x[1:].reshape(self.B,self.T)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--B",type=int,default=16); ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--data",default=None,help="llm.c token .bin for real-data train timing")
    ap.add_argument("--log-every",type=int,default=int(os.environ.get("ENTROPY_LOG_EVERY","5")))
    a=ap.parse_args()
    ge=lambda n,d:int(os.environ.get(n,d))
    V,C,L,H,I,T=ge("ENTROPY_V",32768),ge("ENTROPY_C",768),ge("ENTROPY_L",12),ge("ENTROPY_H",12),ge("ENTROPY_I",2048),ge("ENTROPY_T",1024)
    key=jax.random.PRNGKey(0); p=init(key,V,C,L,H,I)
    import optax
    tm=jax.tree_util.tree_map
    # Fair mode: fp32 master weights + fp32 grads + fp32 Adam moments + bf16 compute,
    # exactly matching the entropy C/CUDA trainer (default JAX bench is pure-bf16, which
    # does ~3-4ms less optimizer/grad work and isn't an apples-to-apples comparison).
    f32_master=int(os.environ.get("ENTROPY_F32_MASTER","0"))
    if f32_master: p=tm(lambda x: x.astype(jnp.float32), p)
    lr=3e-4 if a.data else 1e-4
    opt=optax.adamw(lr,b1=0.9,b2=0.95,weight_decay=0.1); st=opt.init(p)
    if a.data:
        loader=TokenBinLoader(a.data,a.B,T)
        idx_np,tgt_np=loader.next()
        idx=jnp.asarray(idx_np); tgt=jnp.asarray(tgt_np)
    else:
        loader=None
        idx=jax.random.randint(jax.random.PRNGKey(1),(a.B,T),0,V)
        tgt=jax.random.randint(jax.random.PRNGKey(2),(a.B,T),0,V)
    lossf=partial(fwd,C=C,H=H,I=I)
    @jax.jit
    def step(p,st,idx,tgt):
        if f32_master:
            pbf=tm(lambda x: x.astype(jnp.bfloat16), p)            # bf16 copy for compute
            l,g=jax.value_and_grad(lossf)(pbf,idx,tgt)
            g=tm(lambda x: x.astype(jnp.float32), g)               # fp32 grads
        else:
            l,g=jax.value_and_grad(lossf)(p,idx,tgt)
        u,st=opt.update(g,st,p); p=optax.apply_updates(p,u); return p,st,l
    nparam=sum(x.size for x in jax.tree_util.tree_leaves(p))
    print(f"jax {jax.__version__}  {jax.devices()[0].device_kind}  params={nparam/1e6:.1f}M")
    if a.data:
        print(f"training {a.steps} steps: V={V} C={C} L={L} H={H} I={I} T={T} B={a.B} params={nparam/1e6:.1f}M data={a.data}")
        for _ in range(4):
            idx_np,tgt_np=loader.next(); idx=jnp.asarray(idx_np); tgt=jnp.asarray(tgt_np)
            p,st,l=step(p,st,idx,tgt)
        l.block_until_ready()
        t0=time.time(); w0=t0; wsteps=0
        for i in range(a.steps):
            idx_np,tgt_np=loader.next(); idx=jnp.asarray(idx_np); tgt=jnp.asarray(tgt_np)
            p,st,l=step(p,st,idx,tgt); wsteps+=1
            if i%a.log_every==0 or i==a.steps-1:
                l.block_until_ready()
                t=time.time(); dt=t-w0; fl=flops(V,C,L,H,I,T,a.B)*wsteps
                peak=float(os.environ.get("ENTROPY_PEAK_TFLOPS","364"))*1e12
                tok=a.B*T*wsteps
                print(f"step {i:4d}  loss {float(l):.4f}  {dt*1e3/wsteps:.2f} ms/step  {tok/dt:.0f} tok/s  {fl/dt/1e12:.1f} TFLOP/s  MFU {100*fl/dt/peak:.1f}%  elapsed {t-t0:.2f} s")
                w0=t; wsteps=0
        return
    # forward-only (loss, no grad/opt) to isolate fwd vs bwd+opt
    fwd_jit=jax.jit(lossf)
    for _ in range(4): lf=fwd_jit(p,idx,tgt)
    lf.block_until_ready()
    t0=time.time()
    for _ in range(a.steps): lf=fwd_jit(p,idx,tgt)
    lf.block_until_ready()
    dtf=(time.time()-t0)/a.steps
    for _ in range(4): p,st,l=step(p,st,idx,tgt)
    l.block_until_ready()
    t0=time.time()
    for _ in range(a.steps): p,st,l=step(p,st,idx,tgt)
    l.block_until_ready()
    dt=(time.time()-t0)/a.steps; fl=flops(V,C,L,H,I,T,a.B)
    peak=float(os.environ.get("ENTROPY_PEAK_TFLOPS","364"))*1e12
    print(f"  [jax fwd-only {dtf*1e3:.2f} ms]")
    print(f"B={a.B}  step {dt*1e3:.2f} ms   {a.B*T/dt:.0f} tok/s   {fl/dt/1e12:.1f} TFLOP/s   MFU {100*fl/dt/peak:.1f}%")

if __name__=="__main__": main()
