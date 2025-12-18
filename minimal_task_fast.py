import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lambertw_real_principal(x: torch.Tensor, iters: int = 15) -> torch.Tensor:
    dtype_out = x.dtype
    x = x.to(torch.float64)
    w = torch.where(x <= 1.0, x, torch.log(x + 1e-12) - torch.log(torch.log(x + 1e-12) + 1e-12))
    w = torch.clamp(w, min=-0.5)
    for _ in range(iters):
        ew = torch.exp(w)
        f = w * ew - x
        denom = ew * (1.0 + w)
        w = w - f / (denom + 1e-12)
    return w.to(dtype_out)

@torch.no_grad()
def bitflip_(t: torch.Tensor, idx: tuple, bit: int):
    assert t.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if t.dtype is torch.float32:
        iview = t.view(torch.int32); bit = bit & 31
    else:
        iview = t.view(torch.int16); bit = bit & 15
    iview[idx] ^= (1 << bit)

def sdp_attention_injected(q, k, v, *, inj_where="scores", inj_idx=(0,0,0,0), inj_bit=25, attn_mask=None, dropout_p=0.0, is_causal=True):
    d = q.size(-1)
    if inj_where == "q": bitflip_(q, inj_idx, inj_bit)
    if inj_where == "k": bitflip_(k, inj_idx, inj_bit)
    if inj_where == "v": bitflip_(v, inj_idx, inj_bit)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    if is_causal:
        T = scores.size(-2)
        causal = torch.ones(T, T, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
    if inj_where == "scores": bitflip_(scores, inj_idx, inj_bit)
    p = torch.softmax(scores, dim=-1)
    if inj_where == "weights": bitflip_(p, inj_idx, inj_bit)
    out = p @ v
    if inj_where == "out": bitflip_(out, inj_idx, inj_bit)
    return out, scores, p

@torch.no_grad()
def compute_attention_bounds(scores: torch.Tensor, p: torch.Tensor, d: int):
    B,H,T,_ = scores.shape
    sqrt_d = math.sqrt(d); n = T
    top2_vals, top2_idx = torch.topk(scores, k=2, dim=-1)
    a_star = top2_vals[...,0]; second = top2_vals[...,1]
    j_star_idx = top2_idx[...,0]
    w_star = p.gather(-1, j_star_idx.unsqueeze(-1)).squeeze(-1)
    gamma = a_star - second
    Ea = (p * scores).sum(dim=-1)
    epsilon = sqrt_d * (a_star - Ea)
    lower1 = sqrt_d * gamma / (1.0 + torch.exp(gamma))
    middle = sqrt_d * gamma * (1.0 - w_star)
    upper1 = sqrt_d * (a_star - scores.mean(dim=-1))
    lam_arg = torch.tensor((n - 1) / math.e, device=scores.device, dtype=scores.dtype)
    W = lambertw_real_principal(lam_arg)
    cond = gamma >= (W + 1.0)
    term_case1 = sqrt_d * ((n - 1) * torch.exp(-gamma)) / (1.0 + (n - 1) * torch.exp(-gamma)) * gamma
    term_case2 = sqrt_d * W
    upper2 = torch.where(cond, term_case1, term_case2)
    upper = torch.minimum(upper1, upper2)
    return {"a_star": a_star, "w_star": w_star, "gamma": gamma, "epsilon": epsilon, "lower1": lower1, "middle": middle, "upper1": upper1, "upper2": upper2, "upper": upper}

@torch.no_grad()
def compute_injected_attention(scores: torch.Tensor, attn_out: torch.Tensor, x: torch.Tensor, d: int):
    sqrt_d = d ** 0.5
    a_star = scores.max(dim=-1).values
    Ea = (attn_out * x).sum(dim=-1)
    return sqrt_d * a_star - Ea

class TinyCharDataset:
    def __init__(self, text: str, block_size: int = 32, split: float = 0.9):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * split)
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, batch_size: int = 8):
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= self.block_size:
            ix = torch.zeros(batch_size, dtype=torch.long)
        else:
            ix = torch.randint(0, len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y

class FaultySelfAttention(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads; self.head_dim = head_dim

    def forward(self, x, *, inject=False, inj_where="scores", inj_idx=(0,0,0,0), inj_bit=25):
        B,T,D = x.shape; H,Dh = self.n_heads, self.head_dim
        assert D == H*Dh
        xh = x.view(B,T,H,Dh).permute(0,2,1,3).contiguous()
        q=k=v=xh
        out, scores, p = sdp_attention_injected(q,k,v, inj_where=("scores" if inject else "none"), inj_idx=inj_idx, inj_bit=inj_bit, is_causal=True)
        out = out.permute(0,2,1,3).contiguous().view(B,T,D)
        return out, scores, p, xh

class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int=64, n_heads: int=4, head_dim: int=16, block_size: int=32):
        super().__init__()
        assert n_embd == n_heads*head_dim
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.attn = FaultySelfAttention(n_heads, head_dim)
        self.ln = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None, *, inject=False, inj_where="scores", inj_idx=(0,0,0,0), inj_bit=25):
        x = self.embed(idx)
        attn_out, scores, p, xh = self.attn(x, inject=inject, inj_where=inj_where, inj_idx=inj_idx, inj_bit=inj_bit)
        x = self.ln(x + attn_out)
        logits = self.proj(x)
        loss = None
        if targets is not None:
            B,T,V = logits.shape
            loss = F.cross_entropy(logits.view(B*T,V), targets.view(B*T))
        return logits, loss, scores, p, xh

corpus = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "or to take arms against a sea of troubles\n"
    "and by opposing end them.\n"
    "你好，世界！这是一个很小的语料，用来做最小可用演示。\n"
)

block_size = 32
dataset = TinyCharDataset(corpus, block_size=block_size, split=0.9)
vocab_size = dataset.vocab_size
device = device_auto()

torch.manual_seed(0); random.seed(0); np.random.seed(0)
model = TinyLM(vocab_size=vocab_size, n_embd=64, n_heads=4, head_dim=16, block_size=block_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

# short warmup (10 steps)
for step in range(10):
    model.train()
    x,y = dataset.get_batch("train", batch_size=8)
    x,y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)
    _, loss, _, _, _ = model(x,y, inject=False)
    loss.backward(); opt.step()
    if (step+1) % 5 == 0:
        print(f"step {step+1:02d} | train loss {loss.item():.4f}")

# evaluate baseline
model.eval()
with torch.no_grad():
    x,y = dataset.get_batch("val", batch_size=4)
    x,y = x.to(device), y.to(device)
    _, base_loss, base_scores, base_p, base_xh = model(x,y, inject=False)
print(f"\nBaseline val loss: {base_loss.item():.6f}")

# inject
b,h,i,j = 0,0, min(2,block_size-1), min(7,block_size-1)
inj_idx = (b,h,i,j)

with torch.no_grad():
    _, inj_loss, scores_f, p_f, xh_f = model(x,y, inject=True, inj_where="scores", inj_idx=inj_idx, inj_bit=25)
print(f"Injected val loss: {inj_loss.item():.6f}")
print(f"Δ loss: {inj_loss.item() - base_loss.item():.6f}")

# bounds
Dh = 16
bounds = compute_attention_bounds(base_scores, base_p, d=Dh)
eps = 1e-9
lhs_ok = (bounds["lower1"] <= bounds["middle"] + eps).all().item()
mid_ok = (bounds["middle"] <= bounds["epsilon"] + eps).all().item()
rhs_ok = (bounds["epsilon"] <= bounds["upper"] + eps).all().item()
print("\nBound checks (baseline):")
print(f"lower1 ≤ middle: {bool(lhs_ok)}")
print(f"middle ≤ epsilon: {bool(mid_ok)}")
print(f"epsilon ≤ upper: {bool(rhs_ok)}")

row = {k: (v[b, h, i].item() if v.ndim == 3 else float(v[b, h, i])) for k, v in bounds.items() if k in
       ("gamma", "w_star", "epsilon", "lower1", "middle", "upper1", "upper2", "upper")}
print(f"\n[b={b}, h={h}, i={i}] bounds snapshot:")
for k in ("gamma", "w_star", "epsilon", "lower1", "middle", "upper1", "upper2", "upper"):
    print(f"{k:>8s}: {row[k]:.6f}")

value_injected = compute_injected_attention(base_scores, attn_out=(base_p @ base_xh), x=base_xh, d=Dh)
print(f"\nInjected-epsilon baseline snapshot (b={b},h={h},i={i}): {value_injected[b,h,i].item():.6f}")

abs_err = (((p_f @ xh_f) - (base_p @ base_xh)).abs().max()).item()
print(f"Max abs err (attn_out, injected vs base): {abs_err:.6f}")

print("\nDone.")