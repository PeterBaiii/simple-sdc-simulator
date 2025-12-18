import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, Tuple
import numpy as np
from scipy.special import lambertw

from utils.debug import debug
from utils.check_nan import check_nan

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def bitflip_(t: torch.Tensor, idx: tuple, bit: int):
    """
    In-place bit flip for float tensors (fp16/bf16/fp32).
    idx: tuple indexing into t (e.g., (b,h,i,j))
    bit: which bit to flip (int)
    """
    assert t.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if t.dtype is torch.float32:
        iview = t.view(torch.int32)
        bit = bit & 31
    else:
        iview = t.view(torch.int16)
        bit = bit & 15
    iview[idx] ^= (1 << bit)

def sdp_attention_injected(q, k, v, *,
                           inj_where="none",
                           inj_idx=(0,0,0,0),
                           inj_bit=0,
                           attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    Scaled dot-product attention with optional fault injection.
    Shapes:
      q, k, v: (B, H, T, Dh)
    Returns:
      out: (B, H, T, Dh), scores, p
    """
    if inj_where == "q": bitflip_(q, inj_idx, inj_bit)
    if inj_where == "k": bitflip_(k, inj_idx, inj_bit)
    if inj_where == "v": bitflip_(v, inj_idx, inj_bit)

    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    
    ###################################
    ###################################
    # if inj_where == "flag":
    #     debug(scores[0][0][0])
    #     debug(q[0][0][0])
    #     for i in range(5):
    #         debug(k[0][0][i])
            
    #     debug(k[0, 0, 0, :] == k.transpose(-2, -1)[0, 0, :, 0])
    #     # debug((q[0][0][0] * k[0][0][0]).sum() / math.sqrt(d))
    #     # debug((q[0][0][0] * k[0][0][1]).sum() / math.sqrt(d))
    #     debug((q[0, 0, 0, :] * k.transpose(-2, -1)[0, 0, :, 1]).sum() / math.sqrt(d), scores[0, 0, 0, 1])
    ###################################
    ###################################

    if attn_mask is not None:
        scores = scores + attn_mask

    if is_causal:
        T = scores.size(-2)
        causal = torch.ones(T, T, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))

    if inj_where == "scores": bitflip_(scores, inj_idx, inj_bit)

    p = torch.softmax(scores, dim=-1)

    if inj_where == "weights": bitflip_(p, inj_idx, inj_bit)

    if dropout_p and dropout_p > 0:
        p = F.dropout(p, p=dropout_p, training=True)

    out = p @ v
    if inj_where == "out": bitflip_(out, inj_idx, inj_bit)

    #####################################################
    #####################################################
    # finite = torch.isfinite(scores)
    # scores= torch.where(finite, scores, torch.zeros_like(scores))

    # all_masked = ~finite.any(dim=-1, keepdim=True)
    # p = torch.where(all_masked, torch.zeros_like(p), p)
    #####################################################
    #####################################################
    
    return out, scores, p

@torch.no_grad()
def compute_attention_bounds(scores: torch.Tensor, p: torch.Tensor, d: int):
    """
    scores, p: (B, H, T, T)
    d: head_dim
    Returns a dict of bound tensors (B,H,T)
    """
    B, H, T, _ = scores.shape
    device = scores.device
    dtype = scores.dtype
    sqrt_d = math.sqrt(d)
    n = T

    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    top2_vals, _ = torch.topk(scores, k=2, dim=-1)
    a_star = top2_vals[..., 0] # (B, H, N)
    second = top2_vals[..., 1] # (B, H, N)
    # from utils.return_top2 import top2_distinct_lastdim
    # a_star, second = top2_distinct_lastdim(scores)
    # j_star_idx = top2_idx[..., 0] # index tensor (B, H, N)
    w_star = p.max(dim=-1).values # (B, H, N)
    # w_star = p.gather(-1, j_star_idx.unsqueeze(-1)).squeeze(-1) # (B, H, N)
    gamma = a_star - second
    
    # weak condition, at least acquire ==> K == V
    Ea = (p * scores).sum(dim=-1) # (B, H, N)
    x = torch.nan_to_num(Ea, nan=0.0)
    epsilon = sqrt_d * (a_star - Ea)

    ##################################
    ##################################
    # debug(scores[0][0][2])
    # debug(scores[0][0][0], p[0][0][0])
    # check_nan(a_star, name="a_star")
    # check_nan(second, name="second")
    # check_nan(gamma, name="gamma")
    # check_nan(p, name="p")
    # check_nan(scores, name="scores")
    # check_nan(Ea, name="Ea")
    ##################################
    ##################################
    
    lower1 = sqrt_d * gamma / (1.0 + torch.exp(gamma))
    middle = sqrt_d * gamma * (1.0 - w_star)
    upper1 = sqrt_d * (a_star - scores.mean(dim=-1))

    lam_arg = torch.tensor((n - 1) / math.e, device=device, dtype=dtype)
    W_np = np.asarray(lambertw(lam_arg.detach().cpu().numpy(), 0).real)
    W = torch.as_tensor(W_np, device=lam_arg.device, dtype=lam_arg.dtype)
    cond = gamma >= (W + 1.0)

    term_case1 = sqrt_d * ((n - 1) * torch.exp(-gamma)) / (1.0 + (n - 1) * torch.exp(-gamma)) * gamma
    term_case2 = sqrt_d * W
    upper2 = torch.where(cond, term_case1, term_case2)
    upper = torch.minimum(upper1, upper2)

    return {
        "a_star": a_star, "w_star": w_star, "gamma": gamma,
        "epsilon": epsilon, "lower1": lower1, "middle": middle,
        "upper1": upper1, "upper2": upper2, "upper": upper
    }

@torch.no_grad()
def compute_injected_attention(scores: torch.Tensor, attn_inj: torch.Tensor, q: torch.Tensor, d: int):
    """
    Compute epsilon-like quantity after injection:
        sqrt(d) * max_j a_ij - <Attn_inj(x_i), x_i>
    Constraints:
        x must be the query tensor
    """
    sqrt_d = d ** 0.5
    a_star = scores.max(dim=-1).values
    Ea = (attn_inj * q).sum(dim=-1)
    return sqrt_d * a_star - Ea

# -----------------------------
# Tiny real task: char-level LM
# -----------------------------

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
    
    def get_batch(self, split: str, batch_size: int = 16):
        data = self.train_data if split == "train" else self.val_data
        # 1 digit shift to the left, because the training process is "next token prediction"
        max_start = len(data) - self.block_size - 1

        # slice the data into batches, each batch contains a block of characters with fixed size 
        if max_start < 0:
            # ensure the same length of x and y
            x = data[:-1]
            y = data[1:]
            # Repeat this single sequence to form a batch.
            x = x.unsqueeze(0).repeat(batch_size, 1) # (B * N)
            y = y.unsqueeze(0).repeat(batch_size, 1)
        else:
            # If the data is long enough, sample random starting positions.
            ix = torch.randint(0, max_start + 1, (batch_size,))
            x = torch.stack([data[i : i+self.block_size] for i in ix])
            y = torch.stack([data[i+1 : i+self.block_size+1] for i in ix])
            
        return x, y

class FaultySelfAttention(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, causal: bool = True, p_drop: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.causal = causal
        self.p_drop = p_drop

        # learnable parameters
        D = n_heads * head_dim
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_kv = nn.Linear(D, D, bias=False)

    def forward(self, x, *, inject: bool = False,
                inj_where="none", inj_idx=(0,0,0,0), inj_bit=0):
        
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        assert D == H * Dh, "Embedding dim must equal H*Dh in this minimal demo."

        # shared K and V
        q = self.W_q(x)
        kv = self.W_kv(x)
        k, v = kv, kv

        # reshape tensors into their multihead version, while keeping the main memory consistent
        q = q.view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()

        # dynamically call the self-defined injected attention
        if inject:
            out, scores, p = sdp_attention_injected(
                q, k, v,
                inj_where=inj_where,
                inj_idx=inj_idx,
                inj_bit=inj_bit,
                attn_mask=None,
                dropout_p=self.p_drop,
                is_causal=self.causal
            )
        else:
            out, scores, p = sdp_attention_injected(
                q, k, v,
                inj_where="none",
                inj_idx=(0,0,0,0),
                inj_bit=0,
                attn_mask=None,
                dropout_p=self.p_drop,
                is_causal=self.causal
            )

        # merge the output of multihead
        out = out.permute(0,2,1,3).contiguous().view(B, T, D)
        return out, scores, p, (q, k, v)

# single layer transformer
class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 64, n_heads: int = 4, head_dim: int = 16, block_size: int = 32):
        super().__init__()
        assert n_embd == n_heads * head_dim, "For this minimal demo, n_embd must equal n_heads * head_dim."
        self.block_size = block_size
        # mapping into a lookup table
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.attn = FaultySelfAttention(n_heads=n_heads, head_dim=head_dim, causal=True, p_drop=0.0)
        self.ln = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, *, inject=False, inj_where="none", inj_idx=(0,0,0,0), inj_bit=0):
        x = self.embed(idx)
        
        ####################################
        ####################################
        # if not self.training and not inj_where == "none":
        #     inj_where = "flag"
        # keep causal true during training, false during evaluation
        # self.attn.causal = self.training
        ####################################
        ####################################
            
        attn_out, scores, p, (q, k, v) = self.attn(x, inject=inject, inj_where=inj_where, inj_idx=inj_idx, inj_bit=inj_bit)
        x = self.ln(x + attn_out)
        logits = self.proj(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

        return logits, loss, scores, p, (q, k, v)

# -----------------------------
# Build a tiny corpus
# -----------------------------

corpus = (
    "In the beginning was the word, and the word carried thought.\n"
    "Language became the bridge between silence and meaning,\n"
    "between chaos and the first spark of understanding.\n"
    "To seek knowledge is to chase the faint glow of reason through the fog of uncertainty.\n"
    "Every idea is a small rebellion against ignorance, a fragile candle in the storm.\n"
    "Machines now read our words, imitate our voices, and dream in numbers.\n"
    "Yet behind every algorithm lies a question older than code itself:\n"
    "what does it mean to understand?\n"
    "The universe writes in mathematics, and humans translate it into stories.\n"
    "The sea whispers equations to the shore, and the stars blink in binary rhythm.\n"
    "If thought is computation, perhaps the soul is recursion without termination.\n"
    "In the quiet of midnight, the mind becomes a theater of infinite possibilities,\n"
    "where memory and imagination exchange masks.\n"
    "Somewhere between logic and poetry, truth hides, waiting to be redefined.\n"
    "And when the sun rises, all the unsolved problems still wait,\n"
    "patient, elegant, indifferent to our dreams.\n"
    "Thus continues the long conversation between human and machine,\n"
    "between the known and the unknowable.\n"
)


block_size = 32
dataset = TinyCharDataset(corpus, block_size=block_size, split=0.9)
vocab_size = dataset.vocab_size

# -----------------------------
# Train briefly
# -----------------------------

rand_seed = 0
torch.manual_seed(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)

device = device_auto()
model = TinyLM(vocab_size=vocab_size, n_embd=64, n_heads=4, head_dim=16, block_size=block_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

def run_step(split: str, inject: bool = False, inj_where="none", inj_idx=(0,0,0,0), inj_bit=0):
    is_train = (split == "train")
    model.train(is_train)

    x, y = dataset.get_batch(split, batch_size=16)
    x, y = x.to(device), y.to(device)

    if is_train:
        opt.zero_grad(set_to_none=True)
        _, loss, scores, p, qkv = model(
            x, y, inject=inject, inj_where=inj_where, inj_idx=inj_idx, inj_bit=inj_bit
        )
        loss.backward()
        opt.step()
    else:
        with torch.no_grad():
            _, loss, scores, p, qkv = model(
                x, y, inject=inject, inj_where=inj_where, inj_idx=inj_idx, inj_bit=inj_bit
            )

    q, k, v = (t.detach() for t in qkv)

    return (
        (loss.item() if loss is not None else None),
        scores.detach(),
        p.detach(),
        (q, k, v),
    )


# quick warmup
for step in range(400):
    loss, _, _, _ = run_step("train")
    if (step+1) % 10 == 0:
        print(f"step {step+1:03d} | train loss {loss:.4f}")

# -----------------------------
# Evaluate baseline vs injected
# -----------------------------

with torch.no_grad():
    model.eval()
    x, y = dataset.get_batch("val", batch_size=8)
    x, y = x.to(device), y.to(device)
    logits, base_loss, base_scores, base_p, base_qkv= model(x, y, inject=False)

print(f"\nBaseline val loss: {base_loss.item():.6f}")

# Choose an injection site that's in-bounds for (B,H,T,T).
# We'll inject a bit flip into the attention *scores* tensor at (b=0,h=0,i=2,j=7) if possible.
b, h, i, j = 0, 0, min(2, block_size-1), min(7, block_size-1)
inj_idx = (b, h, i, j)

with torch.no_grad():
    model.eval()
    logits_f, loss_f, scores_f, p_f, qkv_f = model(x, y, inject=True, inj_where="scores", inj_idx=inj_idx, inj_bit=25)

print(f"Injected val loss: {loss_f.item():.6f}")
print(f"Δ loss (inj - base): {loss_f.item() - base_loss.item():.6f}")

# -----------------------------
# Compute and check bounds on baseline scores/weights
# -----------------------------
Dh = 16
bounds = compute_attention_bounds(base_scores, base_p, d=Dh)

#######################################
#######################################
from utils.bound_fixing import hist_tensor_diff
# hist_tensor_diff(bounds)
#######################################
#######################################

eps = 1e-6
lhs_ok = (bounds["lower1"] <= bounds["middle"] + eps).all().item()
mid_ok = (bounds["middle"] <= bounds["epsilon"] + eps).all().item()
rhs_ok = (bounds["epsilon"] <= bounds["upper"] + eps).all().item()

print("\nBound checks on baseline attention:")
print(f"lower1 ≤ middle: {bool(lhs_ok)}")
print(f"middle ≤ epsilon: {bool(mid_ok)}")
print(f"epsilon ≤ upper: {bool(rhs_ok)}")

row = {k: (v[b, h, i].item() if v.ndim == 3 else float(v[b, h, i])) for k, v in bounds.items() if k in
       ("gamma", "w_star", "epsilon", "lower1", "middle", "upper1", "upper2", "upper")}

print(f"\n[b={b}, h={h}, i={i}] bounds snapshot:")
for k in ("gamma", "w_star", "epsilon", "lower1", "middle", "upper1", "upper2", "upper"):
    debug(f"{k:>8s}: {row[k]:.9f}")

# -----------------------------
# Compare epsilon via injected attention (using identity q=k=v=x)
# -----------------------------
value_injected = compute_injected_attention(base_scores, attn_inj=(p_f @ qkv_f[2]), q=base_qkv[0], d=Dh)
debug(f"\nInjected-epsilon baseline snapshot (b={b},h={h},i={i}): {value_injected[b,h,i].item():.9f}")

# Show "abs err" between baseline and injected forward on the per-head hidden output
# for the chosen site, to echo the user's printout idea (but here we compare whole tensors).
abs_err = (((p_f @ qkv_f[2]) - (base_p @ base_qkv[2])).abs().max() ).item()
print(f"Max abs err between injected-attn-out and baseline-attn-out: {abs_err:.6f}")

print("\nDone. This script trained a tiny LM briefly, evaluated baseline vs. injected attention, and checked the bounds.")