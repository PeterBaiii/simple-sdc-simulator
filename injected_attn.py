import math, torch
import re
from scipy.special import lambertw
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def bitflip_(t: torch.Tensor, idx: tuple, bit: int):
    """
        比特翻转
        idx索引张量中要翻转的分量
        bit给出对应分量要翻转的比特位
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
                            inj_where="scores",
                            inj_idx=(0,0,0,0),
                            inj_bit=10,
                            attn_mask=None, dropout_p=0.0, is_causal=False):
    """
        带故障注入的注意力机制计算
        inj_where指示了注入故障的位置
        q, k, v是
    """
    
    if inj_where == "q": bitflip_(q, inj_idx, inj_bit)
    if inj_where == "k": bitflip_(k, inj_idx, inj_bit)
    if inj_where == "v": bitflip_(v, inj_idx, inj_bit)

    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    
    if attn_mask is not None:
        scores = scores + attn_mask
    
    if is_causal:
        i = scores.size(-2); j = scores.size(-1)
        causal = torch.ones(i, j, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))

    if inj_where == "scores": bitflip_(scores, inj_idx, inj_bit)

    p = torch.softmax(scores, dim=-1)
    
    if inj_where == "weights": bitflip_(p, inj_idx, inj_bit)

    if dropout_p and dropout_p > 0:
        p = F.dropout(p, p=dropout_p, training=True)

    out = p @ v
    if inj_where == "out": bitflip_(out, inj_idx, inj_bit)
    
    return out

sigma = 5.0 / 3.0
B, H, L, Dh = 2, 4, 8, 16 # batch大小, attn头数
x = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.float32) * sigma
q, k, v = (x.clone(),) * 3
# q = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.float32)
# k = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.float32)
# v = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.float32)

# 随机翻转QK矩阵的一个位置
y_fault = sdp_attention_injected(q.clone(), k.clone(), v.clone(),
                                    inj_where="scores", inj_idx=(0,0,2,7), inj_bit=25)

# baseline
with torch.no_grad():
    scores = (q @ k.transpose(-2,-1)) / (Dh**0.5)
    p = torch.softmax(scores, dim=-1)
    y_base = p @ v

print("abs err:", (y_fault - y_base).abs().max().item())    

@torch.no_grad()
def compute_attention_bounds(scores: torch.Tensor, p: torch.Tensor, d: int):
    """
        scores和p分别为B * H * L * L的中间结果张量, d是隐藏维度
    """
    B, H, L, _ = scores.shape
    device = scores.device
    dtype = scores.dtype
    sqrt_d = math.sqrt(d)
    n = L

    # 找到最后一个维度top2的值
    top2_vals, top2_idx = torch.topk(scores, k=2, dim=-1)
    a_star = top2_vals[..., 0]
    second = top2_vals[..., 1]
    j_star_idx = top2_idx[..., 0]

    # 根据j*索引得到softmax的对应值
    w_star = p.gather(-1, j_star_idx.unsqueeze(-1)).squeeze(-1)

    # 待求的margin
    gamma = a_star - second

    # 利用广播机制得到x_i Attn(x_i)
    Ea = (p * scores).sum(dim=-1)
    epsilon = sqrt_d * (a_star - Ea)

    # 较小的下界
    lower1 = sqrt_d * gamma / (1.0 + torch.exp(gamma))

    # 中间下界
    middle = sqrt_d * gamma * (1.0 - w_star)

    # 均值上界
    upper1 = sqrt_d * (a_star - scores.mean(dim=-1))

    # 分段上界
    lam_arg = torch.tensor((n - 1) / math.e, device=device, dtype=dtype)
    # 同样的广播机制进行扩张
    W_np = np.asarray(lambertw(lam_arg.detach().cpu().numpy(), 0).real)
    W = torch.as_tensor(W_np, device=lam_arg.device, dtype=lam_arg.dtype)
    cond = gamma >= (W + 1.0)

    # 两段计算结果
    term_case1 = sqrt_d * ((n - 1) * torch.exp(-gamma)) / (1.0 + (n - 1) * torch.exp(-gamma)) * gamma
    term_case2 = sqrt_d * W
    upper2 = torch.where(cond, term_case1, term_case2)

    upper = torch.minimum(upper1, upper2)

    return {
        "a_star": a_star, "w_star": w_star, "gamma": gamma,
        "epsilon": epsilon, "lower1": lower1, "middle": middle,
        "upper1": upper1, "upper2": upper2, "upper": upper
    }

bounds = compute_attention_bounds(scores, p, Dh)

# 首先验证不等式在数值计算上成立
eps = 1e-9
lhs_ok   = (bounds["lower1"] <= bounds["middle"] + eps).all()
mid_ok   = (bounds["middle"] <= bounds["epsilon"] + eps).all()
rhs1_ok  = (bounds["epsilon"] <= bounds["upper"] + eps).all()

print(f"check lower1 ≤ middle: {bool(lhs_ok)}")
print(f"check middle ≤ epsilon: {bool(mid_ok)}")
print(f"check epsiln ≤ upper: {bool(rhs1_ok)}")

# 提取注入位置的bound值
b, h, i = 0, 0, 2
row = {k: v[b, h, i].item() if v.ndim == 3 else v[b, h, i].item() for k, v in bounds.items() if k in
       ("gamma", "w_star", "epsilon", "lower1", "middle", "upper1", "upper2", "upper")}

print(f"[b={b}, h={h}, i={i}]")
for k in ("gamma", "w_star", "epsilon", "lower1", "middle", "upper1", "upper2", "upper"):
    print(f"{k:>8s}: {row[k]:.6f}")

@torch.no_grad()
def compute_injected_attention(scores: torch.Tensor, attn: torch.Tensor, x: torch.Tensor, d: int):
    """
        计算注入错误后的epsilon
    """
    sqrt_d = d ** 0.5
    a_star = scores.max(dim=-1).values
    Ea = (x * attn).sum(dim=-1)
    return sqrt_d * a_star - Ea

value = compute_injected_attention(scores, y_fault, x, Dh)

print(f"injected: {value[b, h, i].item():.6f}")

# 检验不同方法计算的x_i Attn(x_i)
res1 = (Dh ** 0.5) * (scores * p).sum(dim=-1)
res2 = (y_base * x).sum(dim=-1)
print("abs err: ", (res1 - res2).abs().max().item())
