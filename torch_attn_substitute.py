import torch, math
import torch.nn.functional as F
from contextlib import contextmanager

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
    
@contextmanager
def inject_into_sdp(where="scores", idx=(0,0,0,0), bit=10):
    orig = F.scaled_dot_product_attention

    def faulty_sdp(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        d = q.size(-1)
        s = (q @ k.transpose(-2, -1)) * (scale if scale is not None else (1.0/math.sqrt(d)))
        if attn_mask is not None: s = s + attn_mask
        if is_causal:
            i, j = s.size(-2), s.size(-1)
            causal = torch.ones(i, j, dtype=torch.bool, device=s.device).triu(1)
            s = s.masked_fill(causal, float("-inf"))
        if where == "scores": bitflip_(s, idx, bit)
        p = torch.softmax(s, dim=-1)
        if where == "weights": bitflip_(p, idx, bit)
        if dropout_p and dropout_p > 0:
            p = F.dropout(p, p=dropout_p, training=True)
        out = p @ v
        if where == "out": bitflip_(out, idx, bit)
        return out

    # 为保证一定走 Python 路径，可关闭 Flash/高效内核（可选）
    from torch.nn.attention import sdpa_kernel
    with sdpa_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        F.scaled_dot_product_attention = faulty_sdp
        try:
            yield
        finally:
            F.scaled_dot_product_attention = orig

# 用法：对任意现有模型生效
mha = torch.nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True, device="cuda")
x = torch.randn(2, 8, 64, device="cuda")

with inject_into_sdp(where="scores", idx=(0,0,0,3), bit=25):
    y_fault, _ = mha(x, x, x)

y_base, _ = mha(x, x, x)
print("diff:", (y_fault - y_base).abs().max().item())
