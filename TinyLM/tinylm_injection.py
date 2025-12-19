import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
from scipy.special import lambertw

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def bitflip_(t: torch.Tensor, idx: tuple, bit: int):
    """
    In-place bit flip for float tensors (fp16/bf16/fp32).
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
    """
    if inj_where == "q": bitflip_(q, inj_idx, inj_bit)
    if inj_where == "k": bitflip_(k, inj_idx, inj_bit)
    if inj_where == "v": bitflip_(v, inj_idx, inj_bit)

    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

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
    
    return out, scores, p

@torch.no_grad()
def compute_attention_bounds(scores: torch.Tensor, p: torch.Tensor, d: int):
    """
    Compute attention bounds.
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
    a_star = top2_vals[..., 0]
    second = top2_vals[..., 1]
    w_star = p.max(dim=-1).values
    gamma = a_star - second
    
    Ea = (p * scores).sum(dim=-1)
    epsilon = sqrt_d * (a_star - Ea)
    
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
    Compute epsilon after injection: sqrt(d) * max_j a_ij - <Attn_inj(x_i), x_i>
    """
    sqrt_d = d ** 0.5
    a_star = scores.max(dim=-1).values
    Ea = (attn_inj * q).sum(dim=-1)
    return sqrt_d * a_star - Ea

# Dataset class
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
        max_start = len(data) - self.block_size - 1

        if max_start < 0:
            x = data[:-1]
            y = data[1:]
            x = x.unsqueeze(0).repeat(batch_size, 1)
            y = y.unsqueeze(0).repeat(batch_size, 1)
        else:
            ix = torch.randint(0, max_start + 1, (batch_size,))
            x = torch.stack([data[i : i+self.block_size] for i in ix])
            y = torch.stack([data[i+1 : i+self.block_size+1] for i in ix])
            
        return x, y

# Attention module
class FaultySelfAttention(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, causal: bool = True, p_drop: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.causal = causal
        self.p_drop = p_drop

        D = n_heads * head_dim
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_kv = nn.Linear(D, D, bias=False)

    def forward(self, x, *, inject: bool = False,
                inj_where="none", inj_idx=(0,0,0,0), inj_bit=0):
        
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.W_q(x)
        kv = self.W_kv(x)
        k, v = kv, kv

        q = q.view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()

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

        out = out.permute(0,2,1,3).contiguous().view(B, T, D)
        return out, scores, p, (q, k, v)

# Language model
class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 64, n_heads: int = 4, head_dim: int = 16, block_size: int = 32):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.attn = FaultySelfAttention(n_heads=n_heads, head_dim=head_dim, causal=True, p_drop=0.0)
        self.ln = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, *, inject=False, inj_where="none", inj_idx=(0,0,0,0), inj_bit=0):
        x = self.embed(idx)
        attn_out, scores, p, (q, k, v) = self.attn(x, inject=inject, inj_where=inj_where, inj_idx=inj_idx, inj_bit=inj_bit)
        x = self.ln(x + attn_out)
        logits = self.proj(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

        return logits, loss, scores, p, (q, k, v)

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
# with open("synthetic_attention_corpus.txt", "r", encoding="utf-8") as f:
#     corpus = f.read()

# MAX_CHARS = 500_000
# corpus = corpus[:MAX_CHARS]
# ===========================
# Main Testing Script
# ===========================

print("="*80)
print("Tiny Language Model Fault Injection Test")
print("="*80)

# Setup
rand_seed = 42
torch.manual_seed(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)

block_size = 32
dataset = TinyCharDataset(corpus, block_size=block_size, split=0.9)
vocab_size = dataset.vocab_size

device = device_auto()
print(f"Using device: {device}")

# Initialize and train model
model = TinyLM(vocab_size=vocab_size, n_embd=64, n_heads=4, head_dim=16, block_size=block_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"\nTraining model on tiny corpus...")
print(f"Vocabulary size: {vocab_size}")
print(f"Block size: {block_size}")

# Training
for step in range(1000):
    model.train()
    x, y = dataset.get_batch("train", batch_size=16)
    x, y = x.to(device), y.to(device)
    
    opt.zero_grad(set_to_none=True)
    _, loss, _, _, _ = model(x, y, inject=False)
    loss.backward()
    opt.step()
    
    if (step+1) % 100 == 0:
        print(f"Step {step+1:03d} | train loss {loss.item():.4f}")

print("\nTraining complete. Starting fault injection tests...")

# Test configuration
B_test, H_test = 8, 4  # batch size, number of heads
Dh = 16  # head dimension
injection_locations = ["q", "scores", "weights", "out"]  # K=V, so no separate k, v
bit_range = range(0, 32)
n_test_batches = 16  # test on 16 different batches

# Injection position (will test on position (0, 0, 2, 7) if valid)
b, h = 0, 0
i_pos, j_pos = min(2, block_size-1), min(7, block_size-1)
inj_idx = (b, h, i_pos, j_pos)

print(f"\nTest Configuration:")
print(f"  Injection locations: {injection_locations}")
print(f"  Bit range: {bit_range.start}-{bit_range.stop-1}")
print(f"  Test batches: {n_test_batches}")
print(f"  Injection index: {inj_idx}")
print(f"  Extract position: [b={b}, h={h}, i={i_pos}]")
print(f"  Total tests: {len(injection_locations)} × {n_test_batches} × {len(bit_range)} = {len(injection_locations) * n_test_batches * len(bit_range)}")

# Storage for results
all_results = {}

# Run tests
model.eval()

for inj_where in injection_locations:
    print(f"\n{'#'*80}")
    print(f"# Testing Injection Location: {inj_where.upper()}")
    print(f"{'#'*80}")
    
    all_results[inj_where] = {
        "detection_count": 0,
        "total_tests": 0,
        "bit_stats": {str(bit): {"detected": 0, "total": 0} for bit in bit_range},
        "batch_results": []
    }
    
    for batch_idx in range(n_test_batches):
        print(f"\n--- Batch {batch_idx + 1}/{n_test_batches} ---")
        
        # Get a test batch
        torch.manual_seed(rand_seed + batch_idx)
        x, y = dataset.get_batch("val", batch_size=B_test)
        x, y = x.to(device), y.to(device)
        
        # Baseline
        with torch.no_grad():
            _, base_loss, base_scores, base_p, base_qkv = model(x, y, inject=False)
        
        # Compute bounds
        bounds = compute_attention_bounds(base_scores, base_p, Dh)
        
        # Extract bounds at position (b, h, i_pos)
        upper = bounds["upper"][b, h, i_pos].item()
        middle = bounds["middle"][b, h, i_pos].item()
        epsilon_base = bounds["epsilon"][b, h, i_pos].item()
        
        print(f"Baseline: upper={upper:.6f}, middle={middle:.6f}, epsilon={epsilon_base:.6f}")
        
        batch_result = {
            "batch_idx": batch_idx,
            "baseline_loss": base_loss.item(),
            "bounds": {
                "upper": upper,
                "middle": middle,
                "epsilon": epsilon_base
            },
            "detected_bits": [],
            "undetected_bits": []
        }
        
        # Test each bit
        for bit in bit_range:
            with torch.no_grad():
                _, loss_f, scores_f, p_f, qkv_f = model(
                    x, y, inject=True, inj_where=inj_where, inj_idx=inj_idx, inj_bit=bit
                )
            
            # Compute injected epsilon
            epsilon_injected = compute_injected_attention(
                base_scores, 
                attn_inj=(p_f @ qkv_f[2]), 
                q=base_qkv[0], 
                d=Dh
            )
            eps_val = epsilon_injected[b, h, i_pos].item()
            
            # Detection condition: upper <= epsilon OR epsilon <= middle
            tolerance = 1e-6
            detected = (upper + tolerance <= eps_val) or (eps_val <= middle - tolerance)
            
            all_results[inj_where]["total_tests"] += 1
            all_results[inj_where]["bit_stats"][str(bit)]["total"] += 1
            
            if detected:
                all_results[inj_where]["detection_count"] += 1
                all_results[inj_where]["bit_stats"][str(bit)]["detected"] += 1
                batch_result["detected_bits"].append(bit)
            else:
                batch_result["undetected_bits"].append(bit)
        
        all_results[inj_where]["batch_results"].append(batch_result)
        
        n_detected = len(batch_result["detected_bits"])
        print(f"Detected: {n_detected}/{len(bit_range)} bits")
        if n_detected > 0:
            print(f"  Detected bits: {batch_result['detected_bits'][:10]}{'...' if n_detected > 10 else ''}")
    
    # Summary for this location
    res = all_results[inj_where]
    rate = 100 * res["detection_count"] / res["total_tests"] if res["total_tests"] > 0 else 0
    print(f"\n{'='*80}")
    print(f"Summary for location: {inj_where}")
    print(f"{'='*80}")
    print(f"Total detection rate: {res['detection_count']}/{res['total_tests']} ({rate:.2f}%)")
    
    # Bit-wise statistics
    print(f"\nDetection rate by bit position:")
    for bit in bit_range:
        stats = res["bit_stats"][str(bit)]
        bit_rate = 100 * stats["detected"] / stats["total"] if stats["total"] > 0 else 0
        status = "✓" if bit_rate == 100 else "✗" if bit_rate == 0 else "~"
        print(f"  {status} Bit {bit:2d}: {stats['detected']:2d}/{stats['total']:2d} ({bit_rate:5.1f}%)")

# Final summary
print(f"\n{'='*80}")
print("Final Summary")
print(f"{'='*80}")
print(f"\n{'Location':<12} {'Detection Rate':<20} {'Percentage':<12}")
print("-" * 50)

for inj_where in injection_locations:
    res = all_results[inj_where]
    rate = 100 * res["detection_count"] / res["total_tests"]
    print(f"{inj_where:<12} {res['detection_count']:>3d}/{res['total_tests']:<3d} ({rate:>6.2f}%)".ljust(50))

# Save to JSON
print(f"\n{'='*80}")
print("Saving results to JSON...")
print(f"{'='*80}")

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Prepare output data
output_data = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "model_config": {
            "vocab_size": vocab_size,
            "n_embd": 64,
            "n_heads": 4,
            "head_dim": Dh,
            "block_size": block_size
        },
        "test_config": {
            "injection_index": inj_idx,
            "extract_position": [b, h, i_pos],
            "injection_locations": injection_locations,
            "test_batches": n_test_batches,
            "bit_range": [bit_range.start, bit_range.stop],
            "total_tests": sum(all_results[loc]["total_tests"] for loc in injection_locations),
            "detection_condition": "upper <= epsilon OR epsilon <= middle"
        },
        "training_info": {
            "training_steps": 400,
            "learning_rate": 1e-4,
            "random_seed": rand_seed
        }
    },
    "detailed_results": all_results,
    "summary": {
        "by_injection_location": {},
        "overall": {}
    }
}

# Summary by location
for inj_where in injection_locations:
    res = all_results[inj_where]
    rate = 100 * res["detection_count"] / res["total_tests"]
    output_data["summary"]["by_injection_location"][inj_where] = {
        "detected": res["detection_count"],
        "total": res["total_tests"],
        "rate_percent": round(rate, 2)
    }

# Overall summary
overall_detected = sum(all_results[loc]["detection_count"] for loc in injection_locations)
overall_total = sum(all_results[loc]["total_tests"] for loc in injection_locations)
overall_rate = 100 * overall_detected / overall_total if overall_total > 0 else 0
output_data["summary"]["overall"] = {
    "detected": overall_detected,
    "total": overall_total,
    "rate_percent": round(overall_rate, 2)
}

# Save full results
json_filename = f"tinylm_fault_test_results_{timestamp_str}.json"
with open(json_filename, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✓ Full results saved to: {json_filename}")

# Save simplified summary
simplified_data = {
    "metadata": output_data["metadata"],
    "summary": output_data["summary"]
}

simplified_filename = f"tinylm_fault_test_summary_{timestamp_str}.json"
with open(simplified_filename, 'w') as f:
    json.dump(simplified_data, f, indent=2)

print(f"✓ Simplified summary saved to: {simplified_filename}")
print(f"\nOverall detection rate: {overall_detected}/{overall_total} ({overall_rate:.2f}%)")

print(f"\n{'='*80}")
print("Test completed!")
print(f"{'='*80}")