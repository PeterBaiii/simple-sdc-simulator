import math, torch
import re
import json
from datetime import datetime
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

@torch.no_grad()
def compute_injected_epsilon(scores: torch.Tensor, attn_out: torch.Tensor, x: torch.Tensor, d: int):
    """
        计算注入错误后的epsilon
    """
    sqrt_d = d ** 0.5
    a_star = scores.max(dim=-1).values
    Ea_injected = (x * attn_out).sum(dim=-1)
    return sqrt_d * a_star - Ea_injected

# 测试参数
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

B, H, L, Dh = 2, 4, 8, 16  # batch大小, attn头数
inj_idx = (0, 0, 2, 7)  # 固定的注入位置
b, h, i = 0, 0, 2  # 用于提取结果的位置

injection_locations = ["q", "scores", "weights", "out"]  # 注入位置
sigmas = [1.0, 5.0/3.0]  # 两种标准差
n_vectors = 16  # 测试16个向量
bit_range = range(0, 32)  # 比特位1-32

print("=" * 80)
print("Attention Fault Injection Test")
print("=" * 80)
print(f"Configuration: B={B}, H={H}, L={L}, Dh={Dh}")
print(f"Injection index: {inj_idx}, Extract position: [b={b}, h={h}, i={i}]")
print(f"Testing {len(injection_locations)} injection locations: {injection_locations}")
print(f"Testing {len(sigmas)} standard deviations: {sigmas}")
print(f"Testing {n_vectors} random vectors with {len(bit_range)} bit positions each")
print(f"Total tests: {len(injection_locations)} × {len(sigmas)} × {n_vectors} × {len(bit_range)} = {len(injection_locations) * len(sigmas) * n_vectors * len(bit_range)}")
print("=" * 80)

# 存储所有结果用于最终汇总
all_results = {}

for inj_where in injection_locations:
    print(f"\n{'#'*80}")
    print(f"# Injection Location: {inj_where.upper()}")
    print(f"{'#'*80}")
    
    all_results[inj_where] = {}
    
    for sigma in sigmas:
        print(f"\n{'='*80}")
        print(f"Testing with σ = {sigma:.4f} | Injection Location: {inj_where}")
        print(f"{'='*80}")
        
        detection_count = 0  # 成功检测的次数
        total_tests = 0  # 总测试次数
        
        # 统计每个比特位的检测情况
        bit_detection_stats = {bit: {"detected": 0, "total": 0} for bit in bit_range}
        
        for vec_idx in range(n_vectors):
            # 生成随机向量
            torch.manual_seed(vec_idx)  # 为了可重复性
            x = torch.randn(B, H, L, Dh, device=device, dtype=torch.float32) * sigma
            q, k, v = (x.clone(),) * 3
            
            # 计算baseline
            with torch.no_grad():
                scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
                p = torch.softmax(scores, dim=-1)
                y_base = p @ v
            
            # 计算bounds
            bounds = compute_attention_bounds(scores, p, Dh)
            
            print(f"\n--- Vector {vec_idx + 1}/{n_vectors} ---")
            
            # 提取该位置的bound值
            gamma = bounds["gamma"][b, h, i].item()
            w_star = bounds["w_star"][b, h, i].item()
            lower1 = bounds["lower1"][b, h, i].item()
            middle = bounds["middle"][b, h, i].item()
            upper = bounds["upper"][b, h, i].item()
            
            print(f"Bounds: gamma={gamma:.4f}, w_star={w_star:.4f}")
            print(f"        lower1={lower1:.4f}, middle={middle:.4f}, upper={upper:.4f}")
            
            # 遍历每个比特位
            detected_bits = []
            undetected_bits = []
            
            for bit in bit_range:
                # 注入故障（使用当前的注入位置）
                y_fault = sdp_attention_injected(q.clone(), k.clone(), v.clone(),
                                                inj_where=inj_where, 
                                                inj_idx=inj_idx, 
                                                inj_bit=bit)
                
                # 计算注入后的epsilon
                epsilon_injected = compute_injected_epsilon(scores, y_fault, x, Dh)
                eps_val = epsilon_injected[b, h, i].item()
                
                # 检查是否满足: lower1 <= middle <= epsilon_injected <= upper
                tolerance = 1e-6
                detected = (middle - tolerance >= eps_val or 
                           eps_val >= upper + tolerance)
                
                total_tests += 1
                bit_detection_stats[bit]["total"] += 1
                
                if detected:
                    detection_count += 1
                    bit_detection_stats[bit]["detected"] += 1
                    detected_bits.append(bit)
                else:
                    undetected_bits.append(bit)
            
            print(f"Detected bits ({len(detected_bits)}/{len(bit_range)}): {detected_bits[:10]}{'...' if len(detected_bits) > 10 else ''}")
            if undetected_bits:
                print(f"Undetected bits ({len(undetected_bits)}): {undetected_bits[:10]}{'...' if len(undetected_bits) > 10 else ''}")
        
        # 打印统计结果
        print(f"\n{'='*80}")
        print(f"Summary for σ = {sigma:.4f} | Location: {inj_where}")
        print(f"{'='*80}")
        print(f"Total detection rate: {detection_count}/{total_tests} ({100*detection_count/total_tests:.2f}%)")
        
        # 存储结果
        all_results[inj_where][sigma] = {
            "detection_count": detection_count,
            "total_tests": total_tests,
            "bit_stats": bit_detection_stats
        }
        
        # 按比特位统计
        print(f"\nDetection rate by bit position:")
        for bit in bit_range:
            stats = bit_detection_stats[bit]
            rate = 100 * stats["detected"] / stats["total"] if stats["total"] > 0 else 0
            status = "✓" if rate == 100 else "✗" if rate == 0 else "~"
            print(f"  {status} Bit {bit:2d}: {stats['detected']:2d}/{stats['total']:2d} ({rate:5.1f}%)")

# 最终对比总结
print(f"\n{'='*80}")
print("Final Comparison Summary")
print(f"{'='*80}")
print(f"\n{'Location':<12} {'σ':<8} {'Detection Rate':<20} {'Percentage':<12}")
print("-" * 80)

for inj_where in injection_locations:
    for sigma in sigmas:
        res = all_results[inj_where][sigma]
        rate = 100 * res["detection_count"] / res["total_tests"]
        print(f"{inj_where:<12} {sigma:<8.4f} {res['detection_count']:>3d}/{res['total_tests']:<3d} ({rate:>6.2f}%)".ljust(80))

# 按注入位置汇总
print(f"\n{'='*80}")
print("Summary by Injection Location (across both σ values)")
print(f"{'='*80}")
for inj_where in injection_locations:
    total_detected = sum(all_results[inj_where][sigma]["detection_count"] for sigma in sigmas)
    total_tests = sum(all_results[inj_where][sigma]["total_tests"] for sigma in sigmas)
    rate = 100 * total_detected / total_tests if total_tests > 0 else 0
    print(f"{inj_where:<12}: {total_detected:>4d}/{total_tests:<4d} ({rate:6.2f}%)")

# 按标准差汇总
print(f"\n{'='*80}")
print("Summary by Standard Deviation (across all injection locations)")
print(f"{'='*80}")
for sigma in sigmas:
    total_detected = sum(all_results[inj_where][sigma]["detection_count"] for inj_where in injection_locations)
    total_tests = sum(all_results[inj_where][sigma]["total_tests"] for inj_where in injection_locations)
    rate = 100 * total_detected / total_tests if total_tests > 0 else 0
    print(f"σ = {sigma:.4f}: {total_detected:>4d}/{total_tests:<4d} ({rate:6.2f}%)")

print(f"\n{'='*80}")
print("Test completed!")
print(f"{'='*80}")

# 保存结果到JSON文件
print("\nSaving results to JSON...")

# 准备要保存的数据结构
output_data = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "configuration": {
            "B": B,
            "H": H,
            "L": L,
            "Dh": Dh,
            "injection_index": inj_idx,
            "extract_position": [b, h, i],
            "injection_locations": injection_locations,
            "sigmas": sigmas,
            "n_vectors": n_vectors,
            "bit_range": [bit_range.start, bit_range.stop],
            "total_tests": len(injection_locations) * len(sigmas) * n_vectors * len(bit_range)
        }
    },
    "detailed_results": all_results,
    "summary": {
        "by_injection_location": {},
        "by_sigma": {},
        "overall": {}
    }
}

# 按注入位置汇总
for inj_where in injection_locations:
    total_detected = sum(all_results[inj_where][sigma]["detection_count"] for sigma in sigmas)
    total_tests = sum(all_results[inj_where][sigma]["total_tests"] for sigma in sigmas)
    rate = 100 * total_detected / total_tests if total_tests > 0 else 0
    output_data["summary"]["by_injection_location"][inj_where] = {
        "detected": total_detected,
        "total": total_tests,
        "rate_percent": round(rate, 2)
    }

# 按标准差汇总
for sigma in sigmas:
    total_detected = sum(all_results[inj_where][sigma]["detection_count"] for inj_where in injection_locations)
    total_tests = sum(all_results[inj_where][sigma]["total_tests"] for inj_where in injection_locations)
    rate = 100 * total_detected / total_tests if total_tests > 0 else 0
    output_data["summary"]["by_sigma"][str(sigma)] = {
        "detected": total_detected,
        "total": total_tests,
        "rate_percent": round(rate, 2)
    }

# 总体统计
overall_detected = sum(
    all_results[inj_where][sigma]["detection_count"] 
    for inj_where in injection_locations 
    for sigma in sigmas
)
overall_total = sum(
    all_results[inj_where][sigma]["total_tests"] 
    for inj_where in injection_locations 
    for sigma in sigmas
)
overall_rate = 100 * overall_detected / overall_total if overall_total > 0 else 0
output_data["summary"]["overall"] = {
    "detected": overall_detected,
    "total": overall_total,
    "rate_percent": round(overall_rate, 2)
}

# 保存到JSON文件
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
json_filename = f"attention_fault_test_results_{timestamp_str}.json"
json_filepath = f"./results/{json_filename}"

with open(json_filepath, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✓ Results saved to: {json_filename}")
print(f"  Total tests: {overall_total}")
print(f"  Overall detection rate: {overall_detected}/{overall_total} ({overall_rate:.2f}%)")

# 也创建一个简化版本（不包含每个比特位的详细统计）
simplified_data = {
    "metadata": output_data["metadata"],
    "summary": output_data["summary"],
    "results_by_location_and_sigma": {}
}

for inj_where in injection_locations:
    simplified_data["results_by_location_and_sigma"][inj_where] = {}
    for sigma in sigmas:
        res = all_results[inj_where][sigma]
        simplified_data["results_by_location_and_sigma"][inj_where][str(sigma)] = {
            "detection_count": res["detection_count"],
            "total_tests": res["total_tests"],
            "detection_rate_percent": round(100 * res["detection_count"] / res["total_tests"], 2)
        }

simplified_filename = f"attention_fault_test_summary_{timestamp_str}.json"
simplified_filepath = f"./results/{simplified_filename}"

with open(simplified_filepath, 'w') as f:
    json.dump(simplified_data, f, indent=2)

print(f"✓ Simplified summary saved to: {simplified_filename}")
print(f"\n{'='*80}")