from .debug import debug
import torch

def hist_tensor_diff(bounds, eps=1e-6, bins=None, max_neg_show=10):
    if bins is None:
        bins = [0, 1e-3, 1e-2, 1e-1, 1.0, float("inf")]

    pairs = [
        ("lower1", "middle"),
        ("middle", "epsilon"),
        ("epsilon", "upper"),
    ]

    nan_sets = {}
    
    for (lhs, rhs) in pairs:
        diff = (bounds[rhs] + eps - bounds[lhs]).detach().flatten()
        diff_np = diff.cpu()

        total = diff.numel()
        minv = float(diff_np.min().item())
        maxv = float(diff_np.max().item())

        debug(f"\n=== {lhs} → {rhs} ===")
        debug(f"Total elements: {total}")
        debug(f"Min diff: {minv:.6g}, Max diff: {maxv:.6g}")
        debug(f"eps tolerance: {eps}")

        n_nan = int(torch.isnan(diff_np).sum().item())
        n_inf = int(torch.isinf(diff_np).sum().item())
        if n_nan > 0 or n_inf > 0:
            debug(f"⚠️  Found {n_nan} NaN and {n_inf} Inf values in diff!")
        else:
            debug("No NaN/Inf values detected.")
            
        debug("Value range\t\tCount\tPercent")
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            mask = (diff_np >= low) & (diff_np < high)
            count = int(mask.sum().item())
            ratio = count / total * 100
            if high == float("inf"):
                debug(f"[{low:.3g}, ∞)\t{count:>8d}\t{ratio:6.2f}%")
            else:
                debug(f"[{low:.3g}, {high:.3g})\t{count:>8d}\t{ratio:6.2f}%")

        # ===== 检查负值 =====
        neg_mask = diff_np < 0
        n_neg = int(neg_mask.sum().item())
        if n_neg > 0:
            debug(f"\n⚠️  Found {n_neg} negative diffs → violate inequality!")
            neg_values = diff_np[neg_mask]
            debug(f"  Negative diff range: [{float(neg_values.min()):.6g}, {float(neg_values.max()):.6g}]")

            # 输出前若干个负值详情
            neg_indices = torch.nonzero(diff < 0, as_tuple=False)
            debug(f"  Showing up to {min(max_neg_show, n_neg)} violations (index, lhs, rhs, diff):")
            for idx in range(min(max_neg_show, n_neg)):
                tup = tuple(int(i) for i in neg_indices[idx].tolist())
                lval = bounds[lhs].flatten()[tup[-1]].item() if bounds[lhs].numel() == diff.numel() else None
                rval = bounds[rhs].flatten()[tup[-1]].item() if bounds[rhs].numel() == diff.numel() else None
                dval = neg_values[idx].item()
                debug(f"    idx={tup}, {lhs}={lval:.6g}, {rhs}={rval:.6g}, diff={dval:.6g}")
        else:
            debug("✅ No negative diffs (inequality holds everywhere).")
            
        nan_mask = torch.isnan(diff)
        not_nan_mask = ~nan_mask

        nan_indices = set(torch.nonzero(nan_mask, as_tuple=False).flatten().tolist())
        not_nan_indices = set(torch.nonzero(not_nan_mask, as_tuple=False).flatten().tolist())

        debug(f"\nNaN count: {len(nan_indices)}, Non-NaN count: {len(not_nan_indices)}")

        nan_sets[f"{lhs}->{rhs}"] = {
            "nan": nan_indices,
            "not_nan": not_nan_indices,
        }

    # ======= 结束后做集合间比较 =======
    if len(nan_sets) == 3:
        keys = list(nan_sets.keys())
        debug("\n=== NaN Index Set Comparison Across All Diffs ===")
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                nan_i = "not_nan" if k1 == keys[0] else "nan"
                nan1 = nan_sets[k1][nan_i]
                nan2 = nan_sets[k2]["nan"]

                same_nan = (nan1 == nan2)
                inter_nan = nan1 & nan2
                diff_nan = (nan1 - nan2) | (nan2 - nan1)

                debug(f"\n[{k1}] vs [{k2}]:")
                debug(f"  Same NaN indices? {same_nan}")
                debug(f"  Intersection size: {len(inter_nan)}")
                debug(f"  Symmetric difference size: {len(diff_nan)}")

                if not same_nan and len(diff_nan) <= 10:
                    debug(f"  Differing indices: {sorted(list(diff_nan))}")

        debug("\n=== Done comparing NaN sets ===\n")