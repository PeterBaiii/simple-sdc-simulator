import torch

def top2_distinct_lastdim(x: torch.Tensor, eps: float = 0.0):
    v, idx = torch.sort(x, dim=-1, descending=True)

    keep = torch.ones_like(v, dtype=torch.bool)
    if v.size(-1) > 1:
        if eps == 0.0:
            keep[..., 1:] = v[..., 1:] != v[..., :-1]
        else:
            keep[..., 1:] = (v[..., :-1] - v[..., 1:]) > eps

    c = keep.cumsum(dim=-1)
    first_mask  = keep & (c == 1)
    second_mask = keep & (c == 2)

    arg1 = first_mask.to(torch.int64).argmax(dim=-1)
    has2 = second_mask.any(dim=-1)
    arg2_tmp = second_mask.to(torch.int64).argmax(dim=-1)
    arg2 = torch.where(has2, arg2_tmp, arg1)

    top1 = v.gather(-1, arg1.unsqueeze(-1)).squeeze(-1)
    top2 = v.gather(-1, arg2.unsqueeze(-1)).squeeze(-1)

    # if return_indices:
    #     idx1 = idx.gather(-1, arg1.unsqueeze(-1)).squeeze(-1)
    #     idx2 = idx.gather(-1, arg2.unsqueeze(-1)).squeeze(-1)
    #     return top1, top2, idx1, idx2

    return top1, top2

# if __name__ == "__main__":
#     x = torch.tensor([[1., 5., 5., 3.],
#                   [7., 7., 7., 7.],
#                   [2., 2., 1., 1.]])

#     top1, top2 = top2_distinct_lastdim(x)
    
#     print(top1, top2)