from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prune_layerwise_l1_unstructured(model: nn.Module, amount: float) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")
    return model


def prune_global_l1_unstructured(model: nn.Module, amount: float) -> nn.Module:
    params_to_prune = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            params_to_prune.append((m, "weight"))

    if not params_to_prune:
        return model

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    for (m, pname) in params_to_prune:
        prune.remove(m, pname)

    return model


def prune_layerwise_random_unstructured(model: nn.Module, amount: float) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            prune.random_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")
    return model


def prune_layerwise_structured_ln(
    model: nn.Module,
    amount: float,
    n: int = 2,
    dim: int = 0
) -> nn.Module:
    """
    Structured pruning on Linear weights.
    dim=0 -> prune output neurons (rows)
    dim=1 -> prune input features (cols)
    n=2 -> L2 norm grouping
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            prune.ln_structured(m, name="weight", amount=amount, n=n, dim=dim)
            prune.remove(m, "weight")
    return model


@torch.no_grad()
def apply_nm_sparsity_to_linear_weight(w: torch.Tensor, N: int, M: int, dim: int = 1) -> torch.Tensor:
    """
    Enforce N:M sparsity by magnitude within blocks of size M along dim.
    Typical: N=2, M=4, dim=1 (within each row, group columns in chunks of 4).

    Returns a new tensor (same shape) with pruned entries set to 0.
    """
    if w.dim() != 2:
        return w

    if dim not in (0, 1):
        raise ValueError("dim must be 0 or 1")

    out = w.clone()

    if dim == 0:
        out = out.t() 
    rows, cols = out.shape
    blocks = cols // M
    if blocks == 0:
        return w

    trim_cols = blocks * M
    main = out[:, :trim_cols].view(rows, blocks, M)

    abs_main = main.abs()
    topk = torch.topk(abs_main, k=N, dim=2, largest=True, sorted=False).indices  

    mask = torch.zeros_like(main, dtype=torch.bool)
    mask.scatter_(2, topk, True)

    main_pruned = torch.where(mask, main, torch.zeros_like(main))
    out[:, :trim_cols] = main_pruned.view(rows, trim_cols)

    if dim == 0:
        out = out.t()

    return out


def prune_nm_sparsity(model: nn.Module, N: int = 2, M: int = 4, dim: int = 1) -> nn.Module:
    """
    Apply N:M sparsity to all Linear weights.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = apply_nm_sparsity_to_linear_weight(m.weight.data, N=N, M=M, dim=dim)
    return model