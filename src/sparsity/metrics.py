from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def linear_unstructured_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0


@torch.no_grad()
def linear_structured_sparsity_rows_cols(model: nn.Module) -> Tuple[float, float]:
    """
    Returns fraction of fully-zero rows and cols across all Linear layers.
    Useful for structured pruning analysis.
    """
    row_total = row_zero = 0
    col_total = col_zero = 0

    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight

            row_total += w.size(0)
            col_total += w.size(1)

            row_zero += (w.abs().sum(dim=1) == 0).sum().item()
            col_zero += (w.abs().sum(dim=0) == 0).sum().item()

    row_frac = row_zero / row_total if row_total > 0 else 0.0
    col_frac = col_zero / col_total if col_total > 0 else 0.0
    return row_frac, col_frac


@torch.no_grad()
def nm_sparsity_violation_count(w: torch.Tensor, N: int, M: int, dim: int = 1) -> int:
    """
    Counts how many blocks violate N:M pattern.
    For each block of size M along 'dim', number of nonzeros must be <= N.

    dim=1 means operate along columns (groups inside each row).
    """
    if dim not in (0, 1):
        raise ValueError("dim must be 0 or 1 for 2D weights")

    if w.dim() != 2:
        return 0

    if dim == 0:
        w = w.t()  

    rows, cols = w.shape
    blocks = cols // M
    if blocks == 0:
        return 0

    w_trim = w[:, :blocks * M]
    w_blocks = w_trim.view(rows, blocks, M)
    nnz_per_block = (w_blocks != 0).sum(dim=2)
    violations = (nnz_per_block > N).sum().item()
    return violations
