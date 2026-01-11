from __future__ import annotations

import torch
import torch.nn as nn
import pandas as pd

from transformers import AutoModelForCausalLM

from src.benchmarker import KVCacheBenchmarker

from src.sparsity.pruning import (
    prune_layerwise_l1_unstructured,
    prune_global_l1_unstructured,
    prune_layerwise_random_unstructured,
    prune_layerwise_structured_ln,
    prune_nm_sparsity,
)

from src.sparsity.metrics import (
    linear_unstructured_sparsity,
    linear_structured_sparsity_rows_cols,
    nm_sparsity_violation_count,
)


def make_sparse_model_from_pretrained(
    model_name: str,
    device: str,
    dtype,
    strategy: str,
    amount: float = 0.3,
    structured_dim: int = 0,
    structured_n: int = 2,
    nm_N: int = 2,
    nm_M: int = 4,
    nm_dim: int = 1,
):
    """
    strategy options:
      - "layerwise_l1"        (1) your original
      - "global_l1"           (2)
      - "random"              (3)
      - "structured_ln"       (4)
      - "nm"                  (5) N:M (e.g., 2:4)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if device == "cuda" else None,
    ).to(device)
    model.eval()

    if strategy == "layerwise_l1":
        prune_layerwise_l1_unstructured(model, amount=amount)

    elif strategy == "global_l1":
        prune_global_l1_unstructured(model, amount=amount)

    elif strategy == "random":
        prune_layerwise_random_unstructured(model, amount=amount)

    elif strategy == "structured_ln":
        prune_layerwise_structured_ln(model, amount=amount, n=structured_n, dim=structured_dim)

    elif strategy == "nm":
        # for N:M, we don't use 'amount' (pattern-based)
        prune_nm_sparsity(model, N=nm_N, M=nm_M, dim=nm_dim)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Report
    unstruct = linear_unstructured_sparsity(model)
    row_frac, col_frac = linear_structured_sparsity_rows_cols(model)

    print(f"[{strategy}] unstructured sparsity (Linear weights): {unstruct*100:.2f}%")
    print(f"[{strategy}] structured zero rows: {row_frac*100:.2f}% | zero cols: {col_frac*100:.2f}%")

    if strategy == "nm":
        # quick compliance check on a few linear layers
        violations = 0
        checked = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                violations += nm_sparsity_violation_count(m.weight.data, N=nm_N, M=nm_M, dim=nm_dim)
                checked += 1
                if checked >= 5:
                    break
        print(f"[{strategy}] N:M violations in first {checked} Linear layers: {violations}")

    return model


def run_all_sparsity_strategies(
    model_name,
    tokenizer,
    device,
    dtype,
    short_prompts,
    long_prompts,
    strategies,
    methods=("no_cache", "full_cache", "sliding_window"),
    window_size=256,
    max_new_tokens=64,
):
    rows = []

    for s in strategies:
        strategy = s["strategy"]
        print("\n" + "="*80)
        print(f"Running strategy: {strategy} | config: {s}")

        # build model
        model = make_sparse_model_from_pretrained(
            model_name=model_name,
            device=device,
            dtype=dtype,
            **s
        )

        bench = KVCacheBenchmarker(model, tokenizer, device=device)

        for prompt_group, prompts in [("short", short_prompts), ("long", long_prompts)]:
            for method in methods:
                if method == "no_cache" and prompt_group == "long":
                    continue  # skip very slow case
                stats = bench.benchmark_method(
                    prompts,
                    method=method,
                    max_new_tokens=max_new_tokens,
                    window_size=window_size,
                )
                stats["prompt_group"] = prompt_group
                stats["sparsity_strategy"] = strategy
                stats["linear_unstructured_sparsity"] = linear_unstructured_sparsity(model)

                row_frac, col_frac = linear_structured_sparsity_rows_cols(model)
                stats["linear_zero_row_frac"] = row_frac
                stats["linear_zero_col_frac"] = col_frac

                rows.append(stats)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)