"""Example script demonstrating quantized KV cache."""

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarking import KVCacheBenchmarker
from src.core import Config
from src.evaluation import text_similarity


def main():
    """Run quantized KV cache benchmarking."""
    # Initialize configuration
    config = Config(model_name="gpt2", max_new_tokens=64)

    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype if config.device == "cuda" else None,
    ).to(config.device)
    model.eval()
    print("Model loaded.")

    # Create benchmarker
    bench = KVCacheBenchmarker(model, tokenizer, device=config.device)

    # Define prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    # Benchmark quantization methods
    quant_methods = ["quant_int8", "quant_int4", "quant_mixed"]
    results = []

    print("\nBenchmarking quantized caching methods...")

    # Get baseline (full cache)
    print("  Running full_cache (baseline)...")
    baseline_texts = []
    for p in prompts:
        text, _ = bench.generate_with_cache(p, max_new_tokens=config.max_new_tokens)
        baseline_texts.append(text)

    stats = bench.benchmark_method(
        prompts,
        method="full_cache",
        max_new_tokens=config.max_new_tokens,
    )
    results.append(stats)

    # Test quantized methods
    for method in quant_methods:
        print(f"  Running {method}...")

        # Benchmark
        stats = bench.benchmark_method(
            prompts,
            method=method,
            max_new_tokens=config.max_new_tokens,
        )

        # Compute accuracy
        similarities = []
        for i, p in enumerate(prompts):
            mode = method.replace("quant_", "")
            text, _, _ = bench.generate_with_quantized_kv(p, config.max_new_tokens, mode=mode)
            sim = text_similarity(baseline_texts[i], text)
            similarities.append(sim)

        stats["avg_similarity"] = sum(similarities) / len(similarities)
        results.append(stats)

    # Display results
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("QUANTIZED CACHE BENCHMARK RESULTS")
    print("=" * 80)
    print(
        df[["method", "tokens_per_sec", "est_kv_cache_mb_avg", "avg_similarity"]].to_string(
            index=False
        )
    )
    print("\n" + "=" * 80)

    # Show compression ratio
    for _, row in df.iterrows():
        if row["method"] != "full_cache":
            print(
                f"{row['method']:15s}: Cache={row['est_kv_cache_mb_avg']:.2f}MB, "
                f"Similarity={row.get('avg_similarity', float('nan')):.3f}"
            )


if __name__ == "__main__":
    main()
