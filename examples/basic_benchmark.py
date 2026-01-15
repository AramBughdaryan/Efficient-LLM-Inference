"""Example script demonstrating basic KV cache benchmarking."""

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarking import KVCacheBenchmarker
from src.core import Config


def main():
    """Run basic KV cache benchmarking comparison."""
    # Initialize configuration
    config = Config(model_name="gpt2", max_new_tokens=64)

    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}")
    print(f"Dtype: {config.dtype}")

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
    short_prompts = [
        "Explain what reinforcement learning is in one paragraph.",
        "Summarize the plot of Romeo and Juliet in three sentences.",
        "Describe the benefits of using GPUs for deep learning.",
        "Explain the importance of efficient inference in large language models.",
    ]

    # Benchmark different methods
    methods = ["no_cache", "full_cache", "sliding_window"]
    results = []

    print("\nBenchmarking different caching methods...")
    for method in methods:
        print(f"  Running {method}...")
        stats = bench.benchmark_method(
            short_prompts,
            method=method,
            max_new_tokens=config.max_new_tokens,
            window_size=256,
        )
        stats["prompt_group"] = "short"
        results.append(stats)

    # Display results
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n" + "=" * 80)

    # Show speedup
    baseline_tps = df[df["method"] == "no_cache"]["tokens_per_sec"].values[0]
    for _, row in df.iterrows():
        if row["method"] != "no_cache":
            speedup = row["tokens_per_sec"] / baseline_tps
            print(f"{row['method']:20s}: {speedup:.2f}x speedup vs no_cache")


if __name__ == "__main__":
    main()
