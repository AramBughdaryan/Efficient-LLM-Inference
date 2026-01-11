"""Example script demonstrating comprehensive benchmarking."""

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarking import KVCacheBenchmarker
from src.core import Config


def main():
    """Run comprehensive benchmarking across all methods."""
    # Initialize configuration
    config = Config(model_name="gpt2", max_new_tokens=64)
    
    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}\n")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype if config.device == "cuda" else None,
    ).to(config.device)
    model.eval()
    print("Model loaded.\n")
    
    # Create benchmarker
    bench = KVCacheBenchmarker(model, tokenizer, device=config.device)
    
    # Define prompts
    short_prompts = [
        "Explain reinforcement learning.",
        "What is deep learning?",
        "Describe neural networks.",
    ]
    
    base_text = (
        "In recent years, large language models have become central to many applications. "
        "They are used for chatbots, coding assistants, search, translation, and more. "
    )
    long_prompts = [(base_text * 40)[:1000], (base_text * 80)[:2000]]
    
    # Methods to test
    all_methods = [
        ("no_cache", {}),
        ("full_cache", {}),
        ("sliding_window", {"window_size": 256}),
        ("quant_int8", {}),
        ("quant_int4", {}),
        ("paged_attention", {"block_size": 64}),
        ("chunked_cache", {"chunk_size": 64, "keep_last": 256}),
    ]
    
    results = []
    
    # Benchmark short prompts
    print("=" * 80)
    print("BENCHMARKING SHORT PROMPTS")
    print("=" * 80)
    for method, kwargs in all_methods:
        print(f"  Running {method}...")
        try:
            stats = bench.benchmark_method(
                short_prompts,
                method=method,
                max_new_tokens=config.max_new_tokens,
                **kwargs,
            )
            stats["prompt_group"] = "short"
            results.append(stats)
        except Exception as e:
            print(f"    Error: {e}")
    
    # Benchmark long prompts
    print("\n" + "=" * 80)
    print("BENCHMARKING LONG PROMPTS")
    print("=" * 80)
    for method, kwargs in all_methods:
        print(f"  Running {method}...")
        try:
            stats = bench.benchmark_method(
                long_prompts,
                method=method,
                max_new_tokens=config.max_new_tokens,
                **kwargs,
            )
            stats["prompt_group"] = "long"
            results.append(stats)
        except Exception as e:
            print(f"    Error: {e}")
    
    # Save and display results
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Group by prompt type
    for group in ["short", "long"]:
        print(f"\n{group.upper()} PROMPTS:")
        sub = df[df["prompt_group"] == group]
        display_cols = ["method", "tokens_per_sec", "elapsed_sec", "est_kv_cache_mb_avg"]
        print(sub[display_cols].to_string(index=False))
    
    print(f"\nFull results saved to: benchmark_results.csv")


if __name__ == "__main__":
    main()
