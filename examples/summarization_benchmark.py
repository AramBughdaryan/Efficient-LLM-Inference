"""Example: Benchmarking summarization with different caching methods."""

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarking import SummarizationBenchmark
from src.core import Config
from src.datasets import load_cnn_dailymail


def main():
    """Run summarization benchmark on CNN/DailyMail dataset."""
    # Configuration
    config = Config(model_name="Qwen/Qwen2.5-7B", max_new_tokens=128)

    print("=" * 80)
    print("SUMMARIZATION BENCHMARK")
    print("=" * 80)
    print(f"\nModel: {config.model_name}")
    print(f"Device: {config.device}")

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

    # Load dataset
    print("\nLoading CNN/DailyMail dataset...")
    dataset = load_cnn_dailymail(split="test", max_samples=20)
    print(f"Loaded {len(dataset)} samples")

    # Create benchmark
    benchmark = SummarizationBenchmark(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=config.device,
    )

    # Compare different caching methods
    print("\n" + "=" * 80)
    print("COMPARING CACHING METHODS")
    print("=" * 80)

    methods = [
        "full_cache",
        "sliding_window",
        "quant_int8",
    ]

    method_configs = {
        "sliding_window": {"window_size": 256},
    }

    results_df = benchmark.compare_methods(
        methods=methods,
        num_samples=10,
        max_new_tokens=128,
        method_configs=method_configs,
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(
        results_df[
            ["method", "elapsed_sec", "avg_time_per_sample", "rouge1_f", "rouge2_f", "rougeL_f"]
        ].to_string(index=False)
    )

    # Benchmark sliding window with different sizes
    print("\n" + "=" * 80)
    print("SLIDING WINDOW VARIANTS")
    print("=" * 80)

    variants_df = benchmark.benchmark_with_variants(
        base_method="sliding_window",
        variants=[
            {"window_size": 128},
            {"window_size": 256},
            {"window_size": 512},
        ],
        num_samples=5,
        max_new_tokens=128,
    )

    print("\n" + "=" * 80)
    print("WINDOW SIZE COMPARISON")
    print("=" * 80)
    print(
        variants_df[["variant", "elapsed_sec", "rouge1_f", "rouge2_f", "rougeL_f"]].to_string(
            index=False
        )
    )

    # Save results
    results_df.to_csv("summarization_results.csv", index=False)
    variants_df.to_csv("summarization_variants.csv", index=False)
    print("\n✅ Results saved to:")
    print("  - summarization_results.csv")
    print("  - summarization_variants.csv")


if __name__ == "__main__":
    main()
