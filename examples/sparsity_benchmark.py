import torch
from transformers import AutoTokenizer

from src.sparsity.sparse_model import run_all_sparsity_strategies

# If you want to reuse prompts from other example scripts, import them too.

def main():
    MODEL_NAME = "gpt2"  # replace
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    short_prompts = [
        "Write a short email about meeting tomorrow.",
        "Explain attention in one paragraph.",
    ]

    long_prompts = [
        "Explain transformers in detail with examples. " * 20,
    ]

    strategies = [
        {"strategy": "layerwise_l1", "amount": 0.3},
        {"strategy": "global_l1",    "amount": 0.3},
        {"strategy": "random",       "amount": 0.3},
        {"strategy": "structured_ln","amount": 0.2, "structured_dim": 0, "structured_n": 2},
        {"strategy": "nm",           "nm_N": 2, "nm_M": 4, "nm_dim": 1},
    ]

    df = run_all_sparsity_strategies(
        model_name=MODEL_NAME,
        tokenizer=tokenizer,
        device=DEVICE,
        dtype=DTYPE,
        short_prompts=short_prompts,
        long_prompts=long_prompts,
        strategies=strategies,
        methods=("no_cache", "full_cache", "sliding_window"),
        window_size=256,
        max_new_tokens=64,
    )

    print(df)
    df.to_csv("sparsity_benchmark_results.csv", index=False)

if __name__ == "__main__":
    main()
