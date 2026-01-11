# Summarization Benchmarking

This module provides a comprehensive interface for benchmarking summarization quality and performance with different caching mechanisms.

## Features

### 📊 Datasets Supported
- **CNN/DailyMail** - News article summarization (most popular)
- **XSum** - Extreme summarization
- **SAMSum** - Dialogue summarization

### 🎯 Quality Metrics
- **ROUGE-1, ROUGE-2, ROUGE-L** - Standard summarization metrics
- Precision, Recall, F-measure for each

### ⚡ Performance Metrics
- Generation speed (tokens/sec)
- Time per sample
- Memory usage

### 🔧 Caching Methods
- Full cache
- Sliding window
- Quantized (INT4/INT8/mixed)
- Paged attention
- Chunked cache

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.benchmarking import SummarizationBenchmark
from src.datasets import load_cnn_dailymail
from src.core import Config

# Load model
config = Config(model_name="gpt2")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device)

# Load dataset
dataset = load_cnn_dailymail(split="test", max_samples=100)

# Create benchmark
benchmark = SummarizationBenchmark(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    device=config.device,
)

# Compare methods
results = benchmark.compare_methods(
    methods=["full_cache", "sliding_window", "quant_int8"],
    num_samples=10,
    max_new_tokens=128,
)

print(results[["method", "rouge1_f", "rouge2_f", "rougeL_f", "elapsed_sec"]])
```

### Compare Method Variants

```python
# Test different window sizes
results = benchmark.benchmark_with_variants(
    base_method="sliding_window",
    variants=[
        {"window_size": 128},
        {"window_size": 256},
        {"window_size": 512},
    ],
    num_samples=10,
)
```

### Custom Configuration

```python
# Custom method configurations
results = benchmark.compare_methods(
    methods=["sliding_window", "quant_int4", "paged_attention"],
    num_samples=20,
    max_new_tokens=150,
    method_configs={
        "sliding_window": {"window_size": 256},
        "paged_attention": {"block_size": 64},
    }
)
```

## API Reference

### `SummarizationBenchmark`

Main class for benchmarking summarization.

**Methods:**

- `generate_summaries()` - Generate summaries with specified method
- `evaluate_quality()` - Compute ROUGE scores
- `benchmark_configuration()` - Benchmark a single configuration
- `compare_methods()` - Compare multiple caching methods
- `benchmark_with_variants()` - Test different parameter variants

### `SummarizationDataset`

Dataset wrapper for easy loading and formatting.

**Methods:**

- `__len__()` - Get dataset size
- `__getitem__()` - Get single sample
- `get_batch()` - Get batch of samples
- `get_samples()` - Get n samples from offset
- `create_prompts()` - Format articles as prompts

### Helper Functions

```python
# Quick dataset loaders
from src.datasets import (
    load_cnn_dailymail,
    load_xsum,
    load_samsum,
)

dataset = load_cnn_dailymail(split="test", max_samples=100)
```

## Example Output

```
================================================================================
COMPARING CACHING METHODS
================================================================================

Benchmarking: full_cache
Samples: 10, Max tokens: 128
  Elapsed: 12.34s (1.23s/sample)
  ROUGE-1 F1: 0.3456
  ROUGE-2 F1: 0.1234
  ROUGE-L F1: 0.2987

Benchmarking: sliding_window
Samples: 10, Max tokens: 128
  Elapsed: 11.89s (1.19s/sample)
  ROUGE-1 F1: 0.3398
  ROUGE-2 F1: 0.1198
  ROUGE-L F1: 0.2945

Benchmarking: quant_int8
Samples: 10, Max tokens: 128
  Elapsed: 10.23s (1.02s/sample)
  ROUGE-1 F1: 0.3442
  ROUGE-2 F1: 0.1221
  ROUGE-L F1: 0.2976

================================================================================
RESULTS SUMMARY
================================================================================
        method  elapsed_sec  avg_time_per_sample  rouge1_f  rouge2_f  rougeL_f
    full_cache        12.34                 1.23    0.3456    0.1234    0.2987
sliding_window        11.89                 1.19    0.3398    0.1198    0.2945
    quant_int8        10.23                 1.02    0.3442    0.1221    0.2976
```

## Running the Example

```bash
# Install dependencies
pip install -e .

# Run summarization benchmark
python examples/summarization_benchmark.py
```

## Use Cases

### 1. Model Comparison

Test different models on the same dataset:

```python
models = ["gpt2", "gpt2-medium", "gpt2-large"]
for model_name in models:
    # Load model, create benchmark, compare methods
    ...
```

### 2. Attention Variant Testing

When you add new attention mechanisms:

```python
# After implementing MQA or GQA
benchmark.compare_methods(
    methods=["full_cache_mqa", "full_cache_gqa", "full_cache"],
    ...
)
```

### 3. Hyperparameter Tuning

Find optimal cache parameters:

```python
# Find best window size
for size in [64, 128, 256, 512, 1024]:
    results = benchmark.benchmark_configuration(
        method="sliding_window",
        window_size=size,
        ...
    )
```

### 4. Quality vs Speed Tradeoff

Analyze tradeoffs between quality and performance:

```python
df = benchmark.compare_methods(methods=[...])
# Plot: ROUGE scores vs speed
df.plot(x='elapsed_sec', y='rouge1_f', kind='scatter')
```

## Installation

Add required dependencies:

```bash
pip install datasets rouge-score
```

Or install with extras:

```bash
pip install -e ".[datasets]"
```

## Future Extensions

The interface is designed to easily support:
- [ ] Multi-document summarization
- [ ] Abstractive vs extractive comparison
- [ ] Different decoding strategies (beam search, sampling)
- [ ] Custom attention mechanisms
- [ ] Model distillation benchmarks
- [ ] Multi-lingual summarization

---

**See also:**
- `examples/summarization_benchmark.py` - Full example
- `src/datasets/` - Dataset loaders
- `src/evaluation/rouge.py` - ROUGE implementation
- `src/benchmarking/summarization.py` - Main implementation
