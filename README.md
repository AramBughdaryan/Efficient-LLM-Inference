# Efficient LLM Inference

A high-performance library for efficient large language model (LLM) inference using various KV-cache optimization techniques.

## Features

This library implements multiple state-of-the-art KV cache optimization strategies:

### 🔧 Cache Strategies
- **Full KV Cache**: Standard caching with complete history
- **Sliding Window**: Memory-efficient fixed-size context window
- **Paged Attention**: Block-based memory allocation (simulated)
- **Chunk-Summary**: Compress old context with mean-pooling

### 🗜️ Quantization
- **INT8 Quantization**: Symmetric per-tensor quantization
- **INT4 Quantization**: 4-bit packed quantization with CUDA kernels
- **Mixed Precision**: INT8 keys + INT4 values for optimal quality/size tradeoff

### ⚡ Performance
- CUDA kernels for fast dequantization
- Comprehensive benchmarking suite
- Memory tracking and profiling utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/efficient-llm-inference.git
cd efficient-llm-inference

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev]"

# Optional: Install notebook dependencies
pip install -e ".[notebooks]"
```

## Quick Start

### Basic Benchmark

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.benchmarker import KVCacheBenchmarker
from src.config import Config

# Initialize
config = Config(model_name="gpt2")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device)

# Create benchmarker
bench = KVCacheBenchmarker(model, tokenizer, device=config.device)

# Run benchmark
prompts = ["Explain machine learning in simple terms."]
results = bench.benchmark_method(
    prompts,
    method="full_cache",
    max_new_tokens=64
)

print(f"Tokens/sec: {results['tokens_per_sec']:.2f}")
```

### Quantized Cache

```python
# Generate with INT8 quantized cache
text, n_tokens, cache_mb = bench.generate_with_quantized_kv(
    "What is artificial intelligence?",
    max_new_tokens=64,
    mode="int8"
)

print(f"Generated {n_tokens} tokens using {cache_mb:.2f} MB cache")
```

### Running Examples

```bash
# Basic benchmark comparison
python -m examples.basic_benchmark

# Quantized cache demonstration
python -m examples.quantized_cache

# Comprehensive benchmark (all methods)
python examples.comprehensive_benchmark
```

## Project Structure

```
Efficient-LLM-Inference/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration classes
│   ├── utils.py              # Memory tracking utilities
│   ├── cuda_kernels.py       # CUDA extension for quantization
│   ├── quantization.py       # Quantization utilities
│   ├── cache.py              # Cache implementations
│   ├── benchmarker.py        # Main benchmarking class
│   └── metrics.py            # Evaluation metrics
├── examples/
│   ├── basic_benchmark.py    # Simple benchmark example
│   ├── quantized_cache.py    # Quantization example
│   └── comprehensive_benchmark.py  # Full benchmark suite
├── tests/                    # Unit tests (to be added)
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Benchmarking

The library provides comprehensive benchmarking across multiple dimensions:

### Supported Methods
- `no_cache`: Baseline without KV caching
- `full_cache`: Standard KV cache
- `sliding_window`: Fixed-size context window
- `quant_int8`: INT8 quantized cache
- `quant_int4`: INT4 quantized cache
- `quant_mixed`: Mixed INT8/INT4 cache
- `paged_attention`: Block-based allocation
- `chunked_cache`: Chunk-summary compression

### Metrics Collected
- **Throughput**: Tokens per second
- **Latency**: Total generation time
- **Memory**: CPU and GPU peak usage
- **Cache Size**: Estimated KV cache footprint
- **Quality**: Text similarity and perplexity

## Configuration

The library uses dataclasses for configuration:

```python
from src.config import Config, QuantizationConfig, CacheConfig

# Main config
config = Config(
    model_name="gpt2",
    device="cuda",
    seed=42,
    max_new_tokens=64
)

# Quantization config
quant_config = QuantizationConfig(
    mode="int8",
    eps=1e-8
)

# Cache config
cache_config = CacheConfig(
    window_size=256,
    block_size=64,
    chunk_size=64,
    keep_last=256
)
```

## Future Extensions

### Planned Features
- [ ] Additional attention variants for lightweight models
- [ ] Benchmark dataset integration
- [ ] Multi-GPU support
- [ ] Flash Attention integration
- [ ] More quantization schemes (GPTQ, AWQ)
- [ ] Streaming inference support

### Attention Variants (Coming Soon)
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Linear attention mechanisms

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (once implemented)
pytest tests/

# Format code
ruff src/ examples/

# Linting
ruff check src/ examples/
```

## Performance Tips

1. **Use CUDA**: The library includes optimized CUDA kernels for quantization
2. **Choose the right method**: 
   - Short sequences: Full cache is usually fastest
   - Long sequences: Consider sliding window or quantization
   - Memory-constrained: Use INT4 quantization
3. **Batch processing**: Process multiple prompts together when possible
4. **Monitor memory**: Use the provided utilities to track GPU/CPU usage
