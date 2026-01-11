"""Efficient LLM Inference Library.

This library provides various KV-cache optimization techniques for efficient
large language model inference, including:
- Quantization (INT4, INT8, Mixed)
- Paged attention simulation
- Sliding window caching
- Chunk-summary caching
- Summarization benchmarking
"""

__version__ = "0.1.0"

from .benchmarking.benchmarker import KVCacheBenchmarker
from .benchmarking.summarization import SummarizationBenchmark
from .core.config import BenchmarkConfig, CacheConfig, Config, QuantizationConfig
from .datasets import load_cnn_dailymail, load_samsum, load_xsum

__all__ = [
    "Config",
    "QuantizationConfig",
    "CacheConfig",
    "BenchmarkConfig",
    "KVCacheBenchmarker",
    "SummarizationBenchmark",
    "load_cnn_dailymail",
    "load_xsum",
    "load_samsum",
    "__version__",
]
