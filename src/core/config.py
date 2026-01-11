"""Core configuration classes."""

import random
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class Config:
    """Main configuration for LLM inference benchmarking.

    Attributes:
        model_name: HuggingFace model identifier
        device: Device to run inference on ("cuda" or "cpu")
        dtype: Data type for model weights
        seed: Random seed for reproducibility
        max_new_tokens: Default number of tokens to generate
        batch_size: Batch size for inference
    """

    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = field(
        default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32
    )
    seed: int = 42
    max_new_tokens: int = 64
    batch_size: int = 1

    def __post_init__(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


@dataclass
class QuantizationConfig:
    """Configuration for KV cache quantization.

    Attributes:
        mode: Quantization mode ("int8", "int4", or "mixed")
        eps: Small epsilon value to avoid division by zero
    """

    mode: Literal["int8", "int4", "mixed"] = "int8"
    eps: float = 1e-8


@dataclass
class CacheConfig:
    """Configuration for cache strategies.

    Attributes:
        window_size: Size of sliding window for sliding-window cache
        block_size: Block size for paged attention
        chunk_size: Chunk size for chunk-summary caching
        keep_last: Number of recent tokens to keep exact in chunk-summary
    """

    window_size: int = 256
    block_size: int = 64
    chunk_size: int = 64
    keep_last: int = 256


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments.

    Attributes:
        methods: List of caching methods to benchmark
        window_sizes: Window sizes to test for sliding-window cache
        block_sizes: Block sizes to test for paged attention
        chunk_sizes: Chunk sizes to test for chunk-summary caching
    """

    methods: list[str] = field(default_factory=lambda: ["no_cache", "full_cache", "sliding_window"])
    window_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    block_sizes: list[int] = field(default_factory=lambda: [32, 64, 128])
    chunk_sizes: list[int] = field(default_factory=lambda: [32, 64, 128])
