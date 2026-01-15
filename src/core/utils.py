"""Core utility functions."""

import os
from typing import Optional

import psutil
import torch


def get_cpu_mem_mb() -> float:
    """Return current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


def reset_gpu_peak(device: str = "cuda") -> None:
    """Reset GPU peak memory statistics."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_peak_mb(device: str = "cuda") -> Optional[float]:
    """Return peak GPU memory usage in MB.

    Args:
        device: Device name ("cuda" or "cpu")

    Returns:
        Peak memory in MB if CUDA is available, None otherwise
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return None


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Calculate memory footprint of a tensor in bytes.

    Args:
        tensor: PyTorch tensor

    Returns:
        Number of bytes occupied by the tensor
    """
    return tensor.numel() * tensor.element_size()


def mb(num_bytes: int) -> float:
    """Convert bytes to megabytes.

    Args:
        num_bytes: Number of bytes

    Returns:
        Size in megabytes
    """
    return num_bytes / (1024**2)


def kv_bytes_fp(k: torch.Tensor, v: torch.Tensor) -> int:
    """Calculate total bytes used by key and value tensors.

    Args:
        k: Key tensor
        v: Value tensor

    Returns:
        Total bytes used by both tensors
    """
    return tensor_bytes(k) + tensor_bytes(v)
