"""Quantization utilities for KV cache compression."""

from typing import Tuple

import torch

from ..cuda.extensions import get_cuda_extension


def quantize_int8_per_tensor(
    x: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor INT8 quantization.
    
    Quantizes tensor to int8 using symmetric quantization:
    q = clamp(round(x/scale), -127, 127)
    
    Args:
        x: Input tensor to quantize
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Tuple of (quantized tensor, scale factor)
    """
    # Quantize in float32 for stable scale computation
    x_fp32 = x.float()
    max_abs = x_fp32.abs().max()
    scale = (max_abs / 127.0).clamp(min=eps)
    q = torch.clamp((x_fp32 / scale).round(), -127, 127).to(torch.int8)
    return q, scale.to(x.dtype)


def quantize_int4_per_tensor_packed(
    x: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Symmetric per-tensor INT4 quantization with packing.
    
    Quantizes tensor to int4 (range [-8, 7]) and packs two values per uint8.
    
    Args:
        x: Input tensor to quantize
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Tuple of (packed tensor, scale factor, original last dimension)
    """
    x_fp32 = x.float()
    max_abs = x_fp32.abs().max()
    scale = (max_abs / 7.0).clamp(min=eps)  # int4 signed max is 7
    q = torch.clamp((x_fp32 / scale).round(), -8, 7).to(torch.int8)

    # Pack along last dimension
    orig_last = q.size(-1)
    if orig_last % 2 == 1:
        # Pad one element to make even
        q = torch.cat([q, torch.zeros_like(q[..., :1])], dim=-1)

    # Convert signed int4 -> unsigned nibble [0..15] by adding 8
    q_u = (q + 8).to(torch.uint8)

    hi = q_u[..., 0::2]  # even indices
    lo = q_u[..., 1::2]  # odd indices
    packed = (hi << 4) | lo  # uint8

    return packed, scale.to(x.dtype), orig_last


def dequantize_int8_per_tensor(
    q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize INT8 tensor to target dtype.
    
    Uses CUDA kernel if available for fp16, otherwise falls back to Python.
    
    Args:
        q: Quantized int8 tensor
        scale: Scale factor used during quantization
        out_dtype: Target output dtype
        
    Returns:
        Dequantized tensor
    """
    kvq_ext = get_cuda_extension()
    
    if kvq_ext is not None and q.is_cuda and out_dtype == torch.float16:
        # Use fast CUDA kernel
        return kvq_ext.dequant_int8_to_fp16(q.contiguous(), float(scale))
    else:
        # Fallback to Python implementation
        return (q.float() * scale.float()).to(out_dtype)


def dequantize_int4_per_tensor_packed(
    packed: torch.Tensor,
    scale: torch.Tensor,
    orig_last_dim: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize packed INT4 tensor to target dtype.
    
    Uses CUDA kernel if available for fp16, otherwise falls back to Python.
    
    Args:
        packed: Packed uint8 tensor (2 int4 values per byte)
        scale: Scale factor used during quantization
        orig_last_dim: Original size of last dimension before padding
        out_dtype: Target output dtype
        
    Returns:
        Dequantized tensor
    """
    kvq_ext = get_cuda_extension()
    
    if kvq_ext is not None and packed.is_cuda and out_dtype == torch.float16:
        # Use fast CUDA kernel
        out = kvq_ext.dequant_int4_packed_to_fp16(
            packed.contiguous(), float(scale), int(orig_last_dim)
        )
        return out[..., :orig_last_dim]
    else:
        # Fallback to Python implementation
        hi = (packed >> 4) & 0x0F
        lo = packed & 0x0F
        q_u = torch.empty(
            (*packed.shape[:-1], packed.shape[-1] * 2),
            device=packed.device,
            dtype=torch.uint8,
        )
        q_u[..., 0::2] = hi
        q_u[..., 1::2] = lo
        q = (q_u.to(torch.int16) - 8).to(torch.int8)
        q = q[..., :orig_last_dim]
        return (q.float() * scale.float()).to(out_dtype)


class QuantizedLayerKV:
    """Stores quantized KV cache for one transformer layer.
    
    Supports multiple quantization modes:
    - "int8": Both K and V in INT8
    - "int4": Both K and V in INT4 (packed)
    - "mixed": K in INT8, V in INT4 (trades quality vs compression)
    
    Attributes:
        mode: Quantization mode
        device: Device to store tensors on
        compute_dtype: Data type for dequantized output
    """

    def __init__(
        self, mode: str = "int8", device: str = "cuda", compute_dtype: torch.dtype = torch.float16
    ):
        """Initialize quantized layer cache.
        
        Args:
            mode: Quantization mode ("int8", "int4", or "mixed")
            device: Device to store tensors on
            compute_dtype: Data type for dequantized output
        """
        assert mode in ["int8", "int4", "mixed"], f"Invalid mode: {mode}"
        self.mode = mode
        self.device = device
        self.compute_dtype = compute_dtype

        # Storage for quantized tensors and metadata
        self.k_store: list[torch.Tensor] = []
        self.v_store: list[torch.Tensor] = []
        self.k_scales: list[torch.Tensor] = []
        self.v_scales: list[torch.Tensor] = []
        self.k_meta: list[int | None] = []  # for int4: orig_last_dim
        self.v_meta: list[int | None] = []

    @torch.no_grad()
    def append(self, k_1tok: torch.Tensor, v_1tok: torch.Tensor) -> None:
        """Append one token's key-value pair.
        
        Args:
            k_1tok: Key tensor of shape [B, H, 1, D]
            v_1tok: Value tensor of shape [B, H, 1, D]
        """
        if self.mode == "int8":
            kq, ks = quantize_int8_per_tensor(k_1tok)
            vq, vs = quantize_int8_per_tensor(v_1tok)
            self.k_store.append(kq)
            self.v_store.append(vq)
            self.k_scales.append(ks)
            self.v_scales.append(vs)
            self.k_meta.append(None)
            self.v_meta.append(None)

        elif self.mode == "int4":
            kp, ks, k_last = quantize_int4_per_tensor_packed(k_1tok)
            vp, vs, v_last = quantize_int4_per_tensor_packed(v_1tok)
            self.k_store.append(kp)
            self.v_store.append(vp)
            self.k_scales.append(ks)
            self.v_scales.append(vs)
            self.k_meta.append(k_last)
            self.v_meta.append(v_last)

        elif self.mode == "mixed":
            # K in INT8 (more sensitive), V in INT4 (more compressible)
            kq, ks = quantize_int8_per_tensor(k_1tok)
            vp, vs, v_last = quantize_int4_per_tensor_packed(v_1tok)
            self.k_store.append(kq)
            self.v_store.append(vp)
            self.k_scales.append(ks)
            self.v_scales.append(vs)
            self.k_meta.append(None)
            self.v_meta.append(v_last)

    @torch.no_grad()
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full dequantized K,V tensors.
        
        Returns:
            Tuple of (K, V) tensors of shape [B, H, T, D]
        """
        if len(self.k_store) == 0:
            raise ValueError("Empty cache")

        if self.mode == "int8":
            k_list = [
                dequantize_int8_per_tensor(self.k_store[i], self.k_scales[i], self.compute_dtype)
                for i in range(len(self.k_store))
            ]
            v_list = [
                dequantize_int8_per_tensor(self.v_store[i], self.v_scales[i], self.compute_dtype)
                for i in range(len(self.v_store))
            ]

        elif self.mode == "int4":
            k_list = [
                dequantize_int4_per_tensor_packed(
                    self.k_store[i], self.k_scales[i], self.k_meta[i], self.compute_dtype  # type: ignore
                )
                for i in range(len(self.k_store))
            ]
            v_list = [
                dequantize_int4_per_tensor_packed(
                    self.v_store[i], self.v_scales[i], self.v_meta[i], self.compute_dtype  # type: ignore
                )
                for i in range(len(self.v_store))
            ]

        elif self.mode == "mixed":
            k_list = [
                dequantize_int8_per_tensor(self.k_store[i], self.k_scales[i], self.compute_dtype)
                for i in range(len(self.k_store))
            ]
            v_list = [
                dequantize_int4_per_tensor_packed(
                    self.v_store[i], self.v_scales[i], self.v_meta[i], self.compute_dtype  # type: ignore
                )
                for i in range(len(self.v_store))
            ]

        k = torch.cat(k_list, dim=2)
        v = torch.cat(v_list, dim=2)
        return k, v

    def estimated_bytes(self) -> int:
        """Estimate memory footprint of stored quantized KV.
        
        Returns:
            Approximate bytes used (excludes Python list overhead)
        """
        total = 0
        # Stored tensors
        for t in self.k_store:
            total += t.numel() * t.element_size()
        for t in self.v_store:
            total += t.numel() * t.element_size()

        # Scales
        for s in self.k_scales:
            total += s.numel() * s.element_size()
        for s in self.v_scales:
            total += s.numel() * s.element_size()

        return total


class QuantizedKVCache:
    """Multi-layer quantized KV cache container.
    
    Holds quantized caches for all transformer layers.
    
    Attributes:
        layers: List of QuantizedLayerKV instances
    """

    def __init__(
        self,
        n_layers: int,
        mode: str = "int8",
        device: str = "cuda",
        compute_dtype: torch.dtype = torch.float16,
    ):
        """Initialize multi-layer quantized cache.
        
        Args:
            n_layers: Number of transformer layers
            mode: Quantization mode
            device: Device to store tensors on
            compute_dtype: Data type for dequantized output
        """
        self.layers = [
            QuantizedLayerKV(mode=mode, device=device, compute_dtype=compute_dtype)
            for _ in range(n_layers)
        ]

    @torch.no_grad()
    def append_from_past(self, past_key_values: tuple) -> None:
        """Append the last token from model's past_key_values.
        
        Args:
            past_key_values: Tuple of (k, v) pairs from model output
        """
        for i, (k, v) in enumerate(past_key_values):
            self.layers[i].append(k[:, :, -1:, :], v[:, :, -1:, :])

    @torch.no_grad()
    def init_from_prompt_past(self, past_key_values: tuple) -> None:
        """Initialize cache from full prompt output.
        
        Args:
            past_key_values: Tuple of (k, v) pairs from model output
        """
        for layer_idx, (k, v) in enumerate(past_key_values):
            T = k.size(2)
            for t in range(T):
                self.layers[layer_idx].append(k[:, :, t : t + 1, :], v[:, :, t : t + 1, :])

    @torch.no_grad()
    def to_past_key_values(self) -> tuple:
        """Convert to HuggingFace-compatible past_key_values format.
        
        Returns:
            Tuple of (k, v) pairs for all layers
        """
        past = []
        for layer in self.layers:
            k, v = layer.get_kv()
            past.append((k, v))
        return tuple(past)

    def estimated_bytes(self) -> int:
        """Total estimated memory footprint across all layers.
        
        Returns:
            Total bytes used
        """
        return sum(layer.estimated_bytes() for layer in self.layers)
