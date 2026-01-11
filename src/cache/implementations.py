"""Cache implementations for KV cache optimization strategies."""

from typing import Tuple

import torch

from ..core.utils import tensor_bytes


class PagedKVCache:
    """Simulated paged KV cache for one transformer layer.
    
    Stores K/V in fixed-size blocks along the sequence dimension.
    This simulates the paged attention memory layout.
    
    Attributes:
        block_size: Number of tokens per block
        device: Device to store tensors on
        dtype: Data type for tensors
    """

    def __init__(self, block_size: int = 64, device: str = "cuda", dtype: torch.dtype = torch.float16):
        """Initialize paged cache.
        
        Args:
            block_size: Number of tokens per block
            device: Device to store tensors on
            dtype: Data type for tensors
        """
        self.block_size = int(block_size)
        self.device = device
        self.dtype = dtype

        self.k_blocks: list[torch.Tensor] = []  # each: [B, H, block_size, D]
        self.v_blocks: list[torch.Tensor] = []  # each: [B, H, block_size, D]
        self.t_filled = 0  # total tokens stored

        # Cached shape after first append
        self._B: int | None = None
        self._H: int | None = None
        self._D: int | None = None

    def num_blocks(self) -> int:
        """Get number of allocated blocks."""
        return len(self.k_blocks)

    def _alloc_block(self, B: int, H: int, D: int) -> None:
        """Allocate a new block."""
        k = torch.empty((B, H, self.block_size, D), device=self.device, dtype=self.dtype)
        v = torch.empty((B, H, self.block_size, D), device=self.device, dtype=self.dtype)
        self.k_blocks.append(k)
        self.v_blocks.append(v)

    @torch.no_grad()
    def append(self, k_1tok: torch.Tensor, v_1tok: torch.Tensor) -> None:
        """Append one token's KV.
        
        Args:
            k_1tok: Key tensor of shape [B, H, 1, D]
            v_1tok: Value tensor of shape [B, H, 1, D]
        """
        assert k_1tok.dim() == 4 and v_1tok.dim() == 4
        B, H, one, D = k_1tok.shape
        assert one == 1
        if self._B is None:
            self._B, self._H, self._D = B, H, D

        # Allocate first block or new block if current is full
        if self.num_blocks() == 0 or (self.t_filled % self.block_size) == 0:
            self._alloc_block(B, H, D)

        blk_idx = self.t_filled // self.block_size
        off = self.t_filled % self.block_size

        # Write into block at [.., off, :]
        self.k_blocks[blk_idx][:, :, off : off + 1, :] = k_1tok
        self.v_blocks[blk_idx][:, :, off : off + 1, :] = v_1tok
        self.t_filled += 1

    @torch.no_grad()
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full K,V tensors by stitching blocks.
        
        Returns:
            Tuple of (K, V) tensors of shape [B, H, T, D]
        """
        if self.t_filled == 0:
            raise ValueError("Empty cache")

        assert self._B is not None and self._H is not None and self._D is not None
        B, H, D = self._B, self._H, self._D
        T = self.t_filled
        k_out = torch.empty((B, H, T, D), device=self.device, dtype=self.dtype)
        v_out = torch.empty((B, H, T, D), device=self.device, dtype=self.dtype)

        t = 0
        for b in range(self.num_blocks()):
            take = min(self.block_size, T - t)
            k_out[:, :, t : t + take, :] = self.k_blocks[b][:, :, :take, :]
            v_out[:, :, t : t + take, :] = self.v_blocks[b][:, :, :take, :]
            t += take
            if t >= T:
                break
        return k_out, v_out

    def allocated_bytes(self) -> int:
        """Total allocated bytes in blocks (includes unused space)."""
        if self.num_blocks() == 0:
            return 0
        return sum(tensor_bytes(x) for x in self.k_blocks) + sum(
            tensor_bytes(x) for x in self.v_blocks
        )

    def used_bytes(self) -> int:
        """Bytes actually used by stored tokens."""
        if self.t_filled == 0:
            return 0
        assert self._B is not None and self._H is not None and self._D is not None
        return self.t_filled * self._B * self._H * self._D * self.dtype.itemsize * 2


def trim_kv_sliding_window(past_key_values: tuple, window_size: int) -> tuple:
    """Trim KV tensors to keep only last `window_size` tokens.
    
    Args:
        past_key_values: Tuple of (k, v) pairs
        window_size: Number of recent tokens to keep
        
    Returns:
        Trimmed past_key_values tuple
    """
    trimmed = []
    for k, v in past_key_values:
        if k.size(2) > window_size:
            k = k[:, :, -window_size:, :]
            v = v[:, :, -window_size:, :]
        trimmed.append((k, v))
    return tuple(trimmed)


def chunk_summarize_kv(
    past_key_values: tuple, chunk_size: int, keep_last: int
) -> tuple:
    """Compress older KV by replacing with mean-pooled chunk summaries.
    
    Keep last `keep_last` tokens unchanged. Older tokens are grouped into
    chunks and mean-pooled.
    
    Args:
        past_key_values: Tuple of (k, v) pairs per layer
        chunk_size: Number of tokens to pool into one summary
        keep_last: Number of recent tokens to keep exact
        
    Returns:
        Compressed past_key_values tuple
    """
    summarized = []
    for k, v in past_key_values:
        B, H, T, D = k.shape
        keep_last_eff = min(keep_last, T)
        old_len = T - keep_last_eff

        # If nothing to compress, return as-is
        if old_len <= 0:
            summarized.append((k, v))
            continue

        k_old = k[:, :, :old_len, :]
        v_old = v[:, :, :old_len, :]
        k_recent = k[:, :, old_len:, :]
        v_recent = v[:, :, old_len:, :]

        # Pad old part so it divides into chunks
        pad = (chunk_size - (old_len % chunk_size)) % chunk_size
        if pad > 0:
            k_old = torch.cat(
                [k_old, torch.zeros(B, H, pad, D, device=k.device, dtype=k.dtype)], dim=2
            )
            v_old = torch.cat(
                [v_old, torch.zeros(B, H, pad, D, device=v.device, dtype=v.dtype)], dim=2
            )

        n_chunks = k_old.size(2) // chunk_size

        # Reshape into chunks and mean-pool
        k_chunks = k_old.view(B, H, n_chunks, chunk_size, D).mean(dim=3)
        v_chunks = v_old.view(B, H, n_chunks, chunk_size, D).mean(dim=3)

        # New KV = [summaries] + [recent exact]
        k_new = torch.cat([k_chunks, k_recent], dim=2)
        v_new = torch.cat([v_chunks, v_recent], dim=2)

        summarized.append((k_new, v_new))
    return tuple(summarized)
