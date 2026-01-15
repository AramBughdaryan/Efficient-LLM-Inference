"""Benchmarker for KV cache optimization strategies."""

import time
from typing import Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import DynamicCache

from ..cache import (
    PagedKVCache,
    chunk_summarize_kv,
    trim_kv_sliding_window,
)
from ..quantization import QuantizedKVCache
from ..core.utils import get_cpu_mem_mb, get_gpu_peak_mb, mb, reset_gpu_peak


class KVCacheBenchmarker:
    """Benchmarker for various KV cache strategies.
    
    Supports:
    - No cache baseline
    - Full KV cache
    - Sliding window cache
    - Prefix + window cache (keep prefix + recent tail)
    - Strided sparse cache (keep tail + strided older tokens)
    - Block-sparse cache (keep tail + per-block older tokens)
    - Budget-sparse cache (keep tail + fixed-budget sampled older tokens)
    - Quantized cache (int4/int8/mixed)
    - Paged attention (simulated)
    - Chunk-summary cache
    
    Attributes:
        model: HuggingFace causal language model
        tokenizer: HuggingFace tokenizer
        device: Device model is running on
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        """Initialize benchmarker.
        
        Args:
            model: HuggingFace causal language model
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ---------- KV trimming helpers (sparsity) ----------

    def _trim_kv_prefix_window(self, past_key_values, prefix_len: int, window_size: int):
        """Keep first prefix_len + last window_size tokens (legacy tuple KV)."""
        trimmed = []
        for k, v in past_key_values:
            seq_len = k.size(2)
            if seq_len <= prefix_len + window_size:
                trimmed.append((k, v))
                continue
            k_new = torch.cat([k[:, :, :prefix_len, :], k[:, :, -window_size:, :]], dim=2)
            v_new = torch.cat([v[:, :, :prefix_len, :], v[:, :, -window_size:, :]], dim=2)
            trimmed.append((k_new, v_new))
        return tuple(trimmed)

    def _trim_kv_strided(self, past_key_values, window_size: int, stride: int, prefix_len: int = 0):
        """
        Keep:
          - optional prefix (first prefix_len)
          - dense tail (last window_size)
          - from older region, keep every `stride` token
        """
        assert stride >= 1
        trimmed = []
        for k, v in past_key_values:
            seq_len = k.size(2)
            if seq_len <= prefix_len + window_size:
                trimmed.append((k, v))
                continue

            tail_start = max(prefix_len, seq_len - window_size)

            parts_k, parts_v = [], []
            if prefix_len > 0:
                parts_k.append(k[:, :, :prefix_len, :])
                parts_v.append(v[:, :, :prefix_len, :])

            # older region [prefix_len, tail_start)
            if tail_start > prefix_len:
                idx_old = torch.arange(prefix_len, tail_start, step=stride, device=k.device)
                parts_k.append(k.index_select(2, idx_old))
                parts_v.append(v.index_select(2, idx_old))

            # tail region [tail_start, seq_len)
            parts_k.append(k[:, :, tail_start:, :])
            parts_v.append(v[:, :, tail_start:, :])

            trimmed.append((torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)))
        return tuple(trimmed)

    def _trim_kv_block_old(
        self,
        past_key_values,
        window_size: int,
        block_size: int = 64,
        keep_per_block: int = 8,
        prefix_len: int = 0,
    ):
        """
        Block-based sparsity for older context:
          - keep optional prefix
          - keep dense tail (last window_size)
          - for older tokens (excluding prefix), partition into blocks of size block_size
            and keep the last `keep_per_block` tokens from each block.
        """
        assert block_size >= 1
        assert 1 <= keep_per_block <= block_size

        trimmed = []
        for k, v in past_key_values:
            seq_len = k.size(2)
            if seq_len <= prefix_len + window_size:
                trimmed.append((k, v))
                continue

            tail_start = max(prefix_len, seq_len - window_size)

            parts_k, parts_v = [], []
            if prefix_len > 0:
                parts_k.append(k[:, :, :prefix_len, :])
                parts_v.append(v[:, :, :prefix_len, :])

            # older region: [prefix_len, tail_start)
            old_len = tail_start - prefix_len
            if old_len > 0:
                idx_list = []
                start = prefix_len
                while start < tail_start:
                    end = min(start + block_size, tail_start)
                    keep_start = max(start, end - keep_per_block)
                    idx_list.append(torch.arange(keep_start, end, device=k.device))
                    start = end

                if idx_list:
                    idx_old = torch.cat(idx_list, dim=0)
                    parts_k.append(k.index_select(2, idx_old))
                    parts_v.append(v.index_select(2, idx_old))

            # tail region
            parts_k.append(k[:, :, tail_start:, :])
            parts_v.append(v[:, :, tail_start:, :])

            trimmed.append((torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)))
        return tuple(trimmed)

    def _trim_kv_budget_old(self, past_key_values, window_size: int, old_budget: int = 64, prefix_len: int = 0):
        """
        Keep:
          - optional prefix
          - dense tail (last window_size)
          - a fixed budget of `old_budget` tokens sampled uniformly from older region
        """
        assert old_budget >= 0

        trimmed = []
        for k, v in past_key_values:
            seq_len = k.size(2)
            if seq_len <= prefix_len + window_size:
                trimmed.append((k, v))
                continue

            tail_start = max(prefix_len, seq_len - window_size)

            parts_k, parts_v = [], []
            if prefix_len > 0:
                parts_k.append(k[:, :, :prefix_len, :])
                parts_v.append(v[:, :, :prefix_len, :])

            # older region indices: [prefix_len, tail_start)
            old_len = tail_start - prefix_len
            if old_len > 0 and old_budget > 0:
                if old_len <= old_budget:
                    idx_old = torch.arange(prefix_len, tail_start, device=k.device)
                else:
                    idx_old = torch.linspace(prefix_len, tail_start - 1, steps=old_budget, device=k.device).long()
                    idx_old = torch.unique_consecutive(idx_old)

                parts_k.append(k.index_select(2, idx_old))
                parts_v.append(v.index_select(2, idx_old))

            # tail region
            parts_k.append(k[:, :, tail_start:, :])
            parts_v.append(v[:, :, tail_start:, :])

            trimmed.append((torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)))
        return tuple(trimmed)


    # ---------- Generation methods ----------

    @torch.no_grad()
    def generate_no_cache(
        self, prompt: str, max_new_tokens: int = 32
    ) -> Tuple[str, int]:
        """Baseline: NO KV-CACHE.
        
        At every step we feed the entire sequence (prompt + generated).
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Tuple of (generated text, number of new tokens)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)
        generated = input_ids.clone()
        
        # Get vocabulary size for bounds checking
        vocab_size = self.model.config.vocab_size

        for _ in range(max_new_tokens):
            out = self.model(input_ids=generated, use_cache=False)
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Bounds check to prevent invalid token IDs
            next_token = torch.clamp(next_token, 0, vocab_size - 1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check if we've generated EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        n_new = generated.shape[-1] - input_ids.shape[-1]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, n_new

    @torch.no_grad()
    def generate_with_cache(
        self, prompt: str, max_new_tokens: int = 32
    ) -> Tuple[str, int]:
        """Standard KV-CACHE decoding.
        
        First pass over full prompt, then decode using only the last token
        and cached keys/values.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Tuple of (generated text, number of new tokens)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)

        # Use DynamicCache for new transformers API
        out = self.model(input_ids=input_ids, use_cache=True)
        
        # Convert to DynamicCache if we get a tuple (for compatibility)
        if isinstance(out.past_key_values, tuple):
            past_key_values = DynamicCache.from_legacy_cache(out.past_key_values)
        else:
            past_key_values = out.past_key_values
            
        logits = out.logits[:, -1, :]
        generated = input_ids.clone()

        # Get vocabulary size for bounds checking
        vocab_size = self.model.config.vocab_size

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Bounds check to prevent invalid token IDs
            next_token = torch.clamp(next_token, 0, vocab_size - 1)
            
            generated = torch.cat([generated, next_token], dim=-1)

            out = self.model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]

        n_new = generated.shape[-1] - input_ids.shape[-1]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, n_new

    @torch.no_grad()
    def generate_with_sliding_window(
        self, prompt: str, max_new_tokens: int = 32, window_size: int = 256
    ) -> Tuple[str, int]:
        """Sliding-window KV-cache.
        
        Same as full cache, but after every step we keep only the last
        `window_size` positions along the sequence dimension in KV tensors.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Number of tokens to generate
            window_size: Number of recent tokens to keep
            
        Returns:
            Tuple of (generated text, number of new tokens)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        out = self.model(input_ids=input_ids, use_cache=True)
        
        # Convert to tuple for our sliding window function
        past_kv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, 'to_legacy_cache') else out.past_key_values
        past_kv_tuple = trim_kv_sliding_window(past_kv_tuple, window_size)
        past_key_values = DynamicCache.from_legacy_cache(past_kv_tuple)
        
        logits = out.logits[:, -1, :]
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            out = self.model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            # Convert, trim, and convert back
            past_kv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, 'to_legacy_cache') else out.past_key_values
            past_kv_tuple = trim_kv_sliding_window(past_kv_tuple, window_size)
            past_key_values = DynamicCache.from_legacy_cache(past_kv_tuple)
            
            logits = out.logits[:, -1, :]

        n_new = generated.shape[-1] - input_ids.shape[-1]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, n_new

    @torch.no_grad()
    def generate_with_prefix_window(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        window_size: int = 256,
        prefix_len: int = 32,
    ) -> Tuple[str, int]:
        """Keep prefix + tail window."""
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)

        out = self.model(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        generated = input_ids.clone()

        # convert, trim, convert back
        pkv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
        pkv_tuple = self._trim_kv_prefix_window(pkv_tuple, prefix_len=prefix_len, window_size=window_size)
        pkv = DynamicCache.from_legacy_cache(pkv_tuple)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            out = self.model(input_ids=next_token, use_cache=True, past_key_values=pkv)
            logits = out.logits[:, -1, :]

            pkv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
            pkv_tuple = self._trim_kv_prefix_window(pkv_tuple, prefix_len=prefix_len, window_size=window_size)
            pkv = DynamicCache.from_legacy_cache(pkv_tuple)

        n_new = generated.size(-1) - input_ids.size(-1)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, n_new

    @torch.no_grad()
    def generate_with_strided_cache(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        window_size: int = 256,
        stride: int = 4,
        prefix_len: int = 0,
    ) -> Tuple[str, int]:
        """Keep tail window + strided old tokens (+ optional prefix)."""
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)

        out = self.model(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        generated = input_ids.clone()

        pkv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
        pkv_tuple = self._trim_kv_strided(pkv_tuple, window_size=window_size, stride=stride, prefix_len=prefix_len)
        pkv = DynamicCache.from_legacy_cache(pkv_tuple)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            out = self.model(input_ids=next_token, use_cache=True, past_key_values=pkv)
            logits = out.logits[:, -1, :]

            pkv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
            pkv_tuple = self._trim_kv_strided(pkv_tuple, window_size=window_size, stride=stride, prefix_len=prefix_len)
            pkv = DynamicCache.from_legacy_cache(pkv_tuple)

        n_new = generated.size(-1) - input_ids.size(-1)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, n_new

    @torch.no_grad()
    def generate_with_block_cache(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        window_size: int = 256,
        block_size: int = 64,
        keep_per_block: int = 8,
        prefix_len: int = 0,
    ) -> Tuple[str, int]:
        """Keep tail window + block-sparse older tokens."""
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)

        out = self.model(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        generated = input_ids.clone()

        pkv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
        pkv_tuple = self._trim_kv_block_old(
            pkv_tuple,
            window_size=window_size,
            block_size=block_size,
            keep_per_block=keep_per_block,
            prefix_len=prefix_len,
        )
        pkv = DynamicCache.from_legacy_cache(pkv_tuple)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            out = self.model(input_ids=next_token, use_cache=True, past_key_values=pkv)
            logits = out.logits[:, -1, :]

            pkv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
            pkv_tuple = self._trim_kv_block_old(
                pkv_tuple,
                window_size=window_size,
                block_size=block_size,
                keep_per_block=keep_per_block,
                prefix_len=prefix_len,
            )
            pkv = DynamicCache.from_legacy_cache(pkv_tuple)

        n_new = generated.size(-1) - input_ids.size(-1)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, n_new

    @torch.no_grad()
    def generate_with_budget_cache(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        window_size: int = 256,
        old_budget: int = 64,
        prefix_len: int = 0,
    ) -> Tuple[str, int]:
        """Keep tail window + fixed-budget sampled older tokens (+ optional prefix)."""
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)

        out = self.model(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        generated = input_ids.clone()

        pkv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
        pkv_tuple = self._trim_kv_budget_old(
            pkv_tuple,
            window_size=window_size,
            old_budget=old_budget,
            prefix_len=prefix_len,
        )
        pkv = DynamicCache.from_legacy_cache(pkv_tuple)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            out = self.model(input_ids=next_token, use_cache=True, past_key_values=pkv)
            logits = out.logits[:, -1, :]

            pkv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, "to_legacy_cache") else out.past_key_values
            pkv_tuple = self._trim_kv_budget_old(
                pkv_tuple,
                window_size=window_size,
                old_budget=old_budget,
                prefix_len=prefix_len,
            )
            pkv = DynamicCache.from_legacy_cache(pkv_tuple)

        n_new = generated.size(-1) - input_ids.size(-1)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, n_new


    @torch.no_grad()
    def generate_with_quantized_kv(
        self, prompt: str, max_new_tokens: int = 32, mode: str = "int8"
    ) -> Tuple[str, int, float]:
        """Quantized KV-cache decoding.
        
        Stores KV cache in quantized form (int4/int8/mixed).
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Number of tokens to generate
            mode: Quantization mode ("int8", "int4", or "mixed")
            
        Returns:
            Tuple of (generated text, number of new tokens, cache size in MB)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Initial prompt forward
        out = self.model(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :]

        # Convert to tuple for quantization cache
        past_kv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, 'to_legacy_cache') else out.past_key_values
        
        n_layers = len(past_kv_tuple)
        compute_dtype = torch.float16 if self.device == "cuda" else torch.float32
        qcache = QuantizedKVCache(
            n_layers=n_layers,
            mode=mode,
            device=self.device,
            compute_dtype=compute_dtype,
        )

        # Initialize quantized cache from prompt KV
        qcache.init_from_prompt_past(past_kv_tuple)

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Dequantize KV back to tuple, then convert to Cache for model
            past_tuple = qcache.to_past_key_values()
            past = DynamicCache.from_legacy_cache(past_tuple)

            out = self.model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past,
            )
            logits = out.logits[:, -1, :]

            # Convert back to tuple for quantization cache
            past_kv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, 'to_legacy_cache') else out.past_key_values
            qcache.append_from_past(past_kv_tuple)

        n_new = generated.shape[-1] - input_ids.shape[-1]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        est_mb = mb(qcache.estimated_bytes())
        return text, n_new, est_mb

    @torch.no_grad()
    def generate_with_paged_attention(
        self, prompt: str, max_new_tokens: int = 32, block_size: int = 64
    ) -> Tuple[str, int, float, float, int]:
        """Simulated paged attention.
        
        KV stored in fixed-size blocks. This simulates the memory layout
        but still requires stitching for HuggingFace compatibility.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Number of tokens to generate
            block_size: Number of tokens per block
            
        Returns:
            Tuple of (text, n_new, alloc_mb, used_mb, num_blocks)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        
        # Convert to tuple for paged cache
        past_kv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, 'to_legacy_cache') else out.past_key_values
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        paged_cache = [
            PagedKVCache(block_size=block_size, device=self.device, dtype=dtype)
            for _ in range(len(past_kv_tuple))
        ]
        
        # Initialize cache with prompt
        for layer_idx, (k, v) in enumerate(past_kv_tuple):
            for t in range(k.size(2)):
                paged_cache[layer_idx].append(k[:, :, t : t + 1, :], v[:, :, t : t + 1, :])
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stitch blocks -> contiguous past_key_values tuple -> Cache
            past_tuple = []
            for layer_cache in paged_cache:
                k, v = layer_cache.get_kv()
                past_tuple.append((k, v))
            past = DynamicCache.from_legacy_cache(tuple(past_tuple))
            
            out = self.model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past,
            )
            logits = out.logits[:, -1, :]
            
            # Convert back to tuple and append last token
            past_kv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, 'to_legacy_cache') else out.past_key_values
            for layer_idx, (k, v) in enumerate(past_kv_tuple):
                paged_cache[layer_idx].append(k[:, :, -1:, :], v[:, :, -1:, :])
        
        n_new = generated.shape[-1] - input_ids.shape[-1]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        alloc = sum(lc.allocated_bytes() for lc in paged_cache)
        used = sum(lc.used_bytes() for lc in paged_cache)
        nblocks = sum(lc.num_blocks() for lc in paged_cache)
        
        return text, n_new, mb(alloc), mb(used), nblocks

    @torch.no_grad()
    def generate_with_chunked_cache(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        chunk_size: int = 64,
        keep_last: int = 256,
    ) -> Tuple[str, int, float]:
        """Chunk-summary KV caching.
        
        Older KV are compressed into chunk summaries (mean-pooled).
        Recent tokens are kept exact.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Number of tokens to generate
            chunk_size: Number of tokens to pool into one summary
            keep_last: Number of recent tokens to keep exact
            
        Returns:
            Tuple of (text, n_new, cache_mb)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        out = self.model(input_ids=input_ids, use_cache=True)
        
        # Convert to tuple for chunking, then back to Cache
        past_kv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, 'to_legacy_cache') else out.past_key_values
        past_kv_tuple = chunk_summarize_kv(past_kv_tuple, chunk_size=chunk_size, keep_last=keep_last)
        past = DynamicCache.from_legacy_cache(past_kv_tuple)
        
        logits = out.logits[:, -1, :]
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            out = self.model(input_ids=next_token, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]

            # Update and compress cache
            past_kv_tuple = out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values, 'to_legacy_cache') else out.past_key_values
            past_kv_tuple = chunk_summarize_kv(past_kv_tuple, chunk_size=chunk_size, keep_last=keep_last)
            past = DynamicCache.from_legacy_cache(past_kv_tuple)

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        n_new = generated.shape[-1] - input_ids.shape[-1]

        # Calculate cache size from tuple
        past_kv_tuple = past.to_legacy_cache() if hasattr(past, 'to_legacy_cache') else past
        est_bytes = 0
        for k, v in past_kv_tuple:
            est_bytes += k.numel() * k.element_size()
            est_bytes += v.numel() * v.element_size()
        est_mb = est_bytes / (1024**2)

        return text, n_new, est_mb

    # ---------- Benchmarking ----------

    def benchmark_method(
        self,
        prompts: list[str],
        method: str = "no_cache",
        max_new_tokens: int = 32,
        window_size: int = 256,
        block_size: int = 64,
        chunk_size: int = 64,
        keep_last: int = 256,
        mode: str = "int8",
        prefix_len: int = 32,
        stride: int = 4,
        keep_per_block: int = 8,
        old_budget: int = 64,
    ) -> dict:
        """Run benchmark on a list of prompts.
        
        Args:
            prompts: List of prompt strings
            method: Cache method to use
            max_new_tokens: Number of tokens to generate per prompt
            window_size: Window size for sliding window
            block_size: Block size for paged attention
            chunk_size: Chunk size for chunk-summary
            keep_last: Number of tokens to keep exact in chunk-summary
            mode: Quantization mode for quantized methods
            
        Returns:
            Dictionary with benchmark results
        """
        valid_methods = [
            "no_cache",
            "full_cache",
            "sliding_window",
            "quant_int8",
            "quant_int4",
            "quant_mixed",
            "paged_attention",
            "chunked_cache",
            "prefix_window",
            "strided_cache",
            "block_cache",
            "budget_cache",
        ]
        assert method in valid_methods, f"Invalid method: {method}"

        reset_gpu_peak(self.device)
        start_cpu = get_cpu_mem_mb()
        
        # Use CUDA events for GPU timing, fallback to time.time() for CPU
        if self.device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.time()

        total_new_tokens = 0
        est_cache_mbs = []

        for prompt in prompts:
            if method == "no_cache":
                _, n_new = self.generate_no_cache(prompt, max_new_tokens)
                est_cache_mbs.append(0.0)

            elif method == "full_cache":
                _, n_new = self.generate_with_cache(prompt, max_new_tokens)
                est_cache_mbs.append(float("nan"))

            elif method == "sliding_window":
                _, n_new = self.generate_with_sliding_window(
                    prompt, max_new_tokens, window_size=window_size
                )
                est_cache_mbs.append(float("nan"))

            elif method == "quant_int8":
                _, n_new, est_mb = self.generate_with_quantized_kv(prompt, max_new_tokens, mode="int8")
                est_cache_mbs.append(est_mb)

            elif method == "quant_int4":
                _, n_new, est_mb = self.generate_with_quantized_kv(prompt, max_new_tokens, mode="int4")
                est_cache_mbs.append(est_mb)

            elif method == "quant_mixed":
                _, n_new, est_mb = self.generate_with_quantized_kv(prompt, max_new_tokens, mode="mixed")
                est_cache_mbs.append(est_mb)

            elif method == "paged_attention":
                _, n_new, alloc_mb, used_mb, _ = self.generate_with_paged_attention(
                    prompt, max_new_tokens, block_size=block_size
                )
                est_cache_mbs.append(alloc_mb)

            elif method == "chunked_cache":
                _, n_new, est_mb = self.generate_with_chunked_cache(
                    prompt, max_new_tokens, chunk_size=chunk_size, keep_last=keep_last
                )
                est_cache_mbs.append(est_mb)

            elif method == "prefix_window":
                _, n_new = self.generate_with_prefix_window(
                    prompt, max_new_tokens=max_new_tokens, window_size=window_size, prefix_len=prefix_len
                )
                est_cache_mbs.append(float("nan"))

            elif method == "strided_cache":
                _, n_new = self.generate_with_strided_cache(
                    prompt, max_new_tokens=max_new_tokens, window_size=window_size, stride=stride, prefix_len=prefix_len
                )
                est_cache_mbs.append(float("nan"))

            elif method == "block_cache":
                _, n_new = self.generate_with_block_cache(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    window_size=window_size,
                    block_size=block_size,
                    keep_per_block=keep_per_block,
                    prefix_len=prefix_len,
                )
                est_cache_mbs.append(float("nan"))

            elif method == "budget_cache":
                _, n_new = self.generate_with_budget_cache(
                    prompt, max_new_tokens=max_new_tokens, window_size=window_size, old_budget=old_budget, prefix_len=prefix_len
                )
                est_cache_mbs.append(float("nan"))


            total_new_tokens += n_new

        # Calculate elapsed time using CUDA events or time.time()
        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        else:
            elapsed = time.time() - t0
            
        cpu_used = get_cpu_mem_mb() - start_cpu
        gpu_peak = get_gpu_peak_mb(self.device)
        tps = total_new_tokens / elapsed if elapsed > 0 else float("inf")

        # Average estimated cache size
        est_cache_mb_avg = float("nan")
        import math
        finite = [x for x in est_cache_mbs if isinstance(x, float) and not math.isnan(x)]
        if len(finite) > 0:
            est_cache_mb_avg = sum(finite) / len(finite)

        return {
            "method": method,
            "elapsed_sec": elapsed,
            "total_new_tokens": total_new_tokens,
            "tokens_per_sec": tps,
            "cpu_mem_used_mb": cpu_used,
            "gpu_peak_mb": gpu_peak,
            # "window_size": window_size if method == "sliding_window" else None,
            "window_size": window_size if method in ["sliding_window", "prefix_window", "strided_cache", "block_cache", "budget_cache"] else None,
            "block_size": block_size if method == "paged_attention" else None,
            "chunk_size": chunk_size if method == "chunked_cache" else None,
            "est_kv_cache_mb_avg": est_cache_mb_avg,
            "prefix_len": prefix_len if method in ["prefix_window", "strided_cache", "block_cache", "budget_cache"] else None,
            "stride": stride if method == "strided_cache" else None,
            "keep_per_block": keep_per_block if method == "block_cache" else None,
            "old_budget": old_budget if method == "budget_cache" else None,

        }
