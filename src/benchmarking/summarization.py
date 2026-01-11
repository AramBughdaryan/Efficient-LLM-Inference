"""Summarization benchmark interface."""

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..benchmarking import KVCacheBenchmarker
from ..datasets import SummarizationDataset
from ..evaluation import RougeEvaluator


class SummarizationBenchmark:
    """Unified interface for benchmarking summarization with different configurations.
    
    Supports benchmarking:
    - Different caching mechanisms (full, sliding window, quantized, etc.)
    - Different model modifications
    - Different attention variants
    - Quality metrics (ROUGE scores)
    - Performance metrics (speed, memory)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: SummarizationDataset,
        device: str = "cuda",
    ):
        """Initialize summarization benchmark.
        
        Args:
            model: HuggingFace model for summarization
            tokenizer: HuggingFace tokenizer
            dataset: Summarization dataset
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        
        self.benchmarker = KVCacheBenchmarker(model, tokenizer, device)
        self.rouge_evaluator = RougeEvaluator()
    
    def generate_summaries(
        self,
        articles: List[str],
        method: str = "full_cache",
        max_new_tokens: int = 128,
        instruction: str = "Summarize the following article:\n\n",
        **method_kwargs,
    ) -> Tuple[List[str], float, Dict]:
        """Generate summaries using specified method.
        
        Args:
            articles: List of articles to summarize
            method: Caching method to use
            max_new_tokens: Maximum tokens to generate
            instruction: Instruction text
            **method_kwargs: Additional arguments for the method
            
        Returns:
            Tuple of (summaries, elapsed_time, metrics)
        """
        # Create prompts
        prompts = self.dataset.create_prompts(
            articles,
            instruction=instruction,
            max_article_length=method_kwargs.get('max_article_length', None)
        )
        
        # Generate with timing
        if self.device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.time()
        
        summaries = []
        for prompt in prompts:
            if method == "no_cache":
                text, _ = self.benchmarker.generate_no_cache(prompt, max_new_tokens)
            elif method == "full_cache":
                text, _ = self.benchmarker.generate_with_cache(prompt, max_new_tokens)
            elif method == "sliding_window":
                window_size = method_kwargs.get('window_size', 256)
                text, _ = self.benchmarker.generate_with_sliding_window(
                    prompt, max_new_tokens, window_size=window_size
                )
            elif method.startswith("quant_"):
                mode = method.replace("quant_", "")
                text, _, _ = self.benchmarker.generate_with_quantized_kv(
                    prompt, max_new_tokens, mode=mode
                )
            elif method == "paged_attention":
                block_size = method_kwargs.get('block_size', 64)
                text, _, _, _, _ = self.benchmarker.generate_with_paged_attention(
                    prompt, max_new_tokens, block_size=block_size
                )
            elif method == "chunked_cache":
                chunk_size = method_kwargs.get('chunk_size', 64)
                keep_last = method_kwargs.get('keep_last', 256)
                text, _, _ = self.benchmarker.generate_with_chunked_cache(
                    prompt, max_new_tokens, chunk_size=chunk_size, keep_last=keep_last
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Extract only the generated summary (after the prompt)
            if "Summary:" in text:
                summary = text.split("Summary:")[-1].strip()
            else:
                summary = text[len(prompt):].strip()
            
            summaries.append(summary)
        
        # Calculate elapsed time
        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed = time.time() - t0
        
        # Collect metrics
        metrics = {
            "method": method,
            "num_samples": len(articles),
            "elapsed_sec": elapsed,
            "avg_time_per_sample": elapsed / len(articles),
        }
        
        return summaries, elapsed, metrics
    
    def evaluate_quality(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate summary quality using ROUGE.
        
        Args:
            predictions: Predicted summaries
            references: Reference summaries
            
        Returns:
            ROUGE scores dictionary
        """
        return self.rouge_evaluator.compute_rouge(predictions, references)
    
    def benchmark_configuration(
        self,
        method: str,
        num_samples: int = 10,
        max_new_tokens: int = 128,
        offset: int = 0,
        **method_kwargs,
    ) -> Dict:
        """Benchmark a specific configuration.
        
        Args:
            method: Caching method to use
            num_samples: Number of samples to benchmark
            max_new_tokens: Maximum tokens to generate
            offset: Starting index in dataset
            **method_kwargs: Additional method arguments
            
        Returns:
            Dictionary with comprehensive results
        """
        print(f"\nBenchmarking: {method}")
        print(f"Samples: {num_samples}, Max tokens: {max_new_tokens}")
        
        # Get samples
        articles, references = self.dataset.get_samples(num_samples, offset)
        
        # Generate summaries
        summaries, elapsed, metrics = self.generate_summaries(
            articles,
            method=method,
            max_new_tokens=max_new_tokens,
            **method_kwargs,
        )
        
        # Evaluate quality
        rouge_scores = self.evaluate_quality(summaries, references)
        
        # Combine all results
        results = {
            **metrics,
            "rouge1_f": rouge_scores['rouge1']['fmeasure'],
            "rouge2_f": rouge_scores['rouge2']['fmeasure'],
            "rougeL_f": rouge_scores['rougeL']['fmeasure'],
            "rouge_scores": rouge_scores,
        }
        
        print(f"  Elapsed: {elapsed:.2f}s ({metrics['avg_time_per_sample']:.2f}s/sample)")
        print(f"  ROUGE-1 F1: {rouge_scores['rouge1']['fmeasure']:.4f}")
        print(f"  ROUGE-2 F1: {rouge_scores['rouge2']['fmeasure']:.4f}")
        print(f"  ROUGE-L F1: {rouge_scores['rougeL']['fmeasure']:.4f}")
        
        return results
    
    def compare_methods(
        self,
        methods: List[str],
        num_samples: int = 10,
        max_new_tokens: int = 128,
        offset: int = 0,
        method_configs: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """Compare multiple caching methods.
        
        Args:
            methods: List of methods to compare
            num_samples: Number of samples per method
            max_new_tokens: Maximum tokens to generate
            offset: Starting index in dataset
            method_configs: Optional configurations per method
            
        Returns:
            DataFrame with comparison results
        """
        if method_configs is None:
            method_configs = {}
        
        results = []
        for method in methods:
            config = method_configs.get(method, {})
            result = self.benchmark_configuration(
                method=method,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                offset=offset,
                **config,
            )
            # Remove nested dict for DataFrame
            result_flat = {k: v for k, v in result.items() if k != 'rouge_scores'}
            results.append(result_flat)
        
        df = pd.DataFrame(results)
        return df
    
    def benchmark_with_variants(
        self,
        base_method: str = "full_cache",
        variants: Optional[List[Dict]] = None,
        num_samples: int = 10,
        max_new_tokens: int = 128,
    ) -> pd.DataFrame:
        """Benchmark with different configuration variants.
        
        Useful for testing different:
        - Window sizes
        - Block sizes
        - Quantization modes
        - Etc.
        
        Args:
            base_method: Base caching method
            variants: List of variant configurations
            num_samples: Number of samples
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            DataFrame with results for each variant
        """
        if variants is None:
            if base_method == "sliding_window":
                variants = [
                    {"window_size": 128},
                    {"window_size": 256},
                    {"window_size": 512},
                ]
            elif base_method == "paged_attention":
                variants = [
                    {"block_size": 32},
                    {"block_size": 64},
                    {"block_size": 128},
                ]
            else:
                variants = [{}]
        
        results = []
        for i, config in enumerate(variants):
            print(f"\n--- Variant {i+1}/{len(variants)}: {config} ---")
            result = self.benchmark_configuration(
                method=base_method,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                **config,
            )
            # Add variant info
            result_flat = {k: v for k, v in result.items() if k != 'rouge_scores'}
            result_flat['variant'] = str(config)
            results.append(result_flat)
        
        df = pd.DataFrame(results)
        return df
