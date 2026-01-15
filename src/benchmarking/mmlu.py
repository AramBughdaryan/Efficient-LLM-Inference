"""MMLU benchmark interface."""

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..benchmarking import KVCacheBenchmarker
from ..datasets import MMLUDataset
from ..evaluation import AccuracyEvaluator


class MMLUBenchmark:
    """Unified interface for benchmarking MMLU with different configurations.
    
    Supports benchmarking:
    - Different caching mechanisms (full, sliding window, quantized, etc.)
    - Different model modifications
    - Different attention variants
    - Quality metrics (accuracy)
    - Performance metrics (speed, memory, timing)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: MMLUDataset,
        device: str = "cuda",
    ):
        """Initialize MMLU benchmark.
        
        Args:
            model: HuggingFace model for question answering
            tokenizer: HuggingFace tokenizer
            dataset: MMLU dataset
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        
        self.benchmarker = KVCacheBenchmarker(model, tokenizer, device)
        self.accuracy_evaluator = AccuracyEvaluator()
    
    def generate_answers(
        self,
        questions: List[str],
        choices_list: List[List[str]],
        method: str = "full_cache",
        max_new_tokens: int = 10,
        instruction: str = "The following are multiple choice questions (with answers).\n\n",
        **method_kwargs,
    ) -> Tuple[List[str], float, Dict]:
        """Generate answers using specified method.
        
        Args:
            questions: List of questions
            choices_list: List of choice lists (each with 4 options)
            method: Caching method to use
            max_new_tokens: Maximum tokens to generate
            instruction: Instruction text
            **method_kwargs: Additional arguments for the method
            
        Returns:
            Tuple of (answers, elapsed_time, metrics)
        """
        # Create prompts
        prompts = self.dataset.create_prompts(
            questions,
            choices_list,
            instruction=instruction,
        )
        
        # Generate with timing
        if self.device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.time()
        
        answers = []
        for i, prompt in enumerate(prompts):
            if method == "no_cache":
                text, _ = self.benchmarker.generate_no_cache(prompt, max_new_tokens)
            elif method == "full_cache":
                text, _ = self.benchmarker.generate_with_cache(prompt, max_new_tokens)
            elif method == "sliding_window":
                window_size = method_kwargs.get('window_size', 256)
                text, _ = self.benchmarker.generate_with_sliding_window(
                    prompt, max_new_tokens, window_size=window_size
                )
            elif method == "prefix_window":
                window_size = method_kwargs.get('window_size', 256)
                prefix_len = method_kwargs.get('prefix_len', 32)
                text, _ = self.benchmarker.generate_with_prefix_window(
                    prompt, max_new_tokens, window_size=window_size, prefix_len=prefix_len
                )
            elif method == "strided_cache":
                window_size = method_kwargs.get('window_size', 256)
                stride = method_kwargs.get('stride', 4)
                prefix_len = method_kwargs.get('prefix_len', 0)
                text, _ = self.benchmarker.generate_with_strided_cache(
                    prompt, max_new_tokens, window_size=window_size, stride=stride, prefix_len=prefix_len
                )
            elif method == "block_cache":
                window_size = method_kwargs.get('window_size', 256)
                block_size = method_kwargs.get('block_size', 64)
                keep_per_block = method_kwargs.get('keep_per_block', 8)
                prefix_len = method_kwargs.get('prefix_len', 0)
                text, _ = self.benchmarker.generate_with_block_cache(
                    prompt, max_new_tokens, window_size=window_size, block_size=block_size,
                    keep_per_block=keep_per_block, prefix_len=prefix_len
                )
            elif method == "budget_cache":
                window_size = method_kwargs.get('window_size', 256)
                old_budget = method_kwargs.get('old_budget', 64)
                prefix_len = method_kwargs.get('prefix_len', 0)
                text, _ = self.benchmarker.generate_with_budget_cache(
                    prompt, max_new_tokens, window_size=window_size, old_budget=old_budget, prefix_len=prefix_len
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
            
            # Extract only the generated answer (after the prompt)
            # The answer should be just a few tokens, so we take the generated part
            prompt_len = len(prompt)
            if len(text) > prompt_len:
                answer = text[prompt_len:].strip()
            else:
                answer = text.strip()
            
            # Debug: print first few examples to verify different methods produce different outputs
            if i < 3:  # First 3 samples for debugging
                print(f"    [DEBUG] Sample {i} - Method: {method}")
                print(f"      Generated text: '{answer[:100]}'")
                extracted = self.accuracy_evaluator.extract_answer(answer)
                print(f"      Extracted answer: '{extracted}'")
            
            answers.append(answer)
        
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
            "num_samples": len(questions),
            "elapsed_sec": elapsed,
            "avg_time_per_sample": elapsed / len(questions) if len(questions) > 0 else 0.0,
        }
        
        return answers, elapsed, metrics
    
    def evaluate_quality(
        self,
        predictions: List[str],
        references: List[int],
    ) -> Dict:
        """Evaluate answer quality using accuracy.
        
        Args:
            predictions: Predicted answer texts
            references: Reference answer indices (0=A, 1=B, 2=C, 3=D)
            
        Returns:
            Accuracy metrics dictionary
        """
        return self.accuracy_evaluator.compute_accuracy_by_choice(predictions, references)
    
    def benchmark_configuration(
        self,
        method: str,
        num_samples: int = 10,
        max_new_tokens: int = 10,
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
            Dictionary with comprehensive results including timing
        """
        print(f"\nBenchmarking: {method}")
        print(f"Samples: {num_samples}, Max tokens: {max_new_tokens}")
        
        # Get samples
        questions, choices_list, answer_indices = self.dataset.get_samples(num_samples, offset)
        
        # Generate answers with timing
        answers, elapsed, metrics = self.generate_answers(
            questions,
            choices_list,
            method=method,
            max_new_tokens=max_new_tokens,
            **method_kwargs,
        )
        
        # Evaluate quality
        accuracy_results = self.evaluate_quality(answers, answer_indices)
        
        # Combine all results
        results = {
            **metrics,
            "accuracy": accuracy_results["overall_accuracy"],
            "choice_accuracy": accuracy_results["choice_accuracy"],
            "choice_counts": accuracy_results["choice_counts"],
            "correctness": accuracy_results["correctness"],
            "extracted_answers": accuracy_results["extracted_answers"],
        }
        
        print(f"  Elapsed: {elapsed:.2f}s ({metrics['avg_time_per_sample']:.4f}s/sample)")
        print(f"  Accuracy: {accuracy_results['overall_accuracy']:.4f}")
        print(f"  Correct: {sum(accuracy_results['correctness'])}/{len(accuracy_results['correctness'])}")
        
        # Debug: Show extracted answers for first few samples
        if len(accuracy_results['extracted_answers']) > 0:
            sample_answers = accuracy_results['extracted_answers'][:10]
            print(f"  First 10 extracted answers: {sample_answers}")
            # Count unique answers
            unique_answers = set(sample_answers)
            print(f"  Unique answers in first 10: {unique_answers}")
        
        return results
    
    def compare_methods(
        self,
        methods: List[str],
        num_samples: int = 10,
        max_new_tokens: int = 10,
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
            DataFrame with comparison results including timing
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
            # Flatten nested dicts for DataFrame
            result_flat = {
                k: v for k, v in result.items()
                if k not in ['choice_accuracy', 'choice_counts', 'correctness', 'extracted_answers']
            }
            # Add choice accuracy as separate columns
            if 'choice_accuracy' in result:
                for choice, acc in result['choice_accuracy'].items():
                    result_flat[f'accuracy_{choice}'] = acc
            # Keep extracted_answers for comparison
            if 'extracted_answers' in result:
                result_flat['extracted_answers'] = result['extracted_answers']
            results.append(result_flat)
        
        df = pd.DataFrame(results)
        
        # Debug: Compare extracted answers across methods for first few samples
        if len(results) > 1:
            print("\n" + "=" * 80)
            print("COMPARING EXTRACTED ANSWERS ACROSS METHODS (first 10 samples)")
            print("=" * 80)
            for result in results:
                method = result.get('method', 'unknown')
                extracted = result.get('extracted_answers', [])
                if extracted:
                    print(f"{method}: {extracted[:10]}")
            
            # Check if all methods produce the same answers
            if len(results) > 0 and 'extracted_answers' in results[0]:
                first_method_answers = results[0].get('extracted_answers', [])[:10]
                all_same = all(
                    r.get('extracted_answers', [])[:10] == first_method_answers
                    for r in results[1:] if 'extracted_answers' in r
                )
                if all_same:
                    print("\n⚠️  WARNING: All methods produced identical extracted answers for first 10 samples!")
                    print("   This is EXPECTED if:")
                    print("   - Prompts are processed identically (all methods do full forward pass on prompt)")
                    print("   - Answers are single tokens (A, B, C, D)")
                    print("   - Model uses deterministic generation (argmax)")
                    print("   Differences would appear with longer prompts that get truncated differently.")
                else:
                    print("\n✓ Methods produced different answers (cache differences affected generation)")
        
        # Remove extracted_answers from DataFrame (it's a list, not suitable for DataFrame)
        df_clean = df.drop(columns=['extracted_answers'], errors='ignore')
        return df_clean
    
    def benchmark_with_variants(
        self,
        base_method: str = "full_cache",
        variants: Optional[List[Dict]] = None,
        num_samples: int = 10,
        max_new_tokens: int = 10,
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
            DataFrame with results for each variant including timing
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
            result_flat = {
                k: v for k, v in result.items()
                if k not in ['choice_accuracy', 'choice_counts', 'correctness', 'extracted_answers']
            }
            if 'choice_accuracy' in result:
                for choice, acc in result['choice_accuracy'].items():
                    result_flat[f'accuracy_{choice}'] = acc
            result_flat['variant'] = str(config)
            results.append(result_flat)
        
        df = pd.DataFrame(results)
        return df
