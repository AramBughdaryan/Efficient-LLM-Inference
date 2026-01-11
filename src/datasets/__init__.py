"""Dataset loaders for benchmarking."""

from typing import Dict, List, Optional, Tuple

from datasets import load_dataset


class SummarizationDataset:
    """Wrapper for summarization datasets.
    
    Supports:
    - CNN/DailyMail
    - XSum
    - SAMSum
    """
    
    def __init__(
        self,
        dataset_name: str = "cnn_dailymail",
        version: str = "3.0.0",
        split: str = "test",
        max_samples: Optional[int] = None,
    ):
        """Initialize dataset loader.
        
        Args:
            dataset_name: Name of dataset ("cnn_dailymail", "xsum", "samsum")
            version: Dataset version (for CNN/DailyMail)
            split: Dataset split ("train", "validation", "test")
            max_samples: Maximum number of samples to load (None = all)
        """
        self.dataset_name = dataset_name
        self.version = version
        self.split = split
        self.max_samples = max_samples
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from HuggingFace."""
        print(f"Loading {self.dataset_name} ({self.split} split)...")
        
        if self.dataset_name == "cnn_dailymail":
            self.dataset = load_dataset(
                "cnn_dailymail",
                self.version,
                split=self.split
            )
            self.article_key = "article"
            self.summary_key = "highlights"
            
        elif self.dataset_name == "xsum":
            self.dataset = load_dataset(
                "xsum",
                split=self.split
            )
            self.article_key = "document"
            self.summary_key = "summary"
            
        elif self.dataset_name == "samsum":
            self.dataset = load_dataset(
                "samsum",
                split=self.split
            )
            self.article_key = "dialogue"
            self.summary_key = "summary"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Limit samples if specified
        if self.max_samples is not None:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get article and reference summary.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (article, reference_summary)
        """
        sample = self.dataset[idx]
        return sample[self.article_key], sample[self.summary_key]
    
    def get_batch(self, indices: List[int]) -> Tuple[List[str], List[str]]:
        """Get batch of articles and summaries.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (articles, reference_summaries)
        """
        samples = self.dataset.select(indices)
        articles = [s[self.article_key] for s in samples]
        summaries = [s[self.summary_key] for s in samples]
        return articles, summaries
    
    def get_samples(self, n: int, offset: int = 0) -> Tuple[List[str], List[str]]:
        """Get n samples starting from offset.
        
        Args:
            n: Number of samples
            offset: Starting index
            
        Returns:
            Tuple of (articles, reference_summaries)
        """
        end = min(offset + n, len(self.dataset))
        indices = list(range(offset, end))
        return self.get_batch(indices)
    
    def create_prompts(
        self,
        articles: List[str],
        instruction: str = "Summarize the following article:\n\n",
        max_article_length: Optional[int] = None,
    ) -> List[str]:
        """Create prompts for summarization.
        
        Args:
            articles: List of articles
            instruction: Instruction text to prepend
            max_article_length: Maximum article length in characters
            
        Returns:
            List of formatted prompts
        """
        prompts = []
        for article in articles:
            if max_article_length is not None:
                article = article[:max_article_length]
            prompts.append(f"{instruction}{article}\n\nSummary:")
        return prompts


def load_cnn_dailymail(
    split: str = "test",
    max_samples: Optional[int] = 100,
) -> SummarizationDataset:
    """Load CNN/DailyMail dataset.
    
    Args:
        split: Dataset split
        max_samples: Maximum samples to load
        
    Returns:
        SummarizationDataset instance
    """
    return SummarizationDataset(
        dataset_name="cnn_dailymail",
        version="3.0.0",
        split=split,
        max_samples=max_samples,
    )


def load_xsum(
    split: str = "test",
    max_samples: Optional[int] = 100,
) -> SummarizationDataset:
    """Load XSum dataset.
    
    Args:
        split: Dataset split
        max_samples: Maximum samples to load
        
    Returns:
        SummarizationDataset instance
    """
    return SummarizationDataset(
        dataset_name="xsum",
        split=split,
        max_samples=max_samples,
    )


def load_samsum(
    split: str = "test",
    max_samples: Optional[int] = 100,
) -> SummarizationDataset:
    """Load SAMSum dataset.
    
    Args:
        split: Dataset split
        max_samples: Maximum samples to load
        
    Returns:
        SummarizationDataset instance
    """
    return SummarizationDataset(
        dataset_name="samsum",
        split=split,
        max_samples=max_samples,
    )
