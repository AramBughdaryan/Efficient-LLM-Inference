"""Dataset loaders for benchmarking."""

from typing import List, Optional, Tuple

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
            self.dataset = load_dataset("cnn_dailymail", self.version, split=self.split)
            self.article_key = "article"
            self.summary_key = "highlights"

        elif self.dataset_name == "xsum":
            self.dataset = load_dataset("xsum", split=self.split)
            self.article_key = "document"
            self.summary_key = "summary"

        elif self.dataset_name == "samsum":
            self.dataset = load_dataset("samsum", split=self.split)
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


class MMLUDataset:
    """Wrapper for MMLU (Massive Multitask Language Understanding) dataset.
    
    MMLU is a benchmark for evaluating language models on multiple-choice
    questions across 57 tasks covering STEM, humanities, social sciences, and more.
    """

    def __init__(
        self,
        split: str = "test",
        subject: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize MMLU dataset loader.

        Args:
            split: Dataset split ("dev", "validation", "test")
            subject: Optional subject to filter by (e.g., "abstract_algebra", "anatomy")
                     If None, loads all subjects
            max_samples: Maximum number of samples to load (None = all)
        """
        self.split = split
        self.subject = subject
        self.max_samples = max_samples

        self._load_dataset()

    def _load_dataset(self):
        """Load the MMLU dataset from HuggingFace."""
        print(f"Loading MMLU dataset ({self.split} split)...")
        
        if self.subject:
            # Load specific subject
            try:
                self.dataset = load_dataset("cais/mmlu", self.subject, split=self.split)
                print(f"Loaded subject: {self.subject}")
            except Exception as e:
                raise ValueError(f"Failed to load MMLU subject '{self.subject}': {e}")
        else:
            # Load all subjects (combines all tasks)
            try:
                # Load validation split (dev) or test split
                if self.split == "dev" or self.split == "validation":
                    split_name = "validation"
                else:
                    split_name = "test"
                
                # Load a representative subset - MMLU has many subjects
                # We'll load from a few common subjects
                common_subjects = [
                    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
                    "clinical_knowledge", "college_biology", "college_chemistry",
                    "college_computer_science", "college_mathematics", "college_physics",
                    "computer_security", "conceptual_physics", "econometrics",
                    "electrical_engineering", "elementary_mathematics", "formal_logic",
                    "global_facts", "high_school_biology", "high_school_chemistry",
                    "high_school_computer_science", "high_school_european_history",
                    "high_school_geography", "high_school_government_and_politics",
                    "high_school_macroeconomics", "high_school_mathematics",
                    "high_school_microeconomics", "high_school_physics",
                    "high_school_psychology", "high_school_statistics",
                    "high_school_us_history", "high_school_world_history",
                    "human_aging", "human_sexuality", "international_law",
                    "jurisprudence", "logical_fallacies", "machine_learning",
                    "management", "marketing", "medical_genetics", "miscellaneous",
                    "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
                    "prehistory", "professional_accounting", "professional_law",
                    "professional_medicine", "professional_psychology", "public_relations",
                    "security_studies", "sociology", "us_foreign_policy", "virology",
                    "world_religions"
                ]
                
                # Load first few subjects as default
                datasets_list = []
                for subj in common_subjects[:5]:  # Load first 5 subjects by default
                    try:
                        ds = load_dataset("cais/mmlu", subj, split=split_name)
                        datasets_list.append(ds)
                    except Exception:
                        continue
                
                if not datasets_list:
                    raise ValueError("Failed to load any MMLU subjects")
                
                # Concatenate all subjects
                from datasets import concatenate_datasets
                self.dataset = concatenate_datasets(datasets_list)
                print(f"Loaded {len(common_subjects[:5])} subjects")
            except Exception as e:
                raise ValueError(f"Failed to load MMLU dataset: {e}")

        # Limit samples if specified
        if self.max_samples is not None:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, List[str], int]:
        """Get question, choices, and correct answer.

        Args:
            idx: Sample index

        Returns:
            Tuple of (question, choices_list, correct_answer_index)
        """
        sample = self.dataset[idx]
        question = sample["question"]
        
        # Handle different MMLU dataset formats
        if "choices" in sample:
            # New format: choices is a dictionary or list
            choices_dict = sample["choices"]
            if isinstance(choices_dict, dict):
                # Dictionary format: {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
                choices = [choices_dict["A"], choices_dict["B"], choices_dict["C"], choices_dict["D"]]
            elif isinstance(choices_dict, list):
                # List format: ['...', '...', '...', '...']
                choices = choices_dict
            else:
                raise ValueError(f"Unexpected choices format: {type(choices_dict)}")
        else:
            # Old format: separate keys 'A', 'B', 'C', 'D'
            choices = [sample["A"], sample["B"], sample["C"], sample["D"]]
        
        answer_idx = sample["answer"]  # 'A', 'B', 'C', or 'D'
        return question, choices, answer_idx

    def get_batch(self, indices: List[int]) -> Tuple[List[str], List[List[str]], List[int]]:
        """Get batch of questions, choices, and answers.

        Args:
            indices: List of sample indices

        Returns:
            Tuple of (questions, choices_list, answer_indices)
        """
        samples = self.dataset.select(indices)
        questions = [s["question"] for s in samples]
        
        # Handle different MMLU dataset formats
        choices_list = []
        for s in samples:
            if "choices" in s:
                # New format: choices is a dictionary or list
                choices_dict = s["choices"]
                if isinstance(choices_dict, dict):
                    # Dictionary format: {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
                    choices_list.append([choices_dict["A"], choices_dict["B"], choices_dict["C"], choices_dict["D"]])
                elif isinstance(choices_dict, list):
                    # List format: ['...', '...', '...', '...']
                    choices_list.append(choices_dict)
                else:
                    raise ValueError(f"Unexpected choices format: {type(choices_dict)}")
            else:
                # Old format: separate keys 'A', 'B', 'C', 'D'
                choices_list.append([s["A"], s["B"], s["C"], s["D"]])
        
        answers = [s["answer"] for s in samples]
        print("answers", answers)
        # answer_indices = [ord(a) - ord('A') for a in answers]
        return questions, choices_list, answers

    def get_samples(self, n: int, offset: int = 0) -> Tuple[List[str], List[List[str]], List[int]]:
        """Get n samples starting from offset.

        Args:
            n: Number of samples
            offset: Starting index

        Returns:
            Tuple of (questions, choices_list, answer_indices)
        """
        end = min(offset + n, len(self.dataset))
        indices = list(range(offset, end))
        return self.get_batch(indices)

    def create_prompts(
        self,
        questions: List[str],
        choices_list: List[List[str]],
        instruction: str = "The following are multiple choice questions (with answers).\n\n",
    ) -> List[str]:
        """Create prompts for multiple-choice questions.

        Args:
            questions: List of questions
            choices_list: List of choice lists (each with 4 options)
            instruction: Instruction text to prepend

        Returns:
            List of formatted prompts
        """
        prompts = []
        for question, choices in zip(questions, choices_list):
            prompt = f"{instruction}Question: {question}\n\n"
            prompt += "Choices:\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(ord('A') + i)}. {choice}\n"
            prompt += "\nAnswer:"
            prompts.append(prompt)
        return prompts


def load_mmlu(
    split: str = "test",
    subject: Optional[str] = None,
    max_samples: Optional[int] = 100,
) -> MMLUDataset:
    """Load MMLU dataset.

    Args:
        split: Dataset split ("dev", "validation", "test")
        subject: Optional subject to filter by
        max_samples: Maximum samples to load

    Returns:
        MMLUDataset instance
    """
    return MMLUDataset(
        split=split,
        subject=subject,
        max_samples=max_samples,
    )
