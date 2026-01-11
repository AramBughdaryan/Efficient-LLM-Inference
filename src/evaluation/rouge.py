"""ROUGE metrics for summarization evaluation."""

from typing import Dict, List

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")


class RougeEvaluator:
    """ROUGE metric evaluator for summarization."""
    
    def __init__(self, rouge_types: List[str] = None):
        """Initialize ROUGE evaluator.
        
        Args:
            rouge_types: List of ROUGE types to compute (default: rouge1, rouge2, rougeL)
        """
        if not ROUGE_AVAILABLE:
            raise ImportError(
                "rouge-score package not installed. "
                "Install with: pip install rouge-score"
            )
        
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']
        
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE scores.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        assert len(predictions) == len(references), \
            f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
        
        # Compute scores for each pair
        all_scores = {rouge_type: {'precision': [], 'recall': [], 'fmeasure': []} 
                      for rouge_type in self.rouge_types}
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                all_scores[rouge_type]['precision'].append(scores[rouge_type].precision)
                all_scores[rouge_type]['recall'].append(scores[rouge_type].recall)
                all_scores[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)
        
        # Compute averages
        avg_scores = {}
        for rouge_type in self.rouge_types:
            avg_scores[rouge_type] = {
                'precision': sum(all_scores[rouge_type]['precision']) / len(predictions),
                'recall': sum(all_scores[rouge_type]['recall']) / len(predictions),
                'fmeasure': sum(all_scores[rouge_type]['fmeasure']) / len(predictions),
            }
        
        return avg_scores
    
    def format_scores(self, scores: Dict[str, Dict[str, float]]) -> str:
        """Format ROUGE scores for display.
        
        Args:
            scores: ROUGE scores dictionary
            
        Returns:
            Formatted string
        """
        lines = []
        for rouge_type, metrics in scores.items():
            lines.append(f"{rouge_type.upper()}:")
            lines.append(f"  Precision: {metrics['precision']:.4f}")
            lines.append(f"  Recall:    {metrics['recall']:.4f}")
            lines.append(f"  F-measure: {metrics['fmeasure']:.4f}")
        return "\n".join(lines)
