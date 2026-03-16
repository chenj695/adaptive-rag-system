"""Evaluation metrics for comparing RAG retrieval strategies.

Key Metrics:
- Recall@K: What fraction of relevant chunks are retrieved
- Precision@K: What fraction of retrieved chunks are relevant
- NDCG: Normalized Discounted Cumulative Gain (ranking quality)
- MRR: Mean Reciprocal Rank (position of first relevant result)
- Latency: Response time
- Token Efficiency: Tokens used vs information gained
"""

import logging
import time
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a single retrieval operation."""
    query: str
    retrieved_ids: List[str]  # IDs of retrieved chunks/nodes
    retrieved_scores: List[float]  # Similarity scores
    relevant_ids: Set[str]  # Ground truth relevant IDs
    latency_ms: float
    strategy: str  # e.g., "flat", "raptor_multi_level", "raptor_leaf_only"
    tokens_used: int = 0
    
    @property
    def retrieved_set(self) -> Set[str]:
        return set(self.retrieved_ids)


@dataclass
class EvaluationMetrics:
    """Computed metrics for a retrieval strategy."""
    strategy: str
    
    # Ranking metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    
    # Efficiency metrics
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    avg_tokens_per_query: float = 0.0
    
    # Coverage metrics
    level_coverage: Dict[int, float] = field(default_factory=dict)  # For RAPTOR
    
    # Overall
    num_queries: int = 0


def calculate_recall(
    retrieved: Set[str],
    relevant: Set[str],
    k: Optional[int] = None
) -> float:
    """Calculate Recall@K.
    
    Recall = |Retrieved ∩ Relevant| / |Relevant|
    
    Args:
        retrieved: Set of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)
        k: If provided, only consider top-k retrieved
        
    Returns:
        Recall score between 0 and 1
    """
    if k:
        retrieved = set(list(retrieved)[:k])
    
    if not relevant:
        return 0.0
    
    intersection = len(retrieved & relevant)
    return intersection / len(relevant)


def calculate_precision(
    retrieved: Set[str],
    relevant: Set[str],
    k: Optional[int] = None
) -> float:
    """Calculate Precision@K.
    
    Precision = |Retrieved ∩ Relevant| / |Retrieved|
    
    Args:
        retrieved: Set of retrieved document IDs
        relevant: Set of relevant document IDs
        k: If provided, only consider top-k retrieved
        
    Returns:
        Precision score between 0 and 1
    """
    if k:
        retrieved = set(list(retrieved)[:k])
    
    if not retrieved:
        return 0.0
    
    intersection = len(retrieved & relevant)
    return intersection / len(retrieved)


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_ndcg(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    retrieved_scores: Optional[List[float]] = None,
    k: int = 10
) -> float:
    """Calculate NDCG@K (Normalized Discounted Cumulative Gain).
    
    NDCG measures ranking quality, giving higher scores to relevant
    documents that appear earlier in the results.
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        retrieved_scores: Similarity scores (optional)
        k: Cutoff for calculation
        
    Returns:
        NDCG score between 0 and 1
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    
    # Create relevance scores (1 if relevant, 0 if not)
    relevance = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:k]]
    
    if sum(relevance) == 0:
        return 0.0
    
    # Ideal ranking (all relevant docs first)
    ideal_relevance = sorted(relevance, reverse=True)
    
    try:
        return ndcg_score([ideal_relevance], [relevance])
    except Exception:
        # Manual calculation if sklearn fails
        return _manual_ndcg(relevance, ideal_relevance)


def _manual_ndcg(relevance: List[int], ideal_relevance: List[int]) -> float:
    """Manual NDCG calculation."""
    def dcg(scores):
        return sum((2 ** s - 1) / np.log2(i + 2) for i, s in enumerate(scores))
    
    actual_dcg = dcg(relevance)
    ideal_dcg = dcg(ideal_relevance)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def calculate_mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / rank_of_first_relevant
    If no relevant docs found, MRR = 0
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        
    Returns:
        MRR score between 0 and 1
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(
    results: List[RetrievalResult],
    k_values: List[int] = [1, 3, 5, 10]
) -> EvaluationMetrics:
    """Evaluate a set of retrieval results.
    
    Args:
        results: List of RetrievalResult objects
        k_values: K values for @K metrics
        
    Returns:
        EvaluationMetrics with computed scores
    """
    if not results:
        return EvaluationMetrics(strategy="unknown")
    
    strategy = results[0].strategy
    metrics = EvaluationMetrics(strategy=strategy, num_queries=len(results))
    
    # Collect scores for each metric
    recalls = defaultdict(list)
    precisions = defaultdict(list)
    f1s = defaultdict(list)
    ndcgs = defaultdict(list)
    mrrs = []
    latencies = []
    tokens = []
    
    for result in results:
        # Latency
        latencies.append(result.latency_ms)
        tokens.append(result.tokens_used)
        
        # MRR
        mrrs.append(calculate_mrr(result.retrieved_ids, result.relevant_ids))
        
        # @K metrics
        for k in k_values:
            recall = calculate_recall(
                result.retrieved_set, result.relevant_ids, k=k
            )
            precision = calculate_precision(
                result.retrieved_set, result.relevant_ids, k=k
            )
            
            recalls[k].append(recall)
            precisions[k].append(precision)
            f1s[k].append(calculate_f1(precision, recall))
            
            # NDCG
            ndcg = calculate_ndcg(
                result.retrieved_ids,
                result.relevant_ids,
                result.retrieved_scores,
                k=k
            )
            ndcgs[k].append(ndcg)
    
    # Average scores
    for k in k_values:
        metrics.recall_at_k[k] = np.mean(recalls[k])
        metrics.precision_at_k[k] = np.mean(precisions[k])
        metrics.f1_at_k[k] = np.mean(f1s[k])
        metrics.ndcg_at_k[k] = np.mean(ndcgs[k])
    
    metrics.mrr = np.mean(mrrs)
    metrics.avg_latency_ms = np.mean(latencies)
    metrics.total_tokens = sum(tokens)
    metrics.avg_tokens_per_query = np.mean(tokens)
    
    return metrics


def compare_strategies(
    metrics_list: List[EvaluationMetrics]
) -> Dict:
    """Compare multiple retrieval strategies.
    
    Args:
        metrics_list: List of EvaluationMetrics for different strategies
        
    Returns:
        Comparison dictionary with best strategy per metric
    """
    if not metrics_list:
        return {}
    
    comparison = {
        "strategies": [m.strategy for m in metrics_list],
        "best_by_metric": {},
        "detailed_scores": {}
    }
    
    # Find best for each metric
    metrics_to_compare = [
        ("recall@5", lambda m: m.recall_at_k.get(5, 0)),
        ("precision@5", lambda m: m.precision_at_k.get(5, 0)),
        ("f1@5", lambda m: m.f1_at_k.get(5, 0)),
        ("ndcg@5", lambda m: m.ndcg_at_k.get(5, 0)),
        ("mrr", lambda m: m.mrr),
        ("latency_ms", lambda m: -m.avg_latency_ms),  # Lower is better
    ]
    
    for metric_name, metric_fn in metrics_to_compare:
        best_strategy = max(metrics_list, key=metric_fn)
        best_score = metric_fn(best_strategy)
        comparison["best_by_metric"][metric_name] = {
            "strategy": best_strategy.strategy,
            "score": abs(best_score) if "latency" in metric_name else best_score
        }
    
    # Detailed scores
    for m in metrics_list:
        comparison["detailed_scores"][m.strategy] = {
            "recall@5": m.recall_at_k.get(5, 0),
            "precision@5": m.precision_at_k.get(5, 0),
            "f1@5": m.f1_at_k.get(5, 0),
            "ndcg@5": m.ndcg_at_k.get(5, 0),
            "mrr": m.mrr,
            "latency_ms": m.avg_latency_ms,
            "tokens_per_query": m.avg_tokens_per_query
        }
    
    return comparison


def generate_evaluation_report(
    metrics: EvaluationMetrics,
    output_path: Optional[str] = None
) -> str:
    """Generate a formatted evaluation report.
    
    Args:
        metrics: EvaluationMetrics to report
        output_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        f"EVALUATION REPORT: {metrics.strategy}",
        "=" * 60,
        f"\nNumber of queries: {metrics.num_queries}",
        "\n--- RANKING METRICS ---",
    ]
    
    # @K metrics table
    lines.append("\n{:<15} {:>10} {:>10} {:>10} {:>10}".format(
        "@K", "Recall", "Precision", "F1", "NDCG"
    ))
    lines.append("-" * 60)
    
    for k in sorted(metrics.recall_at_k.keys()):
        lines.append("{:<15} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
            f"@{k}",
            metrics.recall_at_k.get(k, 0),
            metrics.precision_at_k.get(k, 0),
            metrics.f1_at_k.get(k, 0),
            metrics.ndcg_at_k.get(k, 0)
        ))
    
    lines.extend([
        f"\nMRR: {metrics.mrr:.3f}",
        "\n--- EFFICIENCY METRICS ---",
        f"Average latency: {metrics.avg_latency_ms:.1f} ms",
        f"Total tokens: {metrics.total_tokens:,}",
        f"Avg tokens/query: {metrics.avg_tokens_per_query:.0f}",
    ])
    
    if metrics.level_coverage:
        lines.extend([
            "\n--- LEVEL COVERAGE (RAPTOR) ---",
        ])
        for level, coverage in sorted(metrics.level_coverage.items()):
            lines.append(f"Level {level}: {coverage:.2%}")
    
    lines.append("\n" + "=" * 60)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")
    
    return report


def create_ground_truth_from_annotations(
    annotations: List[Dict],
    query_key: str = "question",
    relevant_key: str = "relevant_chunks"
) -> Dict[str, Set[str]]:
    """Create ground truth dictionary from manual annotations.
    
    Args:
        annotations: List of dicts with query and relevant chunk IDs
        query_key: Key for query text
        relevant_key: Key for relevant chunk IDs
        
    Returns:
        Dict mapping queries to sets of relevant IDs
    """
    ground_truth = {}
    
    for ann in annotations:
        query = ann[query_key]
        relevant = set(ann[relevant_key])
        ground_truth[query] = relevant
    
    return ground_truth


def calculate_statistical_significance(
    results_a: List[RetrievalResult],
    results_b: List[RetrievalResult],
    metric: str = "recall@5"
) -> Tuple[float, bool]:
    """Calculate statistical significance between two strategies.
    
    Uses paired t-test to determine if difference is significant.
    
    Args:
        results_a: Results from strategy A
        results_b: Results from strategy B
        metric: Which metric to compare
        
    Returns:
        Tuple of (p_value, is_significant)
    """
    from scipy import stats
    
    # Extract scores
    if metric == "recall@5":
        scores_a = [calculate_recall(r.retrieved_set, r.relevant_ids, k=5) 
                    for r in results_a]
        scores_b = [calculate_recall(r.retrieved_set, r.relevant_ids, k=5) 
                    for r in results_b]
    elif metric == "mrr":
        scores_a = [calculate_mrr(r.retrieved_ids, r.relevant_ids) 
                    for r in results_a]
        scores_b = [calculate_mrr(r.retrieved_ids, r.relevant_ids) 
                    for r in results_b]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Significant if p < 0.05
    is_significant = p_value < 0.05
    
    return p_value, is_significant
