"""Evaluation runner for comparing RAG strategies."""

import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass
import numpy as np

from ..pipeline import Pipeline
from ..raptor.tree_builder import RaptorTreeBuilder
from ..raptor.retriever import RaptorRetriever
from .metrics import (
    RetrievalResult, EvaluationMetrics, evaluate_retrieval,
    compare_strategies, generate_evaluation_report, calculate_statistical_significance
)

logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    """A test query with ground truth."""
    query: str
    relevant_chunk_ids: Set[str]  # Ground truth
    expected_answer: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard


class RAGEvaluator:
    """Evaluate and compare RAG retrieval strategies."""
    
    def __init__(
        self,
        llm_client,
        embedding_model: str = "text-embedding-3-large",
        output_dir: Path = None
    ):
        """Initialize evaluator.
        
        Args:
            llm_client: OpenAI client
            embedding_model: Embedding model name
            output_dir: Where to save evaluation results
        """
        self.llm = llm_client
        self.embedding_model = embedding_model
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.strategies = {}
        self.results = {}
    
    def register_strategy(
        self,
        name: str,
        retriever_fn: Callable[[str], tuple[List[Dict], float]],
        description: str = ""
    ):
        """Register a retrieval strategy.
        
        Args:
            name: Strategy name (e.g., "faiss", "raptor_multi_level")
            retriever_fn: Function that takes query and returns (results, latency_ms)
            description: Human-readable description
        """
        self.strategies[name] = {
            'fn': retriever_fn,
            'description': description
        }
        logger.info(f"Registered strategy: {name}")
    
    def run_evaluation(
        self,
        test_queries: List[TestQuery],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, EvaluationMetrics]:
        """Run evaluation for all registered strategies.
        
        Args:
            test_queries: List of test queries with ground truth
            k_values: K values for @K metrics
            
        Returns:
            Dict mapping strategy names to EvaluationMetrics
        """
        all_metrics = {}
        
        for strategy_name, strategy_info in self.strategies.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {strategy_name}")
            logger.info(f"{'='*60}")
            
            results = []
            
            for i, test_query in enumerate(test_queries):
                logger.info(f"  Query {i+1}/{len(test_queries)}: {test_query.query[:50]}...")
                
                # Run retrieval
                try:
                    retrieved_items, latency_ms = strategy_info['fn'](test_query.query)
                    
                    # Extract IDs and scores
                    retrieved_ids = [item.get('id', item.get('node_id', str(i))) 
                                     for i, item in enumerate(retrieved_items)]
                    retrieved_scores = [item.get('score', item.get('distance', 0)) 
                                        for item in retrieved_items]
                    
                    result = RetrievalResult(
                        query=test_query.query,
                        retrieved_ids=retrieved_ids,
                        retrieved_scores=retrieved_scores,
                        relevant_ids=test_query.relevant_chunk_ids,
                        latency_ms=latency_ms,
                        strategy=strategy_name,
                        tokens_used=0  # Would track if available
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    continue
            
            # Compute metrics
            metrics = evaluate_retrieval(results, k_values)
            all_metrics[strategy_name] = metrics
            
            # Save results
            self.results[strategy_name] = results
            
            # Generate report
            report = generate_evaluation_report(metrics)
            print(report)
            
            report_path = self.output_dir / f"{strategy_name}_report.txt"
            generate_evaluation_report(metrics, str(report_path))
        
        return all_metrics
    
    def generate_comparison_report(
        self,
        metrics: Dict[str, EvaluationMetrics]
    ) -> str:
        """Generate comparison report across strategies."""
        comparison = compare_strategies(list(metrics.values()))
        
        lines = [
            "=" * 80,
            "STRATEGY COMPARISON REPORT",
            "=" * 80,
            "\nBEST STRATEGY BY METRIC:",
        ]
        
        for metric, info in comparison['best_by_metric'].items():
            lines.append(f"  {metric:<20} {info['strategy']:<25} ({info['score']:.3f})")
        
        lines.extend([
            "\n" + "=" * 80,
            "DETAILED SCORES:",
            "=" * 80,
        ])
        
        # Table header
        lines.append("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Strategy", "Recall@5", "Prec@5", "F1@5", "NDCG@5", "MRR"
        ))
        lines.append("-" * 80)
        
        # Table rows
        for strategy, scores in comparison['detailed_scores'].items():
            lines.append("{:<25} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                strategy[:25],
                scores['recall@5'],
                scores['precision@5'],
                scores['f1@5'],
                scores['ndcg@5'],
                scores['mrr']
            ))
        
        lines.extend([
            "\n" + "=" * 80,
            "EFFICIENCY COMPARISON:",
            "=" * 80,
        ])
        
        lines.append("\n{:<25} {:>15} {:>20}".format(
            "Strategy", "Latency (ms)", "Tokens/Query"
        ))
        lines.append("-" * 80)
        
        for strategy, scores in comparison['detailed_scores'].items():
            lines.append("{:<25} {:>15.1f} {:>20.0f}".format(
                strategy[:25],
                scores['latency_ms'],
                scores['tokens_per_query']
            ))
        
        lines.append("\n" + "=" * 80)
        
        report = "\n".join(lines)
        
        # Save to file
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Comparison report saved to {report_path}")
        
        return report


def create_test_queries_from_qa_pairs(
    qa_pairs: List[Dict],
    chunked_reports_dir: Path,
    llm_client
) -> List[TestQuery]:
    """Create test queries from Q&A pairs with automatic ground truth.
    
    Args:
        qa_pairs: List of dicts with 'question', 'answer', 'document'
        chunked_reports_dir: Directory with chunked reports
        llm_client: OpenAI client
        
    Returns:
        List of TestQuery with computed ground truth
    """
    test_queries = []
    
    for qa in qa_pairs:
        question = qa['question']
        expected_answer = qa.get('answer', '')
        document = qa.get('document', '')
        
        # Find relevant chunks by embedding similarity
        relevant_chunks = _find_relevant_chunks(
            question, expected_answer,
            chunked_reports_dir, document,
            llm_client
        )
        
        test_queries.append(TestQuery(
            query=question,
            relevant_chunk_ids=relevant_chunks,
            expected_answer=expected_answer,
            difficulty=qa.get('difficulty', 'medium')
        ))
    
    return test_queries


def _find_relevant_chunks(
    question: str,
    expected_answer: str,
    chunked_reports_dir: Path,
    document_name: str,
    llm_client,
    top_k: int = 5
) -> Set[str]:
    """Find relevant chunks for a Q&A pair.
    
    Uses both question and expected answer to find ground truth chunks.
    """
    relevant = set()
    
    # Get embeddings
    q_emb = _get_embedding(question, llm_client)
    a_emb = _get_embedding(expected_answer, llm_client)
    
    # Combine embeddings (weighted average)
    combined = 0.6 * np.array(q_emb) + 0.4 * np.array(a_emb)
    
    # Search in document
    doc_path = chunked_reports_dir / f"{document_name}.json"
    if not doc_path.exists():
        return relevant
    
    with open(doc_path, 'r') as f:
        data = json.load(f)
    
    chunks = data.get('content', {}).get('chunks', [])
    
    # Score chunks
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get('text', '')
        if not chunk_text:
            continue
        
        c_emb = _get_embedding(chunk_text[:1000], llm_client)
        score = np.dot(combined, c_emb) / (np.linalg.norm(combined) * np.linalg.norm(c_emb))
        
        chunk_id = chunk.get('id', i)
        chunk_scores.append((chunk_id, score))
    
    # Take top-k
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    for chunk_id, _ in chunk_scores[:top_k]:
        relevant.add(str(chunk_id))
    
    return relevant


def _get_embedding(text: str, llm_client) -> List[float]:
    """Get embedding for text."""
    response = llm_client.embeddings.create(
        input=text[:8000],
        model="text-embedding-3-large"
    )
    return response.data[0].embedding


# Pre-defined test queries for common RAG scenarios
DEFAULT_TEST_QUERIES = [
    TestQuery(
        query="What was the company's total revenue in 2023?",
        relevant_chunk_ids={"0", "1", "5"},  # Would be populated
        expected_answer="The revenue was $100M",
        difficulty="easy"
    ),
    TestQuery(
        query="What are the main risks mentioned in the report?",
        relevant_chunk_ids={"10", "12", "15"},
        expected_answer="Main risks include market volatility and regulatory changes",
        difficulty="medium"
    ),
    TestQuery(
        query="How does the company's strategy compare to competitors?",
        relevant_chunk_ids={"20", "25", "30"},
        expected_answer="The company focuses on innovation while competitors prioritize cost reduction",
        difficulty="hard"
    ),
]
