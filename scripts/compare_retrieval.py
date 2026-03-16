#!/usr/bin/env python3
"""Compare RAG retrieval strategies: FAISS vs Chroma vs RAPTOR.

Usage:
    python compare_retrieval.py --docs data/pdf_reports --queries test_queries.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

from src.pipeline import Pipeline, configs
from src.pipeline_raptor import RaptorPipeline, raptor_configs
from src.ingestion_chroma import ChromaIngestor, ChromaRetriever
from src.evaluation.runner import RAGEvaluator, TestQuery, create_test_queries_from_qa_pairs
from src.evaluation.metrics import generate_evaluation_report, compare_strategies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_strategies(evaluator: RAGEvaluator, args):
    """Set up different retrieval strategies for comparison."""
    
    load_dotenv()
    llm = OpenAI()
    root_path = Path(args.root_path)
    
    # Strategy 1: FAISS (Flat Retrieval)
    def faiss_retriever(query: str):
        import time
        start = time.time()
        
        pipeline = Pipeline(root_path, configs["max_nst_o3m"])
        answer = pipeline.query_single(query)
        
        # Extract context from answer
        context = answer.get('retrieved_context', [])
        latency = (time.time() - start) * 1000
        
        return context, latency
    
    evaluator.register_strategy(
        "faiss_flat",
        faiss_retriever,
        "Standard FAISS vector search with flat chunk list"
    )
    
    # Strategy 2: Chroma (if available)
    try:
        chroma_persist_dir = root_path / "data" / "chroma_db"
        
        def chroma_retriever(query: str):
            import time
            start = time.time()
            
            retriever = ChromaRetriever(chroma_persist_dir)
            results = retriever.retrieve(query, n_results=6)
            
            # Format results
            context = [
                {
                    'id': r['metadata'].get('chunk_id', i),
                    'text': r['text'],
                    'score': r['distance'],
                    'page': r['metadata'].get('page', 0)
                }
                for i, r in enumerate(results)
            ]
            
            latency = (time.time() - start) * 1000
            return context, latency
        
        evaluator.register_strategy(
            "chroma",
            chroma_retriever,
            "Chroma vector database with metadata filtering"
        )
    except Exception as e:
        logger.warning(f"Chroma not available: {e}")
    
    # Strategy 3: RAPTOR Multi-Level
    try:
        raptor_pipeline = RaptorPipeline(root_path, raptor_configs["raptor_basic"])
        
        def raptor_multi_retriever(query: str):
            import time
            start = time.time()
            
            answer = raptor_pipeline.query_single(query, use_raptor=True)
            
            # Extract nodes from answer
            nodes = answer.get('retrieved_nodes', [])
            context = [
                {
                    'id': n['id'],
                    'text': n.get('text', ''),
                    'score': n['score'],
                    'level': n['level'],
                    'is_leaf': n['is_leaf']
                }
                for n in nodes
            ]
            
            latency = answer.get('retrieval_time_ms', (time.time() - start) * 1000)
            return context, latency
        
        evaluator.register_strategy(
            "raptor_multi_level",
            raptor_multi_retriever,
            "RAPTOR with multi-level tree traversal"
        )
        
        # Strategy 4: RAPTOR Leaf-Only
        def raptor_leaf_retriever(query: str):
            import time
            start = time.time()
            
            raptor_pipeline.run_config.raptor_strategy = "leaf_only"
            answer = raptor_pipeline.query_single(query, use_raptor=True)
            
            nodes = answer.get('retrieved_nodes', [])
            context = [
                {
                    'id': n['id'],
                    'text': n.get('text', ''),
                    'score': n['score']
                }
                for n in nodes if n.get('is_leaf', True)
            ]
            
            latency = answer.get('retrieval_time_ms', (time.time() - start) * 1000)
            return context, latency
        
        evaluator.register_strategy(
            "raptor_leaf_only",
            raptor_leaf_retriever,
            "RAPTOR using only leaf nodes (like flat retrieval)"
        )
        
    except Exception as e:
        logger.warning(f"RAPTOR not available: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare RAG retrieval strategies")
    parser.add_argument("--root-path", default=".", help="Project root path")
    parser.add_argument("--queries", required=True, help="JSON file with test queries")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    args = parser.parse_args()
    
    # Load test queries
    with open(args.queries, 'r') as f:
        queries_data = json.load(f)
    
    # Convert to TestQuery objects
    test_queries = [
        TestQuery(
            query=q['question'],
            relevant_chunk_ids=set(q.get('relevant_chunks', [])),
            expected_answer=q.get('answer'),
            difficulty=q.get('difficulty', 'medium')
        )
        for q in queries_data
    ]
    
    logger.info(f"Loaded {len(test_queries)} test queries")
    
    # Create evaluator
    load_dotenv()
    llm = OpenAI()
    evaluator = RAGEvaluator(llm, output_dir=Path(args.output_dir))
    
    # Register strategies
    setup_strategies(evaluator, args)
    
    # Run evaluation
    metrics = evaluator.run_evaluation(test_queries)
    
    # Generate comparison
    comparison_report = evaluator.generate_comparison_report(metrics)
    print("\n" + comparison_report)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
