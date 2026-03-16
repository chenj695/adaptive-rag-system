"""Integrated pipeline with RAPTOR support."""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .pipeline import Pipeline, RunConfig, configs
from .raptor.tree_builder import RaptorTreeBuilder
from .raptor.retriever import RaptorRetriever
from .raptor.models import RaptorTree

logger = logging.getLogger(__name__)


@dataclass
class RaptorRunConfig(RunConfig):
    """Extended config with RAPTOR-specific options."""
    
    # RAPTOR settings
    use_raptor: bool = True
    raptor_max_levels: int = 3
    raptor_cluster_threshold: float = 0.5
    raptor_top_k_per_level: int = 2
    raptor_strategy: str = "multi_level"  # multi_level, leaf_only, root_to_leaf
    
    # Storage
    raptor_trees_dir: str = "data/raptor_trees"


class RaptorPipeline(Pipeline):
    """Extended pipeline with RAPTOR tree construction and retrieval."""
    
    def __init__(self, root_path: Path, run_config: RaptorRunConfig = None):
        """Initialize RAPTOR pipeline.
        
        Args:
            root_path: Project root directory
            run_config: RaptorRunConfig instance
        """
        super().__init__(root_path, run_config)
        self.run_config = run_config or RaptorRunConfig()
        
        # Set up RAPTOR directories
        self.raptor_trees_dir = self.root_path / self.run_config.raptor_trees_dir
        self.raptor_trees_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RAPTOR components if enabled
        if self.run_config.use_raptor:
            self.tree_builder = RaptorTreeBuilder(
                llm_client=self._get_llm_client(),
                embedding_model=self.run_config.answering_model,
                max_levels=self.run_config.raptor_max_levels,
                cluster_threshold=self.run_config.raptor_cluster_threshold
            )
            self.raptor_retriever = RaptorRetriever(
                llm_client=self._get_llm_client(),
                top_k_per_level=self.run_config.raptor_top_k_per_level
            )
            
            # Try to load existing trees
            self._load_existing_trees()
    
    def _get_llm_client(self):
        """Get or create LLM client."""
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _load_existing_trees(self):
        """Load existing RAPTOR trees if available."""
        if self.raptor_trees_dir.exists():
            try:
                self.raptor_retriever.load_trees_from_directory(self.raptor_trees_dir)
                logger.info(f"Loaded {len(self.raptor_retriever.trees)} RAPTOR trees")
            except Exception as e:
                logger.warning(f"Could not load existing trees: {e}")
    
    def build_raptor_trees(self):
        """Build RAPTOR trees from chunked reports."""
        if not self.run_config.use_raptor:
            logger.info("RAPTOR is disabled")
            return
        
        logger.info("Building RAPTOR trees...")
        
        trees = self.tree_builder.build_trees_from_directory(
            self.chunked_reports_dir,
            self.raptor_trees_dir
        )
        
        # Reload retriever with new trees
        self.raptor_retriever.load_trees_from_directory(self.raptor_trees_dir)
        
        logger.info(f"Built {len(trees)} RAPTOR trees")
    
    def process_parsed_reports(self):
        """Override to include RAPTOR tree building."""
        # Run standard processing
        super().process_parsed_reports()
        
        # Build RAPTOR trees
        if self.run_config.use_raptor:
            self.build_raptor_trees()
    
    def query_single(
        self,
        question: str,
        document_name: Optional[str] = None,
        use_raptor: Optional[bool] = None
    ) -> Dict:
        """Query with optional RAPTOR retrieval.
        
        Args:
            question: Query question
            document_name: Specific document to query
            use_raptor: Override to use RAPTOR (None = use config setting)
            
        Returns:
            Answer dictionary
        """
        use_raptor = use_raptor if use_raptor is not None else self.run_config.use_raptor
        
        if not use_raptor or not self.raptor_retriever.trees:
            # Fall back to standard retrieval
            return super().query_single(question, document_name)
        
        # RAPTOR retrieval
        import time
        start_time = time.time()
        
        # Get relevant nodes from tree
        results = self.raptor_retriever.retrieve(
            query=question,
            sha1_name=document_name,
            strategy=self.run_config.raptor_strategy
        )
        
        retrieval_time = time.time() - start_time
        
        # Format context for LLM
        context_text = self._format_raptor_context(results)
        
        # Generate answer
        from .questions_processing import OpenAIProcessor
        
        processor = OpenAIProcessor(model=self.run_config.answering_model)
        schema = self._determine_schema(question)
        
        answer = processor.get_answer_from_rag_context(
            question=question,
            rag_context=context_text,
            schema=schema,
            model=self.run_config.answering_model
        )
        
        # Add RAPTOR-specific metadata
        answer['retrieval_method'] = 'raptor'
        answer['retrieval_strategy'] = self.run_config.raptor_strategy
        answer['retrieval_time_ms'] = retrieval_time * 1000
        answer['tree_levels_accessed'] = list(set(r['level'] for r in results))
        answer['retrieved_nodes'] = [
            {
                'id': r['node_id'],
                'level': r['level'],
                'score': r['score'],
                'is_leaf': r['is_leaf'],
                'page': r['page']
            }
            for r in results[:5]
        ]
        
        return answer
    
    def _format_raptor_context(self, results: List[Dict]) -> str:
        """Format RAPTOR retrieval results as context string."""
        sections = []
        
        for i, result in enumerate(results, 1):
            level_indicator = "📝" if result['is_leaf'] else "📊"
            sections.append(
                f"{level_indicator} [Node {i} | Level {result['level']} | Score: {result['score']:.3f}]\n"
                f"{result['text']}\n"
            )
        
        return "\n---\n".join(sections)
    
    def _determine_schema(self, question: str) -> str:
        """Determine answer schema type."""
        question_lower = question.lower()
        
        boolean_starters = ['did ', 'was ', 'were ', 'have ', 'has ', 'is ', 'are ']
        if any(question_lower.startswith(s) for s in boolean_starters):
            return "boolean"
        
        if any(w in question_lower for w in ['how much', 'how many', 'what percentage']):
            return "number"
        
        if any(w in question_lower for w in ['who', 'which company', 'what is the name']):
            return "name"
        
        return "text"
    
    def get_raptor_stats(self) -> Dict:
        """Get statistics about RAPTOR trees."""
        if not self.run_config.use_raptor:
            return {"enabled": False}
        
        stats = {
            "enabled": True,
            "num_trees": len(self.raptor_retriever.trees),
            "trees": {}
        }
        
        for sha1_name in self.raptor_retriever.trees:
            tree_stats = self.raptor_retriever.get_tree_stats(sha1_name)
            stats["trees"][sha1_name] = tree_stats
        
        return stats


# Convenience config presets
raptor_configs = {
    "raptor_basic": RaptorRunConfig(
        use_raptor=True,
        raptor_max_levels=2,
        raptor_strategy="multi_level",
        answering_model="gpt-4o-mini"
    ),
    "raptor_deep": RaptorRunConfig(
        use_raptor=True,
        raptor_max_levels=4,
        raptor_strategy="multi_level",
        answering_model="o3-mini-2025-01-31"
    ),
    "raptor_leaf_only": RaptorRunConfig(
        use_raptor=True,
        raptor_max_levels=3,
        raptor_strategy="leaf_only",
        answering_model="gpt-4o-mini"
    ),
    "standard": RaptorRunConfig(
        use_raptor=False,
        answering_model="gpt-4o-mini"
    )
}
