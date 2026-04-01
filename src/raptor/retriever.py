"""RAPTOR retriever implementation using local embeddings."""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models import RaptorTree, TreeNode
from src.local_embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class RaptorRetriever:
    """Retrieve from RAPTOR tree at multiple levels using local embeddings."""
    
    def __init__(
        self,
        llm_client,
        embedding_model: str = "text-embedding-3-large",
        top_k_per_level: int = 2,
        max_nodes: int = 6
    ):
        """Initialize RAPTOR retriever.
        
        Args:
            llm_client: OpenAI client (for generation, not embeddings)
            embedding_model: Embedding model name (legacy, kept for API compatibility)
            top_k_per_level: How many nodes to retrieve per level
            max_nodes: Maximum total nodes to retrieve
        """
        self.llm = llm_client
        self.embedding_model = embedding_model
        self.top_k_per_level = top_k_per_level
        self.max_nodes = max_nodes
        self.trees: Dict[str, RaptorTree] = {}
        # Use local embedding model for queries
        self.local_embedder = get_embedding_model()
    
    def load_tree(self, tree_path: Path, sha1_name: str):
        """Load a RAPTOR tree from file."""
        with open(tree_path, 'r') as f:
            data = json.load(f)
        
        tree = RaptorTree.from_dict(data)
        self.trees[sha1_name] = tree
        logger.info(f"Loaded tree for {sha1_name}: {len(tree.nodes)} nodes")
    
    def load_trees_from_directory(self, trees_dir: Path):
        """Load all trees from directory."""
        for tree_path in trees_dir.glob("*_raptor.json"):
            sha1_name = tree_path.stem.replace("_raptor", "")
            self.load_tree(tree_path, sha1_name)
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query using local model."""
        embedding = self.local_embedder.encode(query)
        return np.array(embedding).reshape(1, -1)
    
    def _get_node_embedding(self, node: TreeNode) -> Optional[np.ndarray]:
        """Get embedding for node, computing if necessary."""
        if node.embedding is None:
            try:
                embedding = self.local_embedder.encode(node.text[:8000])
                node.embedding = embedding.tolist()
            except Exception as e:
                logger.error(f"Failed to get embedding: {e}")
                return None
        
        return np.array(node.embedding).reshape(1, -1)
    
    def retrieve_from_tree(
        self,
        query: str,
        tree: RaptorTree,
        strategy: str = "multi_level"
    ) -> List[Dict]:
        """Retrieve nodes from a single tree.
        
        Args:
            query: Query text
            tree: RAPTOR tree to search
            strategy: "multi_level", "leaf_only", "root_to_leaf"
            
        Returns:
            List of result dicts with node info and scores
        """
        query_emb = self._get_query_embedding(query)
        results = []
        
        if strategy == "leaf_only":
            # Only search leaf nodes (like standard RAG)
            nodes_to_search = tree.leaves
            
        elif strategy == "root_only":
            # Only search root nodes
            nodes_to_search = tree.roots
            
        elif strategy == "multi_level":
            # Search at multiple levels and combine
            all_nodes = list(tree.nodes.values())
            
            # Group by level
            nodes_by_level = {}
            for node in all_nodes:
                level = node.level
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
            
            # Retrieve from each level
            selected_nodes = []
            for level in sorted(nodes_by_level.keys()):
                level_nodes = nodes_by_level[level]
                
                # Score nodes at this level
                level_scores = []
                for node in level_nodes:
                    node_emb = self._get_node_embedding(node)
                    if node_emb is not None:
                        score = cosine_similarity(query_emb, node_emb)[0][0]
                        level_scores.append((node, score))
                
                # Take top-k from this level
                level_scores.sort(key=lambda x: x[1], reverse=True)
                selected_nodes.extend(level_scores[:self.top_k_per_level])
            
            # Sort combined results
            selected_nodes.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            for node, score in selected_nodes[:self.max_nodes]:
                results.append({
                    'text': node.text,
                    'level': node.level,
                    'score': float(score),
                    'is_leaf': node.is_leaf,
                    'node_id': node.node_id,
                    'source_doc': node.source_doc,
                    'page': node.page,
                    'num_leaf_children': len(node.get_leaf_nodes()),
                    'path_to_root': [n.text[:100] for n in node.get_path_to_root()[1:]]
                })
            
            return results
        
        elif strategy == "root_to_leaf":
            # Start from root, traverse down to most relevant leaves
            current_nodes = tree.roots
            path = []
            
            while current_nodes:
                # Score current level
                level_scores = []
                for node in current_nodes:
                    node_emb = self._get_node_embedding(node)
                    if node_emb is not None:
                        score = cosine_similarity(query_emb, node_emb)[0][0]
                        level_scores.append((node, score))
                
                # Take best node
                level_scores.sort(key=lambda x: x[1], reverse=True)
                best_node, best_score = level_scores[0]
                path.append((best_node, best_score))
                
                # Move to children
                if best_node.children:
                    current_nodes = best_node.children
                else:
                    break
            
            # Return path from root to leaf
            for node, score in path:
                results.append({
                    'text': node.text,
                    'level': node.level,
                    'score': float(score),
                    'is_leaf': node.is_leaf,
                    'node_id': node.node_id,
                    'source_doc': node.source_doc,
                    'page': node.page
                })
            
            return results
        
        # For leaf_only and root_only strategies
        nodes_to_search = nodes_to_search if 'nodes_to_search' in locals() else tree.leaves
        
        node_scores = []
        for node in nodes_to_search:
            node_emb = self._get_node_embedding(node)
            if node_emb is not None:
                score = cosine_similarity(query_emb, node_emb)[0][0]
                node_scores.append((node, score))
        
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        for node, score in node_scores[:self.max_nodes]:
            results.append({
                'text': node.text,
                'level': node.level,
                'score': float(score),
                'is_leaf': node.is_leaf,
                'node_id': node.node_id,
                'source_doc': node.source_doc,
                'page': node.page
            })
        
        return results
    
    def retrieve(
        self,
        query: str,
        sha1_name: Optional[str] = None,
        strategy: str = "multi_level"
    ) -> List[Dict]:
        """Retrieve across all trees or specific document.
        
        Args:
            query: Query text
            sha1_name: Specific document to search (None = all)
            strategy: Retrieval strategy
            
        Returns:
            Combined results from all trees
        """
        all_results = []
        
        if sha1_name and sha1_name in self.trees:
            trees_to_search = {sha1_name: self.trees[sha1_name]}
        else:
            trees_to_search = self.trees
        
        for doc_name, tree in trees_to_search.items():
            results = self.retrieve_from_tree(query, tree, strategy)
            for r in results:
                r['document'] = doc_name
            all_results.extend(results)
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:self.max_nodes]
    
    def get_tree_stats(self, sha1_name: str) -> Dict:
        """Get statistics for a tree."""
        if sha1_name not in self.trees:
            return {"error": "Tree not found"}
        
        tree = self.trees[sha1_name]
        
        # Count nodes at each level
        level_counts = {}
        for node in tree.nodes.values():
            level = node.level
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            "total_nodes": len(tree.nodes),
            "leaf_nodes": len(tree.leaves),
            "root_nodes": len(tree.roots),
            "levels": tree.max_levels,
            "nodes_per_level": level_counts,
            "average_children_per_parent": sum(
                len(n.children) for n in tree.nodes.values()
            ) / max(1, len([n for n in tree.nodes.values() if n.children]))
        }
