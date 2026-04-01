"""RAPTOR tree builder implementation using local embeddings."""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import json
import numpy as np

from .models import TreeNode, RaptorTree
from .clustering import perform_clustering
from src.local_embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class RaptorTreeBuilder:
    """Build RAPTOR tree from document chunks using local embeddings."""
    
    def __init__(
        self,
        llm_client,
        embedding_model: str = "text-embedding-3-large",
        max_levels: int = 3,
        target_chunk_size: int = 1000,
        cluster_threshold: float = 0.5,
        cluster_dim: int = 10,
        summary_length: str = "auto"  # "auto", "short", "medium", "long"
    ):
        """Initialize RAPTOR tree builder.
        
        Args:
            llm_client: OpenAI client for summarization (embeddings use local model)
            embedding_model: Legacy parameter (kept for API compatibility)
            max_levels: Maximum tree levels (0 = leaves only)
            target_chunk_size: Target token count for summary chunks
            cluster_threshold: GMM probability threshold
            cluster_dim: Dimensionality for clustering
            summary_length: Length of summaries
        """
        self.llm = llm_client
        self.embedding_model = embedding_model
        self.max_levels = max_levels
        self.target_chunk_size = target_chunk_size
        self.cluster_threshold = cluster_threshold
        self.cluster_dim = cluster_dim
        self.summary_length = summary_length
        # Use local embedding model
        self.local_embedder = get_embedding_model()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using local model."""
        embedding = self.local_embedder.encode(text[:8000])  # Truncate if too long
        return embedding.tolist()
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using local model."""
        # Local model handles batching efficiently
        embeddings = self.local_embedder.encode(texts)
        return embeddings.tolist()
    
    def _summarize_cluster(self, texts: List[str], level: int) -> str:
        """Generate summary for a cluster of texts.
        
        Args:
            texts: Texts in the cluster
            level: Current tree level
            
        Returns:
            Summary text
        """
        combined = "\n\n---\n\n".join(texts)
        
        # Adjust summary length based on level
        if self.summary_length == "auto":
            max_tokens = 500 if level == 1 else 300
        elif self.summary_length == "short":
            max_tokens = 200
        elif self.summary_length == "medium":
            max_tokens = 400
        else:
            max_tokens = 600
        
        prompt = f"""You are creating a summary for a hierarchical document retrieval system.

Below are related text excerpts from a document. Create a concise summary that captures:
1. The main themes and key information
2. Important facts, figures, and relationships
3. Context that would help answer questions about this content

Keep the summary informative but concise ({max_tokens} tokens max).

TEXTS TO SUMMARIZE:
{combined}

SUMMARY:"""
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful summarization assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: concatenate with truncation
            return combined[:2000] + "..." if len(combined) > 2000 else combined
    
    def build_tree(
        self,
        chunks: List[Dict],
        document_name: str = ""
    ) -> RaptorTree:
        """Build RAPTOR tree from chunks.
        
        Args:
            chunks: List of chunk dicts with 'text', 'page', 'chunk_id'
            document_name: Name of source document
            
        Returns:
            RaptorTree instance
        """
        tree = RaptorTree(max_levels=self.max_levels)
        
        logger.info(f"Building RAPTOR tree for {document_name} with {len(chunks)} chunks")
        
        # Step 1: Create leaf nodes
        leaf_nodes = []
        for i, chunk in enumerate(chunks):
            node = TreeNode(
                text=chunk['text'],
                level=0,
                source_doc=document_name,
                page=chunk.get('page', 0),
                chunk_id=chunk.get('chunk_id', i),
                metadata={
                    'is_leaf': True,
                    'original_index': i
                }
            )
            leaf_nodes.append(node)
            tree.add_node(node)
        
        logger.info(f"Created {len(leaf_nodes)} leaf nodes")
        
        # Step 2: Build tree bottom-up
        current_level_nodes = leaf_nodes
        
        for level in range(1, self.max_levels + 1):
            logger.info(f"Building level {level}...")
            
            if len(current_level_nodes) <= 1:
                logger.info(f"Stopping at level {level-1}: only {len(current_level_nodes)} nodes")
                break
            
            # Get embeddings for current level
            embeddings = self._get_embeddings_batch([n.text for n in current_level_nodes])
            embeddings_array = np.array(embeddings)
            
            # Assign embeddings to nodes
            for node, emb in zip(current_level_nodes, embeddings):
                node.embedding = emb
            
            # Perform clustering
            clusters = perform_clustering(
                embeddings_array,
                dim=self.cluster_dim,
                threshold=self.cluster_threshold,
                verbose=True
            )
            
            logger.info(f"Level {level}: Created {len(clusters)} clusters")
            
            if len(clusters) == 1 and len(clusters[0]) == len(current_level_nodes):
                logger.info("All nodes in one cluster, stopping")
                break
            
            # Create summary nodes for each cluster
            next_level_nodes = []
            
            for cluster_indices in clusters:
                cluster_nodes = [current_level_nodes[i] for i in cluster_indices]
                cluster_texts = [n.text for n in cluster_nodes]
                
                # Generate summary
                summary = self._summarize_cluster(cluster_texts, level)
                
                # Create parent node
                parent = TreeNode(
                    text=summary,
                    level=level,
                    source_doc=document_name,
                    metadata={
                        'is_summary': True,
                        'num_children': len(cluster_nodes),
                        'child_levels': list(set(c.level for c in cluster_nodes))
                    }
                )
                
                # Add to tree (connects parent to children)
                tree.add_node(parent)
                for child in cluster_nodes:
                    child.parent = parent
                    parent.children.append(child)
                
                next_level_nodes.append(parent)
            
            current_level_nodes = next_level_nodes
            logger.info(f"Level {level}: Created {len(next_level_nodes)} summary nodes")
        
        # Compute embeddings for final level
        if current_level_nodes:
            embeddings = self._get_embeddings_batch([n.text for n in current_level_nodes])
            for node, emb in zip(current_level_nodes, embeddings):
                node.embedding = emb
        
        # Set roots
        tree.roots = [n for n in tree.nodes.values() if n.parent is None]
        
        logger.info(f"Tree complete: {len(tree.nodes)} nodes, {len(tree.roots)} roots, {len(tree.leaves)} leaves")
        
        return tree
    
    def build_trees_from_directory(
        self,
        chunked_reports_dir: Path,
        output_dir: Path
    ) -> Dict[str, RaptorTree]:
        """Build RAPTOR trees for all documents in directory.
        
        Args:
            chunked_reports_dir: Directory with chunked JSON files
            output_dir: Where to save trees
            
        Returns:
            Dict mapping sha1_name to RaptorTree
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trees = {}
        
        for json_path in chunked_reports_dir.glob("*.json"):
            logger.info(f"Processing {json_path.name}")
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            metainfo = data.get('metainfo', {})
            sha1_name = metainfo.get('sha1_name', json_path.stem)
            document_name = metainfo.get('document_name', sha1_name)
            
            chunks = data.get('content', {}).get('chunks', [])
            if not chunks:
                logger.warning(f"No chunks in {json_path.name}")
                continue
            
            # Format chunks
            formatted_chunks = [
                {
                    'text': c.get('text', ''),
                    'page': c.get('page', 0),
                    'chunk_id': c.get('id', i)
                }
                for i, c in enumerate(chunks)
            ]
            
            # Build tree
            tree = self.build_tree(formatted_chunks, document_name)
            trees[sha1_name] = tree
            
            # Save tree
            output_path = output_dir / f"{sha1_name}_raptor.json"
            with open(output_path, 'w') as f:
                json.dump(tree.to_dict(), f, indent=2)
            
            logger.info(f"Saved tree to {output_path}")
        
        return trees
