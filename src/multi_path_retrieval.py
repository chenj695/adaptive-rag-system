"""
Multi-Path Retrieval System

Combines multiple retrieval strategies:
1. Semantic Path: Dense vector retrieval (FAISS)
2. Lexical Path: BM25 keyword retrieval
3. RRF Fusion: Reciprocal Rank Fusion to combine results
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from src.reranking import get_reranker
from src.local_embeddings import get_embedding_model

_log = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based lexical retrieval."""
    
    def __init__(self, bm25_dir: Path, documents_dir: Path):
        self.bm25_dir = Path(bm25_dir)
        self.documents_dir = Path(documents_dir)
        self.all_indices = self._load_all_bm25_indices()
    
    def _load_all_bm25_indices(self) -> List[Dict]:
        """Load all BM25 indices and associated documents."""
        all_indices = []
        
        for pkl_file in self.bm25_dir.glob("*.pkl"):
            sha1_name = pkl_file.stem
            json_path = self.documents_dir / f"{sha1_name}.json"
            
            if not json_path.exists():
                _log.warning(f"Document file not found for {sha1_name}")
                continue
            
            with open(json_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            with open(pkl_file, 'rb') as f:
                bm25_index = pickle.load(f)
            
            all_indices.append({
                "sha1_name": sha1_name,
                "bm25_index": bm25_index,
                "document": document
            })
        
        return all_indices
    
    def retrieve_by_document(self, sha1_name: str, query: str,
                             top_n: int = 20) -> List[Dict]:
        """Retrieve relevant chunks using BM25 for a specific document."""
        target_report = None
        for report in self.all_indices:
            if report["sha1_name"] == sha1_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No BM25 index found for sha1 '{sha1_name}'")
            return []
        
        document = target_report["document"]
        bm25_index = target_report["bm25_index"]
        
        # Handle both content formats
        content = document.get("content", {})
        if isinstance(content, dict):
            chunks = content.get("chunks", [])
        else:
            chunks = content
        
        if not chunks:
            _log.warning(f"No chunks found in document {sha1_name}")
            return []
        
        actual_top_n = min(top_n, len(chunks))
        
        # Tokenize query
        tokenized_query = query.split()
        
        # Get BM25 scores
        scores = bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:actual_top_n]
        
        retrieval_results = []
        for idx in top_indices:
            # Safety check
            if idx < 0 or idx >= len(chunks):
                _log.warning(f"BM25 index {idx} out of bounds for chunks list (len={len(chunks)})")
                continue
            chunk = chunks[idx]
            result = {
                "score": float(scores[idx]),
                "page": chunk.get("page", 0),
                "text": chunk.get("text", ""),
                "rank": len(retrieval_results) + 1
            }
            retrieval_results.append(result)
        
        return retrieval_results


class VectorRetriever:
    """FAISS-based vector retrieval using local embeddings."""
    
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = Path(vector_db_dir)
        self.documents_dir = Path(documents_dir)
        self.all_dbs = self._load_all_vector_dbs()
        load_dotenv()
        # Use local embedding model instead of OpenAI API
        self.embedding_model = get_embedding_model()

    def _load_all_vector_dbs(self) -> List[Dict]:
        """Load all FAISS indexes and associated documents."""
        all_dbs = []
        
        for faiss_file in self.vector_db_dir.glob("*.faiss"):
            sha1_name = faiss_file.stem
            json_path = self.documents_dir / f"{sha1_name}.json"
            
            if not json_path.exists():
                _log.warning(f"Document file not found for {sha1_name}")
                continue
            
            with open(json_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            index = faiss.read_index(str(faiss_file))
            all_dbs.append({
                "sha1_name": sha1_name,
                "vector_db": index,
                "document": document
            })
        
        return all_dbs

    def retrieve_by_document(self, sha1_name: str, query: str,
                             top_n: int = 20) -> List[Dict]:
        """Retrieve relevant chunks for a specific document."""
        target_report = None
        for report in self.all_dbs:
            if report["sha1_name"] == sha1_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with sha1 '{sha1_name}'")
            return []
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        
        # Handle both content formats
        content = document.get("content", {})
        if isinstance(content, dict):
            chunks = content.get("chunks", [])
        else:
            chunks = content
        
        if not chunks:
            _log.warning(f"No chunks found in document {sha1_name}")
            return []
        
        # Check for index/chunk mismatch
        index_size = vector_db.ntotal
        chunk_count = len(chunks)
        if index_size != chunk_count:
            _log.warning(f"Mismatch in {sha1_name}: FAISS index has {index_size} vectors, but document has {chunk_count} chunks")
            effective_count = min(index_size, chunk_count)
        else:
            effective_count = chunk_count
        
        actual_top_n = min(top_n, effective_count)
        if actual_top_n <= 0:
            _log.warning(f"Cannot retrieve: effective_count={effective_count}, top_n={top_n}")
            return []
        
        # Get query embedding using local model
        embedding = self.embedding_model.encode(query)
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        
        # Normalize for cosine similarity (same as during indexing)
        faiss.normalize_L2(embedding_array)
        
        # Search FAISS index
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
        
        retrieval_results = []
        for rank, (distance, chunk_idx) in enumerate(zip(distances[0], indices[0]), 1):
            # Safety check for index bounds
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                _log.warning(f"FAISS index {chunk_idx} out of bounds for chunks list (len={len(chunks)})")
                continue
            
            chunk = chunks[chunk_idx]
            result = {
                "score": float(distance),
                "page": chunk.get("page", 0),
                "text": chunk.get("text", ""),
                "rank": rank
            }
            retrieval_results.append(result)
        
        return retrieval_results


class MultiPathRetriever:
    """
    Multi-Path Retrieval combining:
    1. Semantic path (FAISS vector retrieval)
    2. Lexical path (BM25 keyword retrieval)
    3. RRF Fusion to combine results
    """
    
    def __init__(self, vector_db_dir: Path, bm25_dir: Path, documents_dir: Path,
                 rrf_k: int = 60):
        """
        Initialize multi-path retriever.
        
        Args:
            vector_db_dir: Directory containing FAISS indexes
            bm25_dir: Directory containing BM25 pickle files
            documents_dir: Directory containing document JSON files
            rrf_k: Reciprocal Rank Fusion constant (default: 60)
        """
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.bm25_retriever = BM25Retriever(bm25_dir, documents_dir)
        self.rrf_k = rrf_k
        self.reranker = get_reranker()
        
        _log.info(f"MultiPathRetriever initialized with RRF k={rrf_k}")
    
    def _reciprocal_rank_fusion(self, 
                                 semantic_results: List[Dict],
                                 lexical_results: List[Dict],
                                 top_k: int = 20) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Formula: score = 1/(k + rank)
        
        Args:
            semantic_results: Results from vector retrieval
            lexical_results: Results from BM25 retrieval
            top_k: Number of results to return
            
        Returns:
            Fused and sorted results
        """
        # Create score dictionaries
        scores = {}
        
        # Process semantic results
        for result in semantic_results:
            doc_id = f"page_{result['page']}"
            rank = result.get('rank', 1)
            rrf_score = 1.0 / (self.rrf_k + rank)
            
            if doc_id not in scores:
                scores[doc_id] = {
                    'total_score': 0,
                    'page': result['page'],
                    'text': result['text'],
                    'semantic_rank': rank,
                    'lexical_rank': None
                }
            scores[doc_id]['total_score'] += rrf_score
        
        # Process lexical results
        for result in lexical_results:
            doc_id = f"page_{result['page']}"
            rank = result.get('rank', 1)
            rrf_score = 1.0 / (self.rrf_k + rank)
            
            if doc_id not in scores:
                scores[doc_id] = {
                    'total_score': 0,
                    'page': result['page'],
                    'text': result['text'],
                    'semantic_rank': None,
                    'lexical_rank': rank
                }
            scores[doc_id]['total_score'] += rrf_score
            scores[doc_id]['lexical_rank'] = rank
        
        # Sort by total score
        sorted_results = sorted(scores.values(), 
                               key=lambda x: x['total_score'], 
                               reverse=True)
        
        # Add fusion rank
        for i, result in enumerate(sorted_results, 1):
            result['rank'] = i
        
        return sorted_results[:top_k]
    
    def retrieve_by_document(self, sha1_name: str, query: str,
                             semantic_top_n: int = 20,
                             lexical_top_n: int = 20,
                             fusion_top_k: int = 20,
                             use_reranking: bool = False,
                             llm_reranking_sample_size: int = 28,
                             top_n: int = 6) -> List[Dict]:
        """
        Multi-path retrieval for a specific document.
        
        Flow:
        1. Retrieve from semantic path (FAISS)
        2. Retrieve from lexical path (BM25)
        3. Fuse with RRF
        4. Optionally rerank with LLM
        
        Args:
            sha1_name: Document identifier
            query: User query
            semantic_top_n: Number of results from semantic path
            lexical_top_n: Number of results from lexical path
            fusion_top_k: Number of results after RRF fusion
            use_reranking: Whether to apply LLM reranking
            llm_reranking_sample_size: Sample size for LLM reranking
            top_n: Final number of results
            
        Returns:
            Retrieved and ranked documents
        """
        _log.info(f"Multi-path retrieval for '{sha1_name}' with query: '{query[:50]}...'")
        
        # Path 1: Semantic retrieval
        _log.debug("Path 1: Semantic retrieval (FAISS)")
        semantic_results = self.vector_retriever.retrieve_by_document(
            sha1_name=sha1_name,
            query=query,
            top_n=semantic_top_n
        )
        _log.info(f"Semantic path returned {len(semantic_results)} results")
        
        # Path 2: Lexical retrieval
        _log.debug("Path 2: Lexical retrieval (BM25)")
        lexical_results = self.bm25_retriever.retrieve_by_document(
            sha1_name=sha1_name,
            query=query,
            top_n=lexical_top_n
        )
        _log.info(f"Lexical path returned {len(lexical_results)} results")
        
        # Path 3: RRF Fusion
        _log.debug("Path 3: RRF Fusion")
        fused_results = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            lexical_results=lexical_results,
            top_k=fusion_top_k
        )
        _log.info(f"RRF fusion produced {len(fused_results)} results")
        
        # Optionally apply LLM reranking
        if use_reranking:
            _log.debug("Applying LLM reranking")
            reranked_results = self.reranker.rerank_documents(
                query=query,
                documents=fused_results[:llm_reranking_sample_size],
                llm_weight=0.7
            )
            return reranked_results[:top_n]
        
        return fused_results[:top_n]
    
    def get_all_documents(self) -> List[Dict]:
        """Get list of all available documents."""
        return self.vector_retriever.get_all_documents()


class MultiPathRetrieverWithRAPTOR(MultiPathRetriever):
    """
    Extended multi-path retriever that includes RAPTOR hierarchical retrieval.
    
    Paths:
    1. Semantic (FAISS dense vectors)
    2. Lexical (BM25 keyword matching)
    3. Hierarchical (RAPTOR tree traversal)
    """
    
    def __init__(self, vector_db_dir: Path, bm25_dir: Path, documents_dir: Path,
                 raptor_dir: Optional[Path] = None, rrf_k: int = 60):
        super().__init__(vector_db_dir, bm25_dir, documents_dir, rrf_k)
        self.raptor_dir = raptor_dir
        # RAPTOR retriever would be initialized here if available
        _log.info("MultiPathRetrieverWithRAPTOR initialized")
    
    def retrieve_by_document(self, sha1_name: str, query: str,
                             semantic_top_n: int = 15,
                             lexical_top_n: int = 15,
                             fusion_top_k: int = 20,
                             use_reranking: bool = False,
                             top_n: int = 6) -> List[Dict]:
        """
        Three-path retrieval with optional RAPTOR.
        
        Currently falls back to parent implementation (2-path).
        RAPTOR path can be added when tree indices are available.
        """
        return super().retrieve_by_document(
            sha1_name=sha1_name,
            query=query,
            semantic_top_n=semantic_top_n,
            lexical_top_n=lexical_top_n,
            fusion_top_k=fusion_top_k,
            use_reranking=use_reranking,
            top_n=top_n
        )
