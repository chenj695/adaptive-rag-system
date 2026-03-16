import json
import logging
from pathlib import Path
from typing import List, Dict
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

from src.reranking import LLMReranker

_log = logging.getLogger(__name__)


class VectorRetriever:
    """FAISS-based vector retrieval."""
    
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = Path(vector_db_dir)
        self.documents_dir = Path(documents_dir)
        self.all_dbs = self._load_all_vector_dbs()
        load_dotenv()
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    def get_all_documents(self) -> List[Dict]:
        """Get list of all available documents."""
        docs = []
        for db in self.all_dbs:
            metainfo = db["document"].get("metainfo", {})
            docs.append({
                "sha1_name": metainfo.get("sha1_name", ""),
                "document_name": metainfo.get("document_name", ""),
                "filename": metainfo.get("filename", ""),
                "pages_amount": metainfo.get("pages_amount", 0)
            })
        return docs

    def retrieve_by_document(self, sha1_name: str, query: str,
                             top_n: int = 6, 
                             return_parent_pages: bool = False) -> List[Dict]:
        """Retrieve relevant chunks for a specific document."""
        target_report = None
        for report in self.all_dbs:
            if report["sha1_name"] == sha1_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with sha1 '{sha1_name}'")
            raise ValueError(f"No report found with sha1 '{sha1_name}'")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        actual_top_n = min(top_n, len(chunks))
        
        # Get query embedding
        embedding = self.llm.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
        
        retrieval_results = []
        seen_pages = set()
        
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next((p for p in pages if p["page"] == chunk["page"]), None)
            
            if return_parent_pages:
                if parent_page and parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results

    def retrieve_all_pages(self, sha1_name: str) -> List[Dict]:
        """Retrieve all pages from a document."""
        target_report = None
        for report in self.all_dbs:
            if report["sha1_name"] == sha1_name:
                target_report = report
                break
        
        if target_report is None:
            raise ValueError(f"No report found with sha1 '{sha1_name}'")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
        return all_pages


class HybridRetriever:
    """Combines vector retrieval with LLM reranking."""
    
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()

    def get_all_documents(self) -> List[Dict]:
        """Get list of all available documents."""
        return self.vector_retriever.get_all_documents()

    def retrieve_by_document(self, sha1_name: str, query: str,
                             llm_reranking_sample_size: int = 28,
                             documents_batch_size: int = 2,
                             top_n: int = 6,
                             llm_weight: float = 0.7,
                             return_parent_pages: bool = False) -> List[Dict]:
        """
        Hybrid retrieval:
        1. Get top-k from vector DB
        2. Rerank with LLM
        3. Return top_n results
        """
        vector_results = self.vector_retriever.retrieve_by_document(
            sha1_name=sha1_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        
        return reranked_results[:top_n]
