import json
import logging
from pathlib import Path
from typing import List, Dict
import faiss
import numpy as np
from dotenv import load_dotenv

from src.reranking import LLMReranker
from src.local_embeddings import get_embedding_model

_log = logging.getLogger(__name__)


class VectorRetriever:
    """FAISS-based vector retrieval using local embeddings."""

    def __init__(self, vector_db_dir: Path, chunked_reports_dir: Path, documents_dir: Path):
        self.vector_db_dir = Path(vector_db_dir)
        self.chunked_reports_dir = Path(chunked_reports_dir)
        self.documents_dir = Path(documents_dir)
        load_dotenv()
        self.embedding_model = get_embedding_model()
        self.all_dbs = self._load_all_vector_dbs()

    def _load_all_vector_dbs(self) -> List[Dict]:
        """Load all FAISS indexes and associated documents."""
        all_dbs = []

        for faiss_file in self.vector_db_dir.glob("*.faiss"):
            sha1_name = faiss_file.stem
            chunked_json_path = self.chunked_reports_dir / f"{sha1_name}.json"
            merged_json_path = self.documents_dir / f"{sha1_name}.json"

            if not chunked_json_path.exists():
                _log.warning("Chunked document file not found for %s", sha1_name)
                continue

            if not merged_json_path.exists():
                _log.warning("Merged document file not found for %s", sha1_name)
                continue

            with open(chunked_json_path, "r", encoding="utf-8") as f:
                chunked_document = json.load(f)

            with open(merged_json_path, "r", encoding="utf-8") as f:
                merged_document = json.load(f)

            index = faiss.read_index(str(faiss_file))

            content = chunked_document.get("content", {})
            if isinstance(content, dict):
                chunks = content.get("chunks", [])
            elif isinstance(content, list):
                chunks = content
            else:
                chunks = []

            index_size = index.ntotal
            chunk_count = len(chunks)
            if index_size != chunk_count:
                _log.error(
                    "[SKIP] %s: FAISS/document mismatch (index=%s, chunks=%s). "
                    "Please rebuild vector DB for this document.",
                    sha1_name, index_size, chunk_count
                )
                continue

            all_dbs.append({
                "sha1_name": sha1_name,
                "vector_db": index,
                "chunked_document": chunked_document,
                "merged_document": merged_document
            })

        return all_dbs

    def get_all_documents(self) -> List[Dict]:
        docs = []
        for db in self.all_dbs:
            metainfo = db["merged_document"].get("metainfo", {})
            docs.append({
                "sha1_name": metainfo.get("sha1_name", db["sha1_name"]),
                "document_name": metainfo.get("document_name", ""),
                "filename": metainfo.get("filename", ""),
                "pages_amount": metainfo.get("pages_amount", 0),
            })
        return docs

    def retrieve_by_document(
        self,
        sha1_name: str,
        query: str,
        top_n: int = 6,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        target_report = next((r for r in self.all_dbs if r["sha1_name"] == sha1_name), None)
        if target_report is None:
            raise ValueError(f"No report found with sha1 '{sha1_name}'")

        chunked_document = target_report["chunked_document"]
        merged_document = target_report["merged_document"]
        vector_db = target_report["vector_db"]

        chunked_content = chunked_document.get("content", {})
        merged_content = merged_document.get("content", {})

        content = chunked_content if isinstance(chunked_content, dict) else {}
        merged = merged_content if isinstance(merged_content, dict) else {}

        if isinstance(content, dict):
            chunks = content.get("chunks", [])
        elif isinstance(chunked_content, list):
            chunks = chunked_content
        else:
            chunks = []

        pages = merged.get("pages", [])
        if not pages and isinstance(content, dict):
            pages = content.get("pages", [])

        if isinstance(chunked_content, list):
            pages = []

        if not chunks:
            _log.warning("No chunks found in document %s", sha1_name)
            return []

        embedding = self.embedding_model.encode(query)
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding_array)

        # over-fetch then filter invalids, avoid empty results
        k_search = min(vector_db.ntotal, max(top_n * 8, 64))
        if k_search <= 0:
            return []

        distances, indices = vector_db.search(embedding_array, k=k_search)

        retrieval_results = []
        seen_pages = set()

        for distance, chunk_idx in zip(distances[0], indices[0]):
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                continue

            chunk = chunks[chunk_idx]
            chunk_page_num = chunk.get("page", 0)
            score = round(float(distance), 4)

            parent_page = None
            if pages:
                parent_page = next((p for p in pages if p.get("page") == chunk_page_num), None)

            if return_parent_pages:
                if parent_page and parent_page.get("page") not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    retrieval_results.append({
                        "distance": score,
                        "page": parent_page.get("page", chunk_page_num),
                        "text": parent_page.get("text", chunk.get("text", "")),
                    })
            else:
                retrieval_results.append({
                    "distance": score,
                    "page": chunk_page_num,
                    "text": chunk.get("text", ""),
                })

            if len(retrieval_results) >= top_n:
                break

        return retrieval_results

    def retrieve_all_pages(self, sha1_name: str) -> List[Dict]:
        target_report = next((r for r in self.all_dbs if r["sha1_name"] == sha1_name), None)
        if target_report is None:
            raise ValueError(f"No report found with sha1 '{sha1_name}'")

        content = target_report["merged_document"].get("content", {})
        if isinstance(content, dict):
            pages = content.get("pages", [])
        else:
            pages = []
        if not pages:
            return []

        return [{
            "distance": 0.5,
            "page": p.get("page", 0),
            "text": p.get("text", "")
        } for p in sorted(pages, key=lambda x: x.get("page", 0))]


class HybridRetriever:
    """Combines vector retrieval with LLM reranking."""

    def __init__(self, vector_db_dir: Path, chunked_reports_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, chunked_reports_dir, documents_dir)
        self.reranker = LLMReranker()

    def get_all_documents(self) -> List[Dict]:
        return self.vector_retriever.get_all_documents()

    def retrieve_by_document(
        self,
        sha1_name: str,
        query: str,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 6,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        vector_results = self.vector_retriever.retrieve_by_document(
            sha1_name=sha1_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )

        if not vector_results:
            return []

        try:
            reranked_results = self.reranker.rerank_documents(
                query=query,
                documents=vector_results,
                documents_batch_size=documents_batch_size,
                llm_weight=llm_weight,
            )
            return reranked_results[:top_n]
        except Exception as exc:
            _log.exception("Reranker failed, fallback to vector results: %s", exc)
            return vector_results[:top_n]
