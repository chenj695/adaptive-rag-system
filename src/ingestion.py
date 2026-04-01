"""
Vector database ingestion using local sentence-transformers model.
No API calls required - works completely offline.
"""
import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi

__all__ = ['VectorDBIngestor', 'BM25Ingestor']
import faiss
import numpy as np

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")


class BM25Ingestor:
    """Create BM25 keyword search indices."""
    
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and create BM25 indices."""
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))
        
        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Handle both formats
            content = report_data.get('content', {})
            if isinstance(content, dict):
                chunks = content.get('chunks', [])
            else:
                chunks = content
            
            text_chunks = [chunk['text'] for chunk in chunks if 'text' in chunk]
            if not text_chunks:
                print(f"Warning: No text chunks found in {report_path.name}")
                continue
                
            bm25_index = self.create_bm25_index(text_chunks)
            
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
        
        print(f"Processed {len(all_report_paths)} reports for BM25")


class VectorDBIngestor:
    """Create FAISS vector databases using local sentence-transformers model."""
    
    # Default local model - good balance of speed and quality
    # Other options: "all-MiniLM-L6-v2" (faster), "all-mpnet-base-v2" (better quality)
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, model_name: str = None, model_path: str = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for local embeddings.\n"
                "Install with: pip install sentence-transformers\n"
                "Or use the API-based version by setting OPENAI_API_KEY"
            )
        
        # Check for local path first (for offline/air-gapped servers)
        self.model_path = model_path or os.getenv("EMBEDDING_MODEL_PATH")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.model = None  # Lazy load
        
        if self.model_path:
            print(f"Using local embedding model from: {self.model_path}")
        else:
            print(f"Using embedding model: {self.model_name}")

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self.model is None:
            if self.model_path:
                # Load from custom local path (for air-gapped/offline servers)
                print(f"Loading embedding model from: {self.model_path}...")
                self.model = SentenceTransformer(self.model_path)
            else:
                # Load from HuggingFace/cache (requires internet on first run)
                print(f"Loading embedding model: {self.model_name}...")
                self.model = SentenceTransformer(self.model_name)
            
            print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        return self.model

    def _get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings using local sentence-transformers model."""
        model = self._load_model()
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter out empty texts
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            raise ValueError("No valid text to embed")
        
        # Get embeddings - this is done locally, no API calls
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        return embeddings.tolist()

    def _create_vector_db(self, embeddings: List[List[float]]):
        """Create FAISS index from embeddings."""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Inner product index (cosine similarity after normalization)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        return index

    def _process_report(self, report: dict):
        """Process single report."""
        # Handle both formats
        content = report.get('content', {})
        if isinstance(content, dict):
            chunks = content.get('chunks', [])
        else:
            chunks = content
        
        text_chunks = [chunk['text'] for chunk in chunks if 'text' in chunk]
        if not text_chunks:
            raise ValueError("No text chunks found in report")
            
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and create vector DBs."""
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for report_path in tqdm(all_report_paths, desc="Creating vector DBs"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))
        
        print(f"Processed {len(all_report_paths)} reports")
