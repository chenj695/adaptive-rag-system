import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

__all__ = ['VectorDBIngestor', 'BM25Ingestor']
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


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
    """Create FAISS vector databases."""
    
    def __init__(self):
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[float]:
        """Get embeddings from OpenAI API."""
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        
        if isinstance(text, list):
            text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
        else:
            text_chunks = [text]
        
        embeddings = []
        for chunk in text_chunks:
            response = self.llm.embeddings.create(input=chunk, model=model)
            embeddings.extend([embedding.embedding for embedding in response.data])
        
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        """Create FAISS index from embeddings."""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
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
