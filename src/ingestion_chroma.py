"""Chroma-based vector database implementation.

This is an alternative to FAISS with better metadata support.
Install: pip install chromadb
"""

import os
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from tenacity import retry, wait_fixed, stop_after_attempt


class ChromaIngestor:
    """Create and manage Chroma vector database."""
    
    def __init__(self, persist_directory: Path = None):
        """Initialize Chroma client.
        
        Args:
            persist_directory: Where to store Chroma data. 
                             If None, uses in-memory mode.
        """
        load_dotenv()
        self.llm = self._set_up_llm()
        
        # Initialize Chroma client
        if persist_directory:
            persist_directory = Path(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        else:
            # In-memory mode (data lost on restart)
            self.client = chromadb.Client()
    
    def _set_up_llm(self):
        """Initialize OpenAI client for embeddings."""
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm
    
    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
        """Get embeddings from OpenAI API."""
        # Handle empty strings
        texts = [t if t.strip() else " " for t in texts]
        
        # Batch in chunks of 1024 (OpenAI limit)
        all_embeddings = []
        for i in range(0, len(texts), 1024):
            chunk = texts[i:i + 1024]
            response = self.llm.embeddings.create(input=chunk, model=model)
            all_embeddings.extend([e.embedding for e in response.data])
        
        return all_embeddings
    
    def create_or_get_collection(self, name: str = "documents"):
        """Get or create a Chroma collection."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_document(self, 
                     report_path: Path, 
                     collection_name: str = "documents") -> Dict:
        """Add a single document to Chroma.
        
        Args:
            report_path: Path to chunked JSON report
            collection_name: Chroma collection name
            
        Returns:
            Dict with stats about added document
        """
        collection = self.create_or_get_collection(collection_name)
        
        # Load report
        import json
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        metainfo = report_data.get("metainfo", {})
        sha1_name = metainfo.get("sha1_name", report_path.stem)
        document_name = metainfo.get("document_name", sha1_name)
        
        # Check if document already exists
        existing = collection.get(
            where={"sha1_name": sha1_name},
            limit=1
        )
        if existing['ids']:
            print(f"Document {document_name} already exists. Skipping...")
            return {"added": 0, "skipped": True}
        
        # Get chunks
        chunks = report_data.get("content", {}).get("chunks", [])
        if not chunks:
            print(f"No chunks found in {report_path.name}")
            return {"added": 0, "error": "No chunks"}
        
        # Prepare data for Chroma
        texts = []
        ids = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue
            
            texts.append(chunk_text)
            ids.append(f"{sha1_name}_chunk_{i}")
            metadatas.append({
                "sha1_name": sha1_name,
                "document_name": document_name,
                "filename": metainfo.get("filename", ""),
                "page": chunk.get("page", 0),
                "chunk_id": chunk.get("id", i),
                "chunk_type": chunk.get("type", "content"),
                "token_count": chunk.get("length_tokens", 0)
            })
        
        if not texts:
            return {"added": 0, "error": "No valid chunks"}
        
        # Get embeddings
        embeddings = self._get_embeddings(texts)
        
        # Add to Chroma in batches (Chroma has 5461 limit per add)
        batch_size = 5000
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        return {
            "added": len(texts),
            "sha1_name": sha1_name,
            "document_name": document_name
        }
    
    def process_reports(self, 
                        all_reports_dir: Path, 
                        collection_name: str = "documents"):
        """Process all reports and add to Chroma.
        
        Args:
            all_reports_dir: Directory containing chunked JSON reports
            collection_name: Chroma collection to use
        """
        all_report_paths = list(all_reports_dir.glob("*.json"))
        
        if not all_report_paths:
            print(f"No JSON files found in {all_reports_dir}")
            return
        
        print(f"Processing {len(all_report_paths)} reports for Chroma...")
        
        stats = {"added": 0, "skipped": 0, "errors": 0}
        
        for report_path in tqdm(all_report_paths, desc="Adding to Chroma"):
            try:
                result = self.add_document(report_path, collection_name)
                if result.get("skipped"):
                    stats["skipped"] += 1
                elif result.get("error"):
                    stats["errors"] += 1
                else:
                    stats["added"] += result["added"]
            except Exception as e:
                print(f"Error processing {report_path.name}: {e}")
                stats["errors"] += 1
        
        print(f"\nChroma ingestion complete:")
        print(f"  - Total chunks added: {stats['added']}")
        print(f"  - Documents skipped: {stats['skipped']}")
        print(f"  - Errors: {stats['errors']}")
    
    def delete_document(self, sha1_name: str, collection_name: str = "documents"):
        """Delete a document from Chroma."""
        collection = self.create_or_get_collection(collection_name)
        
        # Find all chunks for this document
        results = collection.get(
            where={"sha1_name": sha1_name}
        )
        
        if results['ids']:
            collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks for {sha1_name}")
            return True
        else:
            print(f"Document {sha1_name} not found")
            return False
    
    def list_documents(self, collection_name: str = "documents") -> List[Dict]:
        """List all unique documents in the collection."""
        collection = self.create_or_get_collection(collection_name)
        
        # Get all metadata
        results = collection.get()
        
        # Extract unique documents
        documents = {}
        for metadata in results['metadatas']:
            sha1_name = metadata.get("sha1_name")
            if sha1_name not in documents:
                documents[sha1_name] = {
                    "sha1_name": sha1_name,
                    "document_name": metadata.get("document_name", sha1_name),
                    "filename": metadata.get("filename", ""),
                    "chunk_count": 1
                }
            else:
                documents[sha1_name]["chunk_count"] += 1
        
        return list(documents.values())
    
    def get_stats(self, collection_name: str = "documents") -> Dict:
        """Get collection statistics."""
        collection = self.create_or_get_collection(collection_name)
        count = collection.count()
        
        return {
            "total_chunks": count,
            "unique_documents": len(self.list_documents(collection_name))
        }


class ChromaRetriever:
    """Retrieve documents from Chroma."""
    
    def __init__(self, persist_directory: Path = None, collection_name: str = "documents"):
        """Initialize retriever."""
        load_dotenv()
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if persist_directory:
            self.client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self.client = chromadb.Client()
        
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def _get_query_embedding(self, query: str, model: str = "text-embedding-3-large") -> List[float]:
        """Get embedding for query."""
        response = self.llm.embeddings.create(input=query, model=model)
        return response.data[0].embedding
    
    def retrieve(self, 
                 query: str, 
                 n_results: int = 6,
                 where_filter: Dict = None) -> List[Dict]:
        """Retrieve relevant chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Optional metadata filter (e.g., {"sha1_name": "abc123"})
            
        Returns:
            List of result dictionaries
        """
        query_embedding = self._get_query_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return formatted
    
    def retrieve_by_document(self,
                             query: str,
                             sha1_name: str,
                             n_results: int = 6) -> List[Dict]:
        """Retrieve from a specific document only."""
        return self.retrieve(
            query=query,
            n_results=n_results,
            where_filter={"sha1_name": sha1_name}
        )
    
    def get_all_documents(self) -> List[Dict]:
        """Get list of all documents."""
        results = self.collection.get()
        
        documents = {}
        for metadata in results['metadatas']:
            sha1_name = metadata.get("sha1_name")
            if sha1_name not in documents:
                documents[sha1_name] = {
                    "sha1_name": sha1_name,
                    "document_name": metadata.get("document_name", sha1_name),
                    "filename": metadata.get("filename", ""),
                    "pages_amount": metadata.get("page", 0)
                }
        
        return list(documents.values())
