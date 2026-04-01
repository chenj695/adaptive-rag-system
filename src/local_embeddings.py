"""
Local embedding helper using sentence-transformers.
Works offline with custom model path.
"""
import os
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class LocalEmbeddingModel:
    """Singleton wrapper for local sentence-transformers model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self) -> SentenceTransformer:
        """Get or load the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        if self._model is None:
            # Check for local path first (air-gapped servers)
            model_path = os.getenv("EMBEDDING_MODEL_PATH")
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            
            if model_path:
                print(f"Loading embedding model from: {model_path}")
                self._model = SentenceTransformer(model_path)
            else:
                print(f"Loading embedding model: {model_name}")
                self._model = SentenceTransformer(model_name)
            
            print(f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}")
        
        return self._model
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings."""
        model = self.get_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter empty texts
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            raise ValueError("No valid text to embed")
        
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings


# Global singleton instance
_embedding_model = None


def get_embedding_model() -> LocalEmbeddingModel:
    """Get the global embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = LocalEmbeddingModel()
    return _embedding_model
