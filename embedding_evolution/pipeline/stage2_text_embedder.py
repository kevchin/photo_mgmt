"""
Stage 2: Text Embedding Model

Converts text captions from Stage 1 into vector embeddings.
Supports multiple embedding models with different dimensions.

Key feature: Decoupled from Stage 1, allowing you to:
- Re-embed existing captions with new models without re-processing images
- Support multiple embedding dimensions in the same database
- Upgrade embedding models independently of captioning models
"""

import torch
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Text embedding generator using sentence transformers"""
    
    def __init__(self, 
                 model_id: str = "BAAI/bge-base-en-v1.5",
                 device: Optional[str] = None,
                 normalize_embeddings: bool = True,
                 max_seq_length: int = 512):
        """
        Initialize the text embedding model.
        
        Args:
            model_id: HuggingFace model ID for the embedding model
            device: Device to run on ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to L2-normalize output vectors
            max_seq_length: Maximum sequence length for tokenization
        """
        self.model_id = model_id
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_id} on {self.device}")
        self._load_model()
        
        # Get embedding dimension
        self.dimension = self._get_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(
                self.model_id,
                device=self.device
            )
            
            self.model.max_seq_length = self.max_seq_length
            
            logger.info(f"Embedding model loaded successfully: {self.model_id}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model"""
        try:
            # Try to get dimension from model config
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                dim = self.model.get_sentence_embedding_dimension()
                if dim is not None:
                    return dim
            
            # Fallback: encode a test sentence
            test_embedding = self.encode("test")
            return len(test_embedding)
            
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension automatically: {e}")
            # Common dimensions as fallback
            dimension_map = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "BAAI/bge-small-en-v1.5": 384,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-large-en-v1.5": 1024,
                "thenlper/gte-large": 1024,
                "intfloat/e5-large-v2": 1024,
            }
            
            for model_name, dim in dimension_map.items():
                if model_name in self.model_id:
                    return dim
            
            # Ultimate fallback
            logger.warning("Using default dimension 768")
            return 768
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings with shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query text.
        
        Some models (like BGE) benefit from query-specific prefixes.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector as 1D numpy array
        """
        # Add query prefix for models that benefit from it
        if "bge" in self.model_id.lower():
            query_text = f"Represent this sentence for searching relevant passages: {text}"
        else:
            query_text = text
        
        embeddings = self.encode(query_text)
        return embeddings[0]
    
    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple documents (captions) for storage.
        
        Args:
            texts: List of caption texts
            
        Returns:
            Embedding matrix with shape (n_texts, dimension)
        """
        # Add document prefix for models that benefit from it
        if "bge" in self.model_id.lower():
            prefixed_texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        else:
            prefixed_texts = texts
        
        return self.encode(prefixed_texts)
    
    def similarity(self, 
                   query_embedding: np.ndarray, 
                   document_embeddings: np.ndarray,
                   metric: str = "cosine") -> np.ndarray:
        """
        Compute similarity between query and document embeddings.
        
        Args:
            query_embedding: Query vector (1D or 2D)
            document_embeddings: Document vectors (2D)
            metric: Similarity metric ('cosine', 'dot', 'euclidean')
            
        Returns:
            Array of similarity scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if metric == "cosine" or metric == "dot":
            # Since embeddings are normalized, dot product = cosine similarity
            similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        elif metric == "euclidean":
            distances = np.linalg.norm(document_embeddings - query_embedding, axis=1)
            similarities = -distances  # Convert to similarity (higher is better)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return similarities


def create_embedder(model_id: str, 
                    device: Optional[str] = None,
                    **kwargs) -> TextEmbedder:
    """
    Factory function to create text embedder.
    
    Args:
        model_id: HuggingFace model ID
        device: Device override
        **kwargs: Additional arguments for TextEmbedder
        
    Returns:
        TextEmbedder instance
    """
    return TextEmbedder(model_id=model_id, device=device, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Stage 2: Text Embedding Model")
    print("=" * 50)
    
    # Test with different models
    test_models = [
        "all-MiniLM-L6-v2",      # 384-dim (legacy Florence-2 compatibility)
        "BAAI/bge-base-en-v1.5", # 768-dim
        "BAAI/bge-large-en-v1.5" # 1024-dim
    ]
    
    test_captions = [
        "A golden retriever playing on a beach at sunset",
        "Children building sandcastles near the ocean",
        "Mountain landscape with snow-capped peaks"
    ]
    
    for model_id in test_models:
        print(f"\nTesting model: {model_id}")
        try:
            embedder = create_embedder(model_id)
            print(f"  Dimension: {embedder.dimension}")
            
            # Encode test captions
            embeddings = embedder.encode_documents(test_captions)
            print(f"  Generated embeddings shape: {embeddings.shape}")
            
            # Test query
            query_emb = embedder.encode_query("dogs at beach")
            similarities = embedder.similarity(query_emb, embeddings)
            
            print(f"  Query similarities: {similarities}")
            print(f"  Most similar: '{test_captions[np.argmax(similarities)]}'")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
    print("Stage 2 testing complete!")
