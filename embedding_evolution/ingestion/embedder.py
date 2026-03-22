"""
Embedding generation for text captions using sentence transformers.
Supports multiple embedding models with different dimensions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from config.models import ModelConfig, get_model_config


class Embedder:
    """Generate embeddings for text captions using specified models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Name of the embedding model (e.g., "all-MiniLM-L6-v2")
            device: Device to run model on ("cuda", "cpu", or None for auto-detect)
        """
        self.config = get_model_config(model_name)
        self.model_name = model_name
        
        # Verify this is an embedding model or captioning model
        if self.config.model_type not in [ModelType.EMBEDDING, ModelType.CAPTIONING]:
            raise ValueError(f"Model {model_name} is not suitable for embedding generation")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading embedding model: {self.config.model_id}")
        self.model = SentenceTransformer(self.config.model_id, device=self.device)
        print(f"Embedding model loaded on {self.device}")
        
        # Verify embedding dimension matches expected
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        actual_dim = len(test_embedding)
        if actual_dim != self.config.embedding_dimension:
            print(f"Warning: Expected dimension {self.config.embedding_dimension}, "
                  f"got {actual_dim}")
            # Update config with actual dimension
            self.config.embedding_dimension = actual_dim
    
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize the embedding vector
            
        Returns:
            Numpy array of embedding values
        """
        embedding = self.model.encode(
            text, 
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embedding
    
    def encode_batch(self, texts: List[str], normalize: bool = True, 
                     batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize the embedding vectors
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (len(texts), embedding_dimension)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the embedding dimension for this model."""
        return self.config.embedding_dimension
    
    def get_column_name(self) -> str:
        """Get the database column name for this model's embeddings."""
        return self.config.column_name


def create_embedding_for_caption(caption: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Convenience function to create embedding for a single caption.
    
    Args:
        caption: Caption text to embed
        model_name: Name of embedding model to use
        
    Returns:
        Numpy array of embedding values
    """
    embedder = Embedder(model_name)
    return embedder.encode(caption)


def test_embedder():
    """Test the embedder with sample text."""
    print("Testing all-MiniLM-L6-v2 embedder...")
    embedder = Embedder("all-MiniLM-L6-v2")
    
    test_texts = [
        "Kids playing at the beach on a sunny day",
        "A black and white photo of mountains",
        "Family gathering at Christmas dinner"
    ]
    
    embeddings = embedder.encode_batch(test_texts, show_progress=True)
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Expected dimension: {embedder.get_dimension()}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"\nSimilarity between 'beach' and 'mountains': {similarity:.4f}")
    
    return embeddings


if __name__ == "__main__":
    test_embedder()
