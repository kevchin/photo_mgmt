"""
Joint Vision-Text Embedding Pipeline using Nomic Embed

This module implements direct image-to-text embedding without caption generation.
Both images and text are embedded into the same 768-dimensional space, enabling
direct similarity comparison.

Usage:
    from pipeline.joint_embedder import NomicJointEmbedder
    
    embedder = NomicJointEmbedder(device="cuda")
    
    # Embed an image
    image_embedding = embedder.embed_image("photo.jpg")
    
    # Embed a text query
    text_embedding = embedder.embed_text("kids playing on the beach")
    
    # Compare directly (cosine similarity)
    similarity = cosine_similarity(image_embedding, text_embedding)
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional
from transformers import AutoModel, AutoProcessor
import logging

logger = logging.getLogger(__name__)


class NomicJointEmbedder:
    """
    Joint vision-text embedder using Nomic Embed models.
    
    This embedder uses:
    - nomic-embed-vision-v1.5 for image embeddings
    - nomic-embed-text-v1.5 for text embeddings
    
    Both models produce 768-dimensional vectors in the same embedding space,
    allowing direct comparison between image and text embeddings.
    """
    
    def __init__(
        self,
        vision_model_id: str = "nomic-ai/nomic-embed-vision-v1.5",
        text_model_id: str = "nomic-embed-text-v1.5",
        device: str = "cuda",
        half_precision: bool = True
    ):
        """
        Initialize the joint embedder.
        
        Args:
            vision_model_id: HuggingFace model ID for vision encoder
            text_model_id: HuggingFace model ID for text encoder
            device: Device to run models on ("cuda" or "cpu")
            half_precision: Use FP16 for reduced VRAM usage
        """
        self.device = device
        self.half_precision = half_precision
        self.vision_model_id = vision_model_id
        self.text_model_id = text_model_id
        
        logger.info(f"Loading Nomic vision model: {vision_model_id}")
        logger.info(f"Loading Nomic text model: {text_model_id}")
        
        # Load vision model
        try:
            self.vision_model = AutoModel.from_pretrained(
                vision_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if half_precision else torch.float32,
                device_map=device if device == "cuda" else None
            )
            if device == "cuda" and not half_precision:
                self.vision_model = self.vision_model.to(device)
            self.vision_model.eval()
            
            self.vision_processor = AutoProcessor.from_pretrained(
                vision_model_id,
                trust_remote_code=True
            )
            logger.info("✓ Vision model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise
        
        # Load text model
        try:
            self.text_model = AutoModel.from_pretrained(
                text_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if half_precision else torch.float32,
                device_map=device if device == "cuda" else None
            )
            if device == "cuda" and not half_precision:
                self.text_model = self.text_model.to(device)
            self.text_model.eval()
            
            self.text_processor = AutoProcessor.from_pretrained(
                text_model_id,
                trust_remote_code=True
            )
            logger.info("✓ Text model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            raise
        
        logger.info(f"Nomic Joint Embedder initialized on {device}")
        logger.info(f"Embedding dimension: 768")
    
    @torch.no_grad()
    def embed_image(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed an image into a 768-dimensional vector.
        
        Args:
            image: Image path, PIL Image, or numpy array
            normalize: Whether to normalize the embedding (recommended for similarity search)
        
        Returns:
            numpy array of shape (768,)
        """
        # Load image if path provided
        if isinstance(image, (Path, str)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        # Process image
        inputs = self.vision_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embedding
        if self.half_precision:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        outputs = self.vision_model(**inputs)
        embedding = outputs.image_embeds
        
        # Normalize if requested
        if normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy().flatten()
        
        logger.debug(f"Embedded image: shape={embedding_np.shape}, dtype={embedding_np.dtype}")
        return embedding_np
    
    @torch.no_grad()
    def embed_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed text into a 768-dimensional vector.
        
        Args:
            text: Single string or list of strings
            normalize: Whether to normalize the embedding (recommended for similarity search)
        
        Returns:
            numpy array of shape (768,) for single text, or (n, 768) for multiple texts
        """
        # Handle single string
        if isinstance(text, str):
            text = [text]
            single = True
        else:
            single = False
        
        # Process text
        inputs = self.text_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192  # Nomic supports up to 8192 tokens
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embedding
        if self.half_precision:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        outputs = self.text_model(**inputs)
        embedding = outputs.text_embeds
        
        # Normalize if requested
        if normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy()
        
        if single:
            embedding_np = embedding_np.flatten()
        
        logger.debug(f"Embedded text ({len(text)} items): shape={embedding_np.shape}")
        return embedding_np
    
    @torch.no_grad()
    def embed_batch(
        self,
        images: List[Union[Path, str, Image.Image]],
        batch_size: int = 8,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed multiple images in batches.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Number of images per batch
            normalize: Whether to normalize embeddings
        
        Returns:
            numpy array of shape (n_images, 768)
        """
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
            
            # Load and process images
            pil_images = []
            for img in batch:
                if isinstance(img, (Path, str)):
                    pil_images.append(Image.open(img).convert("RGB"))
                elif isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGB"))
                else:
                    pil_images.append(Image.fromarray(img).convert("RGB"))
            
            # Batch processing
            inputs = self.vision_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.half_precision:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            outputs = self.vision_model(**inputs)
            embeddings = outputs.image_embeds
            
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_dimension(self) -> int:
        """Return the embedding dimension (768 for Nomic models)."""
        return 768
    
    def get_model_info(self) -> dict:
        """Return information about the loaded models."""
        return {
            "vision_model": self.vision_model_id,
            "text_model": self.text_model_id,
            "dimension": 768,
            "device": self.device,
            "half_precision": self.half_precision
        }


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two normalized vectors.
    
    Args:
        vec1: First vector (should be normalized)
        vec2: Second vector (should be normalized)
    
    Returns:
        Cosine similarity score between -1 and 1
    """
    if vec1.ndim > 1:
        vec1 = vec1.flatten()
    if vec2.ndim > 1:
        vec2 = vec2.flatten()
    
    return float(np.dot(vec1, vec2))


# Example usage
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Check if we have an image to test
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python joint_embedder.py <image_path>")
        print("Testing with dummy data...")
        
        # Test initialization
        try:
            embedder = NomicJointEmbedder(device="cuda", half_precision=True)
            print(f"✓ Successfully initialized Nomic Joint Embedder")
            print(f"  Dimension: {embedder.get_dimension()}")
            print(f"  Models: {embedder.get_model_info()}")
        except Exception as e:
            print(f"✗ Failed to initialize: {e}")
            print("\nNote: This requires the nomic-ai models to be downloaded.")
            print("Run: pip install transformers torch pillow")
            sys.exit(1)
        
        sys.exit(0)
    
    # Test with actual image
    try:
        embedder = NomicJointEmbedder(device="cuda", half_precision=True)
        
        # Embed image
        print(f"\nEmbedding image: {image_path}")
        image_emb = embedder.embed_image(image_path)
        print(f"Image embedding shape: {image_emb.shape}")
        print(f"Image embedding norm: {np.linalg.norm(image_emb):.4f}")
        
        # Embed text queries
        queries = [
            "a photo of nature",
            "people at the beach",
            "indoor scene"
        ]
        
        print(f"\nEmbedding text queries:")
        for query in queries:
            text_emb = embedder.embed_text(query)
            similarity = cosine_similarity(image_emb, text_emb)
            print(f"  '{query}': {similarity:.4f}")
        
        # Find best match
        best_idx = np.argmax([cosine_similarity(image_emb, embedder.embed_text(q)) for q in queries])
        print(f"\nBest matching query: '{queries[best_idx]}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
