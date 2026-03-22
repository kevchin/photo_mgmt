"""
2-Stage Pipeline: Orchestrates VLM captioning and text embedding

This is the main pipeline that:
1. Takes images as input
2. Runs Stage 1 (VLM) to generate captions
3. Runs Stage 2 (Text Embedder) to create vectors
4. Stores everything in PostgreSQL with model versioning

Supports the architecture:
[Image] → [VLM] → [Caption Text] → [Embedding Model] → [Vector] → [PostgreSQL]
"""

import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
from PIL import Image

from config.pipeline_config import PipelineConfig
from pipeline.stage1_vlm_captioner import create_vlm, BaseVLM
from pipeline.stage2_text_embedder import create_embedder, TextEmbedder

logger = logging.getLogger(__name__)


class TwoStagePipeline:
    """
    2-Stage Pipeline for image captioning and embedding generation
    
    Stage 1: VLM generates text caption from image
    Stage 2: Text embedding model converts caption to vector
    
    Key features:
    - Decoupled stages allow independent model upgrades
    - Captions are cached, enabling re-embedding without re-running VLM
    - Supports multiple embedding dimensions via model versioning
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the 2-stage pipeline.
        
        Args:
            config: Pipeline configuration with VLM and embedding model settings
        """
        self.config = config
        self.vlm: Optional[BaseVLM] = None
        self.embedder: Optional[TextEmbedder] = None
        
        # Statistics
        self.stats = {
            "images_processed": 0,
            "captions_generated": 0,
            "embeddings_created": 0,
            "errors": 0,
            "cache_hits": 0
        }
    
    def initialize(self):
        """Initialize both stages of the pipeline"""
        logger.info("Initializing 2-Stage Pipeline...")
        
        # Stage 1: Load VLM
        logger.info(f"Loading VLM: {self.config.vlm.model_id} ({self.config.vlm.model_type})")
        self.vlm = create_vlm(
            model_type=self.config.vlm.model_type,
            model_id=self.config.vlm.model_id,
            device=self.config.vlm.device,
            trust_remote_code=self.config.vlm.trust_remote_code
        )
        
        # Stage 2: Load Embedding Model
        logger.info(f"Loading Embedding Model: {self.config.embedding.model_id} ({self.config.embedding.dimension}-dim)")
        self.embedder = create_embedder(
            model_id=self.config.embedding.model_id,
            device=self.config.embedding.device,
            normalize_embeddings=self.config.embedding.normalize_embeddings,
            max_seq_length=self.config.embedding.max_seq_length
        )
        
        logger.info("Pipeline initialization complete!")
        logger.info(f"  VLM: {self.config.vlm.model_id}")
        logger.info(f"  Embedding: {self.config.embedding.model_id} ({self.embedder.dimension}-dim)")
    
    def process_single_image(self, 
                             image_path: Path,
                             use_cache: bool = True) -> Dict[str, Any]:
        """
        Process a single image through both stages.
        
        Args:
            image_path: Path to the image file
            use_cache: Whether to use cached captions if available
            
        Returns:
            Dictionary with caption, embedding, and metadata
        """
        image_path = Path(image_path)
        result = {
            "image_path": str(image_path),
            "caption": None,
            "embedding": None,
            "embedding_dimension": self.embedder.dimension,
            "vlm_model": self.config.vlm.model_id,
            "embedding_model": self.config.embedding.model_id,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
        try:
            # Check cache first
            cache_key = f"{image_path.stem}_{self.config.vlm.model_id.replace('/', '_')}"
            cache_file = self.config.cache_dir / f"{cache_key}.txt"
            
            caption = None
            cache_hit = False
            
            if use_cache and cache_file.exists():
                # Try to load cached caption
                try:
                    caption = cache_file.read_text().strip()
                    logger.debug(f"Cache hit for {image_path.name}")
                    self.stats["cache_hits"] += 1
                    cache_hit = True
                except Exception as e:
                    logger.warning(f"Failed to read cache for {image_path}: {e}")
            
            # Stage 1: Generate caption (if not cached)
            if caption is None:
                logger.info(f"[Stage 1] Generating caption for: {image_path.name}")
                prompt = self.config.vlm.get_prompt()
                
                caption = self.vlm.process_image_path(image_path, prompt)
                result["caption"] = caption
                self.stats["captions_generated"] += 1
                
                # Cache the caption
                if self.config.cache_captions:
                    try:
                        cache_file.write_text(caption)
                        logger.debug(f"Cached caption for {image_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to cache caption: {e}")
            else:
                result["caption"] = caption
                logger.info(f"[Stage 1] Using cached caption for: {image_path.name}")
            
            # Stage 2: Generate embedding
            logger.info(f"[Stage 2] Generating {self.embedder.dimension}-dim embedding for: {image_path.name}")
            embedding = self.embedder.encode_documents([caption])
            
            result["embedding"] = embedding[0]  # Return as 1D array
            result["embedding_dimension"] = len(embedding[0])
            self.stats["embeddings_created"] += 1
            
            self.stats["images_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            result["error"] = str(e)
            self.stats["errors"] += 1
        
        return result
    
    def process_batch(self, 
                      image_paths: List[Path],
                      use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple images through the pipeline.
        
        Args:
            image_paths: List of image paths
            use_cache: Whether to use cached captions
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
            result = self.process_single_image(image_path, use_cache=use_cache)
            results.append(result)
        
        return results
    
    def reprocess_embeddings_only(self,
                                   captions: List[str],
                                   new_embedding_model: str,
                                   new_dimension: int) -> List[np.ndarray]:
        """
        Reprocess existing captions with a new embedding model.
        
        This is the key feature for embedding evolution:
        - No need to re-run the VLM on images
        - Just load stored captions and generate new embeddings
        - Enables upgrading to larger embedding dimensions
        
        Args:
            captions: List of existing caption texts
            new_embedding_model: New embedding model ID
            new_dimension: Expected dimension of new model
            
        Returns:
            List of new embedding vectors
        """
        logger.info(f"Re-processing {len(captions)} captions with new embedding model: {new_embedding_model}")
        
        # Create temporary embedder with new model
        temp_embedder = create_embedder(
            model_id=new_embedding_model,
            device=self.config.embedding.device
        )
        
        logger.info(f"New embedding dimension: {temp_embedder.dimension}")
        
        # Generate new embeddings
        embeddings = temp_embedder.encode_documents(captions)
        
        logger.info(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    def get_stats(self) -> Dict[str, int]:
        """Get pipeline processing statistics"""
        return self.stats.copy()
    
    def print_stats(self):
        """Print pipeline statistics"""
        print("\n" + "="*50)
        print("Pipeline Statistics")
        print("="*50)
        for key, value in self.stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("="*50)


def run_pipeline_example():
    """Example usage of the 2-stage pipeline"""
    
    # Create configuration for Qwen2.5-VL + BGE-large (1024-dim)
    config = PipelineConfig.create_qwen_bge_pipeline(
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
        embedding_model="BAAI/bge-large-en-v1.5",
        database_url="postgresql://postgres:postgres@localhost:5432/photo_archive_evolution"
    )
    
    # Alternative: LLaVA-1.5 + MPNet (768-dim)
    # config = PipelineConfig.create_llava_mpnet_pipeline()
    
    # Initialize pipeline
    pipeline = TwoStagePipeline(config)
    pipeline.initialize()
    
    # Process sample images
    sample_images = [
        Path("/path/to/image1.jpg"),
        Path("/path/to/image2.jpg"),
    ]
    
    # Filter to existing files for testing
    existing_images = [p for p in sample_images if p.exists()]
    
    if existing_images:
        results = pipeline.process_batch(existing_images)
        
        for result in results:
            if result["error"]:
                print(f"Error: {result['error']}")
            else:
                print(f"\nImage: {result['image_path']}")
                print(f"Caption: {result['caption'][:100]}...")
                print(f"Embedding dim: {result['embedding_dimension']}")
                print(f"VLM: {result['vlm_model']}")
                print(f"Embedding Model: {result['embedding_model']}")
        
        pipeline.print_stats()
    else:
        print("No sample images found. Provide valid image paths.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline_example()
