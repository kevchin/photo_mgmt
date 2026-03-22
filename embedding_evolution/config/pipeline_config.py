"""
Configuration for the 2-Stage Embedding Evolution Pipeline

Stage 1: VLM (Vision-Language Model) generates captions from images
Stage 2: Text Embedding Model converts captions to vectors

This separation allows you to:
- Change captioning models without re-running embeddings (if text is stored)
- Change embedding models without re-running VLM (if text is cached)
- Support different vector dimensions independently of caption quality
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class VLMConfig:
    """Configuration for Vision-Language Model (Stage 1)"""
    model_id: str
    model_type: str  # "florence2", "llava", "qwen2vl", "blip2"
    prompt_type: str = "detailed"  # "detailed", "caption", "tags"
    max_tokens: int = 512
    device: str = "cuda"
    trust_remote_code: bool = True
    
    # Model-specific settings
    florence_prompt_map: Dict[str, str] = field(default_factory=lambda: {
        "detailed": "<DETAILED_CAPTION>",
        "caption": "<CAPTION>",
        "tags": "<TAGS>"
    })
    
    llava_prompt_map: Dict[str, str] = field(default_factory=lambda: {
        "detailed": "Describe this image in detail.",
        "caption": "Provide a brief caption for this image.",
        "tags": "List key objects and concepts in this image."
    })
    
    qwen_prompt_map: Dict[str, str] = field(default_factory=lambda: {
        "detailed": "Please provide a detailed description of this image.",
        "caption": "Caption this image briefly.",
        "tags": "What are the main elements in this image?"
    })
    
    def get_prompt(self, prompt_type: Optional[str] = None) -> str:
        """Get the appropriate prompt for the model type"""
        ptype = prompt_type or self.prompt_type
        
        if self.model_type == "florence2":
            return self.florence_prompt_map.get(ptype, self.florence_prompt_map["detailed"])
        elif self.model_type == "llava":
            return self.llava_prompt_map.get(ptype, self.llava_prompt_map["detailed"])
        elif self.model_type == "qwen2vl":
            return self.qwen_prompt_map.get(ptype, self.qwen_prompt_map["detailed"])
        else:
            return "Describe this image in detail."


@dataclass
class EmbeddingModelConfig:
    """Configuration for Text Embedding Model (Stage 2)"""
    model_id: str
    dimension: int  # e.g., 384, 768, 1024, 1536
    model_type: str = "sentence_transformer"  # "sentence_transformer", "openai", "custom"
    device: str = "cuda"
    normalize_embeddings: bool = True
    max_seq_length: int = 512
    
    # Common embedding models and their dimensions
    MODEL_DIMENSIONS: Dict[str, int] = field(default_factory=lambda: {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "thenlper/gte-large": 1024,
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-mistral-7b-instruct": 4096,
        # Nomic Embed models (Vision + Text paired)
        "nomic-embed-vision-v1.5": 768,
        "nomic-embed-text-v1.5": 768,
    })
    
    def __post_init__(self):
        """Auto-detect dimension if not specified"""
        if self.dimension is None and self.model_id in self.MODEL_DIMENSIONS:
            self.dimension = self.MODEL_DIMENSIONS[self.model_id]


@dataclass
class PipelineConfig:
    """Complete 2-Stage Pipeline Configuration"""
    # Database connection
    database_url: str = "postgresql://postgres:postgres@localhost:5432/photo_archive_evolution"
    
    # Stage 1: VLM Configuration
    vlm: VLMConfig = field(default_factory=lambda: VLMConfig(
        model_id="microsoft/Florence-2-base",
        model_type="florence2",
        prompt_type="detailed"
    ))
    
    # Stage 2: Embedding Model Configuration
    embedding: EmbeddingModelConfig = field(default_factory=lambda: EmbeddingModelConfig(
        model_id="BAAI/bge-base-en-v1.5",
        dimension=768
    ))
    
    # Processing settings
    batch_size: int = 8
    num_workers: int = 4
    cache_captions: bool = True  # Store raw captions for re-embedding
    overwrite_existing: bool = False
    
    # Storage paths
    cache_dir: Path = Path("./cache")
    log_dir: Path = Path("./logs")
    
    def __post_init__(self):
        """Create necessary directories"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def create_qwen_bge_pipeline(cls, 
                                  vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                                  embedding_model: str = "BAAI/bge-large-en-v1.5",
                                  database_url: Optional[str] = None) -> 'PipelineConfig':
        """Create a pipeline with Qwen2.5-VL and BGE-large (1024-dim)"""
        
        config = cls(
            vlm=VLMConfig(
                model_id=vlm_model,
                model_type="qwen2vl",
                prompt_type="detailed"
            ),
            embedding=EmbeddingModelConfig(
                model_id=embedding_model,
                dimension=1024
            ),
            database_url=database_url or cls.database_url
        )
        return config
    
    @classmethod
    def create_llava_mpnet_pipeline(cls,
                                     vlm_model: str = "llava-hf/llava-1.5-7b-hf",
                                     embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                                     database_url: Optional[str] = None) -> 'PipelineConfig':
        """Create a pipeline with LLaVA-1.5 and MPNet (768-dim)"""
        
        config = cls(
            vlm=VLMConfig(
                model_id=vlm_model,
                model_type="llava",
                prompt_type="detailed"
            ),
            embedding=EmbeddingModelConfig(
                model_id=embedding_model,
                dimension=768
            ),
            database_url=database_url or cls.database_url
        )
        return config
    
    @classmethod
    def create_nomic_vision_text_pipeline(cls,
                                           vlm_model: str = "microsoft/Florence-2-base",
                                           text_embedding_model: str = "nomic-embed-text-v1.5",
                                           database_url: Optional[str] = None) -> 'PipelineConfig':
        """
        Create a pipeline using Nomic Embed models.
        
        NOTE: nomic-embed-vision-v1.5 is a JOINT embedding model that embeds 
        both images and text into the SAME 768-dim space. This is different from
        the 2-stage pipeline where VLM generates text then text is embedded.
        
        For Nomic Vision-Text paired embeddings, you have TWO options:
        
        OPTION A (Recommended for Nomic): Use joint embedding
          - Image → nomic-embed-vision-v1.5 → 768-dim vector
          - Text query → nomic-embed-text-v1.5 → 768-dim vector
          - Both vectors are directly comparable (same space)
          - No caption generation needed
        
        OPTION B (2-stage with Nomic text embedder):
          - Image → VLM (Florence/LLaVA) → Caption text
          - Caption → nomic-embed-text-v1.5 → 768-dim vector
          - This is what this method creates
        
        Args:
            vlm_model: VLM for generating captions (Stage 1)
            text_embedding_model: Should be "nomic-embed-text-v1.5" for 768-dim
            database_url: PostgreSQL connection string
        """
        
        config = cls(
            vlm=VLMConfig(
                model_id=vlm_model,
                model_type="florence2" if "florence" in vlm_model.lower() else "llava",
                prompt_type="detailed"
            ),
            embedding=EmbeddingModelConfig(
                model_id=text_embedding_model,
                dimension=768
            ),
            database_url=database_url or cls.database_url
        )
        return config
    
    @classmethod
    def create_nomic_joint_pipeline(cls,
                                     vision_model: str = "nomic-ai/nomic-embed-vision-v1.5",
                                     text_model: str = "nomic-embed-text-v1.5",
                                     database_url: Optional[str] = None) -> 'PipelineConfig':
        """
        Create a pipeline using Nomic's JOINT vision-text embedding.
        
        This bypasses caption generation entirely:
          - Image → nomic-embed-vision-v1.5 → 768-dim vector (stored in DB)
          - Text query → nomic-embed-text-v1.5 → 768-dim vector (at search time)
        
        Advantages:
          - Faster ingestion (no caption generation)
          - Direct image-to-text similarity in same embedding space
          - Optimized for cross-modal retrieval
        
        Disadvantages:
          - No human-readable captions stored
          - Cannot search by caption text content, only by similarity
        
        This requires a modified pipeline (see pipeline/joint_embedder.py)
        """
        
        # Special config marker for joint embedding mode
        config = cls(
            vlm=VLMConfig(
                model_id=vision_model,
                model_type="nomic_vision",
                prompt_type="joint_embedding"
            ),
            embedding=EmbeddingModelConfig(
                model_id=text_model,
                dimension=768
            ),
            database_url=database_url or cls.database_url,
            cache_captions=False  # No captions in joint mode
        )
        return config
    
    def save_to_file(self, filepath: Path):
        """Save configuration to YAML/JSON file"""
        import json
        from dataclasses import asdict
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'PipelineConfig':
        """Load configuration from YAML/JSON file"""
        import json
        from dataclasses import fields
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        vlm_data = data.pop('vlm', {})
        embed_data = data.pop('embedding', {})
        
        config = cls(**data)
        config.vlm = VLMConfig(**vlm_data)
        config.embedding = EmbeddingModelConfig(**embed_data)
        
        return config
