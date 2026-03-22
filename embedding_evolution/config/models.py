"""
LLM Model configurations and metadata for embedding evolution.
Defines supported models, their embedding dimensions, and capabilities.
"""
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum


class ModelType(Enum):
    CAPTIONING = "captioning"
    EMBEDDING = "embedding"
    JOINT = "joint"  # Joint image-text embedding (e.g., nomic-embed-vision)


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    name: str
    model_id: str
    model_type: ModelType
    embedding_dimension: int
    description: str
    gpu_memory_required_gb: int = 4
    caption_preset: Optional[str] = None  # e.g., "detailed", "more_detailed", "caption"
    
    @property
    def column_name(self) -> str:
        """Generate the database column name for this model's embeddings."""
        safe_name = self.name.replace("-", "_").replace(".", "_")
        return f"embedding_{safe_name}_{self.embedding_dimension}"


# Pre-defined model configurations
MODELS: Dict[str, ModelConfig] = {
    # Legacy model - baseline
    "florence-2-base": ModelConfig(
        name="florence-2-base",
        model_id="microsoft/Florence-2-base",
        model_type=ModelType.CAPTIONING,
        embedding_dimension=384,
        description="Microsoft Florence-2 base model with detailed captions",
        gpu_memory_required_gb=4,
        caption_preset="detailed"
    ),
    
    # Larger Florence variants
    "florence-2-large": ModelConfig(
        name="florence-2-large",
        model_id="microsoft/Florence-2-large",
        model_type=ModelType.CAPTIONING,
        embedding_dimension=384,  # Same embedding dim, better captions
        description="Microsoft Florence-2 large model with improved caption quality",
        gpu_memory_required_gb=8,
        caption_preset="detailed"
    ),
    
    # LLaVA models - higher quality, larger embeddings
    "llava-1.5-7b": ModelConfig(
        name="llava-1.5-7b",
        model_id="llava-hf/llava-1.5-7b-hf",
        model_type=ModelType.CAPTIONING,
        embedding_dimension=1024,
        description="LLaVA 1.5 7B parameter model",
        gpu_memory_required_gb=16,
        caption_preset=None
    ),
    
    "llava-1.6-34b": ModelConfig(
        name="llava-1.6-34b",
        model_id="llava-hf/llava-v1.6-34b-hf",
        model_type=ModelType.CAPTIONING,
        embedding_dimension=1584,
        description="LLaVA 1.6 34B parameter model with high-quality captions",
        gpu_memory_required_gb=48,
        caption_preset=None
    ),
    
    # BLIP models
    "blip2-opt-2.7b": ModelConfig(
        name="blip2-opt-2.7b",
        model_id="Salesforce/blip2-opt-2.7b",
        model_type=ModelType.CAPTIONING,
        embedding_dimension=768,
        description="BLIP-2 with OPT 2.7B backbone",
        gpu_memory_required_gb=12,
        caption_preset=None
    ),
    
    # Embedding-only models (for re-embedding existing captions)
    "all-MiniLM-L6-v2": ModelConfig(
        name="all-MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        model_type=ModelType.EMBEDDING,
        embedding_dimension=384,
        description="Fast sentence transformer for embedding text captions",
        gpu_memory_required_gb=2,
        caption_preset=None
    ),
    
    "all-mpnet-base-v2": ModelConfig(
        name="all-mpnet-base-v2",
        model_id="sentence-transformers/all-mpnet-base-v2",
        model_type=ModelType.EMBEDDING,
        embedding_dimension=768,
        description="Higher quality sentence transformer",
        gpu_memory_required_gb=4,
        caption_preset=None
    ),
    
    "nomic-embed-text": ModelConfig(
        name="nomic-embed-text",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        model_type=ModelType.EMBEDDING,
        embedding_dimension=768,
        description="Nomic AI text embedding v1.5 - pairs with nomic-embed-vision",
        gpu_memory_required_gb=4,
        caption_preset=None
    ),
    
    "nomic-embed-vision-v1.5": ModelConfig(
        name="nomic-embed-vision-v1.5",
        model_id="nomic-ai/nomic-embed-vision-v1.5",
        model_type=ModelType.JOINT,  # Joint image-text embedding
        embedding_dimension=768,
        description="Nomic AI vision-language model - direct image to 768d embedding",
        gpu_memory_required_gb=6,
        caption_preset=None
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model by name."""
    if model_name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODELS[model_name]


def register_custom_model(config: ModelConfig):
    """Register a custom model configuration."""
    MODELS[config.name] = config


def list_models(model_type: Optional[ModelType] = None) -> list:
    """List all available models, optionally filtered by type."""
    if model_type:
        return [m for m in MODELS.values() if m.model_type == model_type]
    return list(MODELS.values())


def get_models_by_max_gpu_memory(max_memory_gb: int) -> list:
    """Get all models that can run on GPU with specified memory."""
    return [m for m in MODELS.values() if m.gpu_memory_required_gb <= max_memory_gb]
