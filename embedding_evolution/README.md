# Embedding Evolution System - 2-Stage Pipeline

## Architecture Overview

This system implements a **2-stage pipeline** for image captioning and embedding generation:

```
┌─────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌──────────┐     ┌───────────┐
│  Image  │ ──► │ Stage 1: VLM Model   │ ──► │ Caption Text        │ ──► │ Stage 2:  │ ──► │ Vector    │
│         │     │ (Qwen2.5-VL, LLaVA,  │     │ "A golden retriever │     │ Embedding │     │ [1024-d]  │
│         │     │  Florence-2, etc.)   │     │  on a dock at       │     │ Model     │     │           │
│         │     │                      │     │  sunset..."         │     │ (BGE,     │     │           │
└─────────┘     └──────────────────────┘     └─────────────────────┘     │  MPNet,   │          │
                                                                          │  etc.)    │          ▼
                                                                          └───────────┘     ┌───────────┐
                                                                                            │ PostgreSQL│
                                                                                            │ pgvector  │
                                                                                            └───────────┘
```

### Key Benefits of 2-Stage Design

1. **Independent Model Upgrades**: Change the VLM without re-running embeddings (if you keep captions)
2. **Flexible Embedding Dimensions**: Change embedding model without re-processing images
3. **Caption Caching**: Store raw captions to enable re-embedding without GPU-intensive VLM processing
4. **Multi-Model Support**: Store multiple embedding dimensions in the same database

## Quick Start

### 1. Install Dependencies

```bash
cd /workspace/embedding_evolution
pip install -r requirements.txt
```

### 2. Create Evolution Database

```bash
python -c "
from database.evolution_schema import EvolutionDatabase
db = EvolutionDatabase('postgresql://postgres:postgres@localhost:5432/photo_archive_evolution')
db.create_schema()
db.add_embedding_column('all-MiniLM-L6-v2', 384)    # Legacy
db.add_embedding_column('bge-base-en-v1.5', 768)    # Medium
db.add_embedding_column('bge-large-en-v1.5', 1024)  # High quality
"
```

### 3. Run the 2-Stage Pipeline

```python
from config.pipeline_config import PipelineConfig
from pipeline.stage3_orchestrator import TwoStagePipeline
from pathlib import Path

# Configure pipeline: Qwen2.5-VL (Stage 1) + BGE-large (Stage 2)
config = PipelineConfig.create_qwen_bge_pipeline(
    vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
    embedding_model="BAAI/bge-large-en-v1.5"
)

# Initialize and run
pipeline = TwoStagePipeline(config)
pipeline.initialize()

# Process images
images = [Path("/photos/2024/01/15/DSC_0001.jpg")]
results = pipeline.process_batch(images)

for result in results:
    print(f"Caption: {result['caption']}")
    print(f"Embedding dim: {result['embedding_dimension']}")
```

## Supported Models

### Stage 1: Vision-Language Models (VLM)

| Model | Size | VRAM Required | Description |
|-------|------|---------------|-------------|
| **Florence-2-base** | 230MB | ~2GB | Fast, good quality, your current model |
| **LLaVA-1.5-7B** | 7B | ~16GB | Better detail, more VRAM |
| **Qwen2.5-VL-7B** | 7B | ~16GB | Excellent detail, recommended upgrade |
| **LLaVA-1.6-34B** | 34B | ~48GB | State-of-the-art, needs high-end GPU |

### Stage 2: Text Embedding Models

| Model | Dimension | Quality | Use Case |
|-------|-----------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | Good | Legacy compatibility (your current) |
| **all-mpnet-base-v2** | 768 | Better | Balanced performance/quality |
| **BAAI/bge-base-en-v1.5** | 768 | Better | Recommended for general use |
| **BAAI/bge-large-en-v1.5** | 1024 | Best | High-quality search (recommended) |
| **intfloat/e5-mistral-7b** | 4096 | Ultimate | Maximum quality, needs more storage |

## Migration from Legacy System

Your current system uses:
- **VLM**: Florence-2-base with detailed captions
- **Embedding**: 384-dimensional
- **Database**: `image_archive`

The migration process creates a NEW database without affecting your existing one:

```bash
python migrations/migrate_legacy.py \
    --source-db "postgresql://postgres:postgres@localhost:5432/image_archive" \
    --target-db "postgresql://postgres:postgres@localhost:5432/photo_archive_evolution"
```

## Database Schema

Key feature: **Multiple embedding columns per photo**:

```sql
CREATE TABLE photos (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR UNIQUE,
    caption_text TEXT,              -- Raw caption (can be re-embedded!)
    
    -- Multiple embedding columns:
    embedding_384 VECTOR(384),      -- Legacy: Florence-2 + MiniLM
    embedding_768 VECTOR(768),      -- Medium: BGE-base
    embedding_1024 VECTOR(1024),    -- High: Qwen2.5-VL + BGE-large
    embedding_1536 VECTOR(1536),    -- Future models
    
    -- Metadata:
    capture_date TIMESTAMP,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    is_black_and_white BOOLEAN,
    orientation INTEGER
);
```

## Testing

```bash
# Test configuration module
python -c "from config.pipeline_config import PipelineConfig; print('✓ Config OK')"

# Test database schema (requires PostgreSQL)
python database/evolution_schema.py
```

For complete documentation, see `QUICKSTART.md`.
