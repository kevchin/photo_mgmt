# Embedding Evolution System - 2-Stage & Joint Embedding Pipeline

## Architecture Overview

This system implements **TWO pipeline options** for image embedding:

### Option A: 2-Stage Pipeline (Caption-Based)

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

**Best for**: Systems that need human-readable captions, text-based search, and flexibility to re-embed with new models.

### Option B: Joint Embedding Pipeline (Nomic Vision-Text) ⭐ NEW

```
┌─────────┐     ┌─────────────────────────┐     ┌───────────┐     ┌───────────┐
│  Image  │ ──► │ nomic-embed-vision-v1.5 │ ──► │ Vector    │ ──► │ PostgreSQL│
│         │     │                         │     │ [768-d]   │     │ pgvector  │
└─────────┘     └─────────────────────────┘     └───────────┘     └───────────┘
                                                              ▲
┌─────────┐     ┌─────────────────────────┐     ┌───────────┐ │
│  Text   │ ──► │ nomic-embed-text-v1.5   │ ──► │ Vector    │─┘
│  Query  │     │                         │     │ [768-d]   │
└─────────┘     └─────────────────────────┘     └───────────┘
```

**Best for**: Fast ingestion, direct image-to-text similarity, optimized cross-modal retrieval. No captions stored.

### Key Benefits of Dual-Pipeline Design

1. **Choose Your Approach**: Use 2-stage for captions, joint for speed, or BOTH for maximum flexibility
2. **Independent Model Upgrades**: Change models in either pipeline without affecting the other
3. **Multi-Embedding Support**: Store both 384-dim (legacy) and 768-dim (Nomic) in same database
4. **Hybrid Search**: Query across both embedding types for best results

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
db.add_embedding_column('all-MiniLM-L6-v2', 384)              # Legacy Florence-2
db.add_embedding_column('bge-base-en-v1.5', 768)              # Medium quality
db.add_embedding_column('bge-large-en-v1.5', 1024)            # High quality
db.add_embedding_column('nomic-embed-vision-v1.5', 768)       # Nomic joint (NEW!)
"
```

### 3a. Run 2-Stage Pipeline (Caption-Based)

```python
from config.pipeline_config import PipelineConfig
from pipeline.stage3_orchestrator import TwoStagePipeline
from pathlib import Path

# Configure: Qwen2.5-VL (Stage 1) + BGE-large (Stage 2)
config = PipelineConfig.create_qwen_bge_pipeline(
    vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
    embedding_model="BAAI/bge-large-en-v1.5"
)

pipeline = TwoStagePipeline(config)
pipeline.initialize()

# Process images
images = [Path("/photos/2024/01/15/DSC_0001.jpg")]
results = pipeline.process_batch(images)
```

### 3b. Run Joint Embedding Pipeline (Nomic) ⭐ NEW

```bash
# Command line
python ingestion/nomic_joint_ingest.py \
    --photos /path/to/photos \
    --batch-size 4 \
    --limit 100

# Python API
from pipeline.joint_embedder import NomicJointEmbedder

embedder = NomicJointEmbedder(device="cuda")

# Embed image
image_vec = embedder.embed_image("photo.jpg")

# Embed text query
text_vec = embedder.embed_text("kids playing on beach")

# Direct comparison (same embedding space!)
similarity = np.dot(image_vec, text_vec)
```

## Supported Models

### 2-Stage Pipeline Models

#### Stage 1: Vision-Language Models (VLM)

| Model | Size | VRAM Required | Description |
|-------|------|---------------|-------------|
| **Florence-2-base** | 230MB | ~2GB | Fast, good quality, your current model |
| **LLaVA-1.5-7B** | 7B | ~16GB | Better detail, more VRAM |
| **Qwen2.5-VL-7B** | 7B | ~16GB | Excellent detail, recommended upgrade |
| **LLaVA-1.6-34B** | 34B | ~48GB | State-of-the-art, needs high-end GPU |

#### Stage 2: Text Embedding Models

| Model | Dimension | Quality | Use Case |
|-------|-----------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | Good | Legacy compatibility (your current) |
| **all-mpnet-base-v2** | 768 | Better | Balanced performance/quality |
| **BAAI/bge-base-en-v1.5** | 768 | Better | Recommended for general use |
| **nomic-embed-text-v1.5** | 768 | Better | Paired with Nomic vision (NEW!) |
| **BAAI/bge-large-en-v1.5** | 1024 | Best | High-quality search (recommended) |
| **intfloat/e5-mistral-7b** | 4096 | Ultimate | Maximum quality, needs more storage |

### Joint Embedding Pipeline Models ⭐ NEW

| Model Pair | Dimension | VRAM | Description |
|------------|-----------|------|-------------|
| **nomic-embed-vision-v1.5** + **nomic-embed-text-v1.5** | 768 | ~3GB | Optimized cross-modal retrieval, Apache 2.0 license |

**Key Advantage**: Both image and text are embedded into the SAME 768-dim space by design, enabling direct comparison without caption generation.

## Nomic Embed Guide ⭐ NEW

For detailed information about using Nomic Embed models, see **[NOMIC_GUIDE.md](NOMIC_GUIDE.md)** which covers:

- Joint vs. two-stage embedding comparison
- Why Nomic is recommended for your use case
- Hybrid approach (running both pipelines in parallel)
- GPU memory requirements for GTX 970 (4GB)
- Migration path from Florence-2 to Nomic
- Code examples for ingestion and search

### Quick Nomic Test on GTX 970

```bash
# Test with small batch (adjust batch-size if you get OOM errors)
python ingestion/nomic_joint_ingest.py \
    --photos /path/to/test/photos \
    --batch-size 2 \
    --limit 10 \
    --dry-run

# Monitor VRAM usage
watch -n 1 nvidia-smi
```

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

After migration, you can:
1. Continue using your legacy 384-dim embeddings (backward compatible)
2. Add new photos with Nomic 768-dim embeddings
3. Compare search quality between both approaches
4. Gradually migrate old photos if desired

## Database Schema

Key feature: **Multiple embedding columns per photo**:

```sql
CREATE TABLE photos (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR UNIQUE,
    caption_text TEXT,              -- Raw caption (can be re-embedded!)
    
    -- Multiple embedding columns:
    embedding_384 VECTOR(384),      -- Legacy: Florence-2 + MiniLM
    embedding_768 VECTOR(768),      -- Medium: BGE-base or Nomic text
    embedding_1024 VECTOR(1024),    -- High: Qwen2.5-VL + BGE-large
    embedding_nomic_vision VECTOR(768),  -- Nomic joint vision (NEW!)
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

# Test Nomic joint embedder (requires models downloaded)
python pipeline/joint_embedder.py

# Test database schema (requires PostgreSQL)
python database/evolution_schema.py
```

## Recommended Workflow for Your Setup

### Current (GTX 970 4GB)
1. ✅ Keep existing Florence-2 + 384-dim system working
2. ✅ Migrate to evolution database
3. 🆕 Test Nomic joint embedding on small batches
4. 🆕 Ingest NEW photos with BOTH:
   - Legacy 384-dim (for compatibility)
   - Nomic 768-dim (for testing)

### After GPU Upgrade (RTX 3090/4090 24GB+)
1. Switch to Qwen2.5-VL-7B or LLaVA-1.6-34B for better captions
2. Use BGE-large (1024-dim) for highest quality text embeddings
3. Optionally re-process old photos with new models
4. Compare and choose best performing model for your photos

## File Structure

```
embedding_evolution/
├── README.md                  # This file
├── NOMIC_GUIDE.md            # ⭐ NEW: Detailed Nomic guide
├── QUICKSTART.md             # Step-by-step tutorial
├── requirements.txt          # Python dependencies
│
├── config/
│   ├── pipeline_config.py    # Model configurations (updated with Nomic)
│   ├── database.py           # Database connection
│   └── models.py             # Model definitions
│
├── pipeline/
│   ├── stage1_vlm_captioner.py    # Stage 1: VLM captioning
│   ├── stage2_text_embedder.py    # Stage 2: Text embedding
│   ├── stage3_orchestrator.py     # 2-stage pipeline orchestration
│   └── joint_embedder.py          # ⭐ NEW: Nomic joint embedding
│
├── ingestion/
│   ├── photo_ingest.py            # 2-stage ingestion
│   └── nomic_joint_ingest.py      # ⭐ NEW: Nomic joint ingestion
│
├── database/
│   └── evolution_schema.py        # Multi-model schema
│
├── migrations/
│   └── migrate_legacy.py          # Legacy → Evolution migration
│
├── search/
│   └── vector_search.py           # Multi-model search
│
├── utils/
│   └── exif_reader.py             # EXIF metadata extraction
│
└── tests/
    └── test_pipeline.py           # Test suite
```

## Next Steps

1. **Read NOMIC_GUIDE.md** for detailed comparison and recommendations
2. **Run migration**: `python migrations/migrate_legacy.py`
3. **Test Nomic on small batch**: 
   ```bash
   python ingestion/nomic_joint_ingest.py --photos ./test_photos --limit 5 --dry-run
   ```
4. **Compare search quality** between 384-dim and 768-dim embeddings
5. **Decide**: Continue hybrid approach or fully migrate to Nomic

## Additional Resources

- [NOMIC_GUIDE.md](NOMIC_GUIDE.md) - Comprehensive Nomic Embed guide
- [Nomic Embed Vision](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)
- [Nomic Embed Text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
