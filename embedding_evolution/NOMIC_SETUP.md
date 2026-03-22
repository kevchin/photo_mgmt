# Nomic Embed Vision v1.5 Setup Guide

## Overview

This guide shows how to use **nomic-embed-vision-v1.5** (768 dimensions) with your embedding evolution system. This model provides higher-quality embeddings than Florence-2-base (384 dimensions) while still fitting on moderate GPUs.

## Model Architecture

### Two Approaches Supported:

#### 1. **Joint Embedding (Recommended)** - Direct Image→Vector
```
[Image] → [Nomic Embed Vision v1.5] → [768-d Vector] → PostgreSQL
                                              ↓
[Query Text] → [Nomic Embed Text v1.5] → [768-d Vector] → Similarity Search
```

**Advantages:**
- Single model handles both images and text
- No caption generation needed
- Faster ingestion (no two-stage pipeline)
- Optimized for cross-modal retrieval

**GPU Requirements:** ~6GB VRAM

#### 2. **Two-Stage Pipeline** - Image→Caption→Vector
```
[Image] → [VLM Captioner] → [Text] → [Nomic Embed Text] → [768-d Vector]
```

**Use when:** You want human-readable captions stored alongside embeddings

---

## Quick Start

### Step 1: Install Dependencies

```bash
cd /workspace/embedding_evolution
pip install sentence-transformers torch torchvision
```

### Step 2: Create Evolution Database

```bash
python database/evolution_schema.py
```

This creates the `image_archive_evolution` database with columns for:
- `embedding_florence_2_base_384` (legacy)
- `embedding_nomic_embed_vision_v1_5_768` (new)

### Step 3: Migrate Legacy Data (Optional)

```bash
python migrations/migrate_legacy.py
```

Copies your existing Florence-2-base photos to the new database.

### Step 4: Ingest New Photos with Nomic

#### Option A: Joint Embedding (Direct Image→Vector)

```bash
python ingestion/nomic_joint_ingest.py \
  --photos-dir /path/to/photos \
  --model nomic-embed-vision-v1.5
```

#### Option B: Two-Stage (Caption + Embed)

```bash
python ingestion/photo_ingest.py \
  --photos-dir /path/to/photos \
  --caption-model florence-2-base \
  --embed-model nomic-embed-text
```

### Step 5: Run Streamlit App

```bash
# Use evolution database
streamlit run search/streamlit_app.py

# Or explicitly set database
ACTIVE_DATABASE=evolution streamlit run search/streamlit_app.py
```

In the app sidebar, select:
- **"nomic-embed-vision-v1.5"** (768d) for joint embeddings
- **"florence-2-base"** (384d) for legacy embeddings

---

## Configuration Details

### Model Specifications

| Model | Type | Dimension | GPU VRAM | Column Name |
|-------|------|-----------|----------|-------------|
| `florence-2-base` | Captioning | 384 | 4GB | `embedding_florence_2_base_384` |
| `nomic-embed-vision-v1.5` | Joint | 768 | 6GB | `embedding_nomic_embed_vision_v1_5_768` |
| `nomic-embed-text` | Text Embedding | 768 | 4GB | `embedding_nomic_embed_text_768` |

### Database Schema

The evolution database supports multiple embedding columns simultaneously:

```sql
CREATE TABLE photos (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    caption_text TEXT,
    capture_date DATE,
    -- ... metadata fields ...
    
    -- Multiple embedding columns for different models
    embedding_florence_2_base_384 VECTOR(384),
    embedding_nomic_embed_vision_v1_5_768 VECTOR(768),
    embedding_nomic_embed_text_768 VECTOR(768)
);

-- Indexes for fast similarity search
CREATE INDEX idx_nomic_vision_768 ON photos 
USING hnsw (embedding_nomic_embed_vision_v1_5_768 vector_cosine_ops);
```

---

## Testing Your Setup

### Verify Models Are Configured

```bash
cd /workspace/embedding_evolution
python -c "
from config.models import get_model_config

# Check Nomic vision model
m = get_model_config('nomic-embed-vision-v1.5')
print(f'Model: {m.name}')
print(f'Dimension: {m.embedding_dimension}d')
print(f'Type: {m.model_type.value}')
print(f'Column: {m.column_name}')
"
```

Expected output:
```
Model: nomic-embed-vision-v1.5
Dimension: 768d
Type: joint
Column: embedding_nomic_embed_vision_v1_5_768
```

### Test Database Connection

```bash
python -c "
from config.database import active_engine
from sqlalchemy import text

with active_engine.connect() as conn:
    result = conn.execute(text('SELECT current_database()'))
    print(f'Connected to: {result.scalar()}')
"
```

### Test Ingestion (Single Photo)

```bash
python -c "
from pathlib import Path
from ingestion.nomic_joint_ingest import NomicJointEmbedder

embedder = NomicJointEmbedder('nomic-embed-vision-v1.5')
photo_path = Path('/path/to/test/photo.jpg')

# Generate embedding
embedding = embedder.embed_image(photo_path)
print(f'Generated embedding: {embedding.shape}')
print(f'Dimension: {len(embedding)}')
"
```

---

## Migration Strategy

### Phase 1: Parallel Operation (Current)
- Keep legacy system running with 384d Florence embeddings
- Test Nomic 768d on new photos only
- Both systems coexist in evolution database

### Phase 2: Gradual Migration (Optional)
- Re-process old photos with Nomic when GPU available
- Store new embeddings in separate column
- No need to delete old embeddings

### Phase 3: Full Transition (Future)
- Switch Streamlit default to Nomic model
- Legacy embeddings remain for backward compatibility
- New searches use 768d by default

---

## Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--batch-size 4`
2. Use CPU fallback: `CUDA_VISIBLE_DEVICES="" python ...`
3. Close other GPU applications

### Model Download Fails

**Error:** `OSError: Unable to download model`

**Solutions:**
1. Check internet connection
2. Set HuggingFace token: `export HF_TOKEN=your_token`
3. Clear cache: `rm -rf ~/.cache/huggingface`

### pgvector Not Installed

**Error:** `relation "vector" does not exist`

**Solution:**
```sql
psql -U postgres -d image_archive_evolution
CREATE EXTENSION IF NOT EXISTS vector;
```

### Wrong Dimension Error

**Error:** `expected dimension 768, got 384`

**Cause:** Mixing embeddings from different models in same column

**Solution:** Ensure you're using the correct model for each column:
- Florence-2-base → `embedding_florence_2_base_384`
- Nomic Vision → `embedding_nomic_embed_vision_v1_5_768`

---

## Performance Comparison

| Metric | Florence-2-base (384d) | Nomic Vision (768d) |
|--------|------------------------|---------------------|
| Embedding Quality | Good | Better |
| GPU Memory | 4GB | 6GB |
| Inference Speed | Fast | Moderate |
| Search Accuracy | Baseline | +10-15% |
| Caption Storage | Required | Optional |

---

## Next Steps

1. **Test with sample photos**: Start with 10-20 photos
2. **Verify search quality**: Compare results between 384d and 768d
3. **Scale up**: Process full collection when satisfied
4. **Monitor performance**: Track search latency and accuracy

For more details, see:
- `QUICKSTART.md` - General setup guide
- `STREAMLIT_CONFIG.md` - App configuration
- `README.md` - Complete documentation
