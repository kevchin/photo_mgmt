# Nomic Embed Vision-Text Guide

## Overview

You're considering **nomic-embed-vision-v1.5** (768-dim) paired with **nomic-embed-text-v1.5** for your photo archive. This is an excellent choice for cross-modal retrieval, but it works differently from your current 2-stage Florence-2 + text embedding pipeline.

## Key Difference: Joint vs. Two-Stage Embedding

### Your Current Pipeline (Two-Stage)
```
Image → Florence-2-base → Caption Text → all-MiniLM-L6-v2 → 384-dim vector
                                              ↓
                                    Store: caption + vector
```

### Nomic Joint Embedding (Recommended)
```
Image → nomic-embed-vision-v1.5 → 768-dim vector ─┐
                                                   ├→ Direct comparison possible!
Text Query → nomic-embed-text-v1.5 → 768-dim vector ┘
```

### Nomic Two-Stage Alternative (Not Recommended)
```
Image → Florence-2 → Caption → nomic-embed-text-v1.5 → 768-dim vector
```
*This wastes Nomic's capabilities - don't do this.*

## Why Joint Embedding is Better for Nomic

1. **Same Embedding Space**: Both image and text vectors are in the same 768-dim space by design
2. **No Caption Bottleneck**: Image features directly encoded, not filtered through caption generation
3. **Faster Ingestion**: Skip the VLM captioning step entirely
4. **Better Cross-Modal Retrieval**: Trained specifically for image↔text similarity

## Trade-offs

| Aspect | Two-Stage (Florence + Text Embed) | Joint (Nomic Vision-Text) |
|--------|-----------------------------------|---------------------------|
| **Human-readable captions** | ✅ Yes, stored in DB | ❌ No captions stored |
| **Caption-based search** | ✅ Can search caption text | ❌ Similarity only |
| **Ingestion speed** | 🐌 Slower (VLM bottleneck) | ⚡ Faster (direct embedding) |
| **GPU memory** | High (VLM + embedder) | Medium (vision encoder only) |
| **Search flexibility** | High (text + vector) | Medium (vector only) |
| **Embedding dimension** | Configurable (384-4096) | Fixed at 768 |

## Recommendation for Your Use Case

Since you currently:
- Store captions for potential re-embedding
- Have metadata (EXIF, GPS, B&W detection)
- Want to upgrade gradually from 384-dim to higher dimensions

**I recommend a HYBRID approach:**

### Option 1: Run Both Pipelines in Parallel ⭐ RECOMMENDED

Store BOTH types of embeddings in the same database:

```sql
-- Existing column (legacy Florence-2 + MiniLM)
embedding_384 VECTOR(384)

-- New column (Nomic joint vision embedding)
embedding_nomic_vision VECTOR(768)

-- Optional: Future column (better VLM + better text embedder)
embedding_llava_bge VECTOR(1024)
```

**Benefits:**
- Keep your existing caption-based search working
- Add fast Nomic similarity search for new photos
- Gradually test which performs better for your queries
- No need to choose one or the other

### Option 2: Two-Stage with Better Models

If you want to keep captions but upgrade quality:

```
Image → LLaVA-1.6-7B (or Qwen2.5-VL-7B) → Detailed Caption
                                              ↓
                              nomic-embed-text-v1.5 → 768-dim vector
```

**Benefits:**
- Higher quality captions than Florence-2-base
- Still human-readable and searchable
- Uses Nomic's excellent text encoder

**Drawbacks:**
- Doesn't leverage Nomic's vision encoder
- Still has VLM bottleneck

## Implementation

### For Option 1 (Hybrid - Recommended)

The `embedding_evolution` system already supports this! See:

```bash
# Migration creates columns for multiple embedding types
python migrations/migrate_legacy.py

# Ingest with Nomic joint embedding
python ingestion/nomic_joint_ingest.py --photos /path/to/photos

# Search across both embedding types
streamlit run streamlit_app.py
```

### For Option 2 (Two-Stage Upgrade)

```python
from config.pipeline_config import PipelineConfig

# Use LLaVA-1.6 for better captions + Nomic text embedder
config = PipelineConfig(
    vlm=VLMConfig(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        model_type="llava",
        prompt_type="detailed"
    ),
    embedding=EmbeddingModelConfig(
        model_id="nomic-embed-text-v1.5",
        dimension=768
    )
)
```

## Model Requirements

### nomic-embed-vision-v1.5
- **VRAM**: ~2-3 GB (ViT-L/14 based)
- **Type**: Vision encoder only
- **HuggingFace**: `nomic-ai/nomic-embed-vision-v1.5`
- **License**: Apache 2.0

### nomic-embed-text-v1.5
- **VRAM**: ~1-2 GB
- **Type**: Text encoder
- **HuggingFace**: `nomic-embed-text-v1.5`
- **License**: Apache 2.0
- **Context**: 8192 tokens

### Combined VRAM Usage
- **Ingestion** (vision encoder only): ~3 GB
- **Search** (text encoder only): ~2 GB
- **Your GTX 970 (4GB)**: Should work with careful batching!

## Testing on Your GTX 970

Start with a small batch to test VRAM usage:

```bash
# Test Nomic vision embedding on 10 photos
python ingestion/nomic_joint_ingest.py \
  --photos /path/to/test/photos \
  --batch-size 2 \
  --limit 10

# Monitor VRAM
watch -n 1 nvidia-smi
```

If you get OOM errors:
1. Reduce `--batch-size` to 1
2. Use `--half-precision` flag
3. Consider gradient checkpointing

## Migration Path

### Phase 1: Setup (Current)
- ✅ Migrate legacy DB to evolution schema
- ✅ Add `embedding_nomic_vision` column
- ✅ Test Nomic ingestion on small batch

### Phase 2: Dual Operation (1-2 weeks)
- Ingest NEW photos with BOTH:
  - Florence-2 + MiniLM (384-dim, for compatibility)
  - Nomic vision (768-dim, for testing)
- Compare search quality between embeddings

### Phase 3: GPU Upgrade (Future)
- Get new GPU (RTX 3090/4090 with 24GB)
- Switch to LLaVA-1.6-34B or Qwen2.5-VL-7B
- Generate higher quality captions
- Optionally re-embed old photos with Nomic

### Phase 4: Deprecation (Optional)
- If Nomic performs well, phase out 384-dim embeddings
- Keep captions for metadata/search fallback
- Standardize on 768-dim or 1024-dim

## Code Examples

### Example 1: Ingest with Nomic Joint Embedding

```python
from pipeline.joint_embedder import NomicJointEmbedder
from config.database import DatabaseManager

# Initialize
db = DatabaseManager("postgresql://postgres:postgres@localhost:5432/photo_archive_evolution")
embedder = NomicJointEmbedder(device="cuda")

# Process single image
image_path = Path("photos/2024/01/15/beach.jpg")
embedding = embedder.embed_image(image_path)
print(f"Generated {len(embedding)}-dim vector")

# Store in database
db.store_embedding(
    image_path=image_path,
    embedding=embedding,
    model_name="nomic-embed-vision-v1.5",
    dimension=768
)
```

### Example 2: Search with Nomic Text Query

```python
from pipeline.joint_embedder import NomicJointEmbedder
from search.vector_search import VectorSearcher

# Initialize
embedder = NomicJointEmbedder(device="cuda")
searcher = VectorSearcher(db_url="...")

# Convert text query to embedding
query = "kids playing on the beach at sunset"
query_embedding = embedder.embed_text(query)

# Search using Nomic vision embeddings
results = searcher.search_by_vector(
    query_embedding=query_embedding,
    model_column="embedding_nomic_vision",
    top_k=10
)

for result in results:
    print(f"{result['path']}: {result['similarity']:.3f}")
```

### Example 3: Hybrid Search (Both Embeddings)

```python
# Search both 384-dim and 768-dim embeddings
results_384 = searcher.search_by_text(
    query="beach vacation",
    model_column="embedding_384",
    top_k=5
)

results_768 = searcher.search_by_text(
    query="beach vacation",
    model_column="embedding_nomic_vision",
    top_k=5
)

# Merge and re-rank
combined = merge_results(results_384, results_768)
```

## Performance Comparison

Based on MTEB benchmarks:

| Model | Dimension | Image-Text Retrieval | Text-Text Retrieval |
|-------|-----------|---------------------|---------------------|
| all-MiniLM-L6-v2 | 384 | N/A | 61.3 |
| nomic-embed-text-v1.5 | 768 | N/A | 69.4 |
| nomic-embed-vision-v1.5 | 768 | 58.2* | N/A |
| CLIP ViT-L/14 | 768 | 54.8 | N/A |

*nomic-embed-vision-v1.5 is optimized for cross-modal retrieval

## Next Steps

1. **Read the full documentation**: `README.md` in `embedding_evolution/`
2. **Run migration**: `python migrations/migrate_legacy.py`
3. **Test Nomic on small batch**: 
   ```bash
   python ingestion/nomic_joint_ingest.py --photos ./test_photos --limit 5
   ```
4. **Compare search quality** between 384-dim and 768-dim embeddings
5. **Decide**: Continue hybrid approach or fully migrate to Nomic

## Additional Resources

- [Nomic Embed Vision Documentation](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)
- [Nomic Embed Text Documentation](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
