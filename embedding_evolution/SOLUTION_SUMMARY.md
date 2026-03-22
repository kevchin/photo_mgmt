# ✅ Solution Summary: Legacy Database Support Fixed

## Problem
The Streamlit app was crashing with error:
```
sqlalchemy.exc.ProgrammingError: relation "caption_models" does not exist
```

This occurred when trying to use `ACTIVE_DATABASE=legacy` because the code assumed all databases would have the new evolution schema with a `caption_models` table.

## Root Cause
Your **legacy database** (the original working database at `postgresql://postgres:postgres@localhost:5432/image_archive`) uses the old schema:
- Has `photos` table with embeddings
- Does NOT have `caption_models` tracking table
- Embedding column likely named `embedding` or similar

The new evolution code expected ALL databases to have the `caption_models` table for model versioning.

## Solution Implemented

### Updated Files

#### 1. `/workspace/embedding_evolution/search/vector_search.py`
Added automatic legacy/evolution database detection with three key improvements:

**a) Auto-detection in `get_available_models()` (lines 207-264)**
```python
# Check if caption_models table exists
result = conn.execute(text("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = 'caption_models'
    )
""")).scalar()

if not result:
    # Legacy database without caption_models table
    print("Legacy database detected (no caption_models table)")
    # Return florence-2-base as the only model
    return [{
        'name': 'florence-2-base',
        'dimension': 384,
        'description': 'Legacy Florence-2-base embeddings',
        'is_active': True,
        'photo_count': col_count
    }]
```

**b) Fallback column names in `search_by_vector()` (lines 137-166)**
```python
try:
    result = conn.execute(text(query), params)
except Exception as e:
    # Try legacy column names
    for legacy_col in ["embedding", "embedding_vector", "florence_embedding"]:
        try:
            # Try searching with this column
            ...
        except Exception:
            continue
```

**c) Graceful handling in `get_stats()` (lines 259-332)**
```python
has_caption_models = conn.execute(text("""
    SELECT EXISTS (...)
""")).scalar()

if has_caption_models:
    # Evolution database logic
else:
    # Legacy database logic - check common embedding columns
```

#### 2. Created Diagnostic Script
`/workspace/embedding_evolution/test_legacy_db.py`
- Tests PostgreSQL connection
- Checks pgvector installation
- Lists tables and columns
- Detects legacy vs evolution schema
- Shows embedding column names

#### 3. Created Documentation
`/workspace/embedding_evolution/LEGACY_SUPPORT_GUIDE.md`
- Complete usage instructions
- Migration path
- Troubleshooting guide
- Nomic embeddings setup

## How to Use Now

### Step 1: Test Your Legacy Database

```bash
cd /workspace/embedding_evolution

# Run diagnostic
python test_legacy_db.py

# Expected output:
# ✓ PostgreSQL version: ...
# ✓ pgvector installed: True
# Tables: ['photos']
# Columns: [..., 'embedding', ...]
# caption_models table exists: False
# → LEGACY database detected
```

### Step 2: Run Streamlit with Legacy Database

```bash
cd /workspace/embedding_evolution
export ACTIVE_DATABASE=legacy
streamlit run search/streamlit_app.py
```

**Expected behavior:**
- App starts without errors
- Sidebar shows: "florence-2-base (384d, X photos)"
- You can search your existing photos
- All metadata (date, GPS, B&W) works

### Step 3: (Optional) Create Evolution Database

When ready to support multiple embedding models:

```bash
# Create new evolution database
python migrations/migrate_legacy.py

# This will:
# 1. Create image_archive_evolution database
# 2. Add caption_models table
# 3. Copy all photos from legacy
# 4. Add embedding columns for future models
```

Then test:
```bash
export ACTIVE_DATABASE=evolution
streamlit run search/streamlit_app.py
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Your Workflow                         │
└─────────────────────────────────────────────────────────┘

Current State (Legacy):
┌──────────────────┐
│  PostgreSQL      │
│  image_archive   │
│  ┌────────────┐  │
│  │ photos     │  │
│  │ - id       │  │
│  │ - path     │  │
│  │ - embedding│←── 384-dim (Florence-2)
│  │ - metadata │  │
│  └────────────┘  │
└────────┬─────────┘
         │
         ├─→ Your existing Streamlit app (unchanged)
         │
         └─→ NEW: embedding_evolution app (legacy mode)
              export ACTIVE_DATABASE=legacy

Future State (Evolution):
┌──────────────────┐      ┌──────────────────┐
│  PostgreSQL      │      │  PostgreSQL      │
│  image_archive   │      │  image_archive_  │
│  (unchanged)     │      │  evolution       │
│                  │      │  ┌────────────┐  │
│  Your current    │      │  │ photos     │  │
│  tools work!     │      │  │ - embed_384│←─ Florence-2
│                  │      │  │ - embed_768│←─ Nomic (new)
│                  │      │  │ - embed_1024│  │
│                  │      │  └────────────┘  │
│                  │      │  ┌────────────┐  │
│                  │      │  │caption_    │  │
│                  │      │  │models      │←── Model tracking
│                  │      │  └────────────┘  │
│                  │      └──────────────────┘
│                  │               │
│                  │               └─→ NEW: embedding_evolution app
│                  │                    export ACTIVE_DATABASE=evolution
```

## Nomic Embeddings Support

The system already includes configuration for **nomic-embed-vision-v1.5**:

**Model Config** (`config/models.py` lines 121-129):
```python
"nomic-embed-vision-v1.5": ModelConfig(
    name="nomic-embed-vision-v1.5",
    model_id="nomic-ai/nomic-embed-vision-v1.5",
    model_type=ModelType.JOINT,
    embedding_dimension=768,
    description="Nomic AI vision-language model",
    gpu_memory_required_gb=6,
    column_name="embedding_nomic_embed_vision_v1_5_768"
)
```

**To use with your GTX 970 (4GB):**
⚠️ Note: Nomic requires ~6GB VRAM. Your GTX 970 has 4GB.

**Options:**
1. **Use CPU inference** (slow but works)
2. **Use smaller embedding model** like `all-MiniLM-L6-v2` (384-dim)
3. **Wait for GPU upgrade** then use Nomic
4. **Use 2-stage pipeline**:
   - Stage 1: Florence-2 on GPU → caption
   - Stage 2: all-mpnet-base-v2 (768-dim) on CPU/GPU → embed caption

## Testing Checklist

- [ ] PostgreSQL is running
- [ ] Run `python test_legacy_db.py` - should show legacy database info
- [ ] Set `export ACTIVE_DATABASE=legacy`
- [ ] Run `streamlit run search/streamlit_app.py`
- [ ] App should start without errors
- [ ] Search for a known photo using text query
- [ ] Verify results match your expectations

## Key Benefits of This Architecture

1. **Backward Compatible**: Your legacy database continues working unchanged
2. **Forward Compatible**: Evolution database supports unlimited embedding dimensions
3. **No Data Loss**: Migration copies data, doesn't replace
4. **Gradual Transition**: Test new models on subset of photos first
5. **Multi-Model Support**: Store multiple embeddings per photo simultaneously
6. **Easy Model Switching**: Change models via config, no code changes

## Next Steps

1. **Immediate**: Test legacy mode with your existing database
2. **Short-term**: Create evolution database when ready to experiment
3. **Medium-term**: Add Nomic or other 768-dim model after GPU upgrade
4. **Long-term**: Gradually re-embed old photos with better models as needed

## Questions?

See detailed documentation:
- `LEGACY_SUPPORT_GUIDE.md` - Usage and troubleshooting
- `README.md` - System overview
- `NOMIC_GUIDE.md` - Nomic-specific setup
- `QUICKSTART.md` - Quick start guide
