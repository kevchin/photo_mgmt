# Legacy Database Support Guide

## Problem Fixed

The Streamlit app was failing when connecting to a **legacy database** (your original database without the `caption_models` table) because it assumed all databases would have the new evolution schema.

## Solution Implemented

I've updated `/workspace/embedding_evolution/search/vector_search.py` with automatic detection and fallback support:

### 1. **Auto-Detection of Database Type**
- Checks if `caption_models` table exists
- If NOT: treats as legacy database with 384-dim Florence-2 embeddings
- If YES: treats as evolution database with multi-model support

### 2. **Fallback Column Names**
When searching, the code now tries multiple column names in order:
1. Model-specific column (e.g., `embedding_florence_2_base_384`)
2. Legacy columns: `embedding`, `embedding_vector`, `florence_embedding`

### 3. **Graceful Error Handling**
- Returns empty model list instead of crashing if table doesn't exist
- Provides helpful debug messages about which mode is active

## How to Use

### Option A: Test with Legacy Database

Set environment variable to point to your existing working database:

```bash
cd /workspace/embedding_evolution
export ACTIVE_DATABASE=legacy
streamlit run search/streamlit_app.py
```

The app will:
- Auto-detect it's a legacy database (no `caption_models` table)
- Show "florence-2-base (384d)" as the only available model
- Search using your existing 384-dim embeddings
- Display all your photos with metadata

### Option B: Test with Evolution Database

After running the migration:

```bash
cd /workspace/embedding_evolution
export ACTIVE_DATABASE=evolution
streamlit run search/streamlit_app.py
```

The app will:
- Detect the evolution schema with `caption_models` table
- Show all registered models (Florence-2-base + any new ones you add)
- Let you select which model's embeddings to search

## Testing the Fix

Run this diagnostic script first:

```bash
cd /workspace/embedding_evolution
python test_legacy_db.py
```

This will show:
- ✓ PostgreSQL connection status
- ✓ Whether pgvector is installed
- ✓ List of tables
- ✓ Columns in photos table (including embedding columns)
- ✓ Total photo count
- ✓ Whether it's detected as legacy or evolution schema

## Expected Behavior

### Legacy Database Mode
When `ACTIVE_DATABASE=legacy`:
```
Tables in legacy database: ['photos']
Columns in 'photos' table:
  - id: INTEGER
  - file_path: VARCHAR
  - embedding: VECTOR(384)  ← Your existing embedding column
  ...

caption_models table exists: False
→ This appears to be a LEGACY database (no model tracking)
→ The Streamlit app should auto-detect this and use fallback mode
```

Streamlit sidebar will show:
```
Embedding Model
Search using embeddings from:
  florence-2-base (384d, 1234 photos)
```

### Evolution Database Mode
After migration with `ACTIVE_DATABASE=evolution`:
```
Tables in legacy database: ['photos', 'caption_models']
caption_models table exists: True
→ This appears to be an EVOLUTION database (with model tracking)
```

Streamlit sidebar will show:
```
Embedding Model
Search using embeddings from:
  florence-2-base (384d, 1234 photos)
  nomic-embed-vision-v1.5 (768d, 0 photos)  ← Ready for new photos
```

## Migration Path

1. **Keep using your current system** - no changes needed
   ```bash
   # Your existing Streamlit app continues working
   streamlit run /path/to/old/app.py
   ```

2. **Test legacy mode** with new code
   ```bash
   export ACTIVE_DATABASE=legacy
   streamlit run embedding_evolution/search/streamlit_app.py
   ```
   Should work immediately with your existing data

3. **Create evolution database** (when ready)
   ```bash
   python migrations/migrate_legacy.py
   ```

4. **Test evolution mode**
   ```bash
   export ACTIVE_DATABASE=evolution
   streamlit run embedding_evolution/search/streamlit_app.py
   ```

5. **Add new photos with larger embeddings**
   ```bash
   python ingestion/photo_ingest.py --photos-dir /new/photos --caption-model nomic-embed-vision-v1.5
   ```

## Troubleshooting

### "relation caption_models does not exist"
✅ **Expected for legacy databases** - now handled automatically

### "Could not find any valid embedding column"
Check your actual embedding column name:
```bash
python test_legacy_db.py
```
Then update the fallback list in `vector_search.py` line 145 if needed.

### PostgreSQL not running
Start PostgreSQL:
```bash
sudo systemctl start postgresql
# or
pg_ctl -D /var/lib/postgresql/data start
```

## Next Steps for Nomic Embeddings

To test with nomic-embed-vision-v1.5 (768-dim):

1. Install the model:
```bash
pip install timm transformers
```

2. Add to config (`config/models.py`):
```python
"nomic-embed-vision-v1.5": ModelConfig(
    name="nomic-embed-vision-v1.5",
    model_id="nomic-ai/nomic-embed-vision-v1.5",
    model_type=ModelType.JOINT,  # Joint image-text embedding
    embedding_dimension=768,
    description="Nomic joint vision-language embedding model",
    column_name="embedding_nomic_embed_vision_v1_5_768"
)
```

3. Run migration to add the column:
```bash
python migrations/add_model_version.py --model nomic-embed-vision-v1.5
```

4. Ingest new photos:
```bash
python ingestion/nomic_joint_ingest.py --photos-dir /path/to/new/photos
```

5. Search will automatically detect the new model!
