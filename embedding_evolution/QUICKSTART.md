# Quick Start Guide - Embedding Evolution

This guide walks you through migrating your existing photo database to the embedding evolution system and using it with new LLM models.

## Prerequisites

- PostgreSQL with pgvector extension
- Python 3.9+
- Existing legacy database at `postgresql://postgres:postgres@localhost:5432/image_archive`
- GPU (optional but recommended for faster processing)

## Step 1: Install Dependencies

```bash
cd /workspace/embedding_evolution
pip install -r requirements.txt
```

## Step 2: Configure Database Connection

Copy the example environment file and customize if needed:

```bash
cp .env.example .env
```

Edit `.env` if your database credentials differ from the defaults.

## Step 3: Run Tests (Optional)

Verify your setup before migration:

```bash
python tests/test_evolution.py
```

Expected output: Most tests should pass. Some may show warnings if the evolution database doesn't exist yet (that's normal).

## Step 4: Migrate Legacy Database

Run the migration to create the new evolution database:

```bash
python migrations/migrate_legacy.py
```

This will:
- Create a new database `image_archive_evolution`
- Install pgvector extension
- Create the `photos` table with model-versioned embedding columns
- Create the `caption_models` tracking table
- Migrate all your existing photos, captions, and metadata
- Preserve your original 384-dimension Florence-2-base embeddings

**Your original database remains unchanged!**

## Step 5: Verify Migration

Check that the migration succeeded:

```bash
python migrations/add_model_version.py --list
```

You should see `florence-2-base` registered with 384 dimensions.

## Step 6: Test Search Functionality

Launch the Streamlit search app:

```bash
streamlit run search/streamlit_app.py
```

The app will open in your browser. Try searching for "kids at the beach" or other descriptions.

## Using with New Models

### Adding Support for a Larger Model

When you upgrade to a GPU with more memory, add support for larger models:

```bash
# Add LLaVA 1.6 34B support (requires ~48GB GPU memory)
python migrations/add_model_version.py --add-model llava-1.6-34b

# Or add LLaVA 1.5 7B (requires ~16GB GPU memory)
python migrations/add_model_version.py --add-model llava-1.5-7b
```

This creates a new embedding column with the appropriate dimension (e.g., 1584 for LLaVA 1.6).

### Ingesting New Photos with Different Model

Add new photos using a specific captioning model:

```bash
# Using Florence-2-base (works on 4GB GPU)
python ingestion/photo_ingest.py \
    --photos-dir /path/to/new/photos \
    --caption-model florence-2-base

# Using LLaVA 1.5 7B (requires 16GB GPU)
python ingestion/photo_ingest.py \
    --photos-dir /path/to/new/photos \
    --caption-model llava-1.5-7b

# Using LLaVA 1.6 34B (requires 48GB GPU)
python ingestion/photo_ingest.py \
    --photos-dir /path/to/new/photos \
    --caption-model llava-1.6-34b
```

The ingestion tool will:
- Extract EXIF metadata (date, GPS, B&W detection, orientation)
- Generate AI captions using the specified model
- Store everything in the appropriate database columns

### Searching Across Different Models

In the Streamlit app, use the sidebar to select which model's embeddings to search:
- Photos ingested with Florence-2-base will be searchable using the 384-dim embeddings
- Photos ingested with LLaVA will be searchable using their respective higher-dimension embeddings

## Architecture Overview

### Database Schema

The evolution database has:
- **photos table**: Contains all photo metadata plus multiple embedding columns
  - `embedding_florence_2_base_384` - VECTOR(384)
  - `embedding_llava_1_5_7b_1024` - VECTOR(1024)
  - `embedding_llava_1_6_34b_1584` - VECTOR(1584)
  - Plus more as you add them
- **caption_models table**: Tracks which models are registered

### Key Design Decisions

1. **Multiple Vector Columns**: Each embedding model gets its own column with the correct dimension
2. **HNSW Indexes**: Each embedding column has an optimized index for fast similarity search
3. **Raw Caption Storage**: Captions are stored as text, enabling future re-embedding without re-processing images
4. **Model Tracking**: The `caption_models` table tracks which models have been used

### Why Not CSV?

Storing everything in PostgreSQL provides:
- ACID compliance for data integrity
- Unified querying (metadata filters + vector search in one query)
- Optimized HNSW indexes for fast approximate nearest neighbor search
- No sync issues between files and database

## Common Workflows

### Workflow 1: Gradual GPU Upgrade

1. **Current (4GB GPU)**: Continue using Florence-2-base for new photos
2. **Upgrade to 16GB GPU**: Add LLaVA 1.5 7B support, start using it for new photos
3. **Upgrade to 48GB GPU**: Add LLaVA 1.6 34B support for highest quality captions

Old photos remain searchable with their original embeddings.

### Workflow 2: Re-embedding Old Photos

If you want to upgrade old photos to use a better model:

```bash
# This would require a custom script to:
# 1. Query photos with NULL in the new embedding column
# 2. Use the stored caption_text to generate new embeddings
# 3. Update the new embedding column
```

Note: This is not implemented yet but can be added if needed.

### Workflow 3: Testing New Models

Before committing to a new model:

```bash
# Test with a small batch first
python ingestion/photo_ingest.py \
    --photos-dir /path/to/test/photos \
    --caption-model llava-1.6-34b \
    --limit 10

# Check results in Streamlit app
streamlit run search/streamlit_app.py
```

## Troubleshooting

### "pgvector extension not found"
```sql
-- Connect to your database and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

### "CUDA out of memory"
- Use a smaller model (e.g., florence-2-base instead of llava-1.6-34b)
- Reduce batch size in the ingestion script
- Enable 4-bit quantization (already enabled for large LLaVA models)

### "Column does not exist"
Run the migration script again or add the model explicitly:
```bash
python migrations/add_model_version.py --add-model model-name
```

## Next Steps

1. **Run the migration**: `python migrations/migrate_legacy.py`
2. **Test search**: `streamlit run search/streamlit_app.py`
3. **Add new photos**: Use the ingestion tool with your preferred model
4. **Upgrade GPU**: Add support for larger models as hardware allows

For more details, see the main [README.md](README.md).
