# Database Setup Status

## Current State

✅ **PostgreSQL 15** installed and running on `localhost:5432`
✅ **pgvector extension** compiled and installed for PostgreSQL 15
✅ **Database `image_archive`** created with pgvector extension enabled
✅ **Configuration file** `.env` created with proper settings

## Issue Identified

The `image_archive` database is currently **empty** (no tables). This is expected since we just installed PostgreSQL fresh.

### Your Existing Database

You mentioned you have an existing working database at:
```
postgresql://postgres:postgres@localhost:5432/image_archive
```

This database should contain:
- A table named `images` (not `photos`) with your photo metadata
- 384-dimensional embeddings from Florence-2-base
- EXIF data (dates, GPS, B&W detection, orientation)

## Next Steps

### Option 1: Connect to Your Existing Database

If your existing database is on a different PostgreSQL instance or port, update the `.env` file:

```bash
# Edit .env with your actual database connection
LEGACY_DB_URL=postgresql://postgres:postgres@YOUR_HOST:YOUR_PORT/image_archive
```

### Option 2: Verify Table Name

Run the test script again to confirm the table name:

```bash
cd /workspace/embedding_evolution
python test_legacy_db.py
```

Expected output should show:
```
Tables in legacy database: ['images']
Columns in 'images' table:
  - id: INTEGER
  - file_path: VARCHAR
  - caption_text: TEXT
  - embedding: VECTOR(384)  # or similar
  ...
```

### Option 3: Create Sample Data for Testing

If you want to test the system before connecting your real database:

```bash
# Run the evolution schema creation
python database/evolution_schema.py

# This will create the new evolved database structure
```

## Streamlit App Usage

Once connected to your database with the `images` table:

```bash
cd /workspace/embedding_evolution
export ACTIVE_DATABASE=legacy
streamlit run search/streamlit_app.py
```

The app will now:
1. Auto-detect the `images` table (not hardcoded to `photos`)
2. Detect legacy database mode (no `caption_models` table)
3. Use fallback column names (`embedding`, `embedding_vector`, etc.)
4. Display your existing photos with 384-dim Florence-2 embeddings

## Key Changes Made

1. **Auto-detection of table name**: The code now searches for `photos`, `images`, `pictures`, or `media` tables
2. **Flexible embedding column names**: Tries multiple column names for backward compatibility
3. **Legacy mode detection**: Automatically detects if `caption_models` table exists

## Migration Path

When ready to migrate to the evolution schema:

```bash
# 1. Create evolved database
python database/evolution_schema.py

# 2. Migrate existing data
python migrations/migrate_legacy.py

# 3. Update .env to use evolution database
ACTIVE_DATABASE=evolution

# 4. Test with new Streamlit app
streamlit run search/streamlit_app.py
```

This preserves your original database while creating a forward-compatible version.
