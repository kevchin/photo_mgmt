# Photo Utilities - V1

A comprehensive set of Python tools for deduplicating, organizing, captioning, and searching your photo archive using local AI models.

## Quick Start

### 1. Install Dependencies

```bash
cd image_utils
pip install -r requirements.txt
```

### 2. Setup PostgreSQL Database (Optional but Recommended)

```bash
# Create database
createdb image_archive

# Enable pgvector extension (requires postgresql-XX-pgvector package)
psql -d image_archive -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Initialize schema
export DATABASE_URL="postgresql://user:pass@localhost/image_archive"
python image_database.py init --db "$DATABASE_URL"
```

**Note**: If you get an error about missing `vector.control`, install pgvector:
- **Ubuntu/Debian**: `sudo apt install postgresql-14-pgvector`
- **CentOS/RHEL/Fedora**: `sudo dnf install pgvector_14`
- **macOS**: `brew install pgvector`

Then restart PostgreSQL: `sudo systemctl restart postgresql`

## Available Tools

| Tool | Description |
|------|-------------|
| `image_dedup.py` | Find duplicate and similar images using checksums and perceptual hashing |
| `image_organizer.py` | Organize photos into YEAR/MONTH/DAY structure with metadata extraction |
| `ingest_images.py` | **NEW**: Ingest images from directory or SQLite into PostgreSQL with pgvector |
| `generate_captions.py` | Generate AI captions and embeddings using OpenAI API for semantic search |
| `generate_captions_local.py` | Generate AI captions and embeddings offline using Florence-2 |
| `image_database.py` | Manage PostgreSQL database with pgvector for semantic search |

## Common Workflows

### Find Duplicates
```bash
python image_dedup.py find-dups --dir /path/to/photos --output report.txt
```

### Compare New Photos Against Archive
```bash
python image_dedup.py compare \
    --archive /path/to/archive \
    --new /path/to/new/photos \
    --output comparison_report.txt
```

### Organize Photos
```bash
python image_organizer.py organize \
    --source /path/to/photos \
    --dest /path/to/archive \
    --save-db images.db  # Optional: create SQLite database
```

### **NEW: Incremental Ingestion (Recommended for Adding New Photos)**

The `incremental_ingest.py` script is designed specifically for your use case: adding a small batch of new photos to an existing archive without reprocessing thousands of existing photos.

**Key Benefits:**
- ✅ Only processes NEW photos (fast, even with large archives)
- ✅ Uses PostgreSQL database checksums for deduplication (no archive scanning needed)
- ✅ Automatically organizes by date into YYYY/MM/DD structure
- ✅ Optional AI caption generation
- ✅ Safe dry-run mode to preview changes

```bash
# Basic usage - add new photos to archive and database
python incremental_ingest.py \
    --new-photos /path/to/new/photos \
    --archive /path/to/organized/archive \
    --db "postgresql://user:pass@localhost/image_archive"

# With local AI caption generation (offline, no API key needed)
python incremental_ingest.py \
    --new-photos /path/to/new/photos \
    --archive /path/to/organized/archive \
    --db "postgresql://user:pass@localhost/image_archive" \
    --generate-captions \
    --local-captions

# Dry run - preview what would happen without making changes
python incremental_ingest.py \
    --new-photos /path/to/new/photos \
    --archive /path/to/organized/archive \
    --db "postgresql://user:pass@localhost/image_archive" \
    --dry-run
```

**Workflow:**
1. Scans only the new photos folder (typically < 100 photos)
2. Checks each photo against PostgreSQL database using SHA256 checksums
3. Skips duplicates (already in database)
4. For unique photos:
   - Extracts date metadata (EXIF or file modification time)
   - Copies to YYYY/MM/DD organized archive directory
   - Adds to PostgreSQL database with full metadata
   - Optionally generates AI captions

### Ingest from Existing Organized Directory

If you already have photos organized in a directory structure, use `ingest_images.py`:

**If you already have an organized directory**, skip `image_organizer.py` and ingest directly:

```bash
# From organized directory (no SQLite needed)
python ingest_images.py \
    --source-dir /path/to/organized/photos \
    --postgres-db "$DATABASE_URL"

# With local AI caption generation (Florence-2, runs offline)
python ingest_images.py \
    --source-dir /path/to/organized/photos \
    --postgres-db "$DATABASE_URL" \
    --local-captions
```

**If you used image_organizer.py with --save-db**:

```bash
# From SQLite database (created by image_organizer.py)
python ingest_images.py \
    --sqlite-db images.db \
    --postgres-db "$DATABASE_URL"

# With local AI caption generation
python ingest_images.py \
    --sqlite-db images.db \
    --postgres-db "$DATABASE_URL" \
    --local-captions
```

After ingestion with `--local-captions`, follow the on-screen instructions to run `generate_captions_local.py` which will:
1. Generate captions using Florence-2 (completely offline, no API key needed)
2. Create vector embeddings for semantic search
3. Update your PostgreSQL database automatically

### Generate AI Captions (Offline with Local LLM)

```bash
# Generate captions and embeddings for all images in PostgreSQL
python generate_captions_local.py \
    --db "$DATABASE_URL" \
    --from-db \
    --model microsoft/Florence-2-base \
    --embedding-model all-MiniLM-L6-v2

# Use a more detailed model (slower but better quality)
python generate_captions_local.py \
    --db "$DATABASE_URL" \
    --from-db \
    --model microsoft/Florence-2-large

# Only generate embeddings for existing captions
python generate_captions_local.py \
    --db "$DATABASE_URL" \
    --from-db \
    --embeddings-only
```

**Note**: The first run will download the Florence-2 model (~2GB). Subsequent runs are cached locally.

### Search Photos
```bash
# Search by date range
python image_database.py search-meta \
    --db "$DATABASE_URL" \
    --date-start "2024-01-01" \
    --date-end "2024-12-31"

# Semantic search by caption
python image_database.py search-caption \
    --db "$DATABASE_URL" \
    --query-text "kids playing on beach at sunset" \
    --limit 10
```

## Documentation

For detailed documentation, examples, and troubleshooting, see:
- [Image Utils README](image_utils/README.md) - Complete guide with all features
- [GPS Location Guide](image_utils/GPS_LOCATION_GUIDE.md) - How to work with GPS data

## Requirements

- Python 3.8+
- PostgreSQL 14+ with pgvector (optional, for database features)
- **For local AI caption generation** (`generate_captions_local.py`):
  - `pip install torch transformers pillow sentence-transformers psycopg2-binary`
  - 8GB+ RAM recommended
  - GPU optional but speeds up processing (CUDA or Apple MPS)
  - First run downloads Florence-2 model (~2GB), then cached locally
- **For OpenAI-based captions** (`generate_captions.py`):
  - OpenAI API key
  - `pip install openai pillow psycopg2-binary`

## License

MIT License
