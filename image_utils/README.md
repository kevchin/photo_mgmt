# Image Deduplication and Organization Utilities

A comprehensive set of Python utilities for deduplicating, organizing, captioning, and searching your photo archive using local AI models.

## Features

### Image Deduplication (`image_dedup.py`)
- **Checksum-based deduplication**: Uses MD5 and SHA256 to find exact duplicates
- **Perceptual hashing**: Finds similar images (resized, slightly edited) using pHash, aHash, and dHash
- **Multi-format support**: HEIC, JPG, JPEG, PNG, HEIF
- **Metadata extraction**: EXIF dates, GPS coordinates, camera info
- **Folder comparison**: Compare new photos against your archive

### Image Organization (`image_organizer.py`)
- **Date-based organization**: Organize into YEAR/MONTH/DAY folder structure
- **Metadata extraction**: Parse EXIF dates, GPS coordinates, camera information
- **PostgreSQL database**: Store and search image metadata with pgvector support
- **Natural language search**: Search by date, captions, camera, location keywords
- **Tag support**: Add custom tags to images for better organization

### Local Caption Generation (`generate_captions_local.py`)
- **Microsoft Florence-2 integration**: Generate detailed captions completely offline
- **Multiple caption styles**: Basic, detailed, or very detailed descriptions
- **OCR support**: Extract text from images
- **Local embeddings**: Generate semantic embeddings with sentence-transformers
- **CSV or database output**: Save results to CSV file or directly to PostgreSQL
- **No API costs**: Runs entirely on your hardware

### Database Management (`image_database.py`)
- **pgvector integration**: Semantic similarity search on captions
- **Rich metadata storage**: Dates, GPS, camera info, formats, tags
- **HNSW/IVFFlat indexes**: Fast vector similarity search
- **Combined queries**: Filter by embedding similarity + metadata (date, location, format)

## Installation

```bash
cd image_utils
pip install -r requirements.txt
```

**Note**: The first time you run the caption generator, it will download the Florence-2 model (~2GB for base, ~4GB for large). Subsequent runs will use the cached model.

## Quick Start

### 1. Find Duplicates in a Folder

```bash
python image_dedup.py find-dups --dir /path/to/photos --output report.txt
```

### 2. Compare New Photos Against Archive

```bash
python image_dedup.py compare \
    --archive /path/to/archive \
    --new /path/to/new/photos \
    --output comparison_report.txt
```

### 3. Setup PostgreSQL Database

```bash
# Create database
createdb image_archive

# Enable pgvector extension
psql -d image_archive -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Initialize schema
export DATABASE_URL="postgresql://user:pass@localhost/image_archive"
python image_database.py init --db "$DATABASE_URL"
```

### 4. Organize Photos and Import to Database

```bash
# Organize and save to PostgreSQL
python image_organizer.py organize \
    --source /path/to/photos \
    --dest /path/to/archive \
    --postgres-db "$DATABASE_URL"
```

### 5. Generate Captions with Local AI (Florence-2)

```bash
# Generate captions and embeddings, save directly to database
python generate_captions_local.py \
    --from-db \
    --db "$DATABASE_URL" \
    --model microsoft/Florence-2-base \
    --embedding-model all-MiniLM-L6-v2 \
    --task "<DETAILED_CAPTION>"

# Or generate captions for a folder and save to CSV
python generate_captions_local.py \
    --source /path/to/photos \
    --output captions.csv \
    --model microsoft/Florence-2-base
```

### 6. Search Your Photos

```bash
# Search by metadata (date, location, format)
python image_database.py search-meta \
    --db "$DATABASE_URL" \
    --date-start "2024-12-01" \
    --date-end "2024-12-31"

# Search by caption similarity (requires generating query embedding first)
python image_database.py search-caption \
    --db "$DATABASE_URL" \
    --query-text "kids playing on beach at sunset" \
    --limit 10
```

## Workflow Example

Here's a complete workflow for processing new photos:

```bash
# Step 1: Compare new photos against your archive
python image_dedup.py compare \
    --archive ~/Pictures/Archive \
    --new ~/Downloads/NewPhotos \
    --output comparison.txt

# Step 2: Review the report to see what's unique

# Step 3: Organize the unique photos into your archive
python image_organizer.py organize \
    --source ~/Downloads/NewPhotos \
    --dest ~/Pictures/Archive \
    --postgres-db "$DATABASE_URL"

# Step 4: Generate AI captions and embeddings (runs locally!)
python generate_captions_local.py \
    --from-db \
    --db "$DATABASE_URL" \
    --model microsoft/Florence-2-large \
    --embedding-model all-mpnet-base-v2

# Step 5: Search for specific photos
# Example: "2 kids on a beach taken in Dec 2024"
python image_database.py search-meta \
    --db "$DATABASE_URL" \
    --date-start "2024-12-01" \
    --date-end "2024-12-31"

# Then filter results by caption keywords or use semantic search
python image_database.py search-caption \
    --db "$DATABASE_URL" \
    --query-text "children on beach" \
    --limit 20
```

## Local Caption Generation

The `generate_captions_local.py` utility uses Microsoft's Florence-2 model to generate rich captions completely offline:

### Supported Tasks

- `<CAPTION>`: Brief one-sentence description
- `<DETAILED_CAPTION>`: Detailed 2-3 sentence description (recommended)
- `<MORE_DETAILED_CAPTION>`: Very detailed description with more context
- `<OCR>`: Extract all text from the image
- `<OCR_WITH_REGION>`: Extract text with bounding boxes

### Model Options

- `microsoft/Florence-2-base`: Faster, ~2GB, good quality (recommended for most users)
- `microsoft/Florence-2-large`: Slower, ~4GB, better quality (for best results)

### Embedding Models

- `all-MiniLM-L6-v2`: Fast, 384 dimensions (good balance)
- `all-mpnet-base-v2`: Better quality, 768 dimensions (recommended)
- `bge-small-en-v1.5`: Good alternative
- `bge-large-en-v1.5`: Best quality, slower

### Examples

```bash
# Basic usage with CSV output
python generate_captions_local.py \
    --source ./photos \
    --output captions.csv

# Process database entries, save back to PostgreSQL
python generate_captions_local.py \
    --from-db \
    --db "postgresql://user:pass@localhost/image_archive" \
    --model microsoft/Florence-2-large

# Only generate embeddings for existing captions
python generate_captions_local.py \
    --from-db \
    --db "postgresql://user:pass@localhost/image_archive" \
    --embeddings-only

# Extract OCR text instead of captions
python generate_captions_local.py \
    --source ./documents \
    --output ocr_results.csv \
    --task "<OCR>"

# Use CPU only (slower but works without GPU)
python generate_captions_local.py \
    --source ./photos \
    --output captions.csv \
    --device cpu

# Limit processing for testing
python generate_captions_local.py \
    --from-db \
    --db "$DATABASE_URL" \
    --limit 10
```

## How It Works

### Deduplication Strategy

1. **Exact Duplicates**: Files with identical SHA256 checksums are exact duplicates
2. **Similar Images**: Perceptual hashes (pHash) detect images that look the same but may have different file sizes or formats

### Organization Strategy

1. **Date Priority**: 
   - First tries EXIF `DateTimeOriginal`
   - Falls back to EXIF `DateTime`
   - Finally uses file modification time

2. **Folder Structure**: `YEAR/MONTH/DAY/` (e.g., `2024/12/25/`)

3. **Conflict Resolution**: If filename exists, adds checksum prefix to avoid overwriting

### Caption Generation

1. **Florence-2 Vision Model**: Analyzes image content locally
2. **Sentence Transformers**: Creates 384-768 dimension embeddings for semantic search
3. **pgvector**: Stores embeddings for fast similarity search

### Search Capabilities

The search system supports:
- **Semantic search**: Find photos by meaning, not just keywords
- **Date filtering**: Search within date ranges
- **Location filtering**: Search by GPS coordinates and radius
- **Format filtering**: Filter by HEIC, JPG, PNG, etc.
- **Combined queries**: "beach photos from December 2024 within 10km of San Francisco"

## PostgreSQL Schema

The database stores:
- File paths and checksums
- Capture dates and creation dates
- GPS coordinates (latitude/longitude)
- Camera information (make, model, settings)
- Image dimensions and format
- AI-generated captions
- Caption embeddings (1536-dimensional vectors)
- Custom tags

Indexes include:
- HNSW index for fast vector similarity search
- B-tree indexes for dates, formats, and checksums
- GiST index for geographic queries

## Hardware Requirements

### For Caption Generation

**Minimum (CPU only)**:
- 8GB RAM
- Any modern CPU
- Processing time: ~5-10 seconds per image

**Recommended (GPU)**:
- NVIDIA GPU with 8GB+ VRAM (or Apple M1/M2/M3)
- 16GB+ system RAM
- Processing time: ~1-2 seconds per image

### For Database

- PostgreSQL 15+ with pgvector extension
- 4GB+ RAM for database operations
- SSD storage recommended for large archives

## Troubleshooting

### Model Download Issues

If you have trouble downloading models:
```bash
# Download models manually
git lfs install
git clone https://huggingface.co/microsoft/Florence-2-base

# Then use local path
python generate_captions_local.py \
    --source ./photos \
    --model ./Florence-2-base \
    --output captions.csv
```

### Out of Memory Errors

Reduce batch size or use CPU:
```bash
python generate_captions_local.py \
    --source ./photos \
    --device cpu \
    --workers 1
```

### pgvector Not Available

Install pgvector extension:
```bash
# On Ubuntu/Debian
sudo apt install postgresql-15-pgvector

# Enable in your database
psql -d image_archive -c "CREATE EXTENSION vector;"
```

## File Formats Supported

- **HEIC/HEIF**: Apple's efficient image format
- **JPEG/JPG**: Standard photo format
- **PNG**: Lossless format with transparency

## License

MIT License - Feel free to use and modify for your personal photo management needs.
