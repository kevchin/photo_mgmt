# Photo Utilities

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
    --postgres-db "$DATABASE_URL"
```

### Generate AI Captions (Offline)
```bash
python generate_captions_local.py \
    --from-db \
    --db "$DATABASE_URL" \
    --model microsoft/Florence-2-base
```

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
- 8GB+ RAM recommended for caption generation
- GPU optional but speeds up AI processing

## License

MIT License
