# Incremental Photo Ingestion Guide

## Overview

The `incremental_ingest.py` script is designed to solve a specific problem: **adding new photos to your existing archive without reprocessing thousands of already-indexed photos**.

This is the recommended workflow when you have:
- An existing PostgreSQL database with thousands of photos
- A new folder of photos (typically < 100) that you want to add
- Need to verify they're not duplicates before adding
- Want them organized in YYYY/MM/DD format automatically

## Key Benefits

| Feature | Benefit |
|---------|---------|
| **Database-driven deduplication** | Uses SHA256 checksums in PostgreSQL - no need to scan archive directory |
| **Incremental processing** | Only processes NEW photos, skips existing ones instantly |
| **Automatic organization** | Creates YYYY/MM/DD structure based on EXIF date or file modification time |
| **Safe dry-run mode** | Preview what would happen without making any changes |
| **Optional AI captions** | Generate captions with local Florence-2 (offline) or OpenAI API |
| **Conflict handling** | Automatically renames files if there are naming conflicts |

## How It Works

```
┌─────────────────────┐
│  New Photos Folder  │  ← Scan only this (< 100 files)
│  (unorganized)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Check PostgreSQL   │  ← Query database by SHA256 checksum
│  Database           │     (instant lookup, no archive scanning)
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐   ┌────────────┐
│Duplicate│   │   UNIQUE   │
│ SKIP    │   │  Process   │
└─────────┘   └─────┬──────┘
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
    ┌─────────┐ ┌─────────┐ ┌──────────┐
    │ Extract │ │  Copy   │ │  Add to  │
    │  Date   │ │  to     │ │ Database │
    │ Metadata│ │ Archive │ │          │
    └─────────┘ └─────────┘ └──────────┘
          │         │         │
          └─────────┴─────────┘
                    │
                    ▼
          YYYY/MM/DD/filename.jpg
```

## Usage

### Basic Usage

```bash
cd image_utils

python incremental_ingest.py \
    --new-photos /path/to/new/photos \
    --archive /path/to/organized/archive \
    --db "postgresql://user:password@localhost/image_archive"
```

This will:
1. Scan `/path/to/new/photos` for image files
2. Check each against the PostgreSQL database using SHA256 checksums
3. Skip any duplicates (already in database)
4. For unique photos:
   - Extract date from EXIF or use file modification time
   - Copy to `/path/to/organized/archive/YYYY/MM/DD/`
   - Add metadata to PostgreSQL database

### With Local AI Captions (Recommended - Offline)

```bash
python incremental_ingest.py \
    --new-photos /path/to/new/photos \
    --archive /path/to/organized/archive \
    --db "postgresql://user:password@localhost/image_archive" \
    --generate-captions \
    --local-captions
```

This uses the Florence-2 model (runs locally, no API costs) to generate descriptive captions for each photo.

**Note:** After generating captions, you'll need to create vector embeddings for semantic search:

```bash
python generate_captions_local.py \
    --db "postgresql://user:password@localhost/image_archive" \
    --from-db \
    --embeddings-only
```

### With OpenAI Captions

```bash
python incremental_ingest.py \
    --new-photos /path/to/new/photos \
    --archive /path/to/organized/archive \
    --db "postgresql://user:password@localhost/image_archive" \
    --generate-captions \
    --openai-api-key $OPENAI_API_KEY \
    --caption-model gpt-4o
```

### Dry Run (Preview Changes)

Always recommended for first use:

```bash
python incremental_ingest.py \
    --new-photos /path/to/new/photos \
    --archive /path/to/organized/archive \
    --db "postgresql://user:password@localhost/image_archive" \
    --dry-run
```

This shows you exactly what would happen without copying files or modifying the database.

## Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--new-photos` | Directory containing new photos to ingest | Yes |
| `--archive` | Archive directory where photos will be organized (YYYY/MM/DD) | Yes |
| `--db` | PostgreSQL connection string | Yes |
| `--generate-captions` | Generate AI captions for new photos | No |
| `--local-captions` | Use local Florence-2 model (offline, no API key) | No |
| `--openai-api-key` | OpenAI API key (if using OpenAI for captions) | No |
| `--caption-model` | OpenAI model name (default: gpt-4o) | No |
| `--dry-run` | Preview without making changes | No |
| `--batch-size` | Photos per database commit batch (default: 50) | No |

## Example Output

```
================================================================================
INCREMENTAL PHOTO INGESTION
================================================================================
New photos directory: /home/user/new_photos
Archive directory: /home/user/photo_archive
Dry run: False

Connected to PostgreSQL database

Scanning for new photos...
Found 47 photo files in /home/user/new_photos

Processing 47 photos...
--------------------------------------------------------------------------------
[1/47] Processing: IMG_20240115_143022.jpg
  Skipping (duplicate by checksum): IMG_20240115_143022.jpg

[2/47] Processing: DSC_0042.JPG
  UNIQUE (copied to /home/user/photo_archive/2024/01/15/DSC_0042.JPG)
  Generating caption...
    Caption: A group of people standing on a beach at sunset with waves...

[3/47] Processing: beach_photo.png
  UNIQUE (copied to /home/user/photo_archive/2024/01/15/beach_photo.png)

...

================================================================================
INGESTION SUMMARY
================================================================================
Total photos scanned:     47
Duplicates skipped:       12
Unique photos found:      35
Added to database:        35
Captions generated:       35
Errors:                   0
```

## Comparison with Other Tools

### vs `image_dedup.py compare`

| Feature | `image_dedup.py compare` | `incremental_ingest.py` |
|---------|-------------------------|------------------------|
| Scans archive directory | ✅ Yes (slow for large archives) | ❌ No (uses database) |
| Scans new photos folder | ✅ Yes | ✅ Yes |
| Organizes into YYYY/MM/DD | ❌ No | ✅ Yes |
| Adds to database | ❌ No | ✅ Yes |
| Generates captions | ❌ No | ✅ Optional |
| Best for | One-time comparison | Regular incremental additions |

### vs `ingest_images.py`

| Feature | `ingest_images.py` | `incremental_ingest.py` |
|---------|-------------------|------------------------|
| Source | Organized directory or SQLite DB | Unorganized new photos folder |
| Deduplication | Checks during ingest | Checks BEFORE processing |
| Organization | Assumes already organized | Creates YYYY/MM/DD structure |
| Best for | Initial bulk import | Adding new photos to existing archive |

## Workflow Recommendations

### First-Time Setup (Bulk Import)

If you're starting fresh with a large photo collection:

```bash
# Step 1: Organize all photos into YYYY/MM/DD structure
python image_organizer.py organize \
    --source /path/to/all/photos \
    --dest /path/to/archive \
    --save-db images.db

# Step 2: Ingest into PostgreSQL with captions
python ingest_images.py \
    --sqlite-db images.db \
    --postgres-db "postgresql://user:pass@localhost/image_archive" \
    --local-captions

# Step 3: Generate embeddings for semantic search
python generate_captions_local.py \
    --db "postgresql://user:pass@localhost/image_archive" \
    --from-db
```

### Ongoing Maintenance (Adding New Photos)

For regular additions of new photos:

```bash
# When you get new photos, put them in a folder and run:
python incremental_ingest.py \
    --new-photos /path/to/new_photos \
    --archive /path/to/archive \
    --db "postgresql://user:pass@localhost/image_archive" \
    --generate-captions \
    --local-captions

# Then generate embeddings for the new captions
python generate_captions_local.py \
    --db "postgresql://user:pass@localhost/image_archive" \
    --from-db \
    --embeddings-only
```

## Troubleshooting

### "Module not found" errors

Install required dependencies:

```bash
cd image_utils
pip install -r requirements.txt
```

### Database connection errors

Verify your PostgreSQL connection string:

```bash
# Test connection
psql "postgresql://user:password@localhost/image_archive" -c "SELECT 1"
```

Make sure pgvector extension is enabled:

```bash
psql "postgresql://user:password@localhost/image_archive" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Permission errors when copying files

Ensure the archive directory is writable:

```bash
mkdir -p /path/to/archive
chmod 755 /path/to/archive
```

### Out of memory with local captions

Reduce batch size or use a smaller model:

```bash
python incremental_ingest.py \
    --new-photos /path/to/new_photos \
    --archive /path/to/archive \
    --db "postgresql://user:pass@localhost/image_archive" \
    --generate-captions \
    --local-captions \
    --batch-size 10
```

Or use OpenAI instead (requires API key):

```bash
python incremental_ingest.py \
    --new-photos /path/to/new_photos \
    --archive /path/to/archive \
    --db "postgresql://user:pass@localhost/image_archive" \
    --generate-captions \
    --openai-api-key $OPENAI_API_KEY
```

## Performance Tips

1. **Use dry-run first**: Always test with `--dry-run` to see what will happen
2. **Batch size**: Adjust `--batch-size` based on your system's memory
3. **SSD storage**: Archive on SSD significantly speeds up file operations
4. **Local network**: Keep database server on same machine or fast network
5. **Index maintenance**: Run `VACUUM ANALYZE` on PostgreSQL periodically

## See Also

- [Main README](../README.md) - Overview of all photo utilities
- [image_dedup.py](./image_dedup.py) - Find duplicates using checksums and perceptual hashing
- [image_organizer.py](./image_organizer.py) - Organize photos by date
- [generate_captions_local.py](./generate_captions_local.py) - Generate AI captions offline
- [GPS Location Guide](./GPS_LOCATION_GUIDE.md) - Working with GPS metadata
