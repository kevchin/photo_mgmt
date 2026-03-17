# Black and White Image Detection Guide

This guide explains how to detect black and white (grayscale) images and query them via PostgreSQL.

## Overview

The solution consists of two Python scripts:

1. **`detect_black_white.py`** - Core detection logic for identifying grayscale images
2. **`add_black_white_column.py`** - Database migration and batch processing utility

## How It Works

### Detection Algorithm

An image is considered **black and white (grayscale)** if all pixels have equal R, G, and B values. The detection:

1. Converts the image to RGB format (if not already)
2. Checks if all pixels have R == G == B (within an optional tolerance)
3. Uses NumPy for fast vectorized comparison when available

**Tolerance**: JPEG compression can introduce slight color variations in grayscale images. Use `--tolerance 3-5` for JPEG images to account for compression artifacts.

## Step-by-Step Instructions

### Step 1: Install Dependencies

Make sure you have the required dependencies:

```bash
cd /workspace/image_utils
pip install Pillow numpy psycopg2-binary
```

### Step 2: Add the Column to Your Database

Run the migration script to add the `is_black_and_white` column:

```bash
python add_black_white_column.py postgresql://username:password@localhost:5432/your_database
```

This will:
- Add a `is_black_white BOOLEAN` column to the `images` table
- Create an index for fast queries
- Process all images and populate the column

### Step 3: Query Black and White Images

Once the column is populated, you can query for black and white images:

#### Find all black and white images:
```sql
SELECT * FROM images WHERE is_black_and_white = TRUE;
```

#### Find all color images:
```sql
SELECT * FROM images WHERE is_black_and_white = FALSE;
```

#### Count by type:
```sql
SELECT 
    is_black_and_white,
    COUNT(*) as count
FROM images
GROUP BY is_black_and_white;
```

#### Find black and white images from a specific year:
```sql
SELECT * FROM images 
WHERE is_black_and_white = TRUE 
  AND EXTRACT(YEAR FROM date_created) = 1990;
```

#### Find black and white images with GPS location:
```sql
SELECT file_name, gps_latitude, gps_longitude
FROM images 
WHERE is_black_and_white = TRUE 
  AND gps_latitude IS NOT NULL;
```

#### Get percentage of black and white images:
```sql
SELECT 
    ROUND(
        COUNT(*) FILTER (WHERE is_black_and_white = TRUE)::numeric / 
        COUNT(*)::numeric * 100, 
        2
    ) as grayscale_percentage
FROM images;
```

## Command-Line Options

### Basic Usage

```bash
# Add column and detect all images (default tolerance: 3)
python add_black_white_column.py postgresql://user:pass@localhost/db

# Use custom tolerance (for JPEG compression artifacts)
python add_black_white_column.py postgresql://user:pass@localhost/db --tolerance 5

# Show statistics without processing
python add_black_white_column.py postgresql://user:pass@localhost/db --stats

# Reclassify all images (overwrites existing values)
python add_black_white_column.py postgresql://user:pass@localhost/db --reclassify --tolerance 10

# Suppress progress output
python add_black_white_column.py postgresql://user:pass@localhost/db --no-verbose
```

### Testing Individual Images

You can test the detection on individual images:

```bash
# Test single images
python detect_black_white.py /path/to/image1.jpg /path/to/image2.png

# Use custom tolerance
python detect_black_white.py /path/to/image.jpg --tolerance 5

# Batch mode with summary
python detect_black_white.py /path/to/images/*.jpg --batch

# Use fast numpy-based detection (default)
python detect_black_white.py /path/to/image.jpg --fast
```

## Integration with Existing Workflow

### Adding to Ingestion Pipeline

To automatically detect black and white status during image ingestion, modify your ingestion process to call the detection function:

```python
from detect_black_white import is_black_and_white_fast

# During image processing
is_bw = is_black_and_white_fast(image_path, tolerance=3)

# Add to metadata
metadata = ImageMetadata(
    # ... other fields ...
    is_black_and_white=is_bw
)
```

### Updating the ImageMetadata Class

If you want to store this during initial ingestion, add the field to `ImageMetadata`:

```python
@dataclass
class ImageMetadata:
    # ... existing fields ...
    is_black_and_white: Optional[bool] = None
```

Then update the database schema and insert statements accordingly.

## Performance Considerations

- **Batch Processing**: The script processes images in batches (default: 100) for efficiency
- **Indexing**: An index is created on the `is_black_and_white` column for fast queries
- **NumPy**: The fast detection method uses NumPy for vectorized operations (recommended)
- **Tolerance**: Higher tolerance values may be needed for heavily compressed JPEG images

## Troubleshooting

### Column Already Exists Error

If you get an error about the column already existing, the script handles this gracefully. Just run it again - it will detect the existing column and skip creation.

### Images Not Found

If some image files have been moved or deleted since ingestion, they will be counted in the "not_found" category. The script only processes images where the file still exists.

### Changing Tolerance

If you want to re-run detection with a different tolerance:

```bash
python add_black_white_column.py postgresql://user:pass@localhost/db --reclassify --tolerance 10
```

## Example Output

```
Adding is_black_and_white column to images table...
Successfully added 'is_black_and_white' column

Detecting black and white images...
Found 1500 images to process
Processed 10/1500...
Processed 20/1500...
...
Processed 1500/1500...

Batch detection complete:
  Processed: 1500
  Grayscale: 342
  Color: 1158
  Errors: 0
  Not Found: 0

Final Statistics:
  Total images: 1500
  Grayscale: 342 (22.8%)
  Color: 1158 (77.2%)
  Unknown: 0

Done!
```

## SQL Query Examples

Here are more advanced queries you can run:

```sql
-- Black and white images grouped by year
SELECT 
    EXTRACT(YEAR FROM date_created) as year,
    COUNT(*) FILTER (WHERE is_black_and_white = TRUE) as grayscale,
    COUNT(*) FILTER (WHERE is_black_and_white = FALSE) as color,
    COUNT(*) as total
FROM images
GROUP BY EXTRACT(YEAR FROM date_created)
ORDER BY year;

-- Black and white images by format
SELECT 
    format,
    COUNT(*) FILTER (WHERE is_black_and_white = TRUE) as grayscale,
    COUNT(*) as total,
    ROUND(COUNT(*) FILTER (WHERE is_black_and_white = TRUE)::numeric / COUNT(*)::numeric * 100, 2) as percentage
FROM images
GROUP BY format;

-- Recent black and white images
SELECT file_name, date_created, format
FROM images
WHERE is_black_and_white = TRUE
ORDER BY date_created DESC
LIMIT 20;
```
