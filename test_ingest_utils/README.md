# Photo Ingestion Test Utilities

This directory contains standalone utilities to test the photo ingestion process **before** actually moving, copying, or modifying any files.

## Utilities Overview

### 1. `test_photo_ingest.py` - Main Testing Tool
Comprehensive testing for duplicate detection, target directory, caption generation, B&W detection, and orientation.

### 2. `heic_metadata_extractor.py` - HEIC Metadata Tool
Extract and display comprehensive metadata from HEIC files including EXIF data, GPS coordinates, camera info, and photo settings.

## Main Utility: `test_photo_ingest.py`

This is the primary testing tool that analyzes photos and reports what would happen during ingestion without making any changes.

### Features

The utility tests each photo and determines:

1. **Duplicate Detection**: Checks if the image is an exact or near-duplicate of any photo in your archive directory
2. **Target Directory**: Determines where the photo would be organized (YYYY/MM/DD format based on EXIF date)
3. **Caption Generation**: Optionally generates an AI caption using Florence-2 model
4. **Black & White Detection**: Identifies if the photo is grayscale or color
5. **Orientation Check**: Detects if the photo needs rotation based on EXIF orientation tags

### Installation Requirements

```bash
# Required
pip install Pillow pillow-heif ImageHash

# Optional (for caption generation)
pip install transformers torch sentence-transformers
```

### Usage Examples

#### Test a Single Photo

```bash
cd /workspace/test_ingest_utils

python test_photo_ingest.py /path/to/photo.jpg --archive /path/to/archive_directory
```

#### Test with Caption Generation

```bash
python test_photo_ingest.py /path/to/photo.jpg \
    --archive /path/to/archive_directory \
    --generate-caption
```

#### Test a Directory of Photos

```bash
python test_photo_ingest.py /path/to/photos/ \
    --archive /path/to/archive_directory \
    --batch
```

#### Save Results to JSON

```bash
python test_photo_ingest.py /path/to/photo.jpg \
    --archive /path/to/archive_directory \
    --output results.json
```

#### Quiet Mode (Summary Only)

```bash
python test_photo_ingest.py /path/to/photo.jpg \
    --archive /path/to/archive_directory \
    --quiet
```

### Command Line Options

```
positional arguments:
  photo_paths           Photo file(s) or directory to test

required arguments:
  --archive ARCHIVE     Path to archive directory for duplicate checking

optional arguments:
  --batch               Process directory of photos
  --generate-caption    Generate AI caption (requires Florence-2 model)
  --caption-model MODEL Caption model to use (default: microsoft/Florence-2-base)
  --phash-threshold N   Perceptual hash threshold for near-duplicate detection (default: 5)
  --output FILE         Output file for JSON results
  --quiet, -q           Suppress detailed output (only show summary)
  --verbose, -v         Show verbose output
```

### Sample Output

```
================================================================================
PHOTO TEST RESULTS: IMG_1234.jpg
================================================================================

📁 File: /photos/IMG_1234.jpg
   Size: 2,458,392 bytes | Format: JPEG | Dimensions: 4032x3024
   MD5: a1b2c3d4e5f6...
   SHA256: x9y8z7w6v5u4...

🔄 DUPLICATE CHECK:
   ✅ NOT A DUPLICATE - Safe to ingest

📂 ORGANIZATION:
   Target directory: 2024/01/15
   Full path would be: <archive>/2024/01/15/IMG_1234.jpg
   Date: 2024:01:15 14:30:22
   Parsed: 2024-01-15

🖼️  IMAGE PROPERTIES:
   Type: Color (color)

🔄 ORIENTATION:
   ✅ No rotation needed
   Status: No rotation needed (orientation is correct)

📝 CAPTION: Not generated (use --generate-caption to enable)

================================================================================
SUMMARY
================================================================================
✅ RECOMMENDATION: INGEST this photo
   Action: Move/copy to <archive>/2024/01/15/
```

### Understanding the Results

#### Duplicate Detection
- **Exact duplicates**: Detected by SHA256 checksum (identical files)
- **Near duplicates**: Detected by perceptual hash (similar images, possibly resized or slightly edited)
- The `--phash-threshold` option controls sensitivity (lower = more strict, default: 5)

#### Target Directory
- Based on EXIF DateTimeOriginal tag
- Falls back to file modification time if no EXIF date
- Format: YYYY/MM/DD (e.g., 2024/01/15)
- If day is unknown: YYYY/MM/unknown_day
- If month is unknown: YYYY/unknown_month

#### Black & White Detection
- Analyzes pixel values to determine if all RGB channels are equal
- Tolerance of 3 by default to account for JPEG compression artifacts
- Reports as "grayscale" or "color"

#### Orientation Detection
- Reads EXIF orientation tag (tag 274)
- Reports rotation angle if needed (90°, 180°, or 270°)
- Does not actually rotate the file

### Exit Codes

- `0`: At least one photo is ready to ingest (not a duplicate)
- `1`: All tested photos are duplicates

### Integration with Ingestion Workflow

This tool is designed to be used **before** running your actual ingestion scripts:

1. **Test Phase**: Use `test_photo_ingest.py` to preview what will happen
2. **Review Phase**: Check the output for any issues (duplicates, missing dates, etc.)
3. **Ingest Phase**: Run your actual ingestion script with confidence

Example workflow:

```bash
# Step 1: Test what would happen
python test_ingest_utils/test_photo_ingest.py new_photos/ \
    --archive /data/photo_archive \
    --output test_results.json

# Step 2: Review test_results.json

# Step 3: If everything looks good, run actual ingestion
python image_utils/ingest_images.py new_photos/ \
    --archive /data/photo_archive
```

## Troubleshooting

### "Module not found" errors
Make sure you're running from the correct directory or add the parent directory to PYTHONPATH:

```bash
cd /workspace
export PYTHONPATH=/workspace:$PYTHONPATH
python test_ingest_utils/test_photo_ingest.py ...
```

### HEIC/HEIF support
Install pillow-heif:
```bash
pip install pillow-heif
```

### Caption generation slow on first run
The Florence-2 model needs to be downloaded on first use (several GB). Subsequent runs will be faster.

### Memory errors with large batches
Process photos in smaller batches or use a machine with more RAM.

## See Also

- `/workspace/image_utils/` - Actual ingestion utilities
- `/workspace/image_utils/image_dedup.py` - Standalone deduplication tool
- `/workspace/image_utils/detect_black_white.py` - B&W detection utility
- `/workspace/image_utils/image_orientation.py` - Orientation correction utility
- `/workspace/image_utils/generate_captions_local.py` - Caption generation utility
- `/workspace/image_metadata_extractor.py` - General image metadata extractor

---

## HEIC Metadata Extractor: `heic_metadata_extractor.py`

This utility extracts and displays comprehensive metadata from HEIC files.

### Features

- **EXIF Data**: Camera make/model, lens info, exposure settings
- **GPS Coordinates**: Latitude, longitude, altitude with Google Maps link
- **Dates**: Creation date, modification date
- **Image Info**: Dimensions, format, orientation, file size
- **Photo Settings**: Exposure time, aperture, ISO, focal length, flash
- **Multiple Output Formats**: Human-readable or JSON
- **Batch Processing**: Process single files or entire directories

### Usage Examples

#### Extract Metadata from a Single HEIC File

```bash
cd /workspace/test_ingest_utils

# Detailed human-readable output
python heic_metadata_extractor.py photo.heic

# JSON output (good for scripting)
python heic_metadata_extractor.py photo.heic --json

# Brief summary only
python heic_metadata_extractor.py photo.heic --summary
```

#### Process a Directory of Images

```bash
# Detailed output for all images
python heic_metadata_extractor.py ./photos/

# Table-style summary
python heic_metadata_extractor.py ./photos/ --summary

# JSON output for all images
python heic_metadata_extractor.py ./photos/ --json
```

### Sample Output (Detailed Mode)

```
======================================================================
FILE: IMG_1234.heic
======================================================================

📁 BASIC INFORMATION:
   Full Path: /photos/IMG_1234.heic
   Format: HEIF
   Dimensions: 4032 x 3024 pixels
   Orientation: Landscape
   Color Mode: RGB
   File Size: 2.45 MB

📅 DATES:
   Creation Date: 2024-01-15T14:30:22
   Modified Date: 2024-01-15T14:30:25

📷 CAMERA INFORMATION:
   Camera: Apple iPhone 15 Pro
   Lens: iPhone 15 Pro back triple camera 6.86mm f/1.78

⚙️ PHOTO SETTINGS:
   Exposure: 1/120 sec
   Aperture: f/1.78
   ISO: 50
   Focal Length: 6.86 mm
   Flash: No Flash

🌍 GPS LOCATION:
   Latitude: 37.7749
   Longitude: -122.4194
   Altitude: 15.5 meters
   Google Maps: https://www.google.com/maps?q=37.7749,-122.4194

📋 OTHER EXIF DATA:
   Software: 17.2.1
   ... and 5 more fields
```

### Sample Output (Summary Mode)

```
Found 5 image(s) in ./photos/

Filename                       Size         Created                   GPS
--------------------------------------------------------------------------------
IMG_1234.heic                  2.45 MB      2024-01-15T14:30:22       ✓
IMG_1235.heic                  1.98 MB      2024-01-15T15:45:10       ✓
IMG_1236.heic                  2.12 MB      2024-01-16T09:20:05       
...
```

### Command Line Options

```
positional arguments:
  target          HEIC file or directory to process

optional arguments:
  --json          Output in JSON format
  --summary       Show brief summary only
  -h, --help      Show help message
```

### JSON Output Example

```json
{
  "file": "/photos/IMG_1234.heic",
  "filename": "IMG_1234.heic",
  "format": "HEIF",
  "width": 4032,
  "height": 3024,
  "mode": "RGB",
  "file_size": 2569830,
  "file_size_formatted": "2.45 MB",
  "orientation": "Landscape",
  "creation_date": "2024-01-15T14:30:22",
  "modification_date": "2024-01-15T14:30:25",
  "camera": {
    "make": "Apple",
    "model": "iPhone 15 Pro",
    "camera": "Apple iPhone 15 Pro"
  },
  "settings": {
    "exposure_time": "1/120",
    "f_number": "1.78",
    "iso": "50",
    "focal_length": "6.86"
  },
  "gps": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 15.5,
    "google_maps_link": "https://www.google.com/maps?q=37.7749,-122.4194"
  }
}
```

### Use Cases

1. **Debug HEIC Files**: Check if a HEIC file has valid EXIF data before ingestion
2. **Verify GPS Data**: Confirm location data is present and accurate
3. **Check Camera Settings**: Review exposure, ISO, and other photo settings
4. **Batch Inventory**: Create an inventory of all photos in a directory
5. **Pre-ingestion Validation**: Verify metadata before running ingestion scripts

### Requirements

```bash
pip install Pillow pillow-heif
```
