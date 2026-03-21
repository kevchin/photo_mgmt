#!/usr/bin/env python3
"""
Image Hash and Metadata Extractor

Computes SHA256 and perceptual hash for an image file.
Extracts creation date from image metadata (EXIF) or falls back to file creation date.
Outputs values formatted for easy use in SQL queries to check for duplicates.

Usage:
    python image_hash_checker.py <image_file>

Output includes:
    - SHA256 hash (for exact file match)
    - Perceptual hash (for similar image match)
    - Proposed YYYY/MM/DD directory based on creation date
    - SQL-ready values for duplicate checking
"""

import sys
import os
import hashlib
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import imagehash

# Try to import pillow-heif for HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT_AVAILABLE = True
except ImportError:
    HEIF_SUPPORT_AVAILABLE = False


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_perceptual_hash(filepath: str) -> str:
    """Compute perceptual hash (pHash) of an image."""
    try:
        with Image.open(filepath) as img:
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            phash = imagehash.phash(img)
            return str(phash)
    except Exception as e:
        if not HEIF_SUPPORT_AVAILABLE and filepath.lower().endswith(('.heic', '.heif')):
            raise RuntimeError(
                f"Cannot process HEIC/HEIF file: {filepath}. "
                "Install pillow-heif with: pip install pillow-heif"
            ) from e
        raise


def get_creation_date_and_gps(filepath: str) -> tuple[datetime, tuple[float, float] | None]:
    """
    Get creation date from image EXIF metadata and GPS coordinates.
    Falls back to file modification time if EXIF date not available.
    
    Returns:
        tuple: (creation_date, gps_coords) where gps_coords is (lat, lon) or None
    
    Priority order for date:
    1. DateTimeOriginal (when photo was taken)
    2. DateTime (when image was created/modified)
    3. File modification time
    """
    creation_date = None
    gps_coords = None
    
    try:
        with Image.open(filepath) as img:
            exif_data = img.getexif()
            if exif_data:
                # Try DateTimeOriginal first (when photo was taken)
                if 'DateTimeOriginal' in exif_data:
                    date_str = exif_data['DateTimeOriginal']
                    # Format: "YYYY:MM:DD HH:MM:SS"
                    try:
                        creation_date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass
                
                # Try DateTime as fallback
                if creation_date is None and 'DateTime' in exif_data:
                    date_str = exif_data['DateTime']
                    try:
                        creation_date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass
                
                # Extract GPS coordinates using IFD (same as working sample)
                # GPS info is typically stored under tag ID 34853 (0x8825)
                gps_info = exif_data.get_ifd(0x8825)
                if gps_info:
                    gps_data = {GPSTAGS.get(t, t): v for t, v in gps_info.items()}
                    
                    if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                        lat = get_decimal_from_dms(
                            gps_data['GPSLatitude'], 
                            gps_data.get('GPSLatitudeRef', 'N')
                        )
                        lon = get_decimal_from_dms(
                            gps_data['GPSLongitude'], 
                            gps_data.get('GPSLongitudeRef', 'E')
                        )
                        gps_coords = (lat, lon)
                        
    except Exception:
        # If we can't read EXIF, fall through to file stats
        pass
    
    # Fall back to file modification time if no EXIF date found
    if creation_date is None:
        stat = os.stat(filepath)
        # Use mtime as creation date proxy (ctime is metadata change time on Unix)
        creation_date = datetime.fromtimestamp(stat.st_mtime)
    
    return creation_date, gps_coords


def get_decimal_from_dms(dms, ref):
    """
    Convert DMS (degrees, minutes, seconds) to decimal degrees.
    
    Args:
        dms: Tuple/list of (degrees, minutes, seconds). 
             Can be either:
             - Flat tuple/list of floats: (40.0, 26.0, 46.0)
             - Nested tuple/list of rationals: ((40, 1), (26, 1), (46, 1))
        ref: Reference direction ('N', 'S', 'E', 'W')
        
    Returns:
        float: Decimal degrees (negative for S/W)
    """
    def to_float(val):
        """Convert a value to float, handling rational tuples."""
        if isinstance(val, (tuple, list)) and len(val) == 2:
            # It's a rational (numerator, denominator)
            return float(val[0]) / float(val[1])
        return float(val)
    
    degrees = to_float(dms[0])
    minutes = to_float(dms[1]) / 60.0
    seconds = to_float(dms[2]) / 3600.0
    decimal = degrees + minutes + seconds
    
    if ref in ['S', 'W']:
        return -decimal
    return decimal


def format_date_path(date: datetime) -> str:
    """Format datetime as YYYY/MM/DD directory path."""
    return date.strftime("%Y/%m/%d")


def escape_sql_string(value: str) -> str:
    """Escape a string for safe use in SQL (basic escaping)."""
    return value.replace("'", "''")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image_file>", file=sys.stderr)
        print("\nThis tool computes SHA256 and perceptual hashes for an image,", file=sys.stderr)
        print("and proposes a YYYY/MM/DD directory based on creation date.", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.isfile(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    # Compute hashes
    sha256 = compute_sha256(filepath)
    phash = compute_perceptual_hash(filepath)
    
    # Get creation date, GPS coordinates and format directory path
    creation_date, gps_coords = get_creation_date_and_gps(filepath)
    date_path = format_date_path(creation_date)
    
    # Get filename
    filename = os.path.basename(filepath)
    
    # Output results
    print("=" * 60)
    print("IMAGE HASH AND METADATA ANALYSIS")
    print("=" * 60)
    print(f"\nFile: {filepath}")
    print(f"Filename: {filename}")
    print(f"\nCreation Date: {creation_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Proposed Directory: {date_path}")
    if gps_coords:
        lat, lon = gps_coords
        print(f"Extracted GPS (lat,lon): {lat:.6f}, {lon:.6f}")
    else:
        print("Extracted GPS (lat,lon): Not available")
    print(f"Proposed Full Path: {date_path}/{filename}")
    
    print("\n" + "-" * 60)
    print("HASH VALUES")
    print("-" * 60)
    print(f"SHA256: {sha256}")
    print(f"Perceptual Hash (pHash): {phash}")
    
    print("\n" + "-" * 60)
    print("SQL QUERY EXAMPLES")
    print("-" * 60)
    
    # Escape values for SQL
    escaped_filename = escape_sql_string(filename)
    escaped_date_path = escape_sql_string(date_path)
    escaped_full_path = escape_sql_string(f"{date_path}/{filename}")
    
    # Example table structure assumed:
    # CREATE TABLE images (
    #     id SERIAL PRIMARY KEY,
    #     filename VARCHAR(255),
    #     directory_path VARCHAR(255),
    #     file_path VARCHAR(512),
    #     sha256 CHAR(64),
    #     perceptual_hash VARCHAR(20),
    #     created_at TIMESTAMP
    # );
    
    print("\n-- Check if exact file path already exists:")
    print(f"SELECT id, sha256 FROM images WHERE file_path = '{escaped_full_path}';")
    
    print("\n-- Check if SHA256 hash exists (exact duplicate anywhere):")
    print(f"SELECT id, file_path FROM images WHERE sha256 = '{sha256}';")
    
    print("\n-- Check for perceptually similar images (may be resized/recompressed):")
    print(f"SELECT id, file_path, perceptual_hash FROM images WHERE perceptual_hash = '{phash}';")
    
    print("\n-- Combined check: verify both path and hash uniqueness:")
    print(f"SELECT ")
    print(f"    EXISTS(SELECT 1 FROM images WHERE file_path = '{escaped_full_path}') AS path_exists,")
    print(f"    EXISTS(SELECT 1 FROM images WHERE sha256 = '{sha256}') AS sha256_exists,")
    print(f"    EXISTS(SELECT 1 FROM images WHERE perceptual_hash = '{phash}') AS phash_exists;")
    
    print("\n" + "-" * 60)
    print("PYTHON USAGE EXAMPLE")
    print("-" * 60)
    print(f"""
# In your Python code, you can use these values directly:
sha256_value = "{sha256}"
phash_value = "{phash}"
directory_path = "{date_path}"
file_path = "{date_path}/{filename}"

# Example with psycopg2:
cursor.execute(\"\"\"
    SELECT id FROM images 
    WHERE file_path = %s OR sha256 = %s
\"\"\", (file_path, sha256_value))
""")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
