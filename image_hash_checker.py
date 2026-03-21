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
from PIL.ExifTags import TAGS
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
            exif_data = img._getexif()
            if exif_data:
                # Map tag IDs to names
                exif = {TAGS.get(tag_id, tag_id): value 
                        for tag_id, value in exif_data.items()}
                
                # Try DateTimeOriginal first (when photo was taken)
                if 'DateTimeOriginal' in exif:
                    date_str = exif['DateTimeOriginal']
                    # Format: "YYYY:MM:DD HH:MM:SS"
                    try:
                        creation_date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass
                
                # Try DateTime as fallback
                if creation_date is None and 'DateTime' in exif:
                    date_str = exif['DateTime']
                    try:
                        creation_date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass
                
                # Extract GPS coordinates if available
                if 'GPSInfo' in exif:
                    gps_info = exif['GPSInfo']
                    gps_coords = extract_gps_coordinates(gps_info)
                    
    except Exception:
        # If we can't read EXIF, fall through to file stats
        pass
    
    # Fall back to file modification time if no EXIF date found
    if creation_date is None:
        stat = os.stat(filepath)
        # Use mtime as creation date proxy (ctime is metadata change time on Unix)
        creation_date = datetime.fromtimestamp(stat.st_mtime)
    
    return creation_date, gps_coords


def extract_gps_coordinates(gps_info: dict) -> tuple[float, float] | None:
    """
    Extract latitude and longitude from GPSInfo EXIF data.
    
    Args:
        gps_info: Dictionary containing GPS EXIF tags
        
    Returns:
        tuple: (latitude, longitude) or None if extraction fails
    """
    try:
        # GPS tags reference: https://exiftool.org/TagNames/GPS.html
        lat_ref = gps_info.get(1, 'N')  # GPSLatitudeRef
        lat = gps_info.get(2)           # GPSLatitude
        lon_ref = gps_info.get(3, 'E')  # GPSLongitudeRef
        lon = gps_info.get(4)           # GPSLongitude
        
        if lat is None or lon is None:
            return None
        
        def convert_to_degrees(value):
            """Convert GPS coordinates from [deg, min, sec] to decimal degrees."""
            d = float(value[0][0]) / float(value[0][1])
            m = float(value[1][0]) / float(value[1][1])
            s = float(value[2][0]) / float(value[2][1])
            return d + (m / 60.0) + (s / 3600.0)
        
        lat_decimal = convert_to_degrees(lat)
        lon_decimal = convert_to_degrees(lon)
        
        # Apply reference direction (N/S, E/W)
        if lat_ref == 'S':
            lat_decimal = -lat_decimal
        if lon_ref == 'W':
            lon_decimal = -lon_decimal
        
        return (lat_decimal, lon_decimal)
        
    except Exception:
        return None


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
