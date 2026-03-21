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


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_perceptual_hash(filepath: str) -> str:
    """Compute perceptual hash (pHash) of an image."""
    with Image.open(filepath) as img:
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        phash = imagehash.phash(img)
        return str(phash)


def get_creation_date(filepath: str) -> datetime:
    """
    Get creation date from image EXIF metadata.
    Falls back to file modification time if EXIF date not available.
    
    Priority order:
    1. DateTimeOriginal (when photo was taken)
    2. DateTime (when image was created/modified)
    3. File modification time
    """
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
                        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass
                
                # Try DateTime as fallback
                if 'DateTime' in exif:
                    date_str = exif['DateTime']
                    try:
                        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass
    except Exception:
        # If we can't read EXIF, fall through to file stats
        pass
    
    # Fall back to file modification time
    stat = os.stat(filepath)
    # Use mtime as creation date proxy (ctime is metadata change time on Unix)
    return datetime.fromtimestamp(stat.st_mtime)


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
    
    # Get creation date and format directory path
    creation_date = get_creation_date(filepath)
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
    #     full_path VARCHAR(512),
    #     sha256_hash CHAR(64),
    #     perceptual_hash VARCHAR(20),
    #     created_at TIMESTAMP
    # );
    
    print("\n-- Check if exact file path already exists:")
    print(f"SELECT id, sha256_hash FROM images WHERE full_path = '{escaped_full_path}';")
    
    print("\n-- Check if SHA256 hash exists (exact duplicate anywhere):")
    print(f"SELECT id, full_path FROM images WHERE sha256_hash = '{sha256}';")
    
    print("\n-- Check for perceptually similar images (may be resized/recompressed):")
    print(f"SELECT id, full_path, perceptual_hash FROM images WHERE perceptual_hash = '{phash}';")
    
    print("\n-- Combined check: verify both path and hash uniqueness:")
    print(f"SELECT ")
    print(f"    EXISTS(SELECT 1 FROM images WHERE full_path = '{escaped_full_path}') AS path_exists,")
    print(f"    EXISTS(SELECT 1 FROM images WHERE sha256_hash = '{sha256}') AS sha256_exists,")
    print(f"    EXISTS(SELECT 1 FROM images WHERE perceptual_hash = '{phash}') AS phash_exists;")
    
    print("\n" + "-" * 60)
    print("PYTHON USAGE EXAMPLE")
    print("-" * 60)
    print(f"""
# In your Python code, you can use these values directly:
sha256_value = "{sha256}"
phash_value = "{phash}"
directory_path = "{date_path}"
full_path = "{date_path}/{filename}"

# Example with psycopg2:
cursor.execute(\"\"\"
    SELECT id FROM images 
    WHERE full_path = %s OR sha256_hash = %s
\"\"\", (full_path, sha256_value))
""")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
