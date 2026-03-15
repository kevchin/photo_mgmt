#!/usr/bin/env python3
"""
Deduplicate and organize photos from multiple source directories into a 
YYYY/MM/DD folder structure based on photo date.

Features:
- Content-based deduplication using SHA-256 hashing
- EXIF date extraction (falls back to file modification time)
- Supports JPEG, PNG, HEIC, TIFF, and other common formats
- Handles filename collisions automatically
- Preserves original files (copies to destination)

Usage:
    python dedupe_and_organize.py /path/to/source1 /path/to/source2 /path/to/destination
"""

import os
import sys
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional
from PIL import Image
from PIL.ExifTags import TAGS
import argparse

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files will be skipped.")


def calculate_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_exif_date(filepath: Path) -> Optional[datetime]:
    """Extract date taken from EXIF data."""
    try:
        with Image.open(filepath) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal":
                        try:
                            return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            pass
                    elif tag == "DateTime":
                        try:
                            return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            pass
    except Exception:
        pass
    return None


def get_file_date(filepath: Path) -> datetime:
    """Get date for file: EXIF date if available, otherwise modification time."""
    exif_date = get_exif_date(filepath)
    if exif_date:
        return exif_date
    
    # Fall back to file modification time
    mtime = os.path.getmtime(filepath)
    return datetime.fromtimestamp(mtime)


def get_destination_path(date: datetime, filename: str, destination: Path, existing_files: Set[str]) -> Path:
    """Generate destination path with YYYY/MM/DD structure, handling collisions."""
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    
    dest_dir = destination / year / month / day
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # filename is already a string, so convert to Path for .stem and .suffix
    filename_path = Path(filename)
    base_name = filename_path.stem
    extension = filename_path.suffix.lower()
    
    dest_path = dest_dir / filename
    
    # Handle filename collisions
    counter = 1
    while str(dest_path) in existing_files:
        new_filename = f"{base_name}_{counter}{extension}"
        dest_path = dest_dir / new_filename
        counter += 1
    
    return dest_path


def scan_directory(directory: Path, supported_extensions: Set[str]) -> List[Path]:
    """Recursively scan directory for image files."""
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = Path(root) / file
            if filepath.suffix.lower() in supported_extensions:
                image_files.append(filepath)
    
    return image_files


def process_images(
    source_dirs: List[Path],
    destination: Path,
    dry_run: bool = False
) -> Dict[str, int]:
    """Process images from source directories, deduplicate and organize."""
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'}
    if HEIC_SUPPORT:
        supported_extensions.update({'.heic', '.heif'})
    
    # Collect all image files
    all_images = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Warning: Source directory does not exist: {source_dir}")
            continue
        images = scan_directory(source_dir, supported_extensions)
        all_images.extend(images)
    
    print(f"Found {len(all_images)} image files to process")
    
    # Track processed files
    seen_hashes: Set[str] = set()
    existing_files: Set[str] = set()
    stats = {
        'processed': 0,
        'duplicates': 0,
        'errors': 0,
        'skipped': 0
    }
    
    # Pre-scan destination for existing files
    if destination.exists():
        for root, _, files in os.walk(destination):
            for file in files:
                existing_files.add(str(Path(root) / file))
    
    for i, filepath in enumerate(all_images, 1):
        print(f"\nProcessing [{i}/{len(all_images)}]: {filepath}")
        
        try:
            # Calculate hash
            file_hash = calculate_file_hash(filepath)
            
            if file_hash in seen_hashes:
                print(f"  Skipping duplicate: {filepath}")
                stats['duplicates'] += 1
                continue
            
            # Get date and destination path
            date = get_file_date(filepath)
            dest_path = get_destination_path(date, filepath.name, destination, existing_files)
            
            if dry_run:
                print(f"  Would copy to: {dest_path}")
                print(f"  Date: {date.strftime('%Y-%m-%d')}")
            else:
                # Copy file
                shutil.copy2(filepath, dest_path)
                existing_files.add(str(dest_path))
                print(f"  Copied to: {dest_path}")
                print(f"  Date: {date.strftime('%Y-%m-%d')}")
            
            seen_hashes.add(file_hash)
            stats['processed'] += 1
            
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            stats['errors'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate and organize photos into YYYY/MM/DD structure"
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=str,
        help="Source directories containing photos"
    )
    parser.add_argument(
        "destination",
        type=str,
        help="Destination directory for organized photos"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files"
    )
    
    args = parser.parse_args()
    
    source_dirs = [Path(p) for p in args.sources]
    destination = Path(args.destination)
    
    print(f"Source directories: {source_dirs}")
    print(f"Destination: {destination}")
    print(f"Dry run: {args.dry_run}")
    
    if not args.dry_run:
        destination.mkdir(parents=True, exist_ok=True)
    
    stats = process_images(source_dirs, destination, args.dry_run)
    
    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"  Successfully processed: {stats['processed']}")
    print(f"  Duplicates skipped: {stats['duplicates']}")
    print(f"  Errors: {stats['errors']}")
    print("="*50)


if __name__ == "__main__":
    main()
