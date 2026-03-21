#!/usr/bin/env python3
"""
Photo Deduplication Scanner

Scans directories for photos, calculates SHA256 and perceptual hashes,
determines target YYYY/MM/DD directories based on EXIF DateTime (or creation date),
and generates a CSV file for further processing.

Supports: JPG, JPEG, HEIC, HEIF, PNG, MPO formats
"""

import os
import sys
import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install Pillow imagehash pillow-heif")
    sys.exit(1)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed. HEIC/HEIF files may not be readable.")
    print("Install with: pip install pillow-heif")

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.mpo'}


def calculate_sha256(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def calculate_perceptual_hash(filepath: str) -> Optional[str]:
    """Calculate perceptual hash (pHash) of an image."""
    try:
        with Image.open(filepath) as img:
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Calculate perceptual hash
            phash = imagehash.phash(img)
            return str(phash)
    except Exception as e:
        print(f"  Warning: Could not calculate perceptual hash for {filepath}: {e}")
        return None


def get_exif_datetime(filepath: str) -> Optional[datetime]:
    """Extract DateTime from EXIF metadata, preferring DateTimeOriginal."""
    try:
        with Image.open(filepath) as img:
            exif_data = img.getexif()
            
            if not exif_data:
                return None
            
            # Try DateTimeOriginal first (tag 36867)
            datetime_original = exif_data.get(36867)
            if datetime_original:
                try:
                    return datetime.strptime(datetime_original, "%Y:%m:%d %H:%M:%S")
                except (ValueError, TypeError):
                    pass
            
            # Fall back to DateTime (tag 306)
            datetime_tag = exif_data.get(306)
            if datetime_tag:
                try:
                    return datetime.strptime(datetime_tag, "%Y:%m:%d %H:%M:%S")
                except (ValueError, TypeError):
                    pass
            
            return None
    except Exception as e:
        print(f"  Warning: Could not read EXIF from {filepath}: {e}")
        return None


def get_creation_date(filepath: str) -> datetime:
    """Get file creation/modification date as fallback."""
    stat = os.stat(filepath)
    
    # Try creation time first (works on Windows/macOS)
    try:
        ctime = stat.st_birthtime
        return datetime.fromtimestamp(ctime)
    except AttributeError:
        # On Linux, st_birthtime may not be available, use mtime
        mtime = stat.st_mtime
        return datetime.fromtimestamp(mtime)


def get_target_date(filepath: str) -> datetime:
    """Get target date for directory organization (EXIF DateTime or creation date)."""
    exif_dt = get_exif_datetime(filepath)
    if exif_dt:
        return exif_dt
    return get_creation_date(filepath)


def get_target_directory(filepath: str) -> str:
    """Generate relative YYYY/MM/DD directory path (without base archive dir)."""
    target_date = get_target_date(filepath)
    year = target_date.strftime("%Y")
    month = target_date.strftime("%m")
    day = target_date.strftime("%d")
    
    return os.path.join(year, month, day)


def generate_unique_filename(target_dir: str, original_filename: str, existing_files: set) -> str:
    """Generate a unique filename if there's a collision."""
    base_name = Path(original_filename).stem
    extension = Path(original_filename).suffix.lower()
    
    new_filename = f"{base_name}{extension}"
    
    if new_filename.lower() not in {f.lower() for f in existing_files}:
        return new_filename
    
    # Handle collision by adding counter
    counter = 1
    while True:
        new_filename = f"{base_name}_{counter}{extension}"
        if new_filename.lower() not in {f.lower() for f in existing_files}:
            return new_filename
        counter += 1


def scan_directories(directories: List[str]) -> List[str]:
    """Recursively scan directories for image files."""
    image_files = []
    
    for directory in directories:
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
        
        print(f"Scanning: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    filepath = os.path.join(root, file)
                    image_files.append(filepath)
    
    return image_files


def process_photos(image_files: List[str], output_csv: str):
    """Process all photos and generate CSV with deduplication info."""
    
    print(f"\nProcessing {len(image_files)} images...")
    print("=" * 80)
    
    # Track files by hash for deduplication
    sha256_to_files: Dict[str, List[str]] = defaultdict(list)
    phash_to_files: Dict[str, List[str]] = defaultdict(list)
    
    # First pass: calculate hashes
    results = []
    for idx, filepath in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(filepath)}")
        
        try:
            # Calculate hashes
            sha256 = calculate_sha256(filepath)
            phash = calculate_perceptual_hash(filepath)
            
            # Get target directory (relative path YYYY/MM/DD)
            target_dir = get_target_directory(filepath)
            
            # Store result
            results.append({
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'target_directory': target_dir,
                'sha256': sha256,
                'perceptual_hash': phash if phash else '',
            })
            
            # Track for deduplication
            sha256_to_files[sha256].append(filepath)
            if phash:
                phash_to_files[phash].append(filepath)
                
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            results.append({
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'target_directory': '',
                'sha256': '',
                'perceptual_hash': '',
                'error': str(e)
            })
    
    # Second pass: identify duplicates and assign target filenames
    print("\n" + "=" * 80)
    print("Analyzing duplicates...")
    
    # Track which files are duplicates (by SHA256 - exact match)
    duplicate_files = set()
    for sha256, files in sha256_to_files.items():
        if len(files) > 1:
            # Keep the first file, mark others as duplicates
            print(f"\nDuplicate group (SHA256: {sha256[:16]}...):")
            for i, file in enumerate(files):
                if i == 0:
                    print(f"  [KEEP] {file}")
                else:
                    print(f"  [DUP]  {file}")
                    duplicate_files.add(file)
    
    # Track files per target directory for name collision handling
    dir_files: Dict[str, set] = defaultdict(set)
    
    # Assign target filenames
    for result in results:
        filepath = result['filepath']
        
        # Skip duplicates in output (or mark them)
        is_duplicate = filepath in duplicate_files
        
        target_dir = result['target_directory']
        if target_dir:
            # Generate unique filename within target directory
            new_filename = generate_unique_filename(
                target_dir, 
                result['filename'], 
                dir_files[target_dir]
            )
            dir_files[target_dir].add(new_filename.lower())
            result['new_filename'] = new_filename
            result['is_duplicate'] = is_duplicate
        else:
            result['new_filename'] = result['filename']
            result['is_duplicate'] = is_duplicate
    
    # Write CSV
    print(f"\nWriting results to: {output_csv}")
    
    fieldnames = ['filename', 'filepath', 'target_directory', 'new_filename', 
                  'perceptual_hash', 'sha256', 'is_duplicate']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'filename': result['filename'],
                'filepath': result['filepath'],
                'target_directory': result['target_directory'],
                'new_filename': result.get('new_filename', ''),
                'perceptual_hash': result.get('perceptual_hash', ''),
                'sha256': result.get('sha256', ''),
                'is_duplicate': result.get('is_duplicate', False)
            })
    
    # Summary
    total_files = len(results)
    duplicate_count = sum(1 for r in results if r.get('is_duplicate', False))
    unique_count = total_files - duplicate_count
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images scanned:     {total_files}")
    print(f"Unique images:            {unique_count}")
    print(f"Duplicate images:         {duplicate_count}")
    print(f"Output CSV:               {output_csv}")
    print("\nNext steps:")
    print("  1. Review the CSV file to verify duplicates and target directories")
    print("  2. Use a second tool to move unique files to their target directories")
    print("  3. Optionally delete or archive duplicate files")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Scan photo directories and generate CSV for deduplication and organization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dirs /photos/vacation /photos/family --output photo_plan.csv
  %(prog)s -d ~/Pictures -o plan.csv

Note: The generated CSV contains relative YYYY/MM/DD paths. Use photo_move_executor.py
      with the --archive option to specify where files should be copied.
        """
    )
    
    parser.add_argument(
        '--dirs', '-d',
        nargs='+',
        required=True,
        help='One or more directories to scan for photos (recursive)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='photo_dedup_plan.csv',
        help='Output CSV file path (default: photo_dedup_plan.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHOTO DEDUPLICATION SCANNER")
    print("=" * 80)
    print(f"Source directories: {', '.join(args.dirs)}")
    print(f"Output CSV:         {args.output}")
    print(f"Supported formats:  {', '.join(SUPPORTED_EXTENSIONS)}")
    print("=" * 80)
    
    # Scan for images
    image_files = scan_directories(args.dirs)
    
    if not image_files:
        print("\nNo image files found in the specified directories.")
        sys.exit(0)
    
    # Process and generate CSV
    process_photos(image_files, args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
