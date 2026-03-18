#!/usr/bin/env python3
"""
Add Single Photo Utility

Adds a single photo to the PostgreSQL database with:
- Duplicate check using SHA256 checksum
- Local caption generation (Florence-2 model)
- Copy to archive directory in YYYY/MM/DD format

The photo is only added if:
1. It's not a duplicate (based on SHA256 checksum)
2. A local caption can be successfully generated

Usage:
    # Basic usage - add single photo with local caption
    python add_single_photo.py /path/to/photo.jpg \\
        --db-url postgresql://user:pass@localhost/dbname \\
        --archive-dir /path/to/archive

    # With detailed caption
    python add_single_photo.py /path/to/photo.jpg \\
        --db-url postgresql://user:pass@localhost/dbname \\
        --archive-dir /path/to/archive \\
        --caption-task "<MORE_DETAILED_CAPTION>"

    # Dry run - see what would be done without making changes
    python add_single_photo.py /path/to/photo.jpg \\
        --db-url postgresql://user:pass@localhost/dbname \\
        --archive-dir /path/to/archive \\
        --dry-run
"""

import os
import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    HAS_IMAGE_SUPPORT = True
except ImportError as e:
    print(f"Error: Pillow not installed. Run: pip install Pillow")
    HAS_IMAGE_SUPPORT = False
    sys.exit(1)

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not installed. Perceptual hashing disabled.")

# Import modules from same directory
sys.path.insert(0, str(Path(__file__).parent))
from image_database import ImageDatabase, ImageMetadata

# Lazy import for local caption generator
LOCAL_CAPTION_AVAILABLE = False
FlorenceCaptionGenerator = None
EmbeddingGenerator = None


def _try_import_local_caption():
    """Try to import local caption generator modules"""
    global LOCAL_CAPTION_AVAILABLE, FlorenceCaptionGenerator, EmbeddingGenerator
    if not LOCAL_CAPTION_AVAILABLE:
        try:
            from generate_captions_local import FlorenceCaptionGenerator as FCG, EmbeddingGenerator as EG
            FlorenceCaptionGenerator = FCG
            EmbeddingGenerator = EG
            LOCAL_CAPTION_AVAILABLE = True
        except ImportError as e:
            LOCAL_CAPTION_AVAILABLE = False
            print(f"Warning: Local caption generator not available: {e}")
            print("Install with: pip install transformers torch pillow sentence-transformers")
    return LOCAL_CAPTION_AVAILABLE


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file"""
    import hashlib
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def calculate_perceptual_hash(file_path: Path) -> str:
    """Calculate perceptual hash of an image"""
    if not HAS_IMAGEHASH or not HAS_IMAGE_SUPPORT:
        return ""
    
    try:
        with Image.open(file_path) as img:
            phash = imagehash.phash(img)
            return str(phash)
    except Exception as e:
        print(f"Warning: Could not calculate perceptual hash: {e}")
        return ""


def extract_metadata(file_path: Path) -> dict:
    """Extract metadata from an image file"""
    metadata = {
        'width': 0,
        'height': 0,
        'format': 'UNKNOWN',
        'exif_date': None,
        'date_taken': None,
        'gps_latitude': None,
        'gps_longitude': None,
        'camera_make': None,
        'camera_model': None
    }
    
    if not HAS_IMAGE_SUPPORT:
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        metadata['date_taken'] = mtime
        return metadata
    
    try:
        with Image.open(file_path) as img:
            metadata['width'], metadata['height'] = img.size
            metadata['format'] = img.format or 'UNKNOWN'
            
            # Extract EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                exif_tags = {
                    271: 'make',
                    272: 'model',
                    306: 'datetime',
                    36867: 'datetime_original',
                    34853: 'gps_info'
                }
                exif_data = {exif_tags.get(tid, tid): v for tid, v in exif.items()}
                
                # Camera info
                metadata['camera_make'] = exif_data.get('make')
                metadata['camera_model'] = exif_data.get('model')
                
                # Date - prefer original datetime
                date_str = exif_data.get('datetime_original') or exif_data.get('datetime')
                if date_str:
                    metadata['exif_date'] = date_str
                    for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', 
                                '%Y:%m:%d', '%Y-%m-%d']:
                        try:
                            metadata['date_taken'] = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                
                # GPS information
                gps_info = exif_data.get('gps_info')
                if gps_info:
                    lat_ref = gps_info.get(1)  # 'N' or 'S'
                    lat = gps_info.get(2)      # tuple of (deg, min, sec)
                    lon_ref = gps_info.get(3)  # 'E' or 'W'
                    lon = gps_info.get(4)      # tuple of (deg, min, sec)
                    
                    def convert_gps(value, ref):
                        if not value or len(value) < 3:
                            return None
                        try:
                            def to_float(v):
                                if isinstance(v, (int, float)):
                                    return float(v)
                                elif hasattr(v, 'num') and hasattr(v, 'den'):
                                    return float(v.num) / float(v.den)
                                return float(v)
                            
                            degrees = to_float(value[0])
                            minutes = to_float(value[1])
                            seconds = to_float(value[2])
                            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                            if ref in ['S', 'W']:
                                decimal = -decimal
                            return decimal
                        except:
                            return None
                    
                    if lat and lat_ref:
                        metadata['gps_latitude'] = convert_gps(lat, lat_ref)
                    if lon and lon_ref:
                        metadata['gps_longitude'] = convert_gps(lon, lon_ref)
            
            # Fallback to file modification time if no EXIF date
            if not metadata['date_taken']:
                metadata['date_taken'] = datetime.fromtimestamp(file_path.stat().st_mtime)
    
    except Exception as e:
        print(f"Warning: Could not extract metadata: {e}")
        metadata['date_taken'] = datetime.fromtimestamp(file_path.stat().st_mtime)
    
    return metadata


def get_archive_path(file_path: Path, archive_dir: Path, date_taken: datetime) -> Tuple[Path, str]:
    """
    Determine the archive path for a photo based on its date
    
    Returns:
        Tuple of (destination_path, relative_path_string)
    """
    if date_taken:
        year = date_taken.year
        month = date_taken.month
        day = date_taken.day
        rel_path = f"{year}/{month:02d}/{day:02d}"
    else:
        rel_path = "unknown_date"
    
    dest_subdir = archive_dir / rel_path
    dest_file = dest_subdir / file_path.name
    
    return dest_file, rel_path


def copy_to_archive(source_path: Path, archive_dir: Path, date_taken: datetime, 
                   dry_run: bool = False) -> Tuple[Optional[Path], str]:
    """
    Copy a photo to the archive directory in YYYY/MM/DD format
    
    Returns:
        Tuple of (destination_path or None, status_message)
    """
    dest_file, rel_path = get_archive_path(source_path, archive_dir, date_taken)
    
    # Check if file already exists at destination
    if dest_file.exists():
        # Check if it's the same file
        if calculate_sha256(dest_file) == calculate_sha256(source_path):
            return dest_file, f"ALREADY_IN_ARCHIVE ({rel_path})"
        
        # Different file with same name - generate unique name
        sha_prefix = calculate_sha256(source_path)[:8]
        dest_file = archive_dir / rel_path / f"{source_path.stem}_{sha_prefix}{source_path.suffix}"
    
    if dry_run:
        return dest_file, f"WOULD_COPY_TO ({rel_path})"
    
    # Create directory and copy file
    try:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_path), str(dest_file))
        return dest_file, f"COPIED_TO ({rel_path})"
    except Exception as e:
        return None, f"COPY_FAILED: {str(e)}"


def generate_local_caption(caption_generator, image_path: str, 
                          task: str = "<DETAILED_CAPTION>") -> Optional[str]:
    """Generate a caption using local Florence-2 model"""
    if not LOCAL_CAPTION_AVAILABLE or caption_generator is None:
        return None
    
    try:
        caption = caption_generator.generate_caption(image_path, task=task)
        return caption.strip() if caption else None
    except Exception as e:
        print(f"Warning: Failed to generate local caption: {e}")
        return None


def add_single_photo(photo_path: str, db_url: str, archive_dir: str,
                    caption_task: str = "<DETAILED_CAPTION>",
                    dry_run: bool = False,
                    verbose: bool = False) -> Tuple[bool, str]:
    """
    Add a single photo to the database with caption
    
    Args:
        photo_path: Path to the photo file
        db_url: PostgreSQL connection URL
        archive_dir: Directory where photos will be organized (YYYY/MM/DD)
        caption_task: Florence-2 task prompt for caption generation
        dry_run: If True, don't make any changes
        verbose: Print detailed output
    
    Returns:
        Tuple of (success, message)
    """
    source_path = Path(photo_path)
    archive_path = Path(archive_dir)
    
    # Validate inputs
    if not source_path.exists():
        return False, f"ERROR: Photo not found: {photo_path}"
    
    if not source_path.is_file():
        return False, f"ERROR: Not a file: {photo_path}"
    
    # Initialize local caption generator
    if not _try_import_local_caption():
        return False, "ERROR: Local caption generator not available"
    
    print(f"\n{'=' * 60}")
    print(f"ADDING SINGLE PHOTO")
    print(f"{'=' * 60}")
    print(f"Source: {source_path.absolute()}")
    print(f"Archive: {archive_path.absolute()}")
    print(f"Dry run: {dry_run}")
    print()
    
    # Calculate checksum for duplicate detection
    print("Step 1: Checking for duplicates...")
    sha256 = calculate_sha256(source_path)
    
    if not dry_run:
        try:
            db = ImageDatabase(db_url, embedding_dimensions=1536)
            if db.image_exists(sha256):
                return False, f"SKIPPED: Duplicate photo already exists in database (SHA256: {sha256[:16]}...)"
        except Exception as e:
            return False, f"ERROR: Database connection failed: {e}"
    else:
        print("  (Skipping duplicate check in dry-run mode)")
    
    print(f"  SHA256: {sha256[:16]}... (unique)")
    
    # Extract metadata
    print("\nStep 2: Extracting metadata...")
    meta = extract_metadata(source_path)
    date_taken = meta.get('date_taken')
    
    if date_taken:
        print(f"  Date taken: {date_taken}")
    else:
        print(f"  Date taken: Unknown (using file modification time)")
    
    print(f"  Dimensions: {meta['width']}x{meta['height']}")
    print(f"  Format: {meta['format']}")
    
    if meta.get('gps_latitude') and meta.get('gps_longitude'):
        print(f"  GPS: {meta['gps_latitude']:.4f}, {meta['gps_longitude']:.4f}")
    
    # Copy to archive
    print("\nStep 3: Copying to archive...")
    dest_path, copy_status = copy_to_archive(source_path, archive_path, date_taken, dry_run)
    print(f"  {copy_status}")
    
    if dest_path is None and not dry_run:
        return False, f"FAILED: Could not copy to archive: {copy_status}"
    
    # Generate local caption
    print(f"\nStep 4: Generating local caption...")
    print(f"  Loading Florence-2 model (this may take a moment on first run)...")
    
    try:
        caption_generator = FlorenceCaptionGenerator(model_name="microsoft/Florence-2-base")
    except Exception as e:
        return False, f"ERROR: Failed to load caption model: {e}"
    
    # Use the source path for caption generation (before copy)
    caption = generate_local_caption(caption_generator, str(source_path), caption_task)
    
    if caption is None:
        return False, "FAILED: Could not generate local caption"
    
    print(f"  Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
    
    # Calculate perceptual hash
    phash = calculate_perceptual_hash(source_path)
    
    # Prepare database record
    print(f"\nStep 5: Adding to database...")
    
    if dry_run:
        print("  (Skipping database insert in dry-run mode)")
        return True, f"DRY_RUN_SUCCESS: Would add {source_path.name} to database"
    
    try:
        # Create metadata object
        metadata = ImageMetadata(
            file_path=str(dest_path.absolute()),
            file_name=dest_path.name,
            file_size=dest_path.stat().st_size,
            sha256=sha256,
            perceptual_hash=phash,
            width=meta['width'],
            height=meta['height'],
            format=meta['format'],
            date_created=date_taken,
            date_modified=datetime.fromtimestamp(source_path.stat().st_mtime),
            gps_latitude=meta.get('gps_latitude'),
            gps_longitude=meta.get('gps_longitude'),
            is_black_and_white=False,  # Could add B&W detection if needed
            caption=caption,
            caption_embedding=None,  # Could generate embedding if needed
            tags=[]
        )
        
        # Insert into database
        image_id = db.insert_image(metadata)
        
        if image_id:
            print(f"  Successfully added to database (ID: {image_id})")
            return True, f"SUCCESS: Added {source_path.name} to database with caption"
        else:
            return False, "FAILED: Database insert returned no ID"
    
    except Exception as e:
        return False, f"ERROR: Database insert failed: {e}"


def main():
    parser = argparse.ArgumentParser(
        description='Add a single photo to the database with local caption generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('photo', help='Path to the photo file')
    parser.add_argument('--db-url', required=True, 
                       help='PostgreSQL connection URL (e.g., postgresql://user:pass@localhost/dbname)')
    parser.add_argument('--archive-dir', required=True,
                       help='Archive directory where photos will be organized (YYYY/MM/DD)')
    parser.add_argument('--caption-task', default='<DETAILED_CAPTION>',
                       choices=['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>'],
                       help='Florence-2 task prompt for caption generation')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output')
    
    args = parser.parse_args()
    
    # Execute
    success, message = add_single_photo(
        photo_path=args.photo,
        db_url=args.db_url,
        archive_dir=args.archive_dir,
        caption_task=args.caption_task,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    print(f"\n{'=' * 60}")
    print(message)
    print(f"{'=' * 60}\n")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
