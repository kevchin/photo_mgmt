#!/usr/bin/env python3
"""
Incremental Photo Ingestion Workflow

This script provides an efficient workflow for adding new photos to your 
PostgreSQL-backed photo archive without reprocessing existing photos.

Workflow:
1. Scan new photos folder (< 100 photos typically)
2. Check each photo against PostgreSQL database using SHA256 checksums
3. Skip duplicates (already in database)
4. For unique photos:
   - Extract date metadata (EXIF or file modification time)
   - Copy to YYYY/MM/DD organized archive directory
   - Add to PostgreSQL database with metadata
   - Optionally generate AI captions

Key Benefits:
- Only processes NEW photos (fast, even if archive has thousands)
- Uses database checksums for deduplication (no need to scan archive)
- Organizes photos by date automatically
- Integrates with existing caption generation workflow

Usage Examples:
    # Basic usage - just organize and add to database
    python incremental_ingest.py \\
        --new-photos /path/to/new/photos \\
        --archive-dir /path/to/archive \\
        --db "postgresql://user:pass@localhost/image_archive"

    # With local AI caption generation (offline, no API key needed)
    python incremental_ingest.py \\
        --new-photos /path/to/new/photos \\
        --archive-dir /path/to/archive \\
        --db "postgresql://user:pass@localhost/image_archive" \\
        --generate-captions \\
        --local-captions

    # Dry run - see what would be done without making changes
    python incremental_ingest.py \\
        --new-photos /path/to/new/photos \\
        --archive-dir /path/to/archive \\
        --db "postgresql://user:pass@localhost/image_archive" \\
        --dry-run
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# Import from existing modules
try:
    from image_database import ImageDatabase, ImageMetadata
    POSTGRES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: image_database module not available: {e}")
    POSTGRES_AVAILABLE = False

try:
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_IMAGE_SUPPORT = True
except ImportError as e:
    print(f"Warning: Image libraries not installed: {e}")
    print("Run: pip install Pillow pillow-heif")
    HAS_IMAGE_SUPPORT = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not installed. Perceptual hashing disabled.")

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}


@dataclass
class PhotoInfo:
    """Container for photo information during ingestion"""
    file_path: str
    file_name: str
    file_size: int
    sha256: str
    perceptual_hash: str
    width: int
    height: int
    format: str
    date_taken: Optional[datetime]
    year: Optional[int]
    month: Optional[int]
    day: Optional[int]
    gps_latitude: Optional[float]
    gps_longitude: Optional[float]
    camera_make: Optional[str]
    camera_model: Optional[str]
    caption: Optional[str] = None


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
        print(f"Warning: Could not calculate perceptual hash for {file_path}: {e}")
        return ""


def parse_exif_date(date_str: str) -> Optional[datetime]:
    """Parse EXIF date string to datetime"""
    if not date_str:
        return None
    
    for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', 
                '%Y:%m:%d', '%Y-%m-%d']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def extract_photo_metadata(file_path: Path) -> Dict:
    """Extract metadata from a photo file"""
    metadata = {
        'width': 0,
        'height': 0,
        'format': 'UNKNOWN',
        'exif_date': None,
        'date_taken': None,
        'year': None,
        'month': None,
        'day': None,
        'gps_latitude': None,
        'gps_longitude': None,
        'camera_make': None,
        'camera_model': None
    }
    
    if not HAS_IMAGE_SUPPORT:
        # Fallback to file modification time
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        metadata['date_taken'] = mtime
        metadata['year'] = mtime.year
        metadata['month'] = mtime.month
        metadata['day'] = mtime.day
        return metadata
    
    try:
        with Image.open(file_path) as img:
            metadata['width'], metadata['height'] = img.size
            metadata['format'] = img.format or 'UNKNOWN'
            
            # Extract EXIF data using modern getexif() method
            exif_data = img.getexif()
            
            if exif_data:
                # Get standard tags
                TAGS = {
                    271: 'make',
                    272: 'model',
                    306: 'datetime',
                    36867: 'datetime_original',
                }
                
                # Camera info
                metadata['camera_make'] = exif_data.get(TAGS[271])
                metadata['camera_model'] = exif_data.get(TAGS[272])
                
                # Date - prefer original datetime (DateTimeOriginal), then DateTime
                # Tag 36867 = DateTimeOriginal, Tag 306 = DateTime
                date_str = exif_data.get(36867) or exif_data.get(306)
                if date_str:
                    metadata['exif_date'] = str(date_str)
                    parsed = parse_exif_date(str(date_str))
                    if parsed:
                        metadata['date_taken'] = parsed
                
                # GPS information - use get_ifd for proper access
                # GPS IFD is at tag 0x8825 (34853)
                gps_info = exif_data.get_ifd(0x8825)
                
                if gps_info:
                    # Use GPSTAGS to decode GPS tags properly
                    from PIL.ExifTags import GPSTAGS
                    gps_data = {GPSTAGS.get(t, t): v for t, v in gps_info.items()}
                    
                    # Helper function to convert IFDRational to float (matches image_metadata_extractor.py)
                    def convert_to_degrees(value):
                        """Convert GPS coordinates to decimal degrees."""
                        try:
                            # Handle IFDRational objects (PIL.TiffImagePlugin.IFDRational)
                            def get_float(val):
                                if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                                    return float(val.numerator) / float(val.denominator)
                                return float(val)
                            
                            d = get_float(value[0])
                            m = get_float(value[1])
                            s = get_float(value[2])
                            return d + (m / 60.0) + (s / 3600.0)
                        except (ZeroDivisionError, IndexError, TypeError, AttributeError):
                            return None
                    
                    # Extract latitude
                    if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                        try:
                            lat_dms = gps_data['GPSLatitude']
                            lat_ref = gps_data['GPSLatitudeRef']
                            decimal = convert_to_degrees(lat_dms)
                            if decimal is not None:
                                if lat_ref in ['S', 'W']:
                                    decimal = -decimal
                                metadata['gps_latitude'] = decimal
                        except Exception as e:
                            print(f"Warning: Could not parse GPS latitude: {e}")
                    
                    # Extract longitude
                    if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                        try:
                            lon_dms = gps_data['GPSLongitude']
                            lon_ref = gps_data['GPSLongitudeRef']
                            decimal = convert_to_degrees(lon_dms)
                            if decimal is not None:
                                if lon_ref in ['S', 'W']:
                                    decimal = -decimal
                                metadata['gps_longitude'] = decimal
                        except Exception as e:
                            print(f"Warning: Could not parse GPS longitude: {e}")
            
            # Fallback to file modification time if no EXIF date
            if not metadata['date_taken']:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                metadata['date_taken'] = mtime
    
    except Exception as e:
        print(f"Warning: Could not extract metadata from {file_path}: {e}")
        # Fallback to file modification time
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        metadata['date_taken'] = mtime
    
    # Set year, month, day from date_taken
    if metadata['date_taken']:
        metadata['year'] = metadata['date_taken'].year
        metadata['month'] = metadata['date_taken'].month
        metadata['day'] = metadata['date_taken'].day
    
    return metadata


def scan_new_photos(directory: Path) -> List[Path]:
    """Scan directory for new photo files"""
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return []
    
    photo_files = []
    for ext in SUPPORTED_EXTENSIONS:
        photo_files.extend(directory.glob(f'*{ext}'))
        photo_files.extend(directory.glob(f'*{ext.upper()}'))
    
    # Also check subdirectories
    for ext in SUPPORTED_EXTENSIONS:
        photo_files.extend(directory.glob(f'**/*{ext}'))
        photo_files.extend(directory.glob(f'**/*{ext.upper()}'))
    
    # Remove duplicates and sort
    photo_files = sorted(set(photo_files))
    print(f"Found {len(photo_files)} photo files in {directory}")
    return photo_files


def process_photo(file_path: Path, archive_dir: Path, dry_run: bool = False) -> Tuple[Optional[PhotoInfo], str]:
    """
    Process a single photo: extract metadata and determine archive location
    
    Returns:
        Tuple of (PhotoInfo or None, status message)
    """
    try:
        # Extract metadata
        meta = extract_photo_metadata(file_path)
        
        # Calculate checksums
        sha256 = calculate_sha256(file_path)
        phash = calculate_perceptual_hash(file_path)
        
        # Determine archive path based on date
        if meta['year'] and meta['month'] and meta['day']:
            rel_path = f"{meta['year']}/{meta['month']:02d}/{meta['day']:02d}"
        elif meta['year'] and meta['month']:
            rel_path = f"{meta['year']}/{meta['month']:02d}/unknown_day"
        elif meta['year']:
            rel_path = f"{meta['year']}/unknown_month"
        else:
            rel_path = "unknown_date"
        
        dest_subdir = archive_dir / rel_path
        dest_file = dest_subdir / file_path.name
        
        # Handle filename conflicts
        if dest_file.exists():
            # Check if it's the same file (duplicate)
            if calculate_sha256(dest_file) == sha256:
                return None, f"DUPLICATE (same file exists at {dest_file})"
            
            # Different file with same name - add checksum prefix
            cs_prefix = sha256[:8]
            dest_file = dest_subdir / f"{file_path.stem}_{cs_prefix}{file_path.suffix}"
        
        # Create PhotoInfo object
        photo_info = PhotoInfo(
            file_path=str(file_path.absolute()),
            file_name=file_path.name,
            file_size=file_path.stat().st_size,
            sha256=sha256,
            perceptual_hash=phash,
            width=meta['width'],
            height=meta['height'],
            format=meta['format'],
            date_taken=meta['date_taken'],
            year=meta['year'],
            month=meta['month'],
            day=meta['day'],
            gps_latitude=meta['gps_latitude'],
            gps_longitude=meta['gps_longitude'],
            camera_make=meta['camera_make'],
            camera_model=meta['camera_model'],
            caption=None
        )
        
        status = f"UNIQUE (would copy to {dest_file})" if dry_run else f"UNIQUE (copied to {dest_file})"
        
        # Copy file to archive
        if not dry_run:
            dest_subdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(file_path), str(dest_file))
            photo_info.file_path = str(dest_file.absolute())
        
        return photo_info, status
        
    except Exception as e:
        return None, f"ERROR: {str(e)}"


def initialize_caption_generator(use_local: bool = False, api_key: Optional[str] = None, model: str = "gpt-4o"):
    """Initialize caption generator (either local or OpenAI)"""
    if use_local:
        try:
            from generate_captions_local import FlorenceCaptionGenerator as CaptionGeneratorLocal
            print(f"Using local caption generator (Florence-2)")
            return CaptionGeneratorLocal(model_name="microsoft/Florence-2-base")
        except ImportError as e:
            print(f"Warning: Local caption generator not available: {e}")
            print("Install with: pip install torch transformers")
            return None
        except Exception as e:
            print(f"Warning: Failed to initialize local caption generator: {e}")
            return None
    else:
        if not api_key:
            print("Warning: OpenAI API key not provided. Captions disabled.")
            return None
        try:
            from generate_captions import CaptionGenerator
            print(f"Using OpenAI caption generator (model: {model})")
            return CaptionGenerator(api_key=api_key, model=model)
        except ImportError as e:
            print(f"Warning: OpenAI caption generator not available: {e}")
            print("Install with: pip install openai")
            return None


def generate_caption_for_photo(caption_gen, file_path: Path, use_local: bool = False) -> Optional[str]:
    """Generate a caption for a photo"""
    try:
        if use_local:
            # Local Florence-2 model
            caption = caption_gen.generate_caption(str(file_path))
        else:
            # OpenAI
            caption = caption_gen.generate_caption(str(file_path))
        return caption
    except Exception as e:
        print(f"  Warning: Failed to generate caption: {e}")
        return None


def incremental_ingest(
    new_photos_dir: Path,
    archive_dir: Path,
    db_connection_string: str,
    generate_captions: bool = False,
    use_local_captions: bool = False,
    caption_api_key: Optional[str] = None,
    caption_model: str = "gpt-4o",
    dry_run: bool = False,
    batch_size: int = 50
) -> Dict:
    """
    Main incremental ingestion workflow
    
    Args:
        new_photos_dir: Directory containing new photos to ingest
        archive_dir: Directory where photos will be organized (YYYY/MM/DD)
        db_connection_string: PostgreSQL connection string
        generate_captions: Whether to generate AI captions
        use_local_captions: Use local Florence-2 model instead of OpenAI
        caption_api_key: OpenAI API key (if not using local)
        caption_model: Model name for caption generation
        dry_run: If True, don't make any changes
        batch_size: Number of photos to process before committing to database
    
    Returns:
        Dictionary with statistics about the ingestion
    """
    
    stats = {
        'total_scanned': 0,
        'duplicates_skipped': 0,
        'unique_found': 0,
        'added_to_db': 0,
        'captions_generated': 0,
        'errors': 0
    }
    
    print("=" * 80)
    print("INCREMENTAL PHOTO INGESTION")
    print("=" * 80)
    print(f"New photos directory: {new_photos_dir}")
    print(f"Archive directory: {archive_dir}")
    print(f"Dry run: {dry_run}")
    print()
    
    # Initialize database connection
    if not dry_run:
        if not POSTGRES_AVAILABLE:
            print("Error: PostgreSQL database module not available")
            return stats
        db = ImageDatabase(db_connection_string, embedding_dimensions=1536)
        print(f"Connected to PostgreSQL database")
    else:
        print("Dry run mode - no database operations will be performed")
        db = None
    print()
    
    # Initialize caption generator if requested
    caption_gen = None
    if generate_captions and not dry_run:
        caption_gen = initialize_caption_generator(
            use_local=use_local_captions,
            api_key=caption_api_key,
            model=caption_model
        )
    print()
    
    # Scan new photos
    print(f"Scanning for new photos...")
    photo_files = scan_new_photos(new_photos_dir)
    stats['total_scanned'] = len(photo_files)
    
    if not photo_files:
        print("No photos found to process")
        return stats
    
    print()
    print(f"Processing {len(photo_files)} photos...")
    print("-" * 80)
    
    # Process each photo
    unique_photos: List[PhotoInfo] = []
    batch_count = 0
    
    for i, file_path in enumerate(photo_files, 1):
        print(f"[{i}/{len(photo_files)}] Processing: {file_path.name}")
        
        # Check if already in database (using SHA256)
        if db and not dry_run:
            if db.image_exists(file_path.name):  # Check by filename first (faster)
                print(f"  Skipping (filename exists): {file_path.name}")
                stats['duplicates_skipped'] += 1
                continue
            
            # More thorough check by SHA256
            sha256 = calculate_sha256(file_path)
            if db.image_exists(sha256):
                print(f"  Skipping (duplicate by checksum): {file_path.name}")
                stats['duplicates_skipped'] += 1
                continue
        
        # Process the photo
        photo_info, status = process_photo(file_path, archive_dir, dry_run=dry_run)
        
        if photo_info is None:
            print(f"  {status}")
            stats['duplicates_skipped'] += 1
            continue
        
        print(f"  {status}")
        stats['unique_found'] += 1
        unique_photos.append(photo_info)
        
        # Generate caption if requested
        if caption_gen and generate_captions:
            print(f"  Generating caption...")
            caption = generate_caption_for_photo(
                caption_gen, 
                Path(photo_info.file_path),
                use_local=use_local_captions
            )
            if caption:
                photo_info.caption = caption
                stats['captions_generated'] += 1
                print(f"    Caption: {caption[:80]}...")
        
        # Add to database batch
        if db and not dry_run:
            try:
                metadata = ImageMetadata(
                    file_path=photo_info.file_path,
                    file_name=photo_info.file_name,
                    file_size=photo_info.file_size,
                    sha256=photo_info.sha256,
                    perceptual_hash=photo_info.perceptual_hash,
                    width=photo_info.width,
                    height=photo_info.height,
                    format=photo_info.format,
                    date_created=photo_info.date_taken,
                    date_modified=datetime.fromtimestamp(Path(photo_info.file_path).stat().st_mtime),
                    gps_latitude=photo_info.gps_latitude,
                    gps_longitude=photo_info.gps_longitude,
                    caption=photo_info.caption,
                    caption_embedding=None,  # Will be generated later by generate_captions.py
                    tags=[]
                )
                
                db.insert_image(metadata)
                stats['added_to_db'] += 1
                batch_count += 1
                
                if batch_count >= batch_size:
                    print(f"  Batch committed ({batch_count} photos)")
                    batch_count = 0
                    
            except Exception as e:
                print(f"  ERROR adding to database: {e}")
                stats['errors'] += 1
        
        print()
    
    # Final summary
    print()
    print("=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)
    print(f"Total photos scanned:     {stats['total_scanned']}")
    print(f"Duplicates skipped:       {stats['duplicates_skipped']}")
    print(f"Unique photos found:      {stats['unique_found']}")
    if not dry_run:
        print(f"Added to database:        {stats['added_to_db']}")
        print(f"Captions generated:       {stats['captions_generated']}")
    print(f"Errors:                   {stats['errors']}")
    print()
    
    if dry_run:
        print("NOTE: This was a DRY RUN. No files were copied and no database changes were made.")
        print()
    
    if not dry_run and stats['captions_generated'] > 0 and use_local_captions:
        print("Note: Captions were generated but embeddings were not created.")
        print("To create vector embeddings for semantic search, run:")
        print(f"  python generate_captions_local.py --db \"{db_connection_string}\" --from-db --embeddings-only")
        print()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Incremental Photo Ingestion - Add new photos to archive without reprocessing existing ones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python incremental_ingest.py --new-photos ./new_photos --archive ./archive --db "postgresql://user:pass@localhost/image_archive"

  # With local AI captions (offline, no API key needed)
  python incremental_ingest.py --new-photos ./new_photos --archive ./archive --db "postgresql://user:pass@localhost/image_archive" --generate-captions --local-captions

  # With OpenAI captions
  python incremental_ingest.py --new-photos ./new_photos --archive ./archive --db "postgresql://user:pass@localhost/image_archive" --generate-captions --openai-api-key $OPENAI_API_KEY

  # Dry run (preview what would happen)
  python incremental_ingest.py --new-photos ./new_photos --archive ./archive --db "postgresql://user:pass@localhost/image_archive" --dry-run
        """
    )
    
    parser.add_argument('--new-photos', required=True, 
                        help='Directory containing new photos to ingest')
    parser.add_argument('--archive', required=True,
                        help='Archive directory where photos will be organized (YYYY/MM/DD)')
    parser.add_argument('--db', required=True,
                        help='PostgreSQL connection string (e.g., postgresql://user:pass@localhost/dbname)')
    
    parser.add_argument('--generate-captions', action='store_true',
                        help='Generate AI captions for new photos')
    parser.add_argument('--local-captions', action='store_true',
                        help='Use local Florence-2 model for captions (offline, no API key needed)')
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='OpenAI API key for caption generation (if not using local)')
    parser.add_argument('--caption-model', type=str, default='gpt-4o',
                        help='OpenAI model for caption generation (default: gpt-4o)')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what would be done without making changes')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of photos to process before committing to database (default: 50)')
    
    args = parser.parse_args()
    
    # Run the incremental ingestion
    stats = incremental_ingest(
        new_photos_dir=Path(args.new_photos),
        archive_dir=Path(args.archive),
        db_connection_string=args.db,
        generate_captions=args.generate_captions,
        use_local_captions=args.local_captions,
        caption_api_key=args.openai_api_key,
        caption_model=args.caption_model,
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )
    
    # Exit with error code if there were errors
    if stats['errors'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
