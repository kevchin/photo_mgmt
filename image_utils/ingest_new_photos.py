#!/usr/bin/env python3
"""
New Photo Ingestion Workflow

This script provides a complete workflow for ingesting new photos into your 
PostgreSQL-backed photo archive without reprocessing existing photos.

Workflow:
1. Scan new photos folder (< 100 photos typically)
2. Check each photo against PostgreSQL database using SHA256 checksums
3. Identify unique (non-duplicate) photos
4. Copy unique photos to YYYY/MM/DD organized archive directory
5. Add unique photos to PostgreSQL database with metadata and optional captions

Key Features:
- Only processes NEW photos (not already in database)
- Uses SHA256 checksums for exact duplicate detection
- Uses perceptual hashing for near-duplicate detection
- Organizes by EXIF date or file modification date
- Optional AI caption generation (local or OpenAI)
- Safe dry-run mode to preview changes
- Progress tracking for large batches

Usage Examples:
    # Dry run to see what would be added
    python ingest_new_photos.py \
        --new-photos /path/to/new/photos \
        --archive-dir /path/to/archive \
        --db "postgresql://user:pass@localhost/image_archive" \
        --dry-run

    # Actually ingest new photos with local AI captions
    python ingest_new_photos.py \
        --new-photos /path/to/new/photos \
        --archive-dir /path/to/archive \
        --db "postgresql://user:pass@localhost/image_archive" \
        --local-captions

    # Ingest with OpenAI captions
    python ingest_new_photos.py \
        --new-photos /path/to/new/photos \
        --archive-dir /path/to/archive \
        --db "postgresql://user:pass@localhost/image_archive" \
        --openai-captions \
        --api-key $OPENAI_API_KEY
"""

import os
import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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

# Import from existing modules
sys.path.insert(0, str(Path(__file__).parent))

# Handle optional imports gracefully
DB_AVAILABLE = False
ORGANIZER_AVAILABLE = False

try:
    from image_database import ImageDatabase, ImageMetadata
    DB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: image_database module not available: {e}")
    # Define stub classes for type hints when DB is not available
    class ImageDatabase:
        pass
    class ImageMetadata:
        pass

try:
    from image_organizer import parse_exif_date, extract_metadata as extract_org_metadata, convert_gps_coordinate
    ORGANIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: image_organizer module not available: {e}")


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}


@dataclass
class NewPhotoInfo:
    """Information about a new photo to be ingested"""
    file_path: str
    sha256: str
    perceptual_hash: str
    is_duplicate: bool
    duplicate_id: Optional[int] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    date_created: Optional[datetime] = None
    width: int = 0
    height: int = 0
    format: str = 'UNKNOWN'
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
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


def scan_new_photos(directory: str) -> List[Path]:
    """Scan directory for image files"""
    path = Path(directory)
    if not path.exists():
        print(f"Error: Directory not found: {directory}")
        return []
    
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(path.glob(f'**/*{ext}'))
        image_files.extend(path.glob(f'**/*{ext.upper()}'))
    
    return sorted(image_files)


def check_duplicates(db: ImageDatabase, file_path: Path) -> Tuple[bool, Optional[int]]:
    """Check if photo exists in database by SHA256"""
    sha256 = calculate_sha256(file_path)
    exists = db.image_exists(sha256)
    
    if exists:
        # Find the existing record
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM images WHERE sha256 = %s", (sha256,))
                result = cur.fetchone()
                dup_id = result[0] if result else None
                return True, dup_id
    
    return False, None


def extract_photo_metadata(file_path: Path) -> Dict:
    """Extract metadata from a photo"""
    metadata = {
        'exif_date': None, 'parsed_date': None, 'year': None, 'month': None, 'day': None,
        'gps_latitude': None, 'gps_longitude': None, 'camera_make': None, 'camera_model': None,
        'width': 0, 'height': 0, 'format': 'UNKNOWN'
    }
    
    if not HAS_IMAGE_SUPPORT:
        dt = datetime.fromtimestamp(os.path.getmtime(file_path))
        metadata['parsed_date'] = dt.isoformat()
        metadata['year'], metadata['month'], metadata['day'] = dt.year, dt.month, dt.day
        return metadata
    
    try:
        with Image.open(file_path) as img:
            metadata['width'], metadata['height'] = img.size
            metadata['format'] = img.format or 'UNKNOWN'
            
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                TAGS = {271: 'make', 272: 'model', 306: 'datetime', 36867: 'datetime_original', 34853: 'gps_info'}
                exif_data = {TAGS.get(tid, tid): v for tid, v in exif.items()}
                
                # Date
                date_str = exif_data.get('datetime_original') or exif_data.get('datetime')
                if date_str:
                    metadata['exif_date'] = date_str
                    parsed = parse_exif_date(date_str) if ORGANIZER_AVAILABLE else None
                    if parsed:
                        metadata['parsed_date'] = parsed.isoformat()
                        metadata['year'], metadata['month'], metadata['day'] = parsed.year, parsed.month, parsed.day
                
                # Camera
                metadata['camera_make'] = exif_data.get('make')
                metadata['camera_model'] = exif_data.get('model')
                
                # GPS
                gps_info = exif_data.get('gps_info')
                if gps_info:
                    from image_organizer import convert_gps_coordinate
                    lat_ref, lat = gps_info.get(1), gps_info.get(2)
                    lon_ref, lon = gps_info.get(3), gps_info.get(4)
                    if lat and lat_ref:
                        metadata['gps_latitude'] = convert_gps_coordinate(lat, lat_ref)
                    if lon and lon_ref:
                        metadata['gps_longitude'] = convert_gps_coordinate(lon, lon_ref)
            
            # Fallback to file modification time
            if not metadata['parsed_date']:
                dt = datetime.fromtimestamp(os.path.getmtime(file_path))
                metadata['parsed_date'] = dt.isoformat()
                metadata['year'], metadata['month'], metadata['day'] = dt.year, dt.month, dt.day
                
    except Exception as e:
        print(f"Warning: Could not extract metadata from {file_path}: {e}")
        # Fallback
        dt = datetime.fromtimestamp(os.path.getmtime(file_path))
        metadata['year'], metadata['month'], metadata['day'] = dt.year, dt.month, dt.day
    
    return metadata


def organize_photo(file_path: Path, archive_dir: Path, metadata: Dict, dry_run: bool = False) -> Path:
    """Copy/move photo to YYYY/MM/DD organized structure"""
    year = metadata.get('year', 'unknown')
    month = metadata.get('month', 1)
    day = metadata.get('day', 1)
    
    # Create destination path
    if year and month:
        if day:
            rel_path = f"{year}/{month:02d}/{day:02d}"
        else:
            rel_path = f"{year}/{month:02d}/unknown_day"
    elif year:
        rel_path = f"{year}/unknown_month"
    else:
        rel_path = "unknown_date"
    
    dest_dir = archive_dir / rel_path
    dest_file = dest_dir / file_path.name
    
    # Handle filename conflicts
    if dest_file.exists():
        sha_short = calculate_sha256(file_path)[:8]
        dest_file = dest_dir / f"{file_path.stem}_{sha_short}{file_path.suffix}"
    
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(file_path), str(dest_file))
    else:
        print(f"  Would copy: {file_path} -> {dest_file}")
    
    return dest_file


def generate_caption_local(file_path: Path, caption_generator) -> str:
    """Generate caption using local Florence-2 model"""
    try:
        caption = caption_generator.generate_caption(str(file_path))
        return caption
    except Exception as e:
        print(f"    Warning: Failed to generate caption: {e}")
        return None


def process_new_photos(
    new_photos_dir: str,
    archive_dir: str,
    postgres_conn: str,
    generate_captions: bool = False,
    caption_method: str = 'local',
    api_key: Optional[str] = None,
    model: str = 'gpt-4o',
    dry_run: bool = False,
    move_originals: bool = False
) -> Dict:
    """
    Main workflow to process new photos
    
    Returns statistics about the ingestion
    """
    results = {
        'scanned': 0,
        'duplicates': 0,
        'unique': 0,
        'organized': 0,
        'ingested': 0,
        'captions_generated': 0
    }
    
    # Initialize database
    if not dry_run:
        if not DB_AVAILABLE:
            print("Error: PostgreSQL database module not available")
            return results
        db = ImageDatabase(postgres_conn, embedding_dimensions=1536)
        print(f"Connected to PostgreSQL database")
    else:
        print("Dry run mode - no database operations")
        db = None
    
    # Initialize caption generator if needed
    caption_gen = None
    if generate_captions and not dry_run:
        if caption_method == 'local':
            try:
                from generate_captions_local import CaptionGenerator as LocalCaptionGenerator
                caption_gen = LocalCaptionGenerator(model='microsoft/Florence-2-base')
                print(f"Initialized local caption generator (Florence-2)")
            except Exception as e:
                print(f"Warning: Could not initialize local caption generator: {e}")
                print("Run: pip install torch transformers")
                generate_captions = False
        elif caption_method == 'openai' and api_key:
            try:
                from generate_captions import CaptionGenerator as OpenAICaptionGenerator
                caption_gen = OpenAICaptionGenerator(api_key=api_key, model=model)
                print(f"Initialized OpenAI caption generator ({model})")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI caption generator: {e}")
                generate_captions = False
    
    # Scan new photos
    print(f"\n=== Scanning New Photos: {new_photos_dir} ===")
    new_photo_files = scan_new_photos(new_photos_dir)
    results['scanned'] = len(new_photo_files)
    print(f"Found {len(new_photo_files)} photos to check")
    
    if not new_photo_files:
        print("No photos found to process")
        return results
    
    # Process each photo
    photos_to_ingest = []
    
    print(f"\n=== Checking for Duplicates ===")
    for i, file_path in enumerate(new_photo_files, 1):
        if i % 10 == 0:
            print(f"  Checked {i}/{len(new_photo_files)} photos...")
        
        # Check if duplicate
        if db and not dry_run:
            is_dup, dup_id = check_duplicates(db, file_path)
        else:
            is_dup, dup_id = False, None
        
        if is_dup:
            results['duplicates'] += 1
            print(f"  [{i}/{len(new_photo_files)}] DUPLICATE (ID: {dup_id}): {file_path.name}")
            continue
        
        results['unique'] += 1
        
        # Extract metadata
        metadata = extract_photo_metadata(file_path)
        phash = calculate_perceptual_hash(file_path)
        sha256 = calculate_sha256(file_path)
        
        # Organize photo
        archive_path = organize_photo(file_path, Path(archive_dir), metadata, dry_run)
        results['organized'] += 1
        
        # Generate caption if requested
        caption = None
        if generate_captions and caption_gen and not dry_run:
            print(f"  Generating caption for: {file_path.name}")
            caption = generate_caption_local(file_path, caption_gen) if caption_method == 'local' else caption_gen.generate_caption(str(file_path))
            if caption:
                results['captions_generated'] += 1
                print(f"    Caption: {caption[:80]}...")
        
        # Prepare for ingestion
        photo_info = NewPhotoInfo(
            file_path=str(archive_path.absolute()) if not dry_run else str(archive_path),
            sha256=sha256,
            perceptual_hash=phash,
            is_duplicate=False,
            year=metadata['year'],
            month=metadata['month'],
            day=metadata['day'],
            date_created=datetime.fromisoformat(metadata['parsed_date']) if metadata['parsed_date'] else None,
            width=metadata['width'],
            height=metadata['height'],
            format=metadata['format'],
            gps_latitude=metadata['gps_latitude'],
            gps_longitude=metadata['gps_longitude'],
            camera_make=metadata['camera_make'],
            camera_model=metadata['camera_model'],
            caption=caption
        )
        photos_to_ingest.append(photo_info)
        
        if not dry_run and i % 10 == 0:
            print(f"  Prepared {i} unique photos for ingestion...")
    
    # Ingest into database
    if not dry_run and db and photos_to_ingest:
        print(f"\n=== Ingesting {len(photos_to_ingest)} Photos into Database ===")
        
        for i, photo in enumerate(photos_to_ingest, 1):
            try:
                meta = ImageMetadata(
                    file_path=photo.file_path,
                    file_name=Path(photo.file_path).name,
                    file_size=Path(photo.file_path).stat().st_size,
                    sha256=photo.sha256,
                    perceptual_hash=photo.perceptual_hash,
                    width=photo.width,
                    height=photo.height,
                    format=photo.format,
                    date_created=photo.date_created,
                    date_modified=datetime.now(),
                    gps_latitude=photo.gps_latitude,
                    gps_longitude=photo.gps_longitude,
                    caption=photo.caption,
                    caption_embedding=None,  # Will be generated separately
                    tags=[]
                )
                
                db.insert_image(meta)
                results['ingested'] += 1
                
                if i % 10 == 0:
                    print(f"  Ingested {i}/{len(photos_to_ingest)} photos...")
                    
            except Exception as e:
                print(f"  Error ingesting {photo.file_path}: {e}")
        
        print(f"\nSuccessfully ingested {results['ingested']} photos into database")
    
    # If move_originals, delete originals after successful ingestion
    if move_originals and not dry_run and results['ingested'] > 0:
        print(f"\n=== Removing Original Files ===")
        for photo in photos_to_ingest:
            # Find original file in new_photos_dir
            original_name = Path(photo.file_path).name
            # Remove any _sha256 suffix that was added
            for ext in SUPPORTED_EXTENSIONS:
                if original_name.endswith(ext):
                    base_name = original_name[:-len(ext)]
                    if '_' in base_name:
                        parts = base_name.rsplit('_', 1)
                        if len(parts[1]) == 8 and all(c in '0123456789abcdef' for c in parts[1]):
                            original_name = parts[0] + ext
                    break
            
            # Try to find and delete original
            for orig_path in Path(new_photos_dir).glob(f'**/{original_name}'):
                if orig_path.exists():
                    orig_path.unlink()
                    print(f"  Removed original: {orig_path}")
                    break
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"INGESTION SUMMARY")
    print(f"{'='*60}")
    print(f"Photos scanned:          {results['scanned']}")
    print(f"Duplicates found:        {results['duplicates']}")
    print(f"Unique photos:           {results['unique']}")
    print(f"Photos organized:        {results['organized']}")
    print(f"Photos ingested to DB:   {results['ingested']}")
    print(f"Captions generated:      {results['captions_generated']}")
    print(f"{'='*60}")
    
    if dry_run:
        print(f"\nNOTE: This was a DRY RUN. No files were copied or database changes made.")
        print(f"Re-run without --dry-run to actually ingest the photos.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Ingest new photos into PostgreSQL archive without reprocessing existing photos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview what would be ingested
  python ingest_new_photos.py \\
      --new-photos /path/to/new/photos \\
      --archive-dir /path/to/archive \\
      --db "postgresql://user:pass@localhost/image_archive" \\
      --dry-run

  # Actually ingest with local AI captions (offline)
  python ingest_new_photos.py \\
      --new-photos /path/to/new/photos \\
      --archive-dir /path/to/archive \\
      --db "postgresql://user:pass@localhost/image_archive" \\
      --local-captions

  # Ingest with OpenAI captions
  python ingest_new_photos.py \\
      --new-photos /path/to/new/photos \\
      --archive-dir /path/to/archive \\
      --db "postgresql://user:pass@localhost/image_archive" \\
      --openai-captions \\
      --api-key $OPENAI_API_KEY

  # Move originals to archive instead of copying
  python ingest_new_photos.py \\
      --new-photos /path/to/new/photos \\
      --archive-dir /path/to/archive \\
      --db "postgresql://user:pass@localhost/image_archive" \\
      --move
        """
    )
    
    parser.add_argument('--new-photos', required=True,
                        help='Directory containing new photos to ingest')
    parser.add_argument('--archive-dir', required=True,
                        help='Destination archive directory (YYYY/MM/DD structure)')
    parser.add_argument('--db', required=True,
                        help='PostgreSQL connection string')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files or database')
    parser.add_argument('--move', action='store_true',
                        help='Move originals to archive instead of copying')
    
    # Caption generation options
    caption_group = parser.add_mutually_exclusive_group()
    caption_group.add_argument('--local-captions', action='store_true',
                               help='Generate captions using local Florence-2 model (offline)')
    caption_group.add_argument('--openai-captions', action='store_true',
                               help='Generate captions using OpenAI API')
    
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'),
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', default='gpt-4o',
                        help='OpenAI model for caption generation (default: gpt-4o)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.new_photos).exists():
        print(f"Error: New photos directory does not exist: {args.new_photos}")
        sys.exit(1)
    
    # Create archive dir if it doesn't exist (unless dry run)
    if not args.dry_run and not Path(args.archive_dir).exists():
        Path(args.archive_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created archive directory: {args.archive_dir}")
    
    # Determine caption method
    generate_captions = args.local_captions or args.openai_captions
    caption_method = 'local' if args.local_captions else 'openai' if args.openai_captions else None
    
    if args.openai_captions and not args.api_key:
        print("Error: OpenAI API key required. Use --api-key or set OPENAI_API_KEY env var.")
        sys.exit(1)
    
    # Run the workflow
    process_new_photos(
        new_photos_dir=args.new_photos,
        archive_dir=args.archive_dir,
        postgres_conn=args.db,
        generate_captions=generate_captions,
        caption_method=caption_method,
        api_key=args.api_key,
        model=args.model,
        dry_run=args.dry_run,
        move_originals=args.move
    )


if __name__ == '__main__':
    main()
