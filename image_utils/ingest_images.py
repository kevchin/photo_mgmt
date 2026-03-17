#!/usr/bin/env python3
"""
Image Ingestion Utility

Ingests images from an organized directory or SQLite database into PostgreSQL
with pgvector support. Generates embeddings for semantic search.

Features:
- Read images from directory or SQLite database (from image_organizer.py)
- Extract metadata (EXIF, GPS, dimensions, etc.)
- Generate perceptual hashes for duplicate detection
- Generate AI captions and embeddings (optional)
- Store everything in PostgreSQL with pgvector
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_IMAGE_SUPPORT = True
except ImportError as e:
    print(f"Warning: Image libraries not installed: {e}")
    HAS_IMAGE_SUPPORT = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not installed. Perceptual hashing disabled.")

# Import database module
from image_database import ImageDatabase, ImageMetadata

# Import orientation correction utility
try:
    from image_orientation import prepare_image_for_processing, cleanup_temp_image, correct_image_orientation
    ORIENTATION_AVAILABLE = True
except ImportError:
    ORIENTATION_AVAILABLE = False
    print("Warning: image_orientation module not found. Auto-rotation disabled.")


@dataclass
class ImageInfo:
    """Container for image information during ingestion"""
    file_path: str
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    caption: Optional[str] = None


def extract_metadata_from_file(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from an image file"""
    metadata = {
        'width': 0,
        'height': 0,
        'format': 'UNKNOWN',
        'exif_date': None,
        'gps_latitude': None,
        'gps_longitude': None,
        'camera_make': None,
        'camera_model': None
    }
    
    if not HAS_IMAGE_SUPPORT:
        return metadata
    
    try:
        with Image.open(file_path) as img:
            metadata['width'], metadata['height'] = img.size
            metadata['format'] = img.format or 'UNKNOWN'
            
            # Extract EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                TAGS = {
                    271: 'make',
                    272: 'model',
                    306: 'datetime',
                    36867: 'datetime_original',
                    34853: 'gps_info'
                }
                exif_data = {TAGS.get(tid, tid): v for tid, v in exif.items()}
                
                # Camera info
                metadata['camera_make'] = exif_data.get('make')
                metadata['camera_model'] = exif_data.get('model')
                
                # Date
                date_str = exif_data.get('datetime_original') or exif_data.get('datetime')
                if date_str:
                    metadata['exif_date'] = date_str
                
                # GPS
                gps_info = exif_data.get('gps_info')
                if gps_info:
                    lat_ref = gps_info.get(1)
                    lat = gps_info.get(2)
                    lon_ref = gps_info.get(3)
                    lon = gps_info.get(4)
                    
                    if lat and lat_ref and len(lat) >= 3:
                        try:
                            degrees = float(lat[0]) if isinstance(lat[0], (int, float)) else float(lat[0].num) / float(lat[0].den)
                            minutes = float(lat[1]) if isinstance(lat[1], (int, float)) else float(lat[1].num) / float(lat[1].den)
                            seconds = float(lat[2]) if isinstance(lat[2], (int, float)) else float(lat[2].num) / float(lat[2].den)
                            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                            if lat_ref in ['S', 'W']:
                                decimal = -decimal
                            metadata['gps_latitude'] = decimal
                        except:
                            pass
                    
                    if lon and lon_ref and len(lon) >= 3:
                        try:
                            degrees = float(lon[0]) if isinstance(lon[0], (int, float)) else float(lon[0].num) / float(lon[0].den)
                            minutes = float(lon[1]) if isinstance(lon[1], (int, float)) else float(lon[1].num) / float(lon[1].den)
                            seconds = float(lon[2]) if isinstance(lon[2], (int, float)) else float(lon[2].num) / float(lon[2].den)
                            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                            if lon_ref in ['S', 'W']:
                                decimal = -decimal
                            metadata['gps_longitude'] = decimal
                        except:
                            pass
    except Exception as e:
        print(f"Warning: Could not extract metadata from {file_path}: {e}")
    
    return metadata


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


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse EXIF date string to datetime"""
    if not date_str:
        return None
    
    for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y:%m:%d', '%Y-%m-%d']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def read_sqlite_database(db_path: str) -> List[Dict[str, Any]]:
    """Read image records from SQLite database created by image_organizer.py"""
    if not os.path.exists(db_path):
        print(f"Error: SQLite database not found: {db_path}")
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM images")
    rows = cursor.fetchall()
    
    images = []
    for row in rows:
        images.append({
            'file_path': row['file_path'],
            'year': row['year'],
            'month': row['month'],
            'day': row['day'],
            'gps_latitude': row['gps_latitude'],
            'gps_longitude': row['gps_longitude'],
            'camera_make': row['camera_make'],
            'camera_model': row['camera_model'],
            'caption': row['caption'],
            'checksum_sha256': row['checksum_sha256'],
            'width': row['width'],
            'height': row['height'],
            'format': row['format'],
            'parsed_date': row['parsed_date']
        })
    
    conn.close()
    return images


def scan_directory(dir_path: str, supported_extensions=None) -> List[str]:
    """Scan directory for image files"""
    if supported_extensions is None:
        supported_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    
    path = Path(dir_path)
    if not path.exists():
        print(f"Error: Directory not found: {dir_path}")
        return []
    
    image_files = []
    for ext in supported_extensions:
        image_files.extend(path.glob(f'**/*{ext}'))
        image_files.extend(path.glob(f'**/*{ext.upper()}'))
    
    return [str(f) for f in image_files]


def ingest_from_sqlite(sqlite_db: str, postgres_conn: str, 
                       generate_captions: bool = False,
                       caption_api_key: Optional[str] = None,
                       caption_model: str = "gpt-4o",
                       batch_size: int = 100,
                       dry_run: bool = False,
                       correct_orientation: bool = True) -> int:
    """Ingest images from SQLite database into PostgreSQL"""
    
    print(f"Reading images from SQLite database: {sqlite_db}")
    images = read_sqlite_database(sqlite_db)
    
    if not images:
        print("No images found in SQLite database")
        return 0
    
    print(f"Found {len(images)} images to ingest")
    
    return process_images(
        images=images,
        postgres_conn=postgres_conn,
        generate_captions=generate_captions,
        caption_api_key=caption_api_key,
        caption_model=caption_model,
        batch_size=batch_size,
        dry_run=dry_run,
        source_type="sqlite",
        correct_orientation=correct_orientation
    )


def ingest_from_directory(source_dir: str, postgres_conn: str,
                          generate_captions: bool = False,
                          caption_api_key: Optional[str] = None,
                          caption_model: str = "gpt-4o",
                          batch_size: int = 100,
                          dry_run: bool = False,
                          correct_orientation: bool = True) -> int:
    """Ingest images from a directory into PostgreSQL"""
    
    print(f"Scanning directory: {source_dir}")
    image_files = scan_directory(source_dir)
    
    if not image_files:
        print("No images found in directory")
        return 0
    
    print(f"Found {len(image_files)} images to ingest")
    
    # Convert to dict format for processing
    images = [{'file_path': f} for f in image_files]
    
    return process_images(
        images=images,
        postgres_conn=postgres_conn,
        generate_captions=generate_captions,
        caption_api_key=caption_api_key,
        caption_model=caption_model,
        batch_size=batch_size,
        dry_run=dry_run,
        source_type="directory",
        correct_orientation=correct_orientation
    )


def process_images(images: List[Dict], postgres_conn: str,
                   generate_captions: bool = False,
                   caption_api_key: Optional[str] = None,
                   caption_model: str = "gpt-4o",
                   batch_size: int = 100,
                   dry_run: bool = False,
                   source_type: str = "directory",
                   correct_orientation: bool = True) -> int:
    """Process and ingest images into PostgreSQL"""
    
    # Initialize PostgreSQL database
    if not dry_run:
        db = ImageDatabase(postgres_conn, embedding_dimensions=1536)
    else:
        print("Dry run mode - no database operations will be performed")
        db = None
    
    # Initialize caption generator if needed
    caption_gen = None
    if generate_captions and caption_api_key:
        try:
            from generate_captions import CaptionGenerator
            caption_gen = CaptionGenerator(api_key=caption_api_key, model=caption_model)
            print(f"Caption generation enabled using model: {caption_model}")
        except Exception as e:
            print(f"Warning: Could not initialize caption generator: {e}")
            generate_captions = False
    
    processed_count = 0
    metadata_list = []
    
    for i, img_info in enumerate(images, 1):
        file_path = img_info['file_path']
        
        if i % 10 == 0:
            print(f"Processing {i}/{len(images)}...")
        
        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        # Apply orientation correction before processing
        processing_path = str(path)
        if correct_orientation and ORIENTATION_AVAILABLE:
            was_corrected, corrected_path, reason = correct_image_orientation(
                str(path), overwrite=False
            )
            if was_corrected:
                print(f"  Orientation corrected: {reason}")
                processing_path = corrected_path
                path = Path(corrected_path)  # Update path for subsequent processing
        
        # Calculate checksum (use original file for consistency)
        sha256 = calculate_sha256(Path(file_path))
        
        # Check if already in database (skip if exists)
        if not dry_run and db.image_exists(sha256):
            print(f"  Skipping (duplicate): {path.name}")
            continue
        
        # Extract metadata
        file_meta = extract_metadata_from_file(path)
        
        # Get date information
        date_created = None
        year, month, day = None, None, None
        
        if source_type == "sqlite":
            # Use metadata from SQLite
            year = img_info.get('year')
            month = img_info.get('month')
            day = img_info.get('day')
            if img_info.get('parsed_date'):
                try:
                    date_created = datetime.fromisoformat(img_info['parsed_date'])
                except:
                    pass
        elif file_meta.get('exif_date'):
            date_created = parse_date_string(file_meta['exif_date'])
            if date_created:
                year, month, day = date_created.year, date_created.month, date_created.day
        
        # Fallback to file modification time
        if not date_created:
            date_created = datetime.fromtimestamp(path.stat().st_mtime)
            if not year:
                year, month, day = date_created.year, date_created.month, date_created.day
        
        # Calculate perceptual hash
        phash = calculate_perceptual_hash(path)
        
        # Generate caption if requested
        caption = img_info.get('caption') if source_type == "sqlite" else None
        caption_embedding = None
        
        if generate_captions and caption_gen and not caption:
            print(f"  Generating caption for: {path.name}")
            try:
                caption = caption_gen.generate_caption(str(path))
                print(f"    Caption: {caption[:100]}...")
            except Exception as e:
                print(f"    Warning: Failed to generate caption: {e}")
        
        # Generate embedding for caption if we have one
        if generate_captions and caption and caption_gen:
            try:
                embed_gen = caption_gen  # CaptionGenerator can also generate embeddings
                # Note: You may need to add an embedding method to CaptionGenerator
                # For now, we'll skip embedding generation here and use generate_captions.py later
                print(f"    Note: Run generate_captions.py to create embeddings for captions")
            except Exception as e:
                print(f"    Warning: Failed to generate embedding: {e}")
        
        # Create metadata object
        metadata = ImageMetadata(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_size=path.stat().st_size,
            sha256=sha256,
            perceptual_hash=phash,
            width=file_meta['width'],
            height=file_meta['height'],
            format=file_meta['format'],
            date_created=date_created,
            date_modified=datetime.fromtimestamp(path.stat().st_mtime),
            gps_latitude=img_info.get('gps_latitude') or file_meta['gps_latitude'],
            gps_longitude=img_info.get('gps_longitude') or file_meta['gps_longitude'],
            caption=caption,
            caption_embedding=caption_embedding,
            tags=[]
        )
        
        metadata_list.append(metadata)
        processed_count += 1
        
        # Batch insert
        if len(metadata_list) >= batch_size:
            if not dry_run:
                count = db.batch_insert_images(metadata_list)
                print(f"  Inserted batch of {count} images")
            else:
                print(f"  Would insert batch of {len(metadata_list)} images")
            metadata_list = []
    
    # Insert remaining
    if metadata_list:
        if not dry_run:
            count = db.batch_insert_images(metadata_list)
            print(f"  Inserted final batch of {count} images")
        else:
            print(f"  Would insert final batch of {len(metadata_list)} images")
    
    if not dry_run:
        db.pool.closeall()
    
    print(f"\nIngestion complete. Processed {processed_count} images.")
    return processed_count


def main():
    parser = argparse.ArgumentParser(
        description='Ingest images into PostgreSQL with pgvector support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest from SQLite database (created by image_organizer.py)
  python ingest_images.py --sqlite-db images.db --postgres-db postgresql://user:pass@localhost/image_archive

  # Ingest from directory
  python ingest_images.py --source-dir /organized/photos --postgres-db postgresql://user:pass@localhost/image_archive

  # With local caption generation (Florence-2)
  python ingest_images.py --sqlite-db images.db --postgres-db postgresql://... --local-captions

  # Dry run (preview only)
  python ingest_images.py --sqlite-db images.db --postgres-db postgresql://... --dry-run

After ingestion, run generate_captions_local.py to create captions and embeddings locally:
  python generate_captions_local.py --db postgresql://... --from-db --model microsoft/Florence-2-base --embedding-model all-MiniLM-L6-v2
        """
    )
    
    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--sqlite-db', help='SQLite database from image_organizer.py')
    source_group.add_argument('--source-dir', help='Directory containing organized images')
    
    # Database
    parser.add_argument('--postgres-db', required=True,
                        help='PostgreSQL connection string (e.g., postgresql://user:pass@localhost/dbname)')
    
    # Caption generation
    parser.add_argument('--local-captions', action='store_true',
                        help='Generate captions locally using Florence-2 (requires generate_captions_local.py)')
    parser.add_argument('--caption-model', default='microsoft/Florence-2-base',
                        help='Local model for caption generation (default: microsoft/Florence-2-base)')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                        help='Model for generating embeddings (default: all-MiniLM-L6-v2)')
    
    # Options
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for database inserts (default: 100)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview operations without making changes')
    
    args = parser.parse_args()
    
    # Run ingestion
    if args.sqlite_db:
        count = ingest_from_sqlite(
            sqlite_db=args.sqlite_db,
            postgres_conn=args.postgres_db,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
    else:
        count = ingest_from_directory(
            source_dir=args.source_dir,
            postgres_conn=args.postgres_db,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
    
    if count == 0:
        print("\nNo images were ingested.")
        sys.exit(0)
    
    print(f"\nSuccessfully ingested {count} images into PostgreSQL.")
    
    if args.local_captions:
        print("\n" + "="*60)
        print("Next step: Generate captions and embeddings locally")
        print("="*60)
        print(f"""
Run the following command to generate captions and embeddings:

  python generate_captions_local.py --db "{args.postgres_db}" \\
      --from-db \\
      --model {args.caption_model} \\
      --embedding-model {args.embedding_model}

This will:
  1. Find all images in the database without captions
  2. Generate captions using Florence-2 (runs locally, no API needed)
  3. Create vector embeddings for semantic search
  4. Update the PostgreSQL database automatically

Note: First run may take time to download the models.
""")
    else:
        print("\nNote: To enable semantic search, generate captions and embeddings:")
        print(f"  python generate_captions_local.py --db \"{args.postgres_db}\" --from-db")


if __name__ == '__main__':
    main()
