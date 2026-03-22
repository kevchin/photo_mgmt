#!/usr/bin/env python3
"""
Photo Archive Ingestion Tool

Ingests photos from organized directories (YYYY/MM/DD structure) into PostgreSQL databases.
Supports multiple archive configurations via YAML config file.

Features:
- Extract EXIF metadata (date, GPS, orientation, camera info)
- Detect black & white photos using local LLM
- Generate captions using local LLM
- Store in configurable PostgreSQL database
- Support for multiple archive databases
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
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

# Import our modules
from image_database import ImageDatabase, ImageMetadata
from archive_config_loader import load_config, GlobalConfig, ArchiveConfig


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file"""
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


def extract_exif_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract EXIF metadata from an image file"""
    metadata = {
        'width': 0,
        'height': 0,
        'format': 'UNKNOWN',
        'date_created': None,
        'gps_latitude': None,
        'gps_longitude': None,
        'orientation': 1,
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
            exif = img.getexif()
            if exif:
                # Basic tags
                metadata['orientation'] = exif.get(274, 1)  # Orientation tag
                metadata['camera_make'] = exif.get(271)  # Make
                metadata['camera_model'] = exif.get(272)  # Model
                
                # Date fields
                date_str = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime
                if date_str:
                    for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y:%m:%d']:
                        try:
                            metadata['date_created'] = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                
                # GPS data
                gps_info = exif.get_ifd(0x8825)  # GPS IFD
                if gps_info:
                    gps_data = {GPSTAGS.get(t, t): v for t, v in gps_info.items()}
                    
                    lat_ref = gps_data.get('GPSLatitudeRef')
                    lat = gps_data.get('GPSLatitude')
                    lon_ref = gps_data.get('GPSLongitudeRef')
                    lon = gps_data.get('GPSLongitude')
                    
                    def to_float(val):
                        if isinstance(val, (int, float)):
                            return float(val)
                        elif hasattr(val, 'numerator'):
                            return float(val.numerator) / float(val.denominator)
                        return float(val)
                    
                    if lat and lat_ref and len(lat) >= 3:
                        try:
                            degrees = to_float(lat[0])
                            minutes = to_float(lat[1])
                            seconds = to_float(lat[2])
                            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                            metadata['gps_latitude'] = -decimal if lat_ref in ['S', 'W'] else decimal
                        except Exception as e:
                            print(f"Warning: Could not parse GPS latitude: {e}")
                    
                    if lon and lon_ref and len(lon) >= 3:
                        try:
                            degrees = to_float(lon[0])
                            minutes = to_float(lon[1])
                            seconds = to_float(lon[2])
                            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                            metadata['gps_longitude'] = -decimal if lon_ref in ['S', 'W'] else decimal
                        except Exception as e:
                            print(f"Warning: Could not parse GPS longitude: {e}")
                            
    except Exception as e:
        print(f"Warning: Could not extract EXIF from {file_path}: {e}")
    
    return metadata


def detect_black_and_white_llm(file_path: Path, llm_config) -> bool:
    """Detect if image is black and white using local LLM"""
    try:
        # Import local caption generator
        from generate_captions_local import LocalCaptionGenerator
        
        generator = LocalCaptionGenerator(
            provider=llm_config.provider,
            base_url=llm_config.base_url,
            model=llm_config.model
        )
        
        # Use a simple prompt to check if B&W
        result = generator.analyze_image(str(file_path), prompt="Is this image black and white (grayscale) or color? Answer only YES for black and white or NO for color.")
        
        response = result.strip().upper()
        return 'YES' in response
        
    except Exception as e:
        print(f"Warning: LLM B&W detection failed for {file_path}: {e}")
        # Fallback to simple pixel analysis
        return detect_black_and_white_simple(file_path)


def detect_black_and_white_simple(file_path: Path) -> bool:
    """Simple fallback B&W detection based on color channels"""
    if not HAS_IMAGE_SUPPORT:
        return False
    
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Sample pixels
            pixels = list(img.getdata())
            sample_size = min(1000, len(pixels))
            step = max(1, len(pixels) // sample_size)
            
            bw_count = 0
            for i in range(0, len(pixels), step):
                r, g, b = pixels[i][:3]
                # Check if R, G, B are very similar (within threshold)
                if abs(r - g) < 10 and abs(g - b) < 10 and abs(r - b) < 10:
                    bw_count += 1
            
            # If >90% of sampled pixels are grayscale, consider it B&W
            return bw_count / sample_size > 0.9
            
    except Exception as e:
        print(f"Warning: Simple B&W detection failed: {e}")
        return False


def generate_caption_llm(file_path: Path, llm_config, caption_detail: str = "basic") -> Optional[str]:
    """Generate caption for image using Florence-2 or local LLM
    
    Args:
        file_path: Path to image
        llm_config: Configuration with model settings
        caption_detail: Detail level - 'basic', 'detailed', or 'very_detailed'
    """
    try:
        # Check if using Florence-2 model
        model_name = getattr(llm_config, 'model', '')
        
        if 'florence' in model_name.lower() or 'Florence' in model_name:
            # Use Florence-2 directly
            from generate_captions_local import FlorenceCaptionGenerator
            
            generator = FlorenceCaptionGenerator(
                model_name=model_name if model_name else "microsoft/Florence-2-base",
                device="auto",
                caption_detail=caption_detail
            )
            
            if caption_detail == "very_detailed":
                caption = generator.generate_very_detailed_caption(str(file_path))
            elif caption_detail == "detailed":
                caption = generator.generate_detailed_caption(str(file_path))
            else:
                caption = generator.generate_basic_caption(str(file_path))
                
            return caption
        else:
            # Use Ollama-based local LLM
            from generate_captions_local import LocalCaptionGenerator
            
            generator = LocalCaptionGenerator(
                provider=getattr(llm_config, 'provider', 'ollama'),
                base_url=getattr(llm_config, 'base_url', 'http://localhost:11434'),
                model=model_name
            )
            
            caption = generator.generate_caption(str(file_path))
            return caption
        
    except Exception as e:
        print(f"Warning: Caption generation failed for {file_path}: {e}")
        return None


def scan_photos_directory(root_dir: Path) -> List[Path]:
    """Scan directory for photo files following YYYY/MM/DD structure"""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.tiff'}
    
    photo_files = []
    for ext in supported_extensions:
        photo_files.extend(root_dir.glob(f'**/*{ext}'))
        photo_files.extend(root_dir.glob(f'**/*{ext.upper()}'))
    
    return sorted(photo_files)


def ingest_photos(
    root_dir: Path,
    db: ImageDatabase,
    llm_config,
    generate_captions: bool = True,
    detect_bw: bool = True,
    batch_size: int = 50,
    dry_run: bool = False,
    caption_detail: str = "basic"
) -> int:
    """Ingest photos from directory into database
    
    Args:
        root_dir: Root directory containing photos
        db: Database connection
        llm_config: LLM configuration
        generate_captions: Whether to generate captions
        detect_bw: Whether to detect black & white photos
        batch_size: Batch size for inserts
        dry_run: If True, don't write to database
        caption_detail: Detail level for captions ('basic', 'detailed', 'very_detailed')
    """
    
    print(f"\nScanning for photos in: {root_dir}")
    photo_files = scan_photos_directory(root_dir)
    
    if not photo_files:
        print("No photos found!")
        return 0
    
    print(f"Found {len(photo_files)} photos to process")
    
    ingested_count = 0
    metadata_batch = []
    
    for i, photo_path in enumerate(photo_files, 1):
        if i % 10 == 0:
            print(f"Processing {i}/{len(photo_files)}...")
        
        # Extract EXIF metadata
        exif_meta = extract_exif_metadata(photo_path)
        
        # Calculate hashes
        sha256 = calculate_sha256(photo_path)
        
        # Check for duplicates
        if not dry_run and db.image_exists(sha256):
            print(f"  Skipping duplicate: {photo_path.name}")
            continue
        
        phash = calculate_perceptual_hash(photo_path)
        
        # Get date from EXIF or file path
        date_created = exif_meta['date_created']
        if not date_created:
            # Try to parse from path (YYYY/MM/DD structure)
            try:
                parts = photo_path.parent.parts
                year_idx = None
                for idx, part in enumerate(parts):
                    if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2100:
                        year_idx = idx
                        break
                
                if year_idx and len(parts) > year_idx + 2:
                    year = int(parts[year_idx])
                    month = int(parts[year_idx + 1])
                    day = int(parts[year_idx + 2])
                    date_created = datetime(year, month, day)
            except:
                pass
        
        # Fallback to file modification time
        if not date_created:
            date_created = datetime.fromtimestamp(photo_path.stat().st_mtime)
        
        # Detect black and white
        is_bw = False
        if detect_bw:
            print(f"  Detecting B&W: {photo_path.name}")
            is_bw = detect_black_and_white_llm(photo_path, llm_config)
        
        # Generate caption
        caption = None
        caption_embedding = None
        
        if generate_captions:
            print(f"  Generating caption ({caption_detail}): {photo_path.name}")
            caption = generate_caption_llm(photo_path, llm_config, caption_detail=caption_detail)
            
            if caption and not dry_run:
                # Generate embedding for the caption
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(llm_config.model_name if hasattr(llm_config, 'model_name') else 'all-MiniLM-L6-v2')
                    caption_embedding = model.encode([caption])[0].tolist()
                except Exception as e:
                    print(f"  Warning: Could not generate embedding: {e}")
        
        # Create metadata object
        metadata = ImageMetadata(
            file_path=str(photo_path.absolute()),
            file_name=photo_path.name,
            file_size=photo_path.stat().st_size,
            sha256=sha256,
            perceptual_hash=phash,
            width=exif_meta['width'],
            height=exif_meta['height'],
            format=exif_meta['format'],
            date_created=date_created,
            date_modified=datetime.fromtimestamp(photo_path.stat().st_mtime),
            gps_latitude=exif_meta['gps_latitude'],
            gps_longitude=exif_meta['gps_longitude'],
            is_black_and_white=is_bw,
            caption=caption,
            caption_embedding=caption_embedding,
            tags=None
        )
        
        metadata_batch.append(metadata)
        
        # Batch insert
        if len(metadata_batch) >= batch_size:
            if not dry_run:
                db.batch_insert_images(metadata_batch)
            ingested_count += len(metadata_batch)
            metadata_batch = []
    
    # Insert remaining
    if metadata_batch:
        if not dry_run:
            db.batch_insert_images(metadata_batch)
        ingested_count += len(metadata_batch)
    
    print(f"\nSuccessfully ingested {ingested_count} photos")
    return ingested_count


def main():
    parser = argparse.ArgumentParser(
        description="Ingest photos into archive database with AI captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest photos using default archive config
  %(prog)s --dir ~/Documents/photos1
  
  # Ingest into specific archive by ID
  %(prog)s --dir ~/Downloads/test3 --archive test_llama3
  
  # Dry run (no database writes)
  %(prog)s --dir ~/Documents/photos1 --dry-run
  
  # Skip caption generation (faster)
  %(prog)s --dir ~/Documents/photos1 --no-captions
  
  # Use Florence-2 with detailed captions
  %(prog)s --dir ~/Documents/photos1 --caption-detail detailed --llm-model microsoft/Florence-2-base
  
  # Use Florence-2 with very detailed captions
  %(prog)s --dir ~/Documents/photos1 --caption-detail very_detailed --llm-model microsoft/Florence-2-large
  
  # Use custom config file
  %(prog)s --dir ~/Documents/photos1 --config /path/to/config.yaml
        """
    )
    
    parser.add_argument('--dir', type=Path, required=True,
                       help='Root directory containing photos (YYYY/MM/DD structure)')
    parser.add_argument('--archive', type=str, default=None,
                       help='Archive ID to use (from config file)')
    parser.add_argument('--config', type=Path, default=None,
                       help='Path to archives config YAML file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Process but do not write to database')
    parser.add_argument('--no-captions', action='store_true',
                       help='Skip AI caption generation')
    parser.add_argument('--no-bw-detection', action='store_true',
                       help='Skip black & white detection')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for database inserts')
    parser.add_argument('--caption-detail', type=str, default='basic',
                       choices=['basic', 'detailed', 'very_detailed'],
                       help='Level of detail for AI captions (default: basic)')
    parser.add_argument('--llm-model', type=str, default=None,
                       help='Override LLM model from config (e.g., microsoft/Florence-2-base, llama3:8b)')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration with {len(config.archives)} archives")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Please create an archives_config.yaml file first")
        sys.exit(1)
    
    # Select archive
    if args.archive:
        archive = config.get_archive(args.archive)
        if not archive:
            print(f"Error: Archive '{args.archive}' not found")
            print(f"Available archives: {', '.join(a.id for a in config.archives)}")
            sys.exit(1)
    else:
        archive = config.get_default_archive()
        if not archive:
            print("Error: No default archive configured")
            sys.exit(1)
    
    print(f"\nUsing archive: {archive.name}")
    print(f"Database: {archive.db_connection}")
    print(f"Root directory: {archive.root_dir}")
    
    # Override LLM model if specified on command line, otherwise use archive config
    llm_config = config.llm
    
    # Use archive-specific LLM model if defined
    if archive.llm_model:
        print(f"Using archive-specific LLM model: {archive.llm_model}")
        llm_config.model = archive.llm_model
    
    # Command line override takes precedence
    if args.llm_model:
        print(f"Overriding LLM model to: {args.llm_model}")
        llm_config.model = args.llm_model
    
    # Determine caption detail level (archive config or command line)
    caption_detail = args.caption_detail if args.caption_detail != 'basic' else archive.caption_detail
    if caption_detail != args.caption_detail and archive.caption_detail:
        print(f"Using archive caption detail: {caption_detail}")
    
    # Verify source directory exists
    if not args.dir.exists():
        print(f"Error: Source directory not found: {args.dir}")
        sys.exit(1)
    
    # Initialize database
    if not args.dry_run:
        try:
            db = ImageDatabase(archive.db_connection, embedding_dimensions=config.embedding.dimensions)
            print("Connected to database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            sys.exit(1)
    else:
        db = None
        print("Dry run mode - no database connection")
    
    # Ingest photos
    count = ingest_photos(
        root_dir=args.dir,
        db=db,
        llm_config=llm_config,
        generate_captions=not args.no_captions,
        detect_bw=not args.no_bw_detection,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        caption_detail=args.caption_detail
    )
    
    if not args.dry_run:
        db.close()
    
    print(f"\nDone! Ingested {count} photos into '{archive.name}' archive")


if __name__ == '__main__':
    main()
