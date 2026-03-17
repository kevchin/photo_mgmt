#!/usr/bin/env python3
"""
Add Photos Utility

Adds one or more photos to the PostgreSQL database with:
- Metadata extraction (EXIF, GPS, dimensions)
- Black & white detection
- AI-generated caption (via OpenAI API or local Florence-2 model)

Supports:
- Single file: python add_single_photo.py /path/to/photo.jpg ...
- Directory:   python add_single_photo.py /path/to/photos/ ...

Usage:
    # Single photo without caption
    python add_single_photo.py /path/to/photo.jpg --db-url postgresql://user:pass@localhost/dbname
    
    # Single photo with OpenAI caption
    python add_single_photo.py /path/to/photo.jpg --db-url postgresql://user:pass@localhost/dbname --api-key $OPENAI_API_KEY
    
    # Directory with local captions (Florence-2)
    python add_single_photo.py /path/to/photos/ --db-url postgresql://user:pass@localhost/dbname --local-caption
    
    # Directory with detailed local captions
    python add_single_photo.py /path/to/photos/ --db-url postgresql://user:pass@localhost/dbname --local-caption --caption-task "<MORE_DETAILED_CAPTION>"
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from PIL import Image
    HAS_IMAGE_SUPPORT = True
except ImportError as e:
    print(f"Error: Pillow not installed. Run: pip install Pillow")
    HAS_IMAGE_SUPPORT = False
    sys.exit(1)

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))
from image_database import ImageDatabase, ImageMetadata
from detect_black_white import is_black_and_white

# Try to import local caption generator (lazy import to avoid hard dependency)
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
        except ImportError:
            LOCAL_CAPTION_AVAILABLE = False
    return LOCAL_CAPTION_AVAILABLE


def extract_metadata(file_path: Path) -> dict:
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
        print(f"Warning: Could not extract metadata: {e}")
    
    return metadata


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file"""
    import hashlib
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


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


def generate_caption(image_path: str, api_key: str, model: str = "gpt-4o") -> Optional[str]:
    """Generate a caption for the image using OpenAI API"""
    try:
        from openai import OpenAI
        import base64
        
        client = OpenAI(api_key=api_key)
        
        # Read and encode the image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = """Describe this image briefly in 1-2 sentences. Include main subjects, setting, and any notable elements."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: Failed to generate caption: {e}")
        return None


def generate_local_caption(caption_generator, 
                          image_path: str, 
                          task: str = "<DETAILED_CAPTION>") -> Optional[str]:
    """Generate a caption using local Florence-2 model"""
    if not LOCAL_CAPTION_AVAILABLE or caption_generator is None:
        return None
    
    try:
        caption = caption_generator.generate_caption(image_path, task=task)
        return caption
    except Exception as e:
        print(f"Warning: Failed to generate local caption: {e}")
        return None


def process_single_image(path: Path, db: ImageDatabase, 
                        caption_generator=None,
                        embedding_generator=None,
                        caption_task: str = "<DETAILED_CAPTION>",
                        use_openai: bool = False,
                        api_key: Optional[str] = None,
                        openai_model: str = "gpt-4o") -> Tuple[bool, Optional[int]]:
    """
    Process a single image file and add to database
    
    Returns:
        (success, image_id)
    """
    # Calculate checksum
    sha256 = calculate_sha256(path)
    
    # Check if already exists
    if db.image_exists(sha256):
        print(f"⚠️  Skipping (already exists): {path.name}")
        return False, None
    
    # Extract metadata
    file_meta = extract_metadata(path)
    
    # Get date information
    date_created = None
    year, month, day = None, None, None
    
    if file_meta.get('exif_date'):
        date_created = parse_date_string(file_meta['exif_date'])
        if date_created:
            year, month, day = date_created.year, date_created.month, date_created.day
    
    # Fallback to file modification time
    if not date_created:
        date_created = datetime.fromtimestamp(path.stat().st_mtime)
        year, month, day = date_created.year, date_created.month, date_created.day
    
    # Detect black and white
    is_bw = is_black_and_white(str(path))
    bw_status = "Black & White" if is_bw else "Color"
    
    # Generate caption
    caption = None
    if use_openai and api_key:
        caption = generate_caption(str(path), api_key, openai_model)
    elif caption_generator:
        caption = generate_local_caption(caption_generator, str(path), caption_task)
    
    # Create metadata object
    metadata = ImageMetadata(
        file_path=str(path.absolute()),
        file_name=path.name,
        file_size=path.stat().st_size,
        sha256=sha256,
        perceptual_hash="",
        width=file_meta['width'],
        height=file_meta['height'],
        format=file_meta['format'],
        date_created=date_created,
        date_modified=datetime.fromtimestamp(path.stat().st_mtime),
        gps_latitude=file_meta['gps_latitude'],
        gps_longitude=file_meta['gps_longitude'],
        is_black_and_white=is_bw,
        caption=caption,
        caption_embedding=None,
        tags=[]
    )
    
    # Insert into database
    try:
        image_id = db.insert_image(metadata)
        if image_id:
            status = "✅" if caption else "📷"
            print(f"{status} {path.name} - {bw_status}" + (f" - Captioned" if caption else ""))
            return True, image_id
        else:
            print(f"❌ Failed to insert: {path.name}")
            return False, None
    except Exception as e:
        print(f"❌ Error inserting {path.name}: {e}")
        return False, None


def collect_image_files(input_path: Path) -> List[Path]:
    """Collect all image files from a path (file or directory)"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.gif', '.bmp', '.tiff', '.webp'}
    
    if input_path.is_file():
        if input_path.suffix.lower() in valid_extensions:
            return [input_path]
        else:
            print(f"Warning: Unknown file extension {input_path.suffix}, but will attempt to process")
            return [input_path]
    
    elif input_path.is_dir():
        image_files = []
        for ext in sorted(valid_extensions):
            # Case-insensitive search
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        # Remove duplicates and sort
        image_files = sorted(set(image_files))
        return image_files
    
    return []


def add_photos(input_path: str, db_url: str, 
               api_key: Optional[str] = None,
               openai_model: str = "gpt-4o",
               skip_caption: bool = False,
               use_local_caption: bool = False,
               caption_task: str = "<DETAILED_CAPTION>",
               local_model: str = "microsoft/Florence-2-base",
               max_workers: int = 1) -> Tuple[int, int, int]:
    """
    Add one or more photos to the database
    
    Args:
        input_path: Path to a single image file or directory of images
        db_url: PostgreSQL connection URL
        api_key: OpenAI API key (optional, for caption generation)
        openai_model: Model to use for OpenAI caption generation
        skip_caption: Skip caption generation
        use_local_caption: Use local Florence-2 model instead of OpenAI
        caption_task: Florence-2 task prompt
        local_model: Local model name/path for Florence-2
        max_workers: Maximum parallel workers for processing
    
    Returns:
        (total_processed, successful, failed)
    """
    path = Path(input_path)
    
    # Validate path exists
    if not path.exists():
        print(f"Error: Path not found: {input_path}")
        return 0, 0, 0
    
    # Collect image files
    image_files = collect_image_files(path)
    
    if not image_files:
        print(f"No image files found in: {input_path}")
        return 0, 0, 0
    
    total_files = len(image_files)
    print(f"\n📁 Found {total_files} image file(s) to process")
    print(f"   Source: {path.absolute()}")
    
    # Initialize database
    try:
        db = ImageDatabase(db_url, embedding_dimensions=1536)
    except Exception as e:
        print(f"Error: Could not connect to database: {e}")
        return 0, 0, 0
    
    # Initialize local caption generator if requested
    caption_generator = None
    embedding_generator = None
    
    if use_local_caption and not skip_caption:
        if not LOCAL_CAPTION_AVAILABLE:
            print("⚠️  Local captioning not available. Install with: pip install transformers torch")
            print("   Falling back to no captions")
            use_local_caption = False
        else:
            print(f"\n🤖 Loading local caption model: {local_model}")
            try:
                caption_generator = FlorenceCaptionGenerator(model_name=local_model)
                print("   Model loaded successfully\n")
            except Exception as e:
                print(f"⚠️  Failed to load local model: {e}")
                print("   Falling back to no captions")
                use_local_caption = False
    
    # Process images
    successful = 0
    failed = 0
    skipped = 0
    
    if max_workers == 1 or total_files == 1:
        # Sequential processing
        for img_path in tqdm(image_files, desc="Processing"):
            result, image_id = process_single_image(
                img_path, db,
                caption_generator=caption_generator,
                embedding_generator=embedding_generator,
                caption_task=caption_task,
                use_openai=(api_key and not use_local_caption),
                api_key=api_key,
                openai_model=openai_model
            )
            
            if result:
                successful += 1
            else:
                # Check if skipped (already exists)
                sha256 = calculate_sha256(img_path)
                if db.image_exists(sha256):
                    skipped += 1
                else:
                    failed += 1
    else:
        # Parallel processing
        def process_wrapper(img_path):
            return process_single_image(
                img_path, db,
                caption_generator=caption_generator,
                embedding_generator=embedding_generator,
                caption_task=caption_task,
                use_openai=(api_key and not use_local_caption),
                api_key=api_key,
                openai_model=openai_model
            )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_wrapper, img_path): img_path 
                      for img_path in image_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                img_path = futures[future]
                try:
                    result, image_id = future.result()
                    if result:
                        successful += 1
                    else:
                        sha256 = calculate_sha256(img_path)
                        if db.image_exists(sha256):
                            skipped += 1
                        else:
                            failed += 1
                except Exception as e:
                    print(f"❌ Error processing {img_path.name}: {e}")
                    failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Processing Summary")
    print(f"{'='*60}")
    print(f"   Total files found:     {total_files}")
    print(f"   Successfully added:    {successful}")
    print(f"   Skipped (duplicate):   {skipped}")
    print(f"   Failed:                {failed}")
    print(f"{'='*60}\n")
    
    # Show example queries
    if successful > 0:
        print("💡 Example SQL queries:")
        print(f"   SELECT * FROM images WHERE is_black_and_white = TRUE;")
        print(f"   SELECT * FROM images WHERE caption IS NOT NULL;")
        print(f"   SELECT COUNT(*) FROM images WHERE file_name LIKE '%{image_files[0].suffix}';\n")
    
    return total_files, successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Add photos to PostgreSQL database with metadata and optional captions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a single photo without caption
  python add_single_photo.py /path/to/photo.jpg --db-url postgresql://user:pass@localhost/dbname
  
  # Add a single photo with OpenAI caption
  python add_single_photo.py /path/to/photo.jpg --db-url postgresql://user:pass@localhost/dbname --api-key $OPENAI_API_KEY
  
  # Add all photos from a directory with local captions
  python add_single_photo.py /path/to/photos/ --db-url postgresql://user:pass@localhost/dbname --local-caption
  
  # Add photos with detailed captions using specific Florence-2 task
  python add_single_photo.py /path/to/photos/ --db-url postgresql://user:pass@localhost/dbname --local-caption --caption-task "<MORE_DETAILED_CAPTION>"
  
  # Add photos in parallel (faster for large batches)
  python add_single_photo.py /path/to/photos/ --db-url postgresql://user:pass@localhost/dbname --local-caption --workers 4
        """
    )
    
    parser.add_argument('input_path', help='Path to image file or directory of images')
    parser.add_argument('--db-url', required=True, help='PostgreSQL connection URL')
    parser.add_argument('--api-key', help='OpenAI API key for caption generation')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model for caption (default: gpt-4o)')
    parser.add_argument('--skip-caption', action='store_true', help='Skip caption generation')
    parser.add_argument('--local-caption', action='store_true', help='Use local Florence-2 model for captions')
    parser.add_argument('--caption-task', default='<DETAILED_CAPTION>', 
                       help='Florence-2 task prompt (default: <DETAILED_CAPTION>)')
    parser.add_argument('--local-model', default='microsoft/Florence-2-base',
                       help='Local model name/path (default: microsoft/Florence-2-base)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.api_key and args.local_caption:
        print("⚠️  Warning: Both --api-key and --local-caption specified. Using local captioning.")
    
    if args.workers < 1:
        print("Error: --workers must be at least 1")
        sys.exit(1)
    
    # Process photos
    total, successful, failed = add_photos(
        args.input_path,
        args.db_url,
        api_key=args.api_key,
        openai_model=args.model,
        skip_caption=args.skip_caption,
        use_local_caption=args.local_caption,
        caption_task=args.caption_task,
        local_model=args.local_model,
        max_workers=args.workers
    )
    
    # Exit with appropriate code
    if failed > 0:
        sys.exit(1)
    elif successful == 0 and total > 0:
        sys.exit(2)  # All skipped or failed
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
