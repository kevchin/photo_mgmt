#!/usr/bin/env python3
"""
Local Caption Generator Utility

Generates captions for images using local models like Florence-2.
No external API calls required - runs completely offline.

Supports:
- Microsoft Florence-2 (base and large)
- Output to CSV file or direct PostgreSQL database update
- Batch processing with progress tracking
- Multiple embedding model options (sentence-transformers, etc.)
- Resume capability for interrupted runs

Requirements:
    pip install transformers torch pillow sentence-transformers psycopg2-binary
"""

import os
import sys
import json
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib

# Check for required packages
try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
except ImportError:
    print("Error: Pillow not installed. Run: pip install pillow")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Florence-2 support disabled.")
    print("Run: pip install transformers torch torchvision")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Embedding generation disabled.")
    print("Run: pip install sentence-transformers")

# Import our database module
try:
    from image_database import ImageDatabase, ImageMetadata
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: image_database module not found. Database operations disabled.")


@dataclass
class ImageRecord:
    """Represents an image record for processing"""
    file_path: str
    file_name: str
    sha256: Optional[str] = None
    caption: Optional[str] = None
    caption_embedding: Optional[List[float]] = field(default_factory=list)
    processed: bool = False
    error: Optional[str] = None


class FlorenceCaptionGenerator:
    """Generate captions using Microsoft Florence-2 model locally"""
    
    SUPPORTED_TASKS = [
        "<CAPTION>",           # Basic caption
        "<DETAILED_CAPTION>",  # Detailed description
        "<MORE_DETAILED_CAPTION>",  # Even more detailed
        "<OD>",                # Object detection
        "<DENSE_REGION_CAPTION>",  # Dense region captions
        "<REGION_PROPOSAL>",   # Region proposals
        "<CAPTION_TO_PHRASE_GROUNDING>",  # Caption to phrase grounding
        "<REFERRING_EXPRESSION_SEGMENTATION>",  # Referring expression segmentation
        "<REGION_TO_SEGMENTATION>",  # Region to segmentation
        "<OPEN_VOCABULARY_DETECTION>",  # Open vocabulary detection
        "<REGION_TO_CATEGORY>",  # Region to category
        "<REGION_TO_DESCRIPTION>",  # Region to description
        "<OCR>",               # OCR text extraction
        "<OCR_WITH_REGION>",   # OCR with region information
    ]
    
    def __init__(self, model_name: str = "microsoft/Florence-2-base", 
                 device: str = "auto", trust_remote_code: bool = True):
        """
        Initialize Florence-2 caption generator
        
        Args:
            model_name: HuggingFace model name or local path
                       Options: 
                       - microsoft/Florence-2-base (faster, less detailed)
                       - microsoft/Florence-2-large (slower, more detailed)
                       - Local path to downloaded model
            device: Device to run on ('cuda', 'cpu', 'auto')
            trust_remote_code: Whether to trust remote code (required for Florence-2)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers package not installed")
        
        self.model_name = model_name
        self.device = device
        
        print(f"Loading Florence-2 model: {model_name}...")
        print("(This may take a few minutes on first run)")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=trust_remote_code
        )
        
        # Auto-detect device if requested
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                print("Using Apple MPS GPU")
            else:
                self.device = "cpu"
                print("Using CPU")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
    
    def generate_caption(self, image_path: str, 
                        task: str = "<DETAILED_CAPTION>",
                        max_tokens: int = 256) -> Optional[str]:
        """
        Generate a caption for an image
        
        Args:
            image_path: Path to the image file
            task: Florence-2 task prompt (default: detailed caption)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated caption text, or None on error
        """
        try:
            # Open and prepare image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare input
            prompt = task
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device, torch.float16 if self.device != "cpu" else torch.float32)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    num_beams=1
                )
            
            # Decode result
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Post-process: extract just the caption
            caption = self.processor.post_process_generation(
                generated_text, 
                task=prompt, 
                image_size=(image.width, image.height)
            )
            
            # Extract the actual caption text from the result
            if isinstance(caption, dict):
                # Florence-2 returns a dict with the task key
                caption = list(caption.values())[0]
            
            return str(caption).strip()
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return None
    
    def generate_basic_caption(self, image_path: str) -> Optional[str]:
        """Generate a basic (shorter) caption"""
        return self.generate_caption(image_path, "<CAPTION>")
    
    def generate_detailed_caption(self, image_path: str) -> Optional[str]:
        """Generate a detailed caption"""
        return self.generate_caption(image_path, "<DETAILED_CAPTION>")
    
    def generate_very_detailed_caption(self, image_path: str) -> Optional[str]:
        """Generate a very detailed caption"""
        return self.generate_caption(image_path, "<MORE_DETAILED_CAPTION>")
    
    def extract_ocr(self, image_path: str) -> Optional[str]:
        """Extract text from image using OCR"""
        return self.generate_caption(image_path, "<OCR>")


class EmbeddingGenerator:
    """Generate embeddings using local sentence-transformers models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 device: str = "auto",
                 dimensions: Optional[int] = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Sentence transformer model name or path
                       Popular options:
                       - all-MiniLM-L6-v2 (fast, 384 dims)
                       - all-mpnet-base-v2 (better quality, 768 dims)
                       - bge-small-en-v1.5 (good balance)
                       - bge-large-en-v1.5 (best quality, slower)
            device: Device to run on ('cuda', 'cpu', 'auto')
            dimensions: Force specific dimensions (optional)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers package not installed")
        
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.model = SentenceTransformer(model_name, device=device)
        self.dimensions = dimensions or self.model.get_sentence_embedding_dimension()
        
        print(f"Embedding model loaded ({self.dimensions} dimensions)")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []


def convert_gps_coordinate(value, ref):
    """Convert GPS EXIF data to decimal degrees"""
    if not value or len(value) < 3:
        return None
    try:
        # GPS coordinates are stored as degrees, minutes, seconds
        def to_float(v):
            if isinstance(v, (int, float)):
                return float(v)
            elif hasattr(v, 'num') and hasattr(v, 'den'):
                return float(v.num) / float(v.den)
            else:
                return float(v)
        
        degrees = to_float(value[0])
        minutes = to_float(value[1])
        seconds = to_float(value[2])
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except (Exception, AttributeError):
        return None


def extract_image_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from an image file including GPS coordinates
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary with metadata including GPS coordinates
    """
    metadata = {
        'gps_latitude': None,
        'gps_longitude': None,
        'date_created': None,
        'width': None,
        'height': None,
        'format': None,
        'camera_make': None,
        'camera_model': None
    }
    
    try:
        with Image.open(file_path) as img:
            # Basic info
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['format'] = img.format
            
            # EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                
                # Create reverse mapping of tag IDs to names
                exif_data = {}
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
                
                # Extract camera info
                metadata['camera_make'] = exif_data.get('Make')
                metadata['camera_model'] = exif_data.get('Model')
                
                # Extract date
                date_str = exif_data.get('DateTimeOriginal') or exif_data.get('DateTime')
                if date_str:
                    try:
                        # EXIF dates are in format "YYYY:MM:DD HH:MM:SS"
                        metadata['date_created'] = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S').isoformat()
                    except ValueError:
                        pass
                
                # Extract GPS information
                gps_info = exif_data.get('GPSInfo')
                if gps_info:
                    # GPS tags: 1=lat_ref, 2=latitude, 3=lon_ref, 4=longitude
                    lat_ref = gps_info.get(1)  # 'N' or 'S'
                    lat = gps_info.get(2)      # tuple of (deg, min, sec)
                    lon_ref = gps_info.get(3)  # 'E' or 'W'
                    lon = gps_info.get(4)      # tuple of (deg, min, sec)
                    
                    if lat and lat_ref:
                        metadata['gps_latitude'] = convert_gps_coordinate(lat, lat_ref)
                    if lon and lon_ref:
                        metadata['gps_longitude'] = convert_gps_coordinate(lon, lon_ref)
                        
    except Exception as e:
        print(f"Warning: Could not extract metadata from {file_path}: {e}")
    
    return metadata


def load_images_from_db(db: ImageDatabase, 
                        skip_captions: bool = True) -> List[Dict]:
    """Load images from database that need captions"""
    from psycopg2.extras import RealDictCursor
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if skip_captions:
                cur.execute("""
                    SELECT * FROM images 
                    WHERE caption IS NULL OR caption = ''
                    ORDER BY id
                """)
            else:
                cur.execute("SELECT * FROM images ORDER BY id")
            
            return [dict(row) for row in cur.fetchall()]


def process_image_local(img_record: Dict,
                        caption_gen: Optional[FlorenceCaptionGenerator],
                        embed_gen: Optional[EmbeddingGenerator],
                        task: str = "<DETAILED_CAPTION>",
                        regenerate: bool = False) -> Tuple[bool, Optional[str], Optional[List[float]], bool]:
    """
    Process a single image locally
    
    Returns:
        (success, caption, embedding, was_skipped)
        - was_skipped=True means image already had caption and was skipped (still valid)
        - was_skipped=False means image was newly processed
    """
    file_path = img_record['file_path']
    
    # Check if already processed
    existing_caption = img_record.get('caption')
    if existing_caption and not regenerate:
        # Might still need embedding
        if img_record.get('caption_embedding') is None and embed_gen:
            embedding = embed_gen.generate_embedding(existing_caption)
            if embedding:
                # Return with was_skipped=True to indicate it's already done but valid
                return True, existing_caption, embedding, True
        # Already has caption and embedding, skip
        return True, existing_caption, None, True
    
    # Generate caption
    caption = None
    if caption_gen:
        print(f"  Generating caption for: {file_path}")
        caption = caption_gen.generate_caption(file_path, task=task)
        if not caption:
            return False, None, None, False
        print(f"    Caption: {caption[:80]}...")
    
    # Generate embedding
    embedding = None
    if embed_gen and caption:
        embedding = embed_gen.generate_embedding(caption)
        if not embedding:
            print(f"    Warning: Failed to generate embedding")
    
    return True, caption, embedding, False


def save_to_csv(results: List[Dict], output_file: str):
    """Save results to CSV file"""
    if not results:
        print("No results to save")
        return
    
    fieldnames = ['file_path', 'file_name', 'sha256', 'caption', 'embedding', 
                  'processed_at', 'processed', 'error',
                  'gps_latitude', 'gps_longitude', 'date_created',
                  'width', 'height', 'format', 'camera_make', 'camera_model']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = result.copy()
            # Convert embedding list to JSON string
            if row.get('embedding'):
                row['embedding'] = json.dumps(row['embedding'])
            writer.writerow(row)
    
    print(f"Results saved to: {output_file}")


def update_database_for_image(db: ImageDatabase, result: Dict) -> Tuple[bool, Optional[int]]:
    """Update database with caption and embedding for a single image
    
    Args:
        db: ImageDatabase instance
        result: Dictionary with image processing results
        
    Returns:
        (success, image_id) - success boolean and the database ID if successful
    """
    # Check for processing errors
    if result.get('error'):
        print(f"  Skipping {result['file_name']}: processing failed - {result.get('error')}")
        return False, None
    
    if not result.get('processed'):
        print(f"  Skipping {result['file_name']}: not processed")
        return False, None
    
    if not result.get('caption'):
        print(f"  Skipping {result['file_name']}: no caption generated")
        return False, None
    
    try:
        # Get file modification time for date_modified
        from datetime import datetime as dt
        try:
            date_modified = dt.fromtimestamp(os.path.getmtime(result['file_path']))
        except:
            date_modified = dt.now()
        
        # Calculate SHA256 if not provided
        sha256 = result.get('sha256')
        if not sha256:
            sha256 = hashlib.sha256(open(result['file_path'], 'rb').read()).hexdigest()
        
        # Calculate perceptual hash
        phash = ""
        try:
            from PIL import Image as PILImage
            import imagehash
            with PILImage.open(result['file_path']) as img:
                phash = str(imagehash.phash(img))
        except:
            pass
        
        # Get file size
        try:
            file_size = os.path.getsize(result['file_path'])
        except:
            file_size = 0
        
        # Parse date_created string to datetime if needed
        date_created = result.get('date_created')
        if isinstance(date_created, str):
            try:
                date_created = dt.fromisoformat(date_created)
            except:
                date_created = None
        
        # Create metadata object
        metadata = ImageMetadata(
            file_path=result['file_path'],
            file_name=result['file_name'],
            file_size=file_size,
            sha256=sha256,
            perceptual_hash=phash,
            caption=result.get('caption'),
            caption_embedding=result.get('embedding'),
            date_created=date_created,
            date_modified=date_modified,
            width=result.get('width', 0) or 0,
            height=result.get('height', 0) or 0,
            format=result.get('format', 'UNKNOWN') or 'UNKNOWN',
            gps_latitude=result.get('gps_latitude'),
            gps_longitude=result.get('gps_longitude'),
            tags=None
        )
        
        img_id = db.insert_image(metadata)
        
        # Print confirmation
        if result.get('was_skipped'):
            print(f"  ✓ Updated existing record: {result['file_name']} (ID: {img_id})")
        else:
            print(f"  ✓ Saved new caption to database: {result['file_name']} (ID: {img_id})")
            print(f"    Caption: {result.get('caption')[:80]}...")
        
        return True, img_id
        
    except Exception as e:
        print(f"Error updating database for {result['file_name']}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def update_database(db: ImageDatabase, results: List[Dict]):
    """Update database with captions and embeddings (batch mode)"""
    success_count = 0
    skipped_already_done = 0
    skipped_errors = 0
    
    for i, result in enumerate(results):
        # Check for processing errors
        if result.get('error'):
            print(f"  Skipping {result['file_name']}: processing failed - {result.get('error')}")
            skipped_errors += 1
            continue
        
        if not result.get('processed'):
            print(f"  Skipping {result['file_name']}: not processed")
            skipped_errors += 1
            continue
        
        if not result.get('caption'):
            print(f"  Skipping {result['file_name']}: no caption generated")
            skipped_errors += 1
            continue
        
        # Check if this was already done (had existing caption)
        if result.get('was_skipped'):
            # Still update the database to ensure embedding is saved
            print(f"  Updating existing record for {result['file_name']}...")
        
        success, img_id = update_database_for_image(db, result)
        if success:
            success_count += 1
        else:
            skipped_errors += 1
    
    print(f"\nDatabase update complete:")
    print(f"  Successfully updated: {success_count} records")
    print(f"  Skipped due to errors: {skipped_errors} records")
    print(f"  Already processed (re-saved): {skipped_already_done} records")


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions and embeddings for images using local models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate captions with Florence-2-base and save to CSV
  %(prog)s --source ./photos --output captions.csv --model microsoft/Florence-2-base
  
  # Generate detailed captions and embeddings, save to PostgreSQL
  %(prog)s --db "postgresql://user:pass@localhost/image_archive" \\
      --model microsoft/Florence-2-large \\
      --embedding-model all-MiniLM-L6-v2 \\
      --task "<DETAILED_CAPTION>"
  
  # Only generate embeddings for existing captions
  %(prog)s --db "postgresql://user:pass@localhost/image_archive" \\
      --embeddings-only
  
  # Use CPU only (slower but works without GPU)
  %(prog)s --source ./photos --output captions.csv --device cpu
  
  # Extract OCR text instead of captions
  %(prog)s --source ./photos --output ocr.csv --task "<OCR>"
  
  # Process images from database that lack captions
  %(prog)s --db "postgresql://user:pass@localhost/image_archive" \\
      --from-db --model microsoft/Florence-2-base
        """
    )
    
    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--source', type=str, 
                             help='Directory containing images to process')
    source_group.add_argument('--from-db', action='store_true',
                             help='Process images from PostgreSQL database')
    
    # Output destination
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--output', type=str, 
                             help='Output CSV file path')
    output_group.add_argument('--db', type=str,
                             help='PostgreSQL connection string (for direct update)')
    
    # Model configuration
    parser.add_argument('--model', type=str, 
                       default='microsoft/Florence-2-base',
                       help='Florence-2 model name or path (default: microsoft/Florence-2-base)')
    parser.add_argument('--embedding-model', type=str,
                       default='all-MiniLM-L6-v2',
                       help='Sentence transformer model for embeddings')
    parser.add_argument('--task', type=str, 
                       default='<DETAILED_CAPTION>',
                       choices=['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>', 
                               '<OCR>', '<OCR_WITH_REGION>'],
                       help='Florence-2 task to perform')
    
    # Device and performance
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='Device to run models on (default: auto)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1, use 1 for GPU)')
    
    # Processing options
    parser.add_argument('--regenerate', action='store_true',
                       help='Regenerate captions even if they exist')
    parser.add_argument('--embeddings-only', action='store_true',
                       help='Only generate embeddings for existing captions')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of images to process (0 = all)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip images that already have captions (when using --from-db)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.from_db and not args.db:
        print("Error: --from-db requires --db connection string")
        sys.exit(1)
    
    if args.output and args.db:
        print("Error: Cannot specify both --output and --db")
        sys.exit(1)
    
    # Initialize generators
    caption_gen = None
    embed_gen = None
    
    if not args.embeddings_only:
        print(f"\nInitializing Florence-2 caption generator...")
        try:
            caption_gen = FlorenceCaptionGenerator(
                model_name=args.model,
                device=args.device
            )
        except Exception as e:
            print(f"Failed to initialize caption generator: {e}")
            sys.exit(1)
    
    print(f"\nInitializing embedding generator...")
    try:
        embed_gen = EmbeddingGenerator(
            model_name=args.embedding_model,
            device=args.device
        )
    except Exception as e:
        print(f"Warning: Failed to initialize embedding generator: {e}")
        print("Continuing without embedding generation")
        embed_gen = None
    
    # Get images to process
    images_to_process = []
    
    if args.source:
        # Process from directory
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Error: Source directory does not exist: {args.source}")
            sys.exit(1)
        
        print(f"\nScanning directory: {args.source}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        
        for ext in image_extensions:
            images_to_process.extend(source_path.rglob(f'*{ext}'))
            images_to_process.extend(source_path.rglob(f'*{ext.upper()}'))
        
        # Also check for HEIC files specifically
        images_to_process.extend(source_path.rglob('*.HEIC'))
        
        images_to_process = list(set(images_to_process))  # Remove duplicates
        
        if args.limit > 0:
            images_to_process = images_to_process[:args.limit]
        
        # Convert to dict format and extract metadata including GPS
        images_to_process_dicts = []
        print("Extracting metadata (including GPS coordinates) from images...")
        for i, img in enumerate(images_to_process):
            if i % 10 == 0:
                print(f"  Processing {i}/{len(images_to_process)} images...")
            file_path = str(img)
            metadata = extract_image_metadata(file_path)
            images_to_process_dicts.append({
                'file_path': file_path,
                'file_name': img.name,
                'sha256': None,  # Will be computed if needed
                'caption': None,
                'caption_embedding': None,
                'gps_latitude': metadata.get('gps_latitude'),
                'gps_longitude': metadata.get('gps_longitude'),
                'date_created': metadata.get('date_created'),
                'width': metadata.get('width'),
                'height': metadata.get('height'),
                'format': metadata.get('format'),
                'camera_make': metadata.get('camera_make'),
                'camera_model': metadata.get('camera_model')
            })
        images_to_process = images_to_process_dicts
        
        # Count images with GPS data
        gps_count = sum(1 for img in images_to_process if img.get('gps_latitude') and img.get('gps_longitude'))
        print(f"Found {len(images_to_process)} images to process ({gps_count} with GPS data)\n")
    
    elif args.from_db:
        # Process from database
        if not DB_AVAILABLE:
            print("Error: Database module not available")
            sys.exit(1)
        
        print(f"\nConnecting to database...")
        # Get embedding dimensions from the embedding generator if available
        emb_dims = embed_gen.dimensions if embed_gen else 1536
        db = ImageDatabase(args.db, embedding_dimensions=emb_dims)
        
        images_to_process = load_images_from_db(db, skip_captions=args.skip_existing)
        
        if args.limit > 0:
            images_to_process = images_to_process[:args.limit]
        
        print(f"Found {len(images_to_process)} images to process from database\n")
    
    if not images_to_process:
        print("No images to process!")
        sys.exit(0)
    
    # Process images
    print(f"Processing {len(images_to_process)} images...")
    print(f"Task: {args.task}")
    print(f"Device: {caption_gen.device if caption_gen else args.device}\n")
    
    results = []
    success_count = 0
    error_count = 0
    db_success_count = 0
    db_error_count = 0
    
    # Initialize database connection if needed (for per-image updates)
    db = None
    if args.db and DB_AVAILABLE:
        print("Connecting to database for per-image updates...")
        emb_dims = embed_gen.dimensions if embed_gen else 1536
        db = ImageDatabase(args.db, embedding_dimensions=emb_dims)
    
    # For GPU, use single worker to avoid OOM issues
    workers = 1 if args.device in ['cuda', 'mps'] else args.workers
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_image_local,
                img,
                caption_gen,
                embed_gen,
                args.task,
                args.regenerate
            ): img for img in images_to_process
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            img = futures[future]
            try:
                success, caption, embedding, was_skipped = future.result()
                
                result = {
                    'file_path': img['file_path'],
                    'file_name': img['file_name'],
                    'sha256': img.get('sha256'),
                    'caption': caption,
                    'embedding': embedding,
                    'processed_at': datetime.now().isoformat(),
                    'processed': success,
                    'was_skipped': was_skipped,  # True if already had caption
                    'error': None if success else 'Failed to generate caption',
                    # Include GPS and other metadata
                    'gps_latitude': img.get('gps_latitude'),
                    'gps_longitude': img.get('gps_longitude'),
                    'date_created': img.get('date_created'),
                    'width': img.get('width'),
                    'height': img.get('height'),
                    'format': img.get('format'),
                    'camera_make': img.get('camera_make'),
                    'camera_model': img.get('camera_model')
                }
                
                results.append(result)
                
                if success:
                    success_count += 1
                    
                    # Write to database immediately after successful processing
                    if db is not None:
                        db_success, img_id = update_database_for_image(db, result)
                        if db_success:
                            db_success_count += 1
                        else:
                            db_error_count += 1
                else:
                    error_count += 1
                    print(f"  ✗ Failed to process {img['file_name']}")
                    
            except Exception as e:
                print(f"Error processing {img['file_path']}: {e}")
                results.append({
                    'file_path': img['file_path'],
                    'file_name': img['file_name'],
                    'caption': None,
                    'embedding': None,
                    'processed': False,
                    'error': str(e)
                })
                error_count += 1
            
            # Progress indicator
            if i % 5 == 0 or i == len(images_to_process):
                progress_msg = f"Progress: {i}/{len(images_to_process)} " \
                              f"(success: {success_count}, errors: {error_count})"
                if db is not None:
                    progress_msg += f" [DB writes: {db_success_count} ok, {db_error_count} failed]"
                print(progress_msg)
    
    # Close database connection if we opened it
    if db is not None:
        db.close()
    
    # Save results
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {success_count}")
    print(f"Errors/Failed: {error_count}")
    
    if args.output:
        save_to_csv(results, args.output)
    elif args.db and DB_AVAILABLE:
        # Database writes already happened per-image during processing
        # Just show summary stats
        print(f"\nDatabase updates completed during processing:")
        print(f"  Successful writes: {db_success_count}")
        print(f"  Failed writes: {db_error_count}")
        
        # Show stats
        stats = db.get_statistics()
        print(f"\nDatabase statistics:")
        print(f"Total images: {stats['total_images']}")
        print(f"With captions: {stats['with_captions']}")
        print(f"With embeddings: {stats['with_embeddings']}")
    else:
        print("\nNo output destination specified. Results not saved.")
        print("Use --output <file.csv> or --db <connection_string> to save results")
    
    # Print sample results
    if results:
        print("\n=== Sample Results ===")
        for i, result in enumerate(results[:3]):
            print(f"\n{i+1}. {result['file_name']}")
            if result['caption']:
                print(f"   Caption: {result['caption'][:100]}...")
            if result.get('caption_embedding'):
                dims = len(result['caption_embedding'])
                print(f"   Embedding: {dims} dimensions")


if __name__ == '__main__':
    main()
