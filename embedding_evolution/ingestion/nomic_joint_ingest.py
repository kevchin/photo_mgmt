#!/usr/bin/env python3
"""
Ingest photos using Nomic Joint Vision-Text Embedding

This script processes photos and stores 768-dimensional embeddings directly
from images using nomic-embed-vision-v1.5, without generating captions.

Usage:
    # Process a directory of photos
    python nomic_joint_ingest.py --photos /path/to/photos
    
    # With options
    python nomic_joint_ingest.py \
        --photos /path/to/photos \
        --batch-size 4 \
        --limit 100 \
        --half-precision
    
    # Test mode (no database writes)
    python nomic_joint_ingest.py --photos /path/to/photos --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json

from pipeline.joint_embedder import NomicJointEmbedder
from config.database import DatabaseManager
from utils.exif_reader import read_image_metadata


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_images(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Find all image files in directory recursively."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif']
    
    images = []
    for ext in extensions:
        images.extend(directory.glob(f'**/*{ext}'))
        images.extend(directory.glob(f'**/*{ext.upper()}'))
    
    return sorted(images)


def ingest_photos(
    photos_dir: Path,
    db_url: str,
    batch_size: int = 8,
    limit: Optional[int] = None,
    dry_run: bool = False,
    half_precision: bool = True,
    skip_existing: bool = True
):
    """
    Ingest photos using Nomic joint embedding.
    
    Args:
        photos_dir: Directory containing photos (organized as YYYY/MM/DD/)
        db_url: PostgreSQL connection string
        batch_size: Number of images to process per batch
        limit: Maximum number of photos to process (None for all)
        dry_run: If True, don't write to database
        half_precision: Use FP16 for reduced VRAM
        skip_existing: Skip photos that already have embeddings
    """
    
    # Initialize components
    logger.info("Initializing Nomic Joint Embedder...")
    embedder = NomicJointEmbedder(
        device="cuda",
        half_precision=half_precision
    )
    
    logger.info(f"Connecting to database: {db_url.split('@')[-1]}")
    db = DatabaseManager(db_url)
    db.connect()
    
    # Ensure the embedding column exists
    model_name = "nomic-embed-vision-v1.5"
    dimension = 768
    column_name = f"embedding_{model_name.replace('-', '_').replace('.', '_')}"
    
    logger.info(f"Ensuring embedding column exists: {column_name}")
    if not dry_run:
        db.add_embedding_column(model_name, dimension)
    
    # Find images
    logger.info(f"Searching for images in: {photos_dir}")
    images = find_images(photos_dir)
    logger.info(f"Found {len(images)} images")
    
    # Apply limit
    if limit:
        images = images[:limit]
        logger.info(f"Limited to {len(images)} images")
    
    # Filter out already processed images
    if skip_existing and not dry_run:
        logger.info("Checking for existing embeddings...")
        filtered_images = []
        for img_path in images:
            # Check if this image already has an embedding
            existing = db.get_image_record(str(img_path))
            if existing and existing.get(column_name):
                logger.debug(f"Skipping {img_path}: already embedded")
                continue
            filtered_images.append(img_path)
        
        skipped_count = len(images) - len(filtered_images)
        images = filtered_images
        logger.info(f"Skipped {skipped_count} existing images, processing {len(images)} new images")
    
    if not images:
        logger.info("No new images to process")
        return
    
    # Process images in batches
    processed = 0
    errors = 0
    
    logger.info(f"Starting ingestion (batch_size={batch_size})...")
    start_time = datetime.now()
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(images) - 1) // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} images)")
        
        try:
            # Generate embeddings for batch
            embeddings = embedder.embed_batch(batch, batch_size=batch_size)
            
            # Process each image in batch
            for j, img_path in enumerate(batch):
                try:
                    # Read metadata
                    metadata = read_image_metadata(img_path)
                    
                    # Prepare record
                    record = {
                        'image_path': str(img_path),
                        'caption': metadata.get('description', ''),  # Use EXIF description if available
                        'date_taken': metadata.get('date_taken'),
                        'latitude': metadata.get('latitude'),
                        'longitude': metadata.get('longitude'),
                        'is_black_and_white': metadata.get('is_black_and_white', False),
                        'orientation': metadata.get('orientation', 1),
                        'file_size': img_path.stat().st_size if img_path.exists() else None,
                    }
                    
                    # Add embedding
                    record[column_name] = embeddings[j].tolist()
                    
                    if not dry_run:
                        # Store in database
                        db.upsert_image(record)
                    
                    processed += 1
                    
                    if processed % 10 == 0:
                        logger.info(f"Progress: {processed}/{len(images)} images processed")
                
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    errors += 1
                    continue
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            errors += len(batch)
            continue
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Total images found: {len(images) + (processed if skip_existing else 0)}")
    logger.info(f"  Images processed: {processed}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Time elapsed: {elapsed:.2f}s")
    logger.info(f"  Rate: {processed/elapsed:.2f} images/sec")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("NOTE: This was a dry run - no data was written to database")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest photos using Nomic Joint Vision-Text Embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all photos in a directory
  %(prog)s --photos /home/user/photos
  
  # Process with small batches (for low VRAM GPUs)
  %(prog)s --photos /home/user/photos --batch-size 2
  
  # Test on first 10 photos only
  %(prog)s --photos /home/user/photos --limit 10 --dry-run
  
  # Use full precision (more VRAM, potentially better quality)
  %(prog)s --photos /home/user/photos --no-half-precision
        """
    )
    
    parser.add_argument(
        '--photos', '-p',
        type=Path,
        required=True,
        help='Directory containing photos (searches recursively)'
    )
    
    parser.add_argument(
        '--database-url', '-d',
        type=str,
        default='postgresql://postgres:postgres@localhost:5432/photo_archive_evolution',
        help='PostgreSQL connection string'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Number of images to process per batch (default: 8)'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of photos to process (default: all)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Do not write to database (for testing)'
    )
    
    parser.add_argument(
        '--no-half-precision',
        action='store_true',
        help='Use full FP32 precision instead of FP16 (uses more VRAM)'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-process images that already have embeddings'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate photos directory
    if not args.photos.exists():
        logger.error(f"Photos directory does not exist: {args.photos}")
        sys.exit(1)
    
    if not args.photos.is_dir():
        logger.error(f"Photos path is not a directory: {args.photos}")
        sys.exit(1)
    
    # Run ingestion
    try:
        ingest_photos(
            photos_dir=args.photos,
            db_url=args.database_url,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
            half_precision=not args.no_half_precision,
            skip_existing=not args.no_skip_existing
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
