#!/usr/bin/env python3
"""
Database Migration Script for Black and White Image Detection

This script adds an 'is_black_and_white' column to the images table
and provides functionality to detect and update black and white status.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_database import ImageDatabase
from detect_black_white import is_black_and_white_fast, get_image_color_type


def add_black_white_column(db: ImageDatabase) -> bool:
    """
    Add is_black_and_white column to the images table if it doesn't exist.
    
    Args:
        db: ImageDatabase instance
    
    Returns:
        True if column was added or already exists, False on error
    """
    print("Adding is_black_and_white column to images table...")
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Check if column already exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'images' AND column_name = 'is_black_and_white'
                )
            """)
            
            column_exists = cur.fetchone()[0]
            
            if column_exists:
                print("Column 'is_black_and_white' already exists")
                return True
            
            # Add the column
            cur.execute("""
                ALTER TABLE images 
                ADD COLUMN is_black_and_white BOOLEAN DEFAULT NULL
            """)
            
            # Create index for faster queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_images_is_black_and_white 
                ON images(is_black_and_white)
                WHERE is_black_and_white IS NOT NULL
            """)
            
            conn.commit()
            print("Successfully added 'is_black_and_white' column")
            return True


def detect_and_update_single_image(db: ImageDatabase, file_path: str, 
                                    tolerance: int = 3) -> Optional[bool]:
    """
    Detect if a single image is black and white and update the database.
    
    Args:
        db: ImageDatabase instance
        file_path: Path to the image file
        tolerance: Tolerance for RGB channel differences
    
    Returns:
        True if grayscale, False if color, None if error or not found
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    # Detect if black and white
    is_bw = is_black_and_white_fast(file_path, tolerance=tolerance)
    
    # Update database
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE images 
                SET is_black_and_white = %s, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = %s
            """, (is_bw, file_path))
            
            if cur.rowcount == 0:
                print(f"Warning: Image not found in database: {file_path}")
                return None
            
            conn.commit()
    
    return is_bw


def batch_detect_and_update(db: ImageDatabase, tolerance: int = 3, 
                            batch_size: int = 100, verbose: bool = True) -> dict:
    """
    Detect and update black and white status for all images in the database.
    
    Args:
        db: ImageDatabase instance
        tolerance: Tolerance for RGB channel differences
        batch_size: Number of images to process in each batch
        verbose: Print progress information
    
    Returns:
        Dictionary with statistics:
        {
            'processed': number of images processed,
            'grayscale': number of grayscale images,
            'color': number of color images,
            'errors': number of errors,
            'not_found': number of images where file was not found
        }
    """
    stats = {
        'processed': 0,
        'grayscale': 0,
        'color': 0,
        'errors': 0,
        'not_found': 0
    }
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Get all images that haven't been processed yet
            cur.execute("""
                SELECT id, file_path FROM images 
                WHERE is_black_and_white IS NULL
                ORDER BY id
            """)
            
            images_to_process = cur.fetchall()
            
            if not images_to_process:
                print("All images have already been processed")
                # Return counts of existing classifications
                cur.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE is_black_and_white = TRUE) as grayscale,
                        COUNT(*) FILTER (WHERE is_black_and_white = FALSE) as color
                    FROM images
                """)
                result = cur.fetchone()
                stats['grayscale'] = result[0] or 0
                stats['color'] = result[1] or 0
                stats['processed'] = stats['grayscale'] + stats['color']
                return stats
    
    total = len(images_to_process)
    print(f"Found {total} images to process")
    
    # Process in batches
    batch = []
    batch_count = 0
    
    for img_id, file_path in images_to_process:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                if verbose:
                    print(f"File not found: {file_path}")
                stats['not_found'] += 1
                # Still mark as processed but with NULL or skip
                continue
            
            # Detect if black and white
            is_bw = is_black_and_white_fast(file_path, tolerance=tolerance)
            
            batch.append((is_bw, img_id))
            batch_count += 1
            stats['processed'] += 1
            
            if is_bw:
                stats['grayscale'] += 1
            else:
                stats['color'] += 1
            
            if verbose and stats['processed'] % 10 == 0:
                print(f"Processed {stats['processed']}/{total}...")
            
            # Insert batch when full
            if len(batch) >= batch_size:
                _update_batch(db, batch)
                batch = []
                
        except Exception as e:
            if verbose:
                print(f"Error processing {file_path}: {e}")
            stats['errors'] += 1
    
    # Process remaining batch
    if batch:
        _update_batch(db, batch)
    
    if verbose:
        print(f"\nBatch detection complete:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Grayscale: {stats['grayscale']}")
        print(f"  Color: {stats['color']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Not Found: {stats['not_found']}")
    
    return stats


def _update_batch(db: ImageDatabase, batch: List[tuple]):
    """Update a batch of images with their black/white status"""
    if not batch:
        return
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Use execute_batch for efficiency
            from psycopg2.extras import execute_batch
            
            execute_batch(cur, """
                UPDATE images 
                SET is_black_and_white = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, batch)
            
            conn.commit()


def reclassify_all_images(db: ImageDatabase, tolerance: int = 3, 
                          batch_size: int = 100, verbose: bool = True) -> dict:
    """
    Reclassify all images, overwriting existing classifications.
    
    Useful if you want to change the tolerance or re-run detection.
    
    Args:
        db: ImageDatabase instance
        tolerance: Tolerance for RGB channel differences
        batch_size: Number of images to process in each batch
        verbose: Print progress information
    
    Returns:
        Same dictionary as batch_detect_and_update
    """
    print("Reclassifying all images (this will overwrite existing classifications)...")
    
    # First, reset all values to NULL
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE images SET is_black_and_white = NULL")
            conn.commit()
    
    # Then re-run detection
    return batch_detect_and_update(db, tolerance=tolerance, 
                                   batch_size=batch_size, verbose=verbose)


def get_statistics(db: ImageDatabase) -> dict:
    """
    Get statistics about black and white vs color images in the database.
    
    Args:
        db: ImageDatabase instance
    
    Returns:
        Dictionary with statistics
    """
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE is_black_and_white = TRUE) as grayscale,
                    COUNT(*) FILTER (WHERE is_black_and_white = FALSE) as color,
                    COUNT(*) FILTER (WHERE is_black_and_white IS NULL) as unknown
                FROM images
            """)
            
            result = cur.fetchone()
            
            return {
                'total': result[0] or 0,
                'grayscale': result[1] or 0,
                'color': result[2] or 0,
                'unknown': result[3] or 0
            }


def main():
    parser = argparse.ArgumentParser(
        description='Add black and white detection to image database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add column and detect all images
  python add_black_white_column.py postgresql://user:pass@localhost/db
  
  # Detect with custom tolerance (for JPEG compression artifacts)
  python add_black_white_column.py postgresql://user:pass@localhost/db --tolerance 5
  
  # Just show statistics
  python add_black_white_column.py postgresql://user:pass@localhost/db --stats
  
  # Reclassify all images with new tolerance
  python add_black_white_column.py postgresql://user:pass@localhost/db --reclassify --tolerance 10

SQL Queries after running:
  -- Find all black and white images
  SELECT * FROM images WHERE is_black_and_white = TRUE;
  
  -- Find all color images
  SELECT * FROM images WHERE is_black_and_white = FALSE;
  
  -- Count by type
  SELECT 
      is_black_and_white,
      COUNT(*) as count
  FROM images
  GROUP BY is_black_and_white;
  
  -- Find black and white images from a specific year
  SELECT * FROM images 
  WHERE is_black_and_white = TRUE 
    AND EXTRACT(YEAR FROM date_created) = 1990;
"""
    )
    
    parser.add_argument('database_url', 
                       help='PostgreSQL connection string')
    parser.add_argument('--tolerance', type=int, default=3,
                       help='Tolerance for RGB channel differences (default: 3)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for updates (default: 100)')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics only, do not process')
    parser.add_argument('--reclassify', action='store_true',
                       help='Reclassify all images, overwriting existing values')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Initialize database
    try:
        db = ImageDatabase(args.database_url, embedding_dimensions=1536)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Show statistics only
    if args.stats:
        stats = get_statistics(db)
        print("\nImage Database Statistics:")
        print(f"  Total images: {stats['total']}")
        print(f"  Grayscale: {stats['grayscale']}")
        print(f"  Color: {stats['color']}")
        print(f"  Unknown: {stats['unknown']}")
        db.pool.closeall()
        sys.exit(0)
    
    # Add column if it doesn't exist
    if not add_black_white_column(db):
        print("Failed to add column")
        db.pool.closeall()
        sys.exit(1)
    
    # Process images
    if args.reclassify:
        print("\nReclassifying all images...")
        results = reclassify_all_images(
            db, 
            tolerance=args.tolerance,
            batch_size=args.batch_size,
            verbose=not args.no_verbose
        )
    else:
        print("\nDetecting black and white images...")
        results = batch_detect_and_update(
            db,
            tolerance=args.tolerance,
            batch_size=args.batch_size,
            verbose=not args.no_verbose
        )
    
    # Show final statistics
    print("\nFinal Statistics:")
    final_stats = get_statistics(db)
    print(f"  Total images: {final_stats['total']}")
    print(f"  Grayscale: {final_stats['grayscale']} ({final_stats['grayscale']/final_stats['total']*100:.1f}%)")
    print(f"  Color: {final_stats['color']} ({final_stats['color']/final_stats['total']*100:.1f}%)")
    print(f"  Unknown: {final_stats['unknown']}")
    
    db.pool.closeall()
    print("\nDone!")


if __name__ == "__main__":
    main()
