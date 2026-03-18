#!/usr/bin/env python3
"""
Rotate Files In Place Based on Database Orientation Corrections

This script physically rotates image files based on the orientation_correction
values stored in the database. Use this ONLY if you need actual rotated files
(e.g., for external tool compatibility).

WARNING: This modifies your original files! Always backup first.

Usage:
    python rotate_files_in_place.py <postgres_connection_string> [--backup] [--dry-run]
"""

import sys
import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("Error: psycopg2 required. Install with: pip install psycopg2-binary")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow required. Install with: pip install Pillow")
    sys.exit(1)


def get_images_needing_rotation(conn_string: str, limit: int = 100) -> List[Dict]:
    """Get images that have non-zero orientation corrections"""
    
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute("""
            SELECT id, file_path, file_name, orientation_correction
            FROM images
            WHERE orientation_correction IS NOT NULL 
              AND orientation_correction != 0
              AND orientation_correction != -999  -- Already rotated marker
            LIMIT %s
        """, (limit,))
        
        images = [dict(row) for row in cur.fetchall()]
        return images
        
    finally:
        cur.close()
        conn.close()


def rotate_file(file_path: str, rotation_angle: int, backup: bool = False) -> tuple:
    """
    Rotate a file in place.
    
    Returns:
        Tuple of (success, message)
    """
    
    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    # Create backup if requested
    if backup:
        backup_path = path.with_suffix(path.suffix + '.backup')
        try:
            shutil.copy2(path, backup_path)
            print(f"  Created backup: {backup_path}")
        except Exception as e:
            return False, f"Failed to create backup: {e}"
    
    try:
        # Open and rotate
        with Image.open(path) as img:
            # PIL rotates counter-clockwise, so negate the angle
            rotated_img = img.rotate(-rotation_angle, expand=True)
            
            # Preserve original format and quality
            save_kwargs = {'quality': 95}
            
            # Convert mode if necessary
            if rotated_img.mode in ('RGBA', 'P'):
                rotated_img = rotated_img.convert('RGB')
            
            # Save back to same location
            rotated_img.save(path, **save_kwargs)
        
        return True, f"Successfully rotated {rotation_angle}°"
        
    except Exception as e:
        if backup and 'backup_path' in locals():
            # Try to restore from backup
            try:
                shutil.copy2(backup_path, path)
                os.remove(backup_path)
            except:
                pass
        return False, f"Rotation failed: {e}"


def mark_as_rotated(conn_string: str, image_id: int):
    """Mark an image as physically rotated in the database"""
    
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    
    try:
        # Set to -999 to indicate physical rotation was applied
        # Reset to 0 since no further correction needed
        cur.execute("""
            UPDATE images 
            SET orientation_correction = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (image_id,))
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def rotate_files_in_place(conn_string: str, limit: int = 100, 
                          backup: bool = True, dry_run: bool = False):
    """Main function to rotate files based on database corrections"""
    
    print("=" * 60)
    print("Physical File Rotation")
    print("=" * 60)
    print(f"Limit: {limit} files")
    print(f"Backup originals: {backup}")
    print(f"Dry run: {dry_run}")
    print()
    
    if backup and not dry_run:
        confirm = input("WARNING: This will modify your original image files!\n"
                       "Backups will be created with .backup extension.\n"
                       "Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted")
            return
    
    # Get images needing rotation
    images = get_images_needing_rotation(conn_string, limit)
    
    if not images:
        print("✓ No images need physical rotation")
        return
    
    print(f"Found {len(images)} images to rotate\n")
    
    success_count = 0
    error_count = 0
    
    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img['file_name']}")
        print(f"  Path: {img['file_path']}")
        print(f"  Rotation: {img['orientation_correction']}°")
        
        if dry_run:
            print(f"  [DRY RUN] Would rotate file")
            success_count += 1
        else:
            success, message = rotate_file(
                img['file_path'], 
                img['orientation_correction'],
                backup=backup
            )
            
            if success:
                print(f"  ✓ {message}")
                # Mark as rotated in database
                mark_as_rotated(conn_string, img['id'])
                print(f"  ✓ Updated database")
                success_count += 1
            else:
                print(f"  ✗ Error: {message}")
                error_count += 1
        
        print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print("=" * 60)
    
    if backup and not dry_run:
        print("\nBackup files created with .backup extension")
        print("Review results before deleting backups!")


def main():
    parser = argparse.ArgumentParser(
        description="Physically rotate image files based on database orientation corrections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "postgresql://user:pass@localhost:5432/dbname"
  %(prog)s "postgresql://user:pass@localhost:5432/dbname" --no-backup
  %(prog)s "postgresql://user:pass@localhost:5432/dbname" --dry-run
  %(prog)s "postgresql://user:pass@localhost:5432/dbname" --limit 50
        """
    )
    
    parser.add_argument("connection_string", 
                       help="PostgreSQL connection string")
    parser.add_argument("--limit", "-l", type=int, default=100,
                       help="Maximum number of files to rotate (default: 100)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup files (DANGEROUS!)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    rotate_files_in_place(
        conn_string=args.connection_string,
        limit=args.limit,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
