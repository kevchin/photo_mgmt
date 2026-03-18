#!/usr/bin/env python3
"""
Orientation Correction Database Migration

This script adds an orientation_correction column to the images table
to store rotation angles needed for proper display.

Usage:
    python add_orientation_column.py <postgres_connection_string>
"""

import sys
import psycopg2


def add_orientation_column(conn_string: str):
    """Add orientation_correction column to images table"""
    
    print("Connecting to database...")
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    
    try:
        # Check if column already exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'images' 
                AND column_name = 'orientation_correction'
            )
        """)
        exists = cur.fetchone()[0]
        
        if exists:
            print("✓ orientation_correction column already exists")
            return
        
        # Add the column
        print("Adding orientation_correction column...")
        cur.execute("""
            ALTER TABLE images 
            ADD COLUMN orientation_correction INTEGER DEFAULT 0
        """)
        
        # Add comment to document the column
        cur.execute("""
            COMMENT ON COLUMN images.orientation_correction IS 
            'Rotation angle in degrees (0, 90, 180, 270) to apply for correct display'
        """)
        
        # Create index for quick lookup of images needing correction
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_orientation_correction 
            ON images(orientation_correction) 
            WHERE orientation_correction != 0
        """)
        
        conn.commit()
        print("✓ Successfully added orientation_correction column")
        print("\nColumn details:")
        print("  - Name: orientation_correction")
        print("  - Type: INTEGER")
        print("  - Default: 0 (no rotation needed)")
        print("  - Values: 0, 90, 180, or 270 degrees clockwise")
        print("\nNext steps:")
        print("  1. Run detect_orientation_issues.py to identify images needing rotation")
        print("  2. Your display application should check this column and rotate images accordingly")
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_orientation_column.py <postgres_connection_string>")
        print("\nExample:")
        print("  python add_orientation_column.py \"postgresql://user:pass@localhost:5432/dbname\"")
        sys.exit(1)
    
    conn_string = sys.argv[1]
    add_orientation_column(conn_string)
