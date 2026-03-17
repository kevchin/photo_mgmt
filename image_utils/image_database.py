#!/usr/bin/env python3
"""
Image Deduplication and Organization Utility with PostgreSQL/pgvector Support

Features:
- Deduplicate images using SHA256 checksums and perceptual hashing
- Organize by date (EXIF or file modification time)
- Store metadata in PostgreSQL with pgvector for semantic search
- Generate embeddings from LLM captions for similarity search
- Search by natural language, date, format, GPS location, etc.
"""

import os
import sys
import hashlib
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Image processing
from PIL import Image
from PIL.ExifTags import TAGS
import imagehash
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed. HEIC support disabled.")

# Database
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import SimpleConnectionPool


@dataclass
class ImageMetadata:
    """Container for image metadata"""
    file_path: str
    file_name: str
    file_size: int
    sha256: str
    perceptual_hash: str
    width: int
    height: int
    format: str  # JPEG, PNG, HEIC, etc.
    date_created: Optional[datetime]
    date_modified: datetime
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    is_black_and_white: bool = False
    caption: Optional[str] = None
    caption_embedding: Optional[List[float]] = None
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        if d['date_created']:
            d['date_created'] = d['date_created'].isoformat()
        d['date_modified'] = d['date_modified'].isoformat()
        return d


class ImageDatabase:
    """PostgreSQL database with pgvector support for image metadata and semantic search"""
    
    def __init__(self, connection_string: str, embedding_dimensions: int = 1536):
        """
        Initialize database connection
        
        Args:
            connection_string: PostgreSQL connection string
                e.g., "postgresql://user:password@localhost:5432/dbname"
            embedding_dimensions: Dimension size for caption embeddings (default: 1536)
        """
        self.conn_string = connection_string
        self.embedding_dimensions = embedding_dimensions
        self.pool = SimpleConnectionPool(1, 10, connection_string)
        self._initialize_schema()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    def _initialize_schema(self):
        """Create database schema with pgvector extension"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Check if images table already exists and get its embedding dimension
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'images'
                    )
                """)
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    # Get the existing vector dimension
                    cur.execute("""
                        SELECT atttypmod 
                        FROM pg_attribute 
                        WHERE attrelid = 'images'::regclass 
                        AND attname = 'caption_embedding'
                    """)
                    result = cur.fetchone()
                    if result and result[0] > 0:
                        existing_dims = result[0]
                        if existing_dims != self.embedding_dimensions:
                            print(f"Warning: Existing database has {existing_dims}-dimensional embeddings, "
                                  f"but model produces {self.embedding_dimensions} dimensions.")
                            print(f"Using existing database dimension: {existing_dims}")
                            self.embedding_dimensions = existing_dims
                        else:
                            print(f"Database schema verified ({self.embedding_dimensions} dimensions)")
                    else:
                        print(f"Using configured embedding dimension: {self.embedding_dimensions}")
                    # Table already exists, no need to recreate
                    conn.commit()
                    return
                
                # Create images table with dynamic embedding dimensions
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS images (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT UNIQUE NOT NULL,
                        file_name TEXT NOT NULL,
                        file_size BIGINT NOT NULL,
                        sha256 CHAR(64) NOT NULL,
                        perceptual_hash TEXT NOT NULL,
                        width INTEGER NOT NULL,
                        height INTEGER NOT NULL,
                        format TEXT NOT NULL,
                        date_created TIMESTAMP,
                        date_modified TIMESTAMP NOT NULL,
                        gps_latitude DOUBLE PRECISION,
                        gps_longitude DOUBLE PRECISION,
                        is_black_and_white BOOLEAN DEFAULT FALSE,
                        caption TEXT,
                        caption_embedding vector({self.embedding_dimensions}),
                        tags TEXT[],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_images_perceptual_hash ON images(perceptual_hash)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_images_date_created ON images(date_created)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_images_format ON images(format)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_images_location ON images(gps_latitude, gps_longitude)
                    WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL
                """)
                
                # Create GIN index for tags array
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_images_tags ON images USING GIN(tags)
                """)
                
                # Create HNSW index for vector similarity search (requires pgvector >= 0.5.0)
                # Fall back to IVFFlat if HNSW is not available
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_images_caption_embedding 
                        ON images USING hnsw(caption_embedding vector_cosine_ops)
                    """)
                except psycopg2.Error:
                    # Fallback to IVFFlat if HNSW not supported
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_images_caption_embedding 
                        ON images USING ivfflat(caption_embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)
                
                conn.commit()
                print("Database schema initialized successfully")
    
    def reset_embeddings_column(self, new_dimensions: int):
        """Drop and recreate the caption_embedding column with new dimensions
        
        WARNING: This will delete all existing caption embeddings!
        
        Args:
            new_dimensions: New embedding dimension size
        """
        print(f"Resetting caption embeddings to {new_dimensions} dimensions...")
        print("WARNING: This will delete all existing caption embeddings!")
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Drop the old column and index
                cur.execute("""
                    ALTER TABLE images DROP COLUMN IF EXISTS caption_embedding CASCADE
                """)
                
                # Add new column with correct dimensions
                cur.execute(f"""
                    ALTER TABLE images ADD COLUMN caption_embedding vector({new_dimensions})
                """)
                
                # Recreate the index
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_images_caption_embedding 
                        ON images USING hnsw(caption_embedding vector_cosine_ops)
                    """)
                except psycopg2.Error:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_images_caption_embedding 
                        ON images USING ivfflat(caption_embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)
                
                conn.commit()
                print(f"Successfully reset caption embeddings to {new_dimensions} dimensions")
                print("Note: All caption embeddings have been cleared. Re-run caption generation to populate them.")
    
    def image_exists(self, sha256: str) -> bool:
        """Check if an image with the given SHA256 hash exists"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM images WHERE sha256 = %s", (sha256,))
                return cur.fetchone() is not None
    
    def image_exists_by_path(self, file_path: str) -> bool:
        """Check if an image with the given file path exists"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM images WHERE file_path = %s", (file_path,))
                return cur.fetchone() is not None
    
    def insert_image(self, metadata: ImageMetadata) -> int:
        """Insert a new image record"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO images (
                        file_path, file_name, file_size, sha256, perceptual_hash,
                        width, height, format, date_created, date_modified,
                        gps_latitude, gps_longitude, is_black_and_white, caption, caption_embedding, tags
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_path) DO UPDATE SET
                        file_size = EXCLUDED.file_size,
                        sha256 = EXCLUDED.sha256,
                        perceptual_hash = EXCLUDED.perceptual_hash,
                        width = EXCLUDED.width,
                        height = EXCLUDED.height,
                        format = EXCLUDED.format,
                        date_created = EXCLUDED.date_created,
                        date_modified = EXCLUDED.date_modified,
                        gps_latitude = EXCLUDED.gps_latitude,
                        gps_longitude = EXCLUDED.gps_longitude,
                        is_black_and_white = EXCLUDED.is_black_and_white,
                        caption = EXCLUDED.caption,
                        caption_embedding = EXCLUDED.caption_embedding,
                        tags = EXCLUDED.tags,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    metadata.file_path, metadata.file_name, metadata.file_size,
                    metadata.sha256, metadata.perceptual_hash,
                    metadata.width, metadata.height, metadata.format,
                    metadata.date_created, metadata.date_modified,
                    metadata.gps_latitude, metadata.gps_longitude,
                    metadata.is_black_and_white,
                    metadata.caption, metadata.caption_embedding, metadata.tags
                ))
                img_id = cur.fetchone()[0]
                conn.commit()
                return img_id
    
    def batch_insert_images(self, metadata_list: List[ImageMetadata]) -> int:
        """Batch insert multiple image records"""
        if not metadata_list:
            return 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                values = [
                    (
                        m.file_path, m.file_name, m.file_size, m.sha256, m.perceptual_hash,
                        m.width, m.height, m.format, m.date_created, m.date_modified,
                        m.gps_latitude, m.gps_longitude, m.is_black_and_white, m.caption, m.caption_embedding, m.tags
                    )
                    for m in metadata_list
                ]
                
                execute_batch(cur, """
                    INSERT INTO images (
                        file_path, file_name, file_size, sha256, perceptual_hash,
                        width, height, format, date_created, date_modified,
                        gps_latitude, gps_longitude, is_black_and_white, caption, caption_embedding, tags
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_path) DO UPDATE SET
                        file_size = EXCLUDED.file_size,
                        sha256 = EXCLUDED.sha256,
                        perceptual_hash = EXCLUDED.perceptual_hash,
                        width = EXCLUDED.width,
                        height = EXCLUDED.height,
                        format = EXCLUDED.format,
                        date_created = EXCLUDED.date_created,
                        date_modified = EXCLUDED.date_modified,
                        gps_latitude = EXCLUDED.gps_latitude,
                        gps_longitude = EXCLUDED.gps_longitude,
                        is_black_and_white = EXCLUDED.is_black_and_white,
                        caption = EXCLUDED.caption,
                        caption_embedding = EXCLUDED.caption_embedding,
                        tags = EXCLUDED.tags,
                        updated_at = CURRENT_TIMESTAMP
                """, values)
                
                conn.commit()
                return len(metadata_list)
    
    def find_duplicates_by_sha256(self, sha256: str) -> List[Dict]:
        """Find all images with the same SHA256 hash"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM images WHERE sha256 = %s
                    ORDER BY date_created, file_path
                """, (sha256,))
                return [dict(row) for row in cur.fetchall()]
    
    def find_similar_by_perceptual_hash(self, phash: str, max_hamming_distance: int = 5) -> List[Dict]:
        """Find visually similar images using perceptual hash"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Convert hex phash to bigint for comparison
                phash_int = int(phash, 16)
                cur.execute("""
                    SELECT *, 
                           bit_count((perceptual_hash::bit(64))::bit(64) # %s::bit(64)::bit(64)) as hamming_distance
                    FROM images
                    WHERE bit_count((perceptual_hash::bit(64))::bit(64) # %s::bit(64)::bit(64)) <= %s
                    ORDER BY hamming_distance
                """, (phash_int, phash_int, max_hamming_distance))
                return [dict(row) for row in cur.fetchall()]
    
    def search_by_caption_similarity(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search for images by caption embedding similarity"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                cur.execute("""
                    SELECT *, 
                           1 - (caption_embedding <=> %s::vector) as similarity_score
                    FROM images
                    WHERE caption_embedding IS NOT NULL
                    ORDER BY caption_embedding <=> %s::vector
                    LIMIT %s
                """, (embedding_str, embedding_str, limit))
                return [dict(row) for row in cur.fetchall()]
    
    def search_by_metadata(self, 
                          date_start: Optional[datetime] = None,
                          date_end: Optional[datetime] = None,
                          format_filter: Optional[str] = None,
                          gps_lat: Optional[float] = None,
                          gps_lon: Optional[float] = None,
                          radius_km: Optional[float] = None,
                          tags: Optional[List[str]] = None,
                          has_caption: Optional[bool] = None,
                          limit: int = 100) -> List[Dict]:
        """Search images by various metadata filters"""
        conditions = []
        params = []
        
        if date_start:
            conditions.append("date_created >= %s")
            params.append(date_start)
        
        if date_end:
            conditions.append("date_created <= %s")
            params.append(date_end)
        
        if format_filter:
            conditions.append("format = %s")
            params.append(format_filter.upper())
        
        if gps_lat is not None and gps_lon is not None and radius_km:
            # Use Haversine formula approximation for distance search
            # This is a simplified version; for production use PostGIS
            conditions.append("""
                gps_latitude IS NOT NULL 
                AND gps_longitude IS NOT NULL
                AND (
                    6371 * acos(
                        cos(radians(%s)) * cos(radians(gps_latitude)) *
                        cos(radians(gps_longitude) - radians(%s)) +
                        sin(radians(%s)) * sin(radians(gps_latitude))
                    )
                ) <= %s
            """)
            params.extend([gps_lat, gps_lon, gps_lat, radius_km])
        
        if tags:
            conditions.append("tags @> %s")
            params.append(tags)
        
        if has_caption is not None:
            if has_caption:
                conditions.append("caption IS NOT NULL AND caption != ''")
            else:
                conditions.append("(caption IS NULL OR caption = '')")
        
        query = "SELECT * FROM images"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" ORDER BY date_created DESC LIMIT {limit}"
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                stats = {}
                
                # Total count
                cur.execute("SELECT COUNT(*) as total FROM images")
                stats['total_images'] = cur.fetchone()['total']
                
                # Count by format
                cur.execute("""
                    SELECT format, COUNT(*) as count 
                    FROM images 
                    GROUP BY format 
                    ORDER BY count DESC
                """)
                stats['by_format'] = {row['format']: row['count'] for row in cur.fetchall()}
                
                # Count with captions
                cur.execute("SELECT COUNT(*) as count FROM images WHERE caption IS NOT NULL")
                stats['with_captions'] = cur.fetchone()['count']
                
                # Count with GPS
                cur.execute("""
                    SELECT COUNT(*) as count 
                    FROM images 
                    WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL
                """)
                stats['with_gps'] = cur.fetchone()['count']
                
                # Date range
                cur.execute("""
                    SELECT MIN(date_created) as earliest, MAX(date_created) as latest
                    FROM images
                    WHERE date_created IS NOT NULL
                """)
                row = cur.fetchone()
                stats['date_range'] = {
                    'earliest': row['earliest'].isoformat() if row['earliest'] else None,
                    'latest': row['latest'].isoformat() if row['latest'] else None
                }
                
                return stats
    
    def close(self):
        """Close all database connections"""
        self.pool.closeall()


def main():
    parser = argparse.ArgumentParser(
        description="Image database utility with PostgreSQL/pgvector support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database
  %(prog)s init --db "postgresql://user:pass@localhost/image_archive"
  
  # Get statistics
  %(prog)s stats --db "postgresql://user:pass@localhost/image_archive"
  
  # Search by caption similarity (requires pre-computed embedding)
  %(prog)s search-caption --db "postgresql://user:pass@localhost/image_archive" \\
      --embedding "[0.1,0.2,...]" --limit 10
  
  # Search by metadata
  %(prog)s search-meta --db "postgresql://user:pass@localhost/image_archive" \\
      --date-start "2024-01-01" --date-end "2024-12-31" --format HEIC
  
  # Search by location (within 10km of coordinates)
  %(prog)s search-meta --db "postgresql://user:pass@localhost/image_archive" \\
      --gps-lat 34.0522 --gps-lon -118.2437 --radius-km 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize database schema')
    init_parser.add_argument('--db', required=True, help='PostgreSQL connection string')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument('--db', required=True, help='PostgreSQL connection string')
    
    # Search by caption embedding
    search_caption_parser = subparsers.add_parser('search-caption', 
                                                   help='Search by caption embedding similarity')
    search_caption_parser.add_argument('--db', required=True, help='PostgreSQL connection string')
    search_caption_parser.add_argument('--embedding', required=True, 
                                       help='Query embedding as JSON array string')
    search_caption_parser.add_argument('--limit', type=int, default=10, 
                                       help='Number of results (default: 10)')
    
    # Search by metadata
    search_meta_parser = subparsers.add_parser('search-meta', 
                                                help='Search by metadata filters')
    search_meta_parser.add_argument('--db', required=True, help='PostgreSQL connection string')
    search_meta_parser.add_argument('--date-start', help='Start date (YYYY-MM-DD)')
    search_meta_parser.add_argument('--date-end', help='End date (YYYY-MM-DD)')
    search_meta_parser.add_argument('--format', help='Image format filter (e.g., HEIC, JPEG)')
    search_meta_parser.add_argument('--gps-lat', type=float, help='GPS latitude')
    search_meta_parser.add_argument('--gps-lon', type=float, help='GPS longitude')
    search_meta_parser.add_argument('--radius-km', type=float, help='Search radius in km')
    search_meta_parser.add_argument('--tags', nargs='+', help='Tags to filter by')
    search_meta_parser.add_argument('--has-caption', action='store_true', 
                                    help='Only show images with captions')
    search_meta_parser.add_argument('--no-caption', action='store_true',
                                    help='Only show images without captions')
    search_meta_parser.add_argument('--limit', type=int, default=100, 
                                    help='Number of results (default: 100)')
    
    # Reset embeddings command
    reset_parser = subparsers.add_parser('reset-embeddings', 
                                          help='Reset caption embeddings column to new dimensions')
    reset_parser.add_argument('--db', required=True, help='PostgreSQL connection string')
    reset_parser.add_argument('--dimensions', type=int, default=384,
                             help='New embedding dimensions (default: 384 for MiniLM, 1536 for OpenAI)')
    reset_parser.add_argument('--force', action='store_true',
                             help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize database with default embedding dimensions
    db = ImageDatabase(args.db, embedding_dimensions=1536)
    
    try:
        if args.command == 'init':
            print("Database already initialized (schema creation is idempotent)")
            stats = db.get_statistics()
            print(f"Total images: {stats['total_images']}")
        
        elif args.command == 'reset-embeddings':
            if not args.force:
                confirm = input(f"\nAre you sure you want to reset embeddings to {args.dimensions} dimensions? This will DELETE all existing caption embeddings. Type 'yes' to confirm: ")
                if confirm.lower() != 'yes':
                    print("Operation cancelled.")
                    sys.exit(0)
            
            db.reset_embeddings_column(args.dimensions)
            print("\nTo regenerate embeddings, run:")
            print(f"  python generate_captions_local.py --db \"{args.db}\" --from-db --model microsoft/Florence-2-base --embedding-model all-MiniLM-L6-v2")
        
        elif args.command == 'stats':
            stats = db.get_statistics()
            print("\n=== Image Database Statistics ===")
            print(f"Total images: {stats['total_images']}")
            print(f"With captions: {stats['with_captions']}")
            print(f"With GPS data: {stats['with_gps']}")
            print("\nBy format:")
            for fmt, count in stats['by_format'].items():
                print(f"  {fmt}: {count}")
            if stats['date_range']['earliest']:
                print(f"\nDate range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        
        elif args.command == 'search-caption':
            import json
            embedding = json.loads(args.embedding)
            results = db.search_by_caption_similarity(embedding, limit=args.limit)
            print(f"\nFound {len(results)} similar images:\n")
            for i, img in enumerate(results, 1):
                sim_score = img.pop('similarity_score', 'N/A')
                print(f"{i}. {img['file_name']} (similarity: {sim_score:.3f})")
                print(f"   Path: {img['file_path']}")
                if img.get('caption'):
                    print(f"   Caption: {img['caption'][:100]}...")
                print(f"   Date: {img['date_created'] or 'Unknown'}")
                print(f"   Format: {img['format']}")
                if img.get('gps_latitude') and img.get('gps_longitude'):
                    print(f"   Location: {img['gps_latitude']}, {img['gps_longitude']}")
                print()
        
        elif args.command == 'search-meta':
            has_caption = True if args.has_caption else (False if args.no_caption else None)
            date_start = datetime.strptime(args.date_start, '%Y-%m-%d') if args.date_start else None
            date_end = datetime.strptime(args.date_end, '%Y-%m-%d') if args.date_end else None
            
            results = db.search_by_metadata(
                date_start=date_start,
                date_end=date_end,
                format_filter=args.format,
                gps_lat=args.gps_lat,
                gps_lon=args.gps_lon,
                radius_km=args.radius_km,
                tags=args.tags,
                has_caption=has_caption,
                limit=args.limit
            )
            
            print(f"\nFound {len(results)} images matching criteria:\n")
            for i, img in enumerate(results, 1):
                print(f"{i}. {img['file_name']}")
                print(f"   Path: {img['file_path']}")
                print(f"   Date: {img['date_created'] or 'Unknown'}")
                print(f"   Format: {img['format']} ({img['width']}x{img['height']})")
                if img.get('gps_latitude') and img.get('gps_longitude'):
                    print(f"   Location: {img['gps_latitude']}, {img['gps_longitude']}")
                if img.get('caption'):
                    caption_preview = img['caption'][:80] + "..." if len(img['caption']) > 80 else img['caption']
                    print(f"   Caption: {caption_preview}")
                if img.get('tags'):
                    print(f"   Tags: {', '.join(img['tags'][:5])}")
                print()
    
    finally:
        db.close()


if __name__ == '__main__':
    main()
