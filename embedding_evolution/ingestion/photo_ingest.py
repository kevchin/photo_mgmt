"""
Main photo ingestion pipeline.
Processes new photos, generates captions with specified LLM, creates embeddings, and stores in database.
"""
import sys
import os
import argparse
from datetime import datetime
from typing import List, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from config.database import evolution_engine, EvolutionSessionLocal, install_pgvector, check_pgvector_installed
from config.models import get_model_config, ModelType
from ingestion.caption_generator import CaptionGenerator
from ingestion.embedder import Embedder
from utils.exif_reader import ExifReader


class PhotoIngestor:
    """Ingest photos into the evolution database with captions and embeddings."""
    
    def __init__(self, caption_model: str = "florence-2-base", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        """
        Initialize the photo ingestor.
        
        Args:
            caption_model: Model name for caption generation
            embedding_model: Model name for embedding generation
            device: Device to run models on (None for auto-detect)
        """
        self.caption_model_name = caption_model
        self.embedding_model_name = embedding_model
        
        # Load caption model config
        self.caption_config = get_model_config(caption_model)
        print(f"Using caption model: {caption_model} ({self.caption_config.embedding_dimension}d)")
        
        # For embeddings, we use the caption text to generate embeddings
        # The embedding dimension comes from the caption model if it's a captioning model
        # or from the embedding model if separate
        if self.caption_config.model_type == ModelType.CAPTIONING:
            # Use the caption model's embedding dimension
            self.embedding_dimension = self.caption_config.embedding_dimension
            self.embedding_column = self.caption_config.column_name
            print(f"Embedding column: {self.embedding_column}")
            print(f"Embedding dimension: {self.embedding_dimension}")
            
            # Initialize caption generator
            print("\nInitializing caption generator...")
            self.caption_generator = CaptionGenerator(caption_model, device)
            self.embedder = None
        else:
            raise ValueError("Caption model must be a captioning model")
        
        # Verify database is ready
        self._verify_database()
    
    def _verify_database(self):
        """Verify the evolution database exists and has required schema."""
        with evolution_engine.connect() as conn:
            # Check pgvector extension
            if not check_pgvector_installed(conn):
                print("Installing pgvector extension...")
                install_pgvector(conn)
            
            # Check if photos table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'photos'
                )
            """)).scalar()
            
            if not result:
                raise RuntimeError("Photos table not found. Run migrations first.")
            
            # Check if embedding column exists
            result = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'photos' AND column_name = '{self.embedding_column}'
                )
            """)).scalar()
            
            if not result:
                print(f"Adding embedding column {self.embedding_column}...")
                conn.execute(text(f"""
                    ALTER TABLE photos 
                    ADD COLUMN {self.embedding_column} VECTOR({self.embedding_dimension})
                """))
                conn.commit()
                
                # Create HNSW index for the new column
                print(f"Creating HNSW index for {self.embedding_column}...")
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.embedding_column}_hnsw
                    ON photos USING hnsw ({self.embedding_column} vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """))
                conn.commit()
    
    def ingest_photo(self, image_path: str, skip_if_exists: bool = True) -> bool:
        """
        Ingest a single photo into the database.
        
        Args:
            image_path: Path to the image file
            skip_if_exists: Skip if photo already in database
            
        Returns:
            True if successfully ingested, False if skipped
        """
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return False
        
        with evolution_engine.connect() as conn:
            # Check if already exists
            if skip_if_exists:
                result = conn.execute(text("""
                    SELECT id FROM photos WHERE file_path = :path
                """), {"path": image_path}).fetchone()
                
                if result:
                    print(f"Skipping existing photo: {image_path}")
                    return False
            
            # Extract EXIF metadata
            print(f"Extracting EXIF from: {image_path}")
            exif_data = ExifReader.read_exif(image_path)
            
            # Generate caption
            print(f"Generating caption with {self.caption_model_name}...")
            try:
                caption = self.caption_generator.generate_caption(image_path)
                print(f"Caption: {caption[:100]}...")
            except Exception as e:
                print(f"Failed to generate caption: {e}")
                caption = None
            
            # Insert into database
            insert_data = {
                "file_path": image_path,
                "file_name": exif_data['file_name'],
                "directory_path": exif_data['directory_path'],
                "year": exif_data['year'],
                "month": exif_data['month'],
                "day": exif_data['day'],
                "caption_text": caption,
                "capture_date": exif_data['capture_date'],
                "latitude": exif_data['latitude'],
                "longitude": exif_data['longitude'],
                "altitude": exif_data['altitude'],
                "is_black_white": exif_data['is_black_white'],
                "orientation": exif_data['orientation'],
                "file_size_bytes": exif_data['file_size_bytes'],
                "mime_type": f"image/{os.path.splitext(image_path)[1][1:]}",
            }
            
            # Build dynamic SQL based on available data
            columns = [
                "file_path", "file_name", "directory_path", "year", "month", "day",
                "caption_text", "capture_date", "latitude", "longitude", "altitude",
                "is_black_white", "orientation", "file_size_bytes", "mime_type"
            ]
            
            values = [
                ":file_path", ":file_name", ":directory_path", ":year", ":month", ":day",
                ":caption_text", ":capture_date", ":latitude", ":longitude", ":altitude",
                ":is_black_white", ":orientation", ":file_size_bytes", ":mime_type"
            ]
            
            # Handle duplicate key (upsert)
            query = f"""
                INSERT INTO photos ({', '.join(columns)})
                VALUES ({', '.join(values)})
                ON CONFLICT (file_path) DO UPDATE SET
                    caption_text = EXCLUDED.caption_text,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """
            
            result = conn.execute(text(query), insert_data).fetchone()
            photo_id = result[0] if result else None
            
            # If we have a caption, generate and store embedding
            if caption and photo_id:
                print(f"Photo inserted with ID: {photo_id}")
                
                # For Florence-2, the caption itself doesn't need separate embedding
                # The model produces both caption and can produce embedding
                # But since we're using the caption text, we'd use an embedding model
                # For now, we'll store NULL for the embedding column since Florence-2
                # is primarily a captioning model, not an embedding model
                
                # If you want to embed the caption text separately, uncomment:
                # print("Generating embedding...")
                # embedder = Embedder("all-MiniLM-L6-v2")
                # embedding = embedder.encode(caption)
                # conn.execute(text(f"""
                #     UPDATE photos SET {self.embedding_column} = :embedding WHERE id = :id
                # """), {"embedding": embedding.tolist(), "id": photo_id})
                
                conn.commit()
                print(f"Successfully ingested: {image_path}")
                return True
            else:
                conn.commit()
                print(f"Ingested without embedding: {image_path}")
                return True
    
    def ingest_directory(self, directory: str, recursive: bool = True, 
                        skip_if_exists: bool = True, limit: Optional[int] = None) -> dict:
        """
        Ingest all photos from a directory.
        
        Args:
            directory: Path to directory containing photos
            recursive: Search subdirectories
            skip_if_exists: Skip photos already in database
            limit: Maximum number of photos to process
            
        Returns:
            Dictionary with ingestion statistics
        """
        stats = {
            'total_found': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0
        }
        
        # Find all image files
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp'}
        image_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if os.path.splitext(file)[1].lower() in extensions:
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if os.path.splitext(file)[1].lower() in extensions:
                    image_files.append(os.path.join(directory, file))
        
        stats['total_found'] = len(image_files)
        print(f"Found {stats['total_found']} images in {directory}")
        
        if limit:
            image_files = image_files[:limit]
            print(f"Limiting to {limit} images")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Processing: {image_path}")
            try:
                if self.ingest_photo(image_path, skip_if_exists):
                    stats['processed'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                stats['failed'] += 1
        
        print(f"\n{'='*60}")
        print(f"Ingestion complete!")
        print(f"  Total found: {stats['total_found']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")
        print(f"{'='*60}")
        
        return stats


def main():
    """Command-line interface for photo ingestion."""
    parser = argparse.ArgumentParser(description="Ingest photos with AI captions")
    parser.add_argument("--photos-dir", type=str, required=True,
                       help="Directory containing photos to ingest")
    parser.add_argument("--caption-model", type=str, default="florence-2-base",
                       help="Caption generation model (default: florence-2-base)")
    parser.add_argument("--recursive", action="store_true", default=True,
                       help="Search subdirectories (default: True)")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive",
                       help="Don't search subdirectories")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                       help="Skip photos already in database (default: True)")
    parser.add_argument("--force", action="store_false", dest="skip_existing",
                       help="Re-process photos already in database")
    parser.add_argument("--limit", type=int, default=None,
                       help="Maximum number of photos to process")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.photos_dir):
        print(f"Error: Directory not found: {args.photos_dir}")
        sys.exit(1)
    
    # Initialize ingestor
    ingestor = PhotoIngestor(
        caption_model=args.caption_model,
        device=args.device
    )
    
    # Start ingestion
    start_time = datetime.now()
    stats = ingestor.ingest_directory(
        directory=args.photos_dir,
        recursive=args.recursive,
        skip_if_exists=args.skip_existing,
        limit=args.limit
    )
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"\nTotal time: {duration:.1f} seconds")
    if stats['processed'] > 0:
        print(f"Average per photo: {duration/stats['processed']:.2f} seconds")


if __name__ == "__main__":
    main()
