"""
Database Schema for Embedding Evolution

Supports multiple embedding models with different dimensions in the same database.
Each model gets its own vector column, allowing gradual migration and multi-model search.

Key features:
- Separate vector columns per embedding model (e.g., embedding_384, embedding_768, embedding_1024)
- Model registry to track which models are available
- Stores raw captions for re-embedding without re-processing images
- Maintains all metadata (EXIF date, GPS, B&W, orientation)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EvolutionDatabase:
    """
    PostgreSQL database manager for embedding evolution architecture.
    
    Schema design:
    - One row per image
    - Multiple embedding columns (one per model/dimension)
    - Raw caption storage for re-embedding
    - Full metadata support (EXIF, GPS, etc.)
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.conn = None
        self.connected = False
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.database_url)
            self.connected = True
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.connected = False
            logger.info("Disconnected from database")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def create_schema(self):
        """
        Create the complete schema for embedding evolution.
        
        Creates:
        1. caption_models table: Registry of embedding models used
        2. photos table: Main photo metadata and embeddings
        3. Indexes for efficient searching
        """
        if not self.connected:
            self.connect()
        
        cursor = self.conn.cursor()
        
        try:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Table 1: Model registry
            logger.info("Creating caption_models table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS caption_models (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) UNIQUE NOT NULL,
                    model_type VARCHAR(50) NOT NULL,  -- 'vlm' or 'embedding'
                    dimension INTEGER,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT true
                );
            """)
            
            # Table 2: Photos with multi-model embeddings
            logger.info("Creating photos table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    id SERIAL PRIMARY KEY,
                    
                    -- Image identification
                    file_path VARCHAR(1024) UNIQUE NOT NULL,
                    file_name VARCHAR(512) NOT NULL,
                    directory_path VARCHAR(1024),
                    
                    -- Date hierarchy (YYYY/MM/DD)
                    year INTEGER,
                    month INTEGER,
                    day INTEGER,
                    
                    -- EXIF metadata
                    capture_date TIMESTAMP,
                    camera_make VARCHAR(255),
                    camera_model VARCHAR(255),
                    exposure_time VARCHAR(50),
                    f_number REAL,
                    iso_speed INTEGER,
                    focal_length REAL,
                    
                    -- Location data
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    gps_altitude REAL,
                    
                    -- Image properties
                    is_black_and_white BOOLEAN DEFAULT false,
                    orientation INTEGER,
                    width INTEGER,
                    height INTEGER,
                    file_size_bytes BIGINT,
                    
                    -- Caption and embeddings (key feature: multiple models)
                    caption_text TEXT NOT NULL,
                    caption_model VARCHAR(255),  -- Which VLM generated the caption
                    caption_generated_at TIMESTAMP,
                    
                    -- Embedding columns for different models (added dynamically)
                    -- These will be added as needed:
                    -- embedding_384 VECTOR(384),   -- Florence-2 + MiniLM
                    -- embedding_768 VECTOR(768),   -- Florence-2 + MPNet/BGE-base
                    -- embedding_1024 VECTOR(1024), -- Qwen2.5-VL + BGE-large
                    -- embedding_1536 VECTOR(1536), -- Future models
                    -- embedding_4096 VECTOR(4096), -- Large models
                    
                    -- Metadata
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes
            logger.info("Creating indexes...")
            
            # Index on file path
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_photos_file_path 
                ON photos(file_path);
            """)
            
            # Index on date hierarchy
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_photos_date 
                ON photos(year, month, day);
            """)
            
            # Index on capture date
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_photos_capture_date 
                ON photos(capture_date);
            """)
            
            # Index on GPS coordinates (for location-based searches)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_photos_location 
                ON photos(latitude, longitude) 
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
            """)
            
            # Index on B&W flag
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_photos_bw 
                ON photos(is_black_and_white);
            """)
            
            # Index on caption model
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_photos_caption_model 
                ON photos(caption_model);
            """)
            
            # Commit changes
            self.conn.commit()
            logger.info("Schema creation completed successfully!")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating schema: {e}")
            raise
        finally:
            cursor.close()
    
    def add_embedding_column(self, model_name: str, dimension: int):
        """
        Add a new embedding column for a specific model.
        
        This allows adding support for new embedding dimensions without
        affecting existing data or requiring table recreation.
        
        Args:
            model_name: Name of the embedding model (e.g., "bge-large-en-v1.5")
            dimension: Vector dimension (e.g., 384, 768, 1024)
        """
        if not self.connected:
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Sanitize column name
        column_name = f"embedding_{dimension}"
        safe_model_name = model_name.replace("-", "_").replace(".", "_")
        
        try:
            # Check if column already exists
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'photos' 
                AND column_name = %s;
            """, (column_name,))
            
            if cursor.fetchone():
                logger.info(f"Column {column_name} already exists")
                return
            
            # Add new column
            logger.info(f"Adding embedding column: {column_name} VECTOR({dimension})")
            cursor.execute(f"""
                ALTER TABLE photos 
                ADD COLUMN {column_name} VECTOR({dimension});
            """)
            
            # Create HNSW index for fast similarity search
            logger.info(f"Creating HNSW index for {column_name}...")
            index_name = f"idx_photos_{column_name}"
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON photos
                USING hnsw ({column_name} vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            # Register the model
            cursor.execute("""
                INSERT INTO caption_models (model_name, model_type, dimension, description)
                VALUES (%s, 'embedding', %s, %s)
                ON CONFLICT (model_name) DO NOTHING;
            """, (safe_model_name, dimension, f"Embedding model {dimension}-dim"))
            
            self.conn.commit()
            logger.info(f"Successfully added embedding column: {column_name}")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding embedding column: {e}")
            raise
        finally:
            cursor.close()
    
    def insert_photo(self, 
                     file_path: str,
                     caption_text: str,
                     caption_model: str,
                     embedding: np.ndarray,
                     embedding_model: str,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Insert a photo with caption and embedding.
        
        Automatically determines which embedding column to use based on dimension.
        
        Args:
            file_path: Full path to the image file
            caption_text: Generated caption text
            caption_model: VLM model that generated the caption
            embedding: Numpy array of embedding vector
            embedding_model: Embedding model name
            metadata: Optional dict with additional metadata
        """
        if not self.connected:
            self.connect()
        
        cursor = self.conn.cursor()
        
        try:
            # Determine embedding dimension and column name
            dimension = len(embedding)
            column_name = f"embedding_{dimension}"
            
            # Ensure the column exists
            self.add_embedding_column(embedding_model, dimension)
            
            # Extract metadata
            metadata = metadata or {}
            
            # Parse file path for directory structure
            from pathlib import Path
            p = Path(file_path)
            file_name = p.name
            directory_path = str(p.parent)
            
            # Try to extract YYYY/MM/DD from path
            year = month = day = None
            parts = p.parts
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 4:  # Year
                    year = int(part)
                    if i+1 < len(parts) and parts[i+1].isdigit() and len(parts[i+1]) == 2:
                        month = int(parts[i+1])
                        if i+2 < len(parts) and parts[i+2].isdigit() and len(parts[i+2]) <= 2:
                            day = int(parts[i+2])
            
            # Prepare dynamic SQL for embedding column
            query = f"""
                INSERT INTO photos (
                    file_path, file_name, directory_path,
                    year, month, day,
                    caption_text, caption_model, caption_generated_at,
                    {column_name},
                    latitude, longitude, is_black_and_white, orientation,
                    capture_date, camera_make, camera_model,
                    processed_at
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, CURRENT_TIMESTAMP,
                    %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (file_path) DO UPDATE SET
                    caption_text = EXCLUDED.caption_text,
                    caption_model = EXCLUDED.caption_model,
                    {column_name} = EXCLUDED.{column_name},
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id;
            """
            
            # Convert embedding to list for psycopg2
            embedding_list = embedding.tolist()
            
            cursor.execute(query, (
                file_path, file_name, directory_path,
                year, month, day,
                caption_text, caption_model,
                embedding_list,
                metadata.get('latitude'),
                metadata.get('longitude'),
                metadata.get('is_black_and_white', False),
                metadata.get('orientation'),
                metadata.get('capture_date'),
                metadata.get('camera_make'),
                metadata.get('camera_model')
            ))
            
            photo_id = cursor.fetchone()[0]
            self.conn.commit()
            
            logger.info(f"Inserted photo ID {photo_id}: {file_name}")
            return photo_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error inserting photo: {e}")
            raise
        finally:
            cursor.close()
    
    def search_by_vector(self, 
                         query_embedding: np.ndarray,
                         model_dimension: int,
                         limit: int = 10,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for similar photos using vector similarity.
        
        Args:
            query_embedding: Query vector
            model_dimension: Dimension of the embedding model used
            limit: Number of results to return
            filters: Optional filters (date range, location, B&W, etc.)
            
        Returns:
            List of matching photos with similarity scores
        """
        if not self.connected:
            self.connect()
        
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            column_name = f"embedding_{model_dimension}"
            embedding_list = query_embedding.tolist()
            
            # Build WHERE clause for filters
            where_clauses = []
            params = [embedding_list]
            
            if filters:
                if 'year' in filters:
                    where_clauses.append("year = %s")
                    params.append(filters['year'])
                
                if 'start_date' in filters:
                    where_clauses.append("capture_date >= %s")
                    params.append(filters['start_date'])
                
                if 'end_date' in filters:
                    where_clauses.append("capture_date <= %s")
                    params.append(filters['end_date'])
                
                if 'is_black_and_white' in filters:
                    where_clauses.append("is_black_and_white = %s")
                    params.append(filters['is_black_and_white'])
                
                if 'latitude_min' in filters and 'latitude_max' in filters:
                    where_clauses.append("latitude BETWEEN %s AND %s")
                    params.extend([filters['latitude_min'], filters['latitude_max']])
                
                if 'longitude_min' in filters and 'longitude_max' in filters:
                    where_clauses.append("longitude BETWEEN %s AND %s")
                    params.extend([filters['longitude_min'], filters['longitude_max']])
            
            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)
            
            # Execute similarity search
            query = f"""
                SELECT 
                    id, file_path, file_name, caption_text,
                    capture_date, latitude, longitude,
                    is_black_and_white,
                    1 - ({column_name} <=> %s::vector) AS similarity_score
                FROM photos
                WHERE {column_name} IS NOT NULL
                {where_sql}
                ORDER BY {column_name} <=> %s::vector
                LIMIT %s;
            """
            
            params_extended = params + [embedding_list, limit]
            cursor.execute(query, params_extended)
            
            results = cursor.fetchall()
            
            # Convert to list of dicts
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error searching by vector: {e}")
            raise
        finally:
            cursor.close()
    
    def get_available_models(self) -> List[Dict]:
        """Get list of registered embedding models"""
        if not self.connected:
            self.connect()
        
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT model_name, model_type, dimension, description, created_at, is_active
                FROM caption_models
                ORDER BY dimension, model_name;
            """)
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            raise
        finally:
            cursor.close()
    
    def get_photos_without_embedding(self, dimension: int) -> List[Dict]:
        """Get photos that don't have embeddings for a specific dimension"""
        if not self.connected:
            self.connect()
        
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        column_name = f"embedding_{dimension}"
        
        try:
            cursor.execute(f"""
                SELECT id, file_path, file_name, caption_text
                FROM photos
                WHERE {column_name} IS NULL
                AND caption_text IS NOT NULL
                LIMIT 1000;
            """)
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting photos without embedding: {e}")
            raise
        finally:
            cursor.close()


# Example usage
if __name__ == "__main__":
    import numpy as np
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Use the same database URL as the rest of the system
    EVOLUTION_DATABASE_URL = os.getenv(
        "EVOLUTION_DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/image_archive_evolution"
    )
    db_url = EVOLUTION_DATABASE_URL
    
    with EvolutionDatabase(db_url) as db:
        # Create schema
        db.create_schema()
        
        # Add embedding columns for different models
        db.add_embedding_column("all-MiniLM-L6-v2", 384)
        db.add_embedding_column("bge-base-en-v1.5", 768)
        db.add_embedding_column("bge-large-en-v1.5", 1024)
        
        # Get available models
        models = db.get_available_models()
        print("\nAvailable embedding models:")
        for model in models:
            print(f"  {model['model_name']}: {model['dimension']}d")
        
        # Example: Insert a test photo
        test_embedding_384 = np.random.rand(384).astype(np.float32)
        test_embedding_768 = np.random.rand(768).astype(np.float32)
        test_embedding_1024 = np.random.rand(1024).astype(np.float32)
        
        print("\nSchema setup complete!")
