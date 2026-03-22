"""
Migration script to transfer data from legacy database to evolution schema.
Creates new database with model-versioned columns and migrates all existing data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text, inspect
from config.database import (
    legacy_engine, 
    evolution_engine,
    create_evolution_database_if_not_exists,
    install_pgvector,
    EvolutionSessionLocal
)
from config.models import get_model_config


def get_legacy_schema_info():
    """Inspect the legacy database schema to understand existing structure."""
    inspector = inspect(legacy_engine)
    
    # Get all tables
    tables = inspector.get_table_names()
    print(f"Found tables in legacy database: {tables}")
    
    # Get column information for each table
    schema_info = {}
    for table_name in tables:
        columns = inspector.get_columns(table_name)
        schema_info[table_name] = [
            {
                'name': col['name'],
                'type': str(col['type']),
                'nullable': col['nullable'],
                'default': col['default']
            }
            for col in columns
        ]
        
        # Also get indexes
        indexes = inspector.get_indexes(table_name)
        for idx in indexes:
            print(f"  Index on {table_name}.{idx['name']}: {idx['column_names']}")
    
    return schema_info


def create_evolution_schema():
    """Create the evolution database schema with model-versioned columns."""
    print("Creating evolution database...")
    create_evolution_database_if_not_exists()
    
    with evolution_engine.connect() as conn:
        # Install pgvector extension
        print("Installing pgvector extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        
        # Get legacy schema to understand what we're migrating
        print("\nInspecting legacy schema...")
        legacy_info = get_legacy_schema_info()
        
        # Find the main photos table (assume it's called 'photos' or similar)
        photos_table = None
        for table_name in legacy_info.keys():
            if 'photo' in table_name.lower() or 'image' in table_name.lower():
                photos_table = table_name
                break
        
        if not photos_table:
            raise ValueError("Could not find photos/images table in legacy database")
        
        print(f"\nUsing table '{photos_table}' as source")
        
        # Create caption_models tracking table
        print("\nCreating caption_models tracking table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS caption_models (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255) UNIQUE NOT NULL,
                model_id VARCHAR(512) NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT FALSE
            )
        """))
        
        # Create photos table with model-versioned embedding columns
        print("Creating photos table with evolution schema...")
        
        # Build column list based on legacy schema + embedding columns
        legacy_columns = legacy_info[photos_table]
        
        # Start creating the table
        create_stmt = "CREATE TABLE IF NOT EXISTS photos ("
        column_defs = []
        
        # Add standard columns (adjust based on your actual legacy schema)
        column_defs.append("id SERIAL PRIMARY KEY")
        column_defs.append("file_path VARCHAR(1024) UNIQUE NOT NULL")
        column_defs.append("file_name VARCHAR(512) NOT NULL")
        column_defs.append("directory_path VARCHAR(1024)")
        column_defs.append("year INTEGER")
        column_defs.append("month INTEGER")
        column_defs.append("day INTEGER")
        column_defs.append("caption_text TEXT")
        column_defs.append("capture_date TIMESTAMP")
        column_defs.append("latitude DECIMAL(10, 8)")
        column_defs.append("longitude DECIMAL(11, 8)")
        column_defs.append("altitude DECIMAL(10, 2)")
        column_defs.append("is_black_white BOOLEAN DEFAULT FALSE")
        column_defs.append("orientation INTEGER DEFAULT 1")
        column_defs.append("file_size_bytes BIGINT")
        column_defs.append("mime_type VARCHAR(100)")
        column_defs.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        column_defs.append("updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        
        # Add embedding columns for known models
        # Start with Florence-2-base (384)
        florence_config = get_model_config("florence-2-base")
        column_defs.append(f"{florence_config.column_name} VECTOR({florence_config.embedding_dimension})")
        
        # Add placeholder columns for future models (will be populated as needed)
        # These can be added dynamically later, but we'll add common ones now
        future_models = ["llava-1.5-7b", "llava-1.6-34b", "blip2-opt-2.7b"]
        for model_name in future_models:
            try:
                config = get_model_config(model_name)
                column_defs.append(f"{config.column_name} VECTOR({config.embedding_dimension})")
            except ValueError:
                pass  # Model not defined, skip
        
        create_stmt += ", ".join(column_defs)
        create_stmt += ")"
        
        conn.execute(text(create_stmt))
        
        # Create indexes
        print("Creating indexes...")
        
        # Index on file_path for fast lookups
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_photos_file_path ON photos(file_path)"))
        
        # Index on date fields for range queries
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_photos_capture_date ON photos(capture_date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_photos_year_month ON photos(year, month)"))
        
        # Index on GPS for location-based queries
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_photos_location ON photos(latitude, longitude)"))
        
        # HNSW indexes for vector similarity search (one per embedding column)
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{florence_config.column_name}_hnsw 
            ON photos USING hnsw ({florence_config.column_name} vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        
        # Insert model tracking record
        print("Registering model versions...")
        conn.execute(text("""
            INSERT INTO caption_models (model_name, model_id, embedding_dimension, description, is_active)
            VALUES (:model_name, :model_id, :dim, :desc, TRUE)
            ON CONFLICT (model_name) DO UPDATE SET is_active = TRUE
        """), {
            "model_name": florence_config.name,
            "model_id": florence_config.model_id,
            "dim": florence_config.embedding_dimension,
            "desc": florence_config.description
        })
        
        conn.commit()
        print("\nEvolution schema created successfully!")
        
        return photos_table


def migrate_data(source_table: str):
    """Migrate data from legacy database to evolution schema."""
    print("\nStarting data migration...")
    
    # Read from legacy database
    with legacy_engine.connect() as legacy_conn:
        # Get all data from legacy table
        result = legacy_conn.execute(text(f"SELECT * FROM {source_table}"))
        columns = result.keys()
        rows = result.fetchall()
        
        print(f"Found {len(rows)} records to migrate")
        
        if len(rows) == 0:
            print("No data to migrate")
            return
    
    # Insert into evolution database
    migrated_count = 0
    with evolution_engine.connect() as ev_conn:
        for row in rows:
            row_dict = dict(zip(columns, row))
            
            # Map legacy columns to evolution schema
            # Adjust this mapping based on your actual legacy schema
            insert_data = {
                "file_path": row_dict.get('file_path') or row_dict.get('path') or row_dict.get('filepath'),
                "file_name": row_dict.get('file_name') or row_dict.get('filename') or os.path.basename(row_dict.get('file_path', '')),
                "directory_path": row_dict.get('directory_path') or row_dict.get('dir_path'),
                "year": row_dict.get('year'),
                "month": row_dict.get('month'),
                "day": row_dict.get('day'),
                "caption_text": row_dict.get('caption') or row_dict.get('caption_text') or row_dict.get('description'),
                "capture_date": row_dict.get('capture_date') or row_dict.get('date_taken') or row_dict.get('exif_date'),
                "latitude": row_dict.get('latitude') or row_dict.get('lat') or row_dict.get('gps_lat'),
                "longitude": row_dict.get('longitude') or row_dict.get('lon') or row_dict.get('lng') or row_dict.get('gps_lon'),
                "altitude": row_dict.get('altitude') or row_dict.get('alt'),
                "is_black_white": row_dict.get('is_black_white') or row_dict.get('bw') or row_dict.get('black_and_white', False),
                "orientation": row_dict.get('orientation') or row_dict.get('exif_orientation', 1),
                "file_size_bytes": row_dict.get('file_size') or row_dict.get('size_bytes'),
                "mime_type": row_dict.get('mime_type') or row_dict.get('content_type'),
            }
            
            # Map the legacy embedding to the florence-2-base column
            # Try common column names for embeddings
            legacy_embedding = (
                row_dict.get('embedding') or 
                row_dict.get('vector') or 
                row_dict.get('embedding_vector') or
                row_dict.get('florence_embedding')
            )
            
            if legacy_embedding:
                insert_data["embedding_florence_2_base_384"] = legacy_embedding
            
            # Insert the record
            try:
                ev_conn.execute(text("""
                    INSERT INTO photos (
                        file_path, file_name, directory_path, year, month, day,
                        caption_text, capture_date, latitude, longitude, altitude,
                        is_black_white, orientation, file_size_bytes, mime_type,
                        embedding_florence_2_base_384
                    ) VALUES (
                        :file_path, :file_name, :directory_path, :year, :month, :day,
                        :caption_text, :capture_date, :latitude, :longitude, :altitude,
                        :is_black_white, :orientation, :file_size_bytes, :mime_type,
                        :embedding_florence_2_base_384
                    )
                """), insert_data)
                migrated_count += 1
                
                if migrated_count % 100 == 0:
                    print(f"  Migrated {migrated_count} records...")
                    
            except Exception as e:
                print(f"Error migrating record {row_dict.get('file_path', 'unknown')}: {e}")
                continue
        
        ev_conn.commit()
    
    print(f"\nMigration complete! Migrated {migrated_count}/{len(rows)} records")


def verify_migration():
    """Verify that migration was successful."""
    print("\nVerifying migration...")
    
    with evolution_engine.connect() as conn:
        # Count records
        result = conn.execute(text("SELECT COUNT(*) FROM photos")).scalar()
        print(f"Total records in evolution database: {result}")
        
        # Check embedding column
        result = conn.execute(text("""
            SELECT COUNT(*) FROM photos 
            WHERE embedding_florence_2_base_384 IS NOT NULL
        """)).scalar()
        print(f"Records with embeddings: {result}")
        
        # Check model tracking
        result = conn.execute(text("SELECT * FROM caption_models")).fetchall()
        print(f"Registered models: {len(result)}")
        for row in result:
            print(f"  - {row[1]} ({row[3]} dimensions)")
        
        # Sample a record
        result = conn.execute(text("""
            SELECT file_name, caption_text, 
                   CASE WHEN embedding_florence_2_base_384 IS NOT NULL THEN 'Yes' ELSE 'No' END as has_embedding
            FROM photos 
            LIMIT 5
        """)).fetchall()
        print("\nSample records:")
        for row in result:
            print(f"  {row[0]}: {row[1][:50] if row[1] else 'No caption'}... (Embedding: {row[2]})")


def main():
    """Main migration function."""
    print("=" * 60)
    print("Legacy to Evolution Database Migration")
    print("=" * 60)
    
    try:
        # Step 1: Create evolution schema
        source_table = create_evolution_schema()
        
        # Step 2: Migrate data
        migrate_data(source_table)
        
        # Step 3: Verify migration
        verify_migration()
        
        print("\n" + "=" * 60)
        print("Migration completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Verify the new database works with your queries")
        print("2. Run: python tests/test_evolution.py")
        print("3. Add new photos with: python ingestion/photo_ingest.py")
        print("4. Launch UI with: streamlit run search/streamlit_app.py")
        
    except Exception as e:
        print(f"\nMigration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
