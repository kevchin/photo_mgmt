#!/usr/bin/env python3
"""
Test script to verify legacy database connectivity and schema detection.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text, inspect, create_engine
from config.database import LEGACY_DATABASE_URL, get_active_database_url, ACTIVE_DATABASE

def test_legacy_connection():
    """Test connection to legacy database and inspect schema."""
    print("=" * 60)
    print("Testing Legacy Database Connection")
    print("=" * 60)
    print(f"\nActive database setting: {ACTIVE_DATABASE}")
    print(f"Legacy database URL: {LEGACY_DATABASE_URL}")
    print(f"Active database URL: {get_active_database_url()}")
    
    try:
        engine = create_engine(LEGACY_DATABASE_URL, pool_pre_ping=True)
        
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version()")).scalar()
            print(f"\n✓ PostgreSQL version: {result[:50]}...")
            
            # Check pgvector extension
            result = conn.execute(text("""
                SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')
            """)).scalar()
            print(f"✓ pgvector installed: {result}")
            
            # List tables
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            print(f"\nTables in legacy database: {tables}")
            
            # Check for image table (could be 'photos' or 'images')
            image_table = None
            if 'photos' in tables:
                image_table = 'photos'
            elif 'images' in tables:
                image_table = 'images'
            
            if image_table:
                columns = inspector.get_columns(image_table)
                print(f"\nColumns in '{image_table}' table:")
                for col in columns:
                    print(f"  - {col['name']}: {col['type']} (nullable: {col['nullable']})")
                
                # Check for embedding column
                embedding_cols = [c for c in columns if 'embed' in c['name'].lower() or 'vector' in c['name'].lower()]
                if embedding_cols:
                    print(f"\n✓ Found embedding columns: {[c['name'] for c in embedding_cols]}")
                else:
                    print("\n⚠ No obvious embedding columns found")
                
                # Count records
                result = conn.execute(text(f"SELECT COUNT(*) FROM {image_table}")).scalar()
                print(f"\nTotal photos: {result}")
                
                # Check for caption_models table
                has_caption_models = 'caption_models' in tables
                print(f"\ncaption_models table exists: {has_caption_models}")
                
                if not has_caption_models:
                    print("→ This appears to be a LEGACY database (no model tracking)")
                    print("→ The Streamlit app should auto-detect this and use fallback mode")
                else:
                    print("→ This appears to be an EVOLUTION database (with model tracking)")
                    
            else:
                print("\n⚠ No 'photos' or 'images' table found!")
                print("   The database appears to be empty or uses a different table name.")
                
    except Exception as e:
        print(f"\n✗ Error connecting to legacy database: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Legacy database test completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_legacy_connection()
    sys.exit(0 if success else 1)
