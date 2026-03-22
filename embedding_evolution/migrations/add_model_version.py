"""
Add new embedding model support to the evolution database.
Dynamically adds columns and indexes for new LLM models.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from sqlalchemy import text, inspect
from config.database import evolution_engine, install_pgvector, check_pgvector_installed
from config.models import get_model_config, register_custom_model, ModelConfig, ModelType


def add_model_column(model_name: str, dimension: int = None):
    """
    Add a new embedding column for a model.
    
    Args:
        model_name: Name of the model (e.g., "llava-1.6-34b")
        dimension: Embedding dimension (optional, will use config if not provided)
    """
    # Get or create model config
    try:
        config = get_model_config(model_name)
    except ValueError:
        # Model not in predefined list, create custom config
        if dimension is None:
            raise ValueError("Must specify dimension for custom model")
        
        config = ModelConfig(
            name=model_name,
            model_id=f"custom/{model_name}",
            model_type=ModelType.CAPTIONING,
            embedding_dimension=dimension,
            description=f"Custom model: {model_name}"
        )
        register_custom_model(config)
        print(f"Registered custom model: {model_name}")
    
    if dimension is None:
        dimension = config.embedding_dimension
    
    column_name = config.column_name
    print(f"Adding column: {column_name} VECTOR({dimension})")
    
    with evolution_engine.connect() as conn:
        # Ensure pgvector is installed
        if not check_pgvector_installed(conn):
            print("Installing pgvector extension...")
            install_pgvector(conn)
        
        # Check if column already exists
        inspector = inspect(evolution_engine)
        if 'photos' not in inspector.get_table_names():
            raise RuntimeError("Photos table not found. Run migration first.")
        
        columns = [col['name'] for col in inspector.get_columns('photos')]
        
        if column_name in columns:
            print(f"Column {column_name} already exists!")
            
            # Register model in tracking table anyway
            conn.execute(text("""
                INSERT INTO caption_models (model_name, model_id, embedding_dimension, description)
                VALUES (:name, :id, :dim, :desc)
                ON CONFLICT (model_name) DO NOTHING
            """), {
                "name": config.name,
                "id": config.model_id,
                "dim": config.embedding_dimension,
                "desc": config.description
            })
            conn.commit()
            return
        
        # Add the column
        print(f"Creating column {column_name}...")
        conn.execute(text(f"""
            ALTER TABLE photos 
            ADD COLUMN {column_name} VECTOR({dimension})
        """))
        
        # Create HNSW index for fast similarity search
        print(f"Creating HNSW index for {column_name}...")
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{column_name}_hnsw
            ON photos USING hnsw ({column_name} vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        
        # Register model in tracking table
        print(f"Registering model in caption_models table...")
        conn.execute(text("""
            INSERT INTO caption_models (model_name, model_id, embedding_dimension, description, is_active)
            VALUES (:name, :id, :dim, :desc, FALSE)
            ON CONFLICT (model_name) DO UPDATE SET 
                embedding_dimension = EXCLUDED.embedding_dimension,
                description = EXCLUDED.description
        """), {
            "name": config.name,
            "id": config.model_id,
            "dim": config.embedding_dimension,
            "desc": config.description
        })
        
        conn.commit()
        print(f"\n✓ Successfully added {column_name} to photos table")
        print(f"  - Column type: VECTOR({dimension})")
        print(f"  - Index: idx_{column_name}_hnsw (HNSW)")
        print(f"  - Model registered in caption_models table")


def list_current_models():
    """List all models currently registered in the database."""
    with evolution_engine.connect() as conn:
        result = conn.execute(text("""
            SELECT model_name, embedding_dimension, description, is_active, created_at
            FROM caption_models
            ORDER BY created_at DESC
        """))
        
        print("\nRegistered Models:")
        print("-" * 80)
        
        models = result.fetchall()
        if not models:
            print("  No models registered yet.")
            return
        
        for row in models:
            status = "✓ Active" if row.is_active else "○ Inactive"
            print(f"  {row.model_name}")
            print(f"    Dimension: {row.embedding_dimension}")
            print(f"    Description: {row.description}")
            print(f"    Status: {status}")
            print(f"    Created: {row.created_at}")
            print()


def verify_column(model_name: str):
    """Verify that a model's column exists and has data."""
    config = get_model_config(model_name)
    column_name = config.column_name
    
    with evolution_engine.connect() as conn:
        # Check column exists
        inspector = inspect(evolution_engine)
        columns = [col['name'] for col in inspector.get_columns('photos')]
        
        if column_name not in columns:
            print(f"✗ Column {column_name} does not exist")
            return False
        
        # Count non-NULL values
        result = conn.execute(text(f"""
            SELECT COUNT(*) FROM photos WHERE {column_name} IS NOT NULL
        """)).scalar()
        
        total = conn.execute(text("SELECT COUNT(*) FROM photos")).scalar()
        
        print(f"✓ Column {column_name} exists")
        print(f"  Photos with embeddings: {result}/{total} ({100*result/total:.1f}%)")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Add new embedding model support")
    parser.add_argument("--add-model", type=str, help="Add support for a new model")
    parser.add_argument("--dimension", type=int, help="Embedding dimension (for custom models)")
    parser.add_argument("--list", action="store_true", help="List all registered models")
    parser.add_argument("--verify", type=str, help="Verify a model's column exists")
    
    args = parser.parse_args()
    
    if args.list:
        list_current_models()
    
    elif args.add_model:
        add_model_column(args.add_model, args.dimension)
        print("\nNext steps:")
        print(f"1. Ingest photos with: python ingestion/photo_ingest.py --caption-model {args.add_model}")
        print(f"2. Verify with: python migrations/add_model_version.py --verify {args.add_model}")
    
    elif args.verify:
        verify_column(args.verify)
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Add support for LLaVA 1.6 34B")
        print("  python add_model_version.py --add-model llava-1.6-34b")
        print()
        print("  # Add custom model with specific dimension")
        print("  python add_model_version.py --add-model my-custom-model --dimension 1024")
        print()
        print("  # List all registered models")
        print("  python add_model_version.py --list")
        print()
        print("  # Verify a model's column")
        print("  python add_model_version.py --verify florence-2-base")


if __name__ == "__main__":
    main()
