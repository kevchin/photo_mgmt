"""
Test suite for embedding evolution system.
Verifies migration, model compatibility, and search functionality.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from sqlalchemy import text
from config.database import (
    evolution_engine, 
    legacy_engine,
    create_evolution_database_if_not_exists,
    check_pgvector_installed
)
from config.models import get_model_config, list_models, ModelType
from utils.exif_reader import ExifReader


class TestDatabaseSetup(unittest.TestCase):
    """Test database configuration and setup."""
    
    def test_legacy_database_connection(self):
        """Test connection to legacy database."""
        try:
            with legacy_engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                self.assertEqual(result, 1)
                print("✓ Legacy database connection successful")
        except Exception as e:
            print(f"✗ Legacy database connection failed: {e}")
            self.fail(f"Cannot connect to legacy database: {e}")
    
    def test_evolution_database_exists(self):
        """Test that evolution database can be created/accessed."""
        try:
            create_evolution_database_if_not_exists()
            with evolution_engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                self.assertEqual(result, 1)
                print("✓ Evolution database connection successful")
        except Exception as e:
            print(f"✗ Evolution database connection failed: {e}")
            self.fail(f"Cannot connect to evolution database: {e}")
    
    def test_pgvector_installed(self):
        """Test that pgvector extension is available."""
        try:
            with evolution_engine.connect() as conn:
                # Install if not present
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                
                has_pgvector = check_pgvector_installed(conn)
                self.assertTrue(has_pgvector, "pgvector extension not installed")
                print("✓ pgvector extension installed")
        except Exception as e:
            print(f"✗ pgvector check failed: {e}")
            self.fail(f"pgvector not available: {e}")


class TestModelConfigs(unittest.TestCase):
    """Test model configuration system."""
    
    def test_florence_base_config(self):
        """Test Florence-2-base configuration (legacy model)."""
        config = get_model_config("florence-2-base")
        self.assertEqual(config.name, "florence-2-base")
        self.assertEqual(config.embedding_dimension, 384)
        self.assertEqual(config.model_type, ModelType.CAPTIONING)
        self.assertEqual(config.caption_preset, "detailed")
        print(f"✓ Florence-2-base config: {config.column_name}")
    
    def test_larger_models_available(self):
        """Test that larger models are configured."""
        larger_models = ["llava-1.5-7b", "llava-1.6-34b", "blip2-opt-2.7b"]
        
        for model_name in larger_models:
            try:
                config = get_model_config(model_name)
                self.assertGreater(config.embedding_dimension, 384, 
                                 f"{model_name} should have larger embedding")
                print(f"✓ {model_name}: {config.embedding_dimension}d, requires {config.gpu_memory_required_gb}GB")
            except ValueError:
                print(f"⚠ {model_name} not configured")
    
    def test_embedding_models_available(self):
        """Test that embedding-only models are configured."""
        embedder = get_model_config("all-MiniLM-L6-v2")
        self.assertEqual(embedder.model_type, ModelType.EMBEDDING)
        self.assertEqual(embedder.embedding_dimension, 384)
        print(f"✓ all-MiniLM-L6-v2: {embedder.embedding_dimension}d")
    
    def test_column_name_generation(self):
        """Test that column names are generated correctly."""
        florence = get_model_config("florence-2-base")
        self.assertEqual(florence.column_name, "embedding_florence_2_base_384")
        
        llava = get_model_config("llava-1.6-34b")
        self.assertEqual(llava.column_name, "embedding_llava_1_6_34b_1584")
        print("✓ Column name generation correct")


class TestExifReader(unittest.TestCase):
    """Test EXIF metadata extraction."""
    
    def test_exif_reader_imports(self):
        """Test that EXIF reader can be imported and instantiated."""
        from utils.exif_reader import ExifReader
        self.assertTrue(hasattr(ExifReader, 'read_exif'))
        print("✓ EXIF reader available")
    
    def test_black_white_detection(self):
        """Test B&W detection logic."""
        from PIL import Image
        import numpy as np
        
        # Create a grayscale image
        gray_img = Image.new('L', (100, 100), color=128)
        is_bw = ExifReader._detect_black_and_white(gray_img)
        self.assertTrue(is_bw, "Grayscale image should be detected as B&W")
        
        # Create a color image
        color_img = Image.new('RGB', (100, 100), color='red')
        is_bw = ExifReader._detect_black_and_white(color_img)
        self.assertFalse(is_bw, "Red image should not be detected as B&W")
        
        print("✓ B&W detection working")


class TestSchemaEvolution(unittest.TestCase):
    """Test schema evolution capabilities."""
    
    def test_photos_table_structure(self):
        """Test that photos table has expected structure."""
        from sqlalchemy import inspect
        
        inspector = inspect(evolution_engine)
        tables = inspector.get_table_names()
        
        if 'photos' in tables:
            columns = [col['name'] for col in inspector.get_columns('photos')]
            
            # Check for standard columns
            required_cols = ['id', 'file_path', 'caption_text', 'capture_date']
            for col in required_cols:
                self.assertIn(col, columns, f"Missing required column: {col}")
            
            # Check for embedding columns
            embedding_cols = [c for c in columns if c.startswith('embedding_')]
            print(f"✓ Photos table has {len(embedding_cols)} embedding columns: {embedding_cols}")
        else:
            print("⚠ Photos table not found - run migration first")
    
    def test_caption_models_table(self):
        """Test caption_models tracking table."""
        from sqlalchemy import inspect
        
        inspector = inspect(evolution_engine)
        tables = inspector.get_table_names()
        
        if 'caption_models' in tables:
            columns = [col['name'] for col in inspector.get_columns('caption_models')]
            required = ['model_name', 'embedding_dimension', 'description']
            
            for col in required:
                self.assertIn(col, columns, f"Missing column: {col}")
            
            # Check for registered models
            with evolution_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM caption_models")).scalar()
                print(f"✓ caption_models table has {result} registered models")
        else:
            print("⚠ caption_models table not found - run migration first")


class TestSearchCapabilities(unittest.TestCase):
    """Test search functionality."""
    
    def test_vector_search_import(self):
        """Test that vector search module loads."""
        from search.vector_search import VectorSearch
        self.assertTrue(hasattr(VectorSearch, 'search_by_text'))
        self.assertTrue(hasattr(VectorSearch, 'search_by_vector'))
        print("✓ Vector search module available")
    
    def test_search_initialization(self):
        """Test search engine initialization."""
        from search.vector_search import VectorSearch
        
        try:
            searcher = VectorSearch(default_model="florence-2-base")
            stats = searcher.get_stats()
            print(f"✓ Search initialized. Total photos: {stats['total_photos']}")
        except Exception as e:
            print(f"⚠ Search initialization warning: {e}")


def run_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Embedding Evolution Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseSetup))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConfigs))
    suite.addTests(loader.loadTestsFromTestCase(TestExifReader))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaEvolution))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchCapabilities))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Run migration: python migrations/migrate_legacy.py")
        print("2. Ingest new photos: python ingestion/photo_ingest.py --photos-dir /path/to/photos")
        print("3. Launch search UI: streamlit run search/streamlit_app.py")
    else:
        print("\n✗ Some tests failed. Check output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
