"""
Vector similarity search for photos using pgvector.
Supports multiple embedding models and hybrid search strategies.
Configurable database connection.
"""
import sys
import os
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text, create_engine
from config.database import evolution_engine, EvolutionSessionLocal, get_active_database_url, active_engine
from config.models import get_model_config, ModelType
from ingestion.embedder import Embedder


class VectorSearch:
    """Perform vector similarity searches on photo embeddings."""
    
    def __init__(self, default_model: str = "florence-2-base", database_url: Optional[str] = None, table_name: Optional[str] = None):
        """
        Initialize vector search.
        
        Args:
            default_model: Default embedding model to use for searches
            database_url: Optional custom database URL (overrides environment)
            table_name: Optional table name (auto-detected if not provided)
        """
        self.default_model = default_model
        self.default_config = get_model_config(default_model)
        self.embedder = None
        self.table_name = table_name
        
        # Use custom database URL if provided, otherwise use the active engine
        if database_url:
            self.engine = create_engine(database_url, pool_pre_ping=True)
        else:
            self.engine = active_engine
        
        # Auto-detect table name if not provided
        if self.table_name is None:
            self.table_name = self._detect_table_name()
        
        # Initialize embedder for the default model if it's an embedding model
        # or use a compatible sentence transformer
        if self.default_config.model_type == ModelType.EMBEDDING:
            self.embedder = Embedder(default_model)
        else:
            # For captioning models, we'll use a sentence transformer to embed query text
            # that matches the expected dimension
            # This is a simplification - in production you'd use the actual embedding model
            # that matches your caption model's output
            print(f"Note: {default_model} is a captioning model.")
            print("For text-to-vector search, use an embedding model like 'all-MiniLM-L6-v2'")
    
    def _detect_table_name(self) -> str:
        """Auto-detect the main image table name from the database."""
        try:
            with self.engine.connect() as conn:
                # Get all tables, excluding system tables
                result = conn.execute(text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name NOT IN ('spatial_ref_sys', 'geometry_columns', 'raster_columns', 'caption_models')
                    ORDER BY table_name
                """))
                tables = [row[0] for row in result.fetchall()]
                
                # Prefer common names in order
                preferred_names = ['photos', 'images', 'pictures', 'media']
                for name in preferred_names:
                    if name in tables:
                        print(f"Auto-detected table name: {name}")
                        return name
                
                # If no preferred name found, use the first non-system table
                if tables:
                    print(f"Auto-detected table name: {tables[0]}")
                    return tables[0]
                
                raise Exception("No suitable table found in database")
        except Exception as e:
            print(f"Error detecting table name, defaulting to 'photos': {e}")
            return "photos"
    
    def search_by_vector(self, query_vector: np.ndarray, 
                        model_name: Optional[str] = None,
                        limit: int = 20,
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for similar photos using a query vector.
        
        Args:
            query_vector: Query embedding vector as numpy array
            model_name: Specific model's embedding column to search (default: default_model)
            limit: Maximum number of results to return
            filters: Optional filters (date_from, date_to, lat_min, lat_max, lon_min, lon_max, is_black_white)
            
        Returns:
            List of photo records with similarity scores
        """
        if model_name is None:
            model_name = self.default_model
        
        try:
            config = get_model_config(model_name)
            column_name = config.column_name
        except ValueError:
            # Model not in config, assume legacy database
            print(f"Model {model_name} not found in config, trying legacy column names")
            # Try common legacy column names
            column_name = "embedding"
        
        # Convert vector to list for SQL
        query_vector_list = query_vector.tolist()
        
        # Build WHERE clause from filters
        where_clauses = []
        params = {"query_vector": query_vector_list, "limit": limit}
        
        if filters:
            if filters.get('date_from'):
                where_clauses.append("capture_date >= :date_from")
                params['date_from'] = filters['date_from']
            
            if filters.get('date_to'):
                where_clauses.append("capture_date <= :date_to")
                params['date_to'] = filters['date_to']
            
            if filters.get('year_from'):
                where_clauses.append("year >= :year_from")
                params['year_from'] = filters['year_from']
            
            if filters.get('year_to'):
                where_clauses.append("year <= :year_to")
                params['year_to'] = filters['year_to']
            
            if filters.get('lat_min') is not None and filters.get('lat_max') is not None:
                where_clauses.append("latitude BETWEEN :lat_min AND :lat_max")
                params['lat_min'] = filters['lat_min']
                params['lat_max'] = filters['lat_max']
            
            if filters.get('lon_min') is not None and filters.get('lon_max') is not None:
                where_clauses.append("longitude BETWEEN :lon_min AND :lon_max")
                params['lon_min'] = filters['lon_min']
                params['lon_max'] = filters['lon_max']
            
            if filters.get('is_black_white') is not None:
                where_clauses.append("is_black_white = :is_black_white")
                params['is_black_white'] = filters['is_black_white']
        
        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)
        
        # Execute search query - try the specific column first, then fall back to legacy columns
        query = f"""
            SELECT 
                id, file_path, file_name, caption_text, capture_date,
                year, month, day, latitude, longitude, is_black_white,
                orientation,
                1 - ({column_name} <=> :query_vector::vector) AS similarity_score
            FROM {self.table_name}
            WHERE {column_name} IS NOT NULL
            {where_clause}
            ORDER BY {column_name} <=> :query_vector::vector
            LIMIT :limit
        """
        
        results = []
        with self.engine.connect() as conn:
            try:
                result = conn.execute(text(query), params)
            except Exception as e:
                # If the specific column fails, try legacy column names
                print(f"Search with column {column_name} failed: {e}")
                print("Trying legacy column names...")
                
                for legacy_col in ["embedding", "embedding_vector", "florence_embedding"]:
                    try:
                        query = f"""
                            SELECT 
                                id, file_path, file_name, caption_text, capture_date,
                                year, month, day, latitude, longitude, is_black_white,
                                orientation,
                                1 - ({legacy_col} <=> :query_vector::vector) AS similarity_score
                            FROM {self.table_name}
                            WHERE {legacy_col} IS NOT NULL
                            {where_clause}
                            ORDER BY {legacy_col} <=> :query_vector::vector
                            LIMIT :limit
                        """
                        result = conn.execute(text(query), params)
                        column_name = legacy_col
                        print(f"Successfully using legacy column: {legacy_col}")
                        break
                    except Exception:
                        continue
                else:
                    raise Exception("Could not find any valid embedding column in the database")
            
            for row in result:
                results.append({
                    'id': row.id,
                    'file_path': row.file_path,
                    'file_name': row.file_name,
                    'caption_text': row.caption_text,
                    'capture_date': row.capture_date,
                    'year': row.year,
                    'month': row.month,
                    'day': row.day,
                    'latitude': float(row.latitude) if row.latitude else None,
                    'longitude': float(row.longitude) if row.longitude else None,
                    'is_black_white': row.is_black_white,
                    'orientation': row.orientation,
                    'similarity_score': float(row.similarity_score) if row.similarity_score else 0.0
                })
        
        return results
    
    def search_by_text(self, query_text: str,
                      model_name: Optional[str] = None,
                      limit: int = 20,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for similar photos using text query.
        
        Args:
            query_text: Text description of desired photos
            model_name: Specific model's embedding column to search
            limit: Maximum number of results
            filters: Optional filters
            
        Returns:
            List of photo records with similarity scores
        """
        if model_name is None:
            model_name = self.default_model
        
        # We need an embedding model that matches the target column's dimension
        # For simplicity, we'll use all-MiniLM-L6-v2 which produces 384-dim vectors
        # matching Florence-2-base's dimension
        # In production, you'd use the exact embedding model that was used during ingestion
        
        try:
            config = get_model_config(model_name)
            target_dimension = config.embedding_dimension
        except ValueError:
            # Model not in config, assume legacy database with 384-dim embeddings
            print(f"Model {model_name} not found in config, assuming 384-dim legacy embeddings")
            target_dimension = 384
        
        # Use appropriate embedding model based on target dimension
        if target_dimension == 384:
            embedder_model = "all-MiniLM-L6-v2"
        elif target_dimension == 768:
            embedder_model = "all-mpnet-base-v2"
        elif target_dimension == 1024:
            embedder_model = "BAAI/bge-large-en-v1.5"
        else:
            # Fall back to MiniLM and hope dimensions match
            embedder_model = "all-MiniLM-L6-v2"
        
        print(f"Embedding query with {embedder_model}...")
        embedder = Embedder(embedder_model)
        query_vector = embedder.encode(query_text)
        
        return self.search_by_vector(
            query_vector=query_vector,
            model_name=model_name,
            limit=limit,
            filters=filters
        )
    
    def get_available_models(self) -> List[Dict]:
        """Get list of models with available embeddings in the database."""
        try:
            with self.engine.connect() as conn:
                # Check if caption_models table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'caption_models'
                    )
                """)).scalar()
                
                if not result:
                    # Legacy database without caption_models table
                    # Assume it has florence-2-base embeddings in a column
                    print("Legacy database detected (no caption_models table)")
                    # Try to count photos with embeddings
                    try:
                        col_count = conn.execute(text("""
                            SELECT COUNT(*) FROM {self.table_name} 
                            WHERE embedding IS NOT NULL OR embedding_vector IS NOT NULL
                        """)).scalar()
                        return [{
                            'name': 'florence-2-base',
                            'dimension': 384,
                            'description': 'Legacy Florence-2-base embeddings',
                            'is_active': True,
                            'photo_count': col_count
                        }]
                    except:
                        return []
                
                # Evolution database with caption_models table
                result = conn.execute(text("""
                    SELECT model_name, embedding_dimension, description, is_active
                    FROM caption_models
                    ORDER BY created_at DESC
                """))
                
                models = []
                for row in result:
                    # Check if this model's column has any data
                    config = get_model_config(row.model_name)
                    col_count = conn.execute(text(f"""
                        SELECT COUNT(*) FROM {self.table_name} WHERE {config.column_name} IS NOT NULL
                    """)).scalar()
                    
                    models.append({
                        'name': row.model_name,
                        'dimension': row.embedding_dimension,
                        'description': row.description,
                        'is_active': row.is_active,
                        'photo_count': col_count
                    })
                
                return models
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.engine.connect() as conn:
            total_photos = conn.execute(text("SELECT COUNT(*) FROM {self.table_name}")).scalar()
            
            # Check if caption_models table exists (evolution schema)
            has_caption_models = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'caption_models'
                )
            """)).scalar()
            
            models_with_counts = []
            if has_caption_models:
                # Evolution database
                result = conn.execute(text("SELECT * FROM caption_models"))
                for row in result:
                    config = get_model_config(row.model_name)
                    count = conn.execute(text(f"""
                        SELECT COUNT(*) FROM {self.table_name} WHERE {config.column_name} IS NOT NULL
                    """)).scalar()
                    models_with_counts.append({
                        'model': row.model_name,
                        'dimension': row.embedding_dimension,
                        'count': count
                    })
            else:
                # Legacy database - assume florence-2-base embeddings
                try:
                    count = conn.execute(text("""
                        SELECT COUNT(*) FROM {self.table_name} 
                        WHERE embedding IS NOT NULL OR embedding_vector IS NOT NULL OR embedding_florence_2_base_384 IS NOT NULL
                    """)).scalar()
                    if count > 0:
                        models_with_counts.append({
                            'model': 'florence-2-base',
                            'dimension': 384,
                            'count': count
                        })
                except:
                    pass
            
            # Date range
            date_range = conn.execute(text("""
                SELECT MIN(capture_date), MAX(capture_date) FROM {self.table_name}
            """)).fetchone()
            
            # GPS bounds
            gps_bounds = conn.execute(text("""
                SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude)
                FROM {self.table_name} WHERE latitude IS NOT NULL
            """)).fetchone()
            
            # B&W count
            bw_count = conn.execute(text("""
                SELECT COUNT(*) FROM {self.table_name} WHERE is_black_white = TRUE
            """)).scalar()
            
            return {
                'total_photos': total_photos,
                'models': models_with_counts,
                'date_range': {
                    'min': date_range[0],
                    'max': date_range[1]
                },
                'gps_bounds': {
                    'lat_min': float(gps_bounds[0]) if gps_bounds[0] else None,
                    'lat_max': float(gps_bounds[1]) if gps_bounds[1] else None,
                    'lon_min': float(gps_bounds[2]) if gps_bounds[2] else None,
                    'lon_max': float(gps_bounds[3]) if gps_bounds[3] else None
                },
                'black_and_white_count': bw_count
            }


def test_search():
    """Test vector search functionality."""
    print("Testing vector search...")
    
    searcher = VectorSearch(default_model="florence-2-base")
    
    # Get stats
    stats = searcher.get_stats()
    print(f"\nDatabase stats:")
    print(f"  Total photos: {stats['total_photos']}")
    print(f"  Models: {stats['models']}")
    
    # Test text search
    print("\nSearching for 'kids at the beach'...")
    results = searcher.search_by_text("kids playing at the beach", limit=5)
    
    if results:
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  {r['file_name']}: {r['caption_text'][:50] if r['caption_text'] else 'No caption'}...")
            print(f"    Score: {r['similarity_score']:.4f}")
    else:
        print("No results found (database may be empty)")
    
    return results


if __name__ == "__main__":
    test_search()
