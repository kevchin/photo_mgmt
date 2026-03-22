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
    
    def __init__(self, default_model: str = "florence-2-base", database_url: Optional[str] = None):
        """
        Initialize vector search.
        
        Args:
            default_model: Default embedding model to use for searches
            database_url: Optional custom database URL (overrides environment)
        """
        self.default_model = default_model
        self.default_config = get_model_config(default_model)
        self.embedder = None
        
        # Use custom database URL if provided, otherwise use the active engine
        if database_url:
            self.engine = create_engine(database_url, pool_pre_ping=True)
        else:
            self.engine = active_engine
        
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
        
        config = get_model_config(model_name)
        column_name = config.column_name
        
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
        
        # Execute search query
        query = f"""
            SELECT 
                id, file_path, file_name, caption_text, capture_date,
                year, month, day, latitude, longitude, is_black_white,
                orientation,
                1 - ({column_name} <=> :query_vector::vector) AS similarity_score
            FROM photos
            WHERE {column_name} IS NOT NULL
            {where_clause}
            ORDER BY {column_name} <=> :query_vector::vector
            LIMIT :limit
        """
        
        results = []
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            
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
        
        config = get_model_config(model_name)
        
        # Use appropriate embedding model based on target dimension
        if config.embedding_dimension == 384:
            embedder_model = "all-MiniLM-L6-v2"
        elif config.embedding_dimension == 768:
            embedder_model = "all-mpnet-base-v2"
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
        with self.engine.connect() as conn:
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
                    SELECT COUNT(*) FROM photos WHERE {config.column_name} IS NOT NULL
                """)).scalar()
                
                models.append({
                    'name': row.model_name,
                    'dimension': row.embedding_dimension,
                    'description': row.description,
                    'is_active': row.is_active,
                    'photo_count': col_count
                })
            
            return models
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.engine.connect() as conn:
            total_photos = conn.execute(text("SELECT COUNT(*) FROM photos")).scalar()
            
            # Count photos with embeddings for each model
            models_with_counts = []
            result = conn.execute(text("SELECT * FROM caption_models"))
            for row in result:
                config = get_model_config(row.model_name)
                count = conn.execute(text(f"""
                    SELECT COUNT(*) FROM photos WHERE {config.column_name} IS NOT NULL
                """)).scalar()
                models_with_counts.append({
                    'model': row.model_name,
                    'dimension': row.embedding_dimension,
                    'count': count
                })
            
            # Date range
            date_range = conn.execute(text("""
                SELECT MIN(capture_date), MAX(capture_date) FROM photos
            """)).fetchone()
            
            # GPS bounds
            gps_bounds = conn.execute(text("""
                SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude)
                FROM photos WHERE latitude IS NOT NULL
            """)).fetchone()
            
            # B&W count
            bw_count = conn.execute(text("""
                SELECT COUNT(*) FROM photos WHERE is_black_white = TRUE
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
