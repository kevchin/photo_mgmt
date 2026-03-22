# Embedding Evolution Architecture

## Overview
This directory contains tools for managing photo caption embeddings with support for multiple LLM models and embedding dimensions. The architecture allows you to:
- Migrate existing 384-dimension Florence-2-base embeddings to a forward-compatible schema
- Add new photos with different/larger embedding models (e.g., 1584+ dimensions)
- Maintain backward compatibility with existing search functionality
- Support hybrid search across multiple embedding models

## Directory Structure
```
embedding_evolution/
├── config/
│   ├── database.py          # Database connection and configuration
│   └── models.py            # LLM model configurations and metadata
├── migrations/
│   ├── migrate_legacy.py    # Migrate from legacy DB to evolution schema
│   └── add_model_version.py # Add new embedding columns for new models
├── ingestion/
│   ├── caption_generator.py # Generate captions using specified LLM
│   ├── embedder.py          # Create embeddings with model-specific dimensions
│   └── photo_ingest.py      # Main ingestion pipeline for new photos
├── search/
│   ├── vector_search.py     # Multi-model vector similarity search
│   └── streamlit_app.py     # Streamlit UI for evolved search
├── utils/
│   ├── exif_reader.py       # Extract EXIF metadata (date, GPS, B&W, orientation)
│   └── validation.py        # Verify embedding compatibility and integrity
├── tests/
│   └── test_evolution.py    # Test migration and new model onboarding
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Key Features

### 1. Model-Versioned Schema
- Separate vector columns for each embedding model (e.g., `embedding_florence_384`, `embedding_llava_1584`)
- Metadata table tracking model versions, dimensions, and activation dates
- Raw caption text storage for future re-embedding without re-processing images

### 2. Migration Path
- Copy existing data from legacy database to new evolution schema
- Preserve all existing metadata (EXIF date, GPS, B&W flag, orientation)
- Maintain original 384-dimension embeddings as baseline

### 3. Forward Compatibility
- Dynamic column creation for new embedding models
- Configuration-driven model selection during ingestion
- Support for mixed embedding dimensions within same table

### 4. Search Capabilities
- Model-specific search (query only photos with specific embedding version)
- Hybrid search (combine results from multiple embedding models)
- Pre-filtering using metadata (date range, GPS bounds, B&W filter) before vector search

## Quick Start

### Prerequisites
- PostgreSQL with pgvector extension
- Python 3.9+
- Existing legacy database at `postgresql://postgres:postgres@localhost:5432/image_archive`

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Database
Edit `config/database.py` to set up connection strings for:
- Legacy database (source)
- Evolution database (target)

### Step 3: Run Migration
```bash
python migrations/migrate_legacy.py
```
This creates the new database with model-versioned schema and migrates all existing data.

### Step 4: Verify Migration
```bash
python tests/test_evolution.py
```

### Step 5: Add New Photos with Different Model
```bash
python ingestion/photo_ingest.py --model llava-1.6-34b --photos-dir /path/to/new/photos
```

### Step 6: Launch Evolved Search UI
```bash
streamlit run search/streamlit_app.py
```

## Architecture Decisions

### Why Not CSV Storage?
- **ACID Compliance**: PostgreSQL ensures data integrity during concurrent operations
- **Unified Querying**: Single SQL interface for metadata filtering + vector search
- **pgvector Performance**: Optimized HNSW indexes for fast approximate nearest neighbor search
- **No Sync Issues**: Avoids complexity of keeping CSV and database in sync

### Why Multiple Vector Columns vs JSONB?
- **Type Safety**: PostgreSQL enforces correct vector dimensions per column
- **Index Efficiency**: Each vector column can have its own optimized HNSW index
- **Query Performance**: Direct column access is faster than JSONB extraction
- **Clear Semantics**: Explicit schema shows which models are supported

### Handling Mixed Embedding Dimensions
When searching across photos with different embedding models:
1. **Model-Specific Mode**: User selects which model's embeddings to search (recommended)
2. **Hybrid Mode**: Run separate searches per model, merge results with score normalization
3. **Fallback Mode**: If query model doesn't match photo model, use text-based keyword search as fallback

## Future Enhancements
- Automated re-embedding pipeline when upgrading models
- Embedding quality metrics and comparison tools
- Distributed ingestion for large photo batches
- Caching layer for frequently searched queries
