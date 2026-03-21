# Multi-Archive Photo Database Setup

This guide explains how to use the multi-archive configuration system to manage multiple photo databases.

## Overview

The multi-archive system allows you to:
- Maintain separate PostgreSQL databases for different photo collections
- Test different captioning models without affecting your production data
- Switch between archives easily in the Streamlit app
- Use a unified configuration file for all archive settings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   archives_config.yaml                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Archive 1: Production                                │   │
│  │   - DB: postgresql://.../photo_archive              │   │
│  │   - Root: ~/Documents/photos1                       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Archive 2: Test Llama3                               │   │
│  │   - DB: postgresql://.../photo_test_llama3          │   │
│  │   - Root: ~/Downloads/test3                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Production   │  │  Test Llama3  │  │  Test GPT-4   │
│   Database    │  │   Database    │  │   Database    │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Configuration File

The `archives_config.yaml` file defines all your archives:

```yaml
archives:
  - name: "Production Archive"
    id: "prod_v1"
    db_connection: "postgresql://user:password@localhost:5432/photo_archive"
    root_dir: "~/Documents/photos1"
    description: "Main production photo archive"
    
  - name: "Test Llama 3 Captions"
    id: "test_llama3"
    db_connection: "postgresql://user:password@localhost:5432/photo_test_llama3"
    root_dir: "~/Downloads/test3"
    description: "Testing Llama 3 generated captions"

default_archive_id: "prod_v1"

embedding:
  model_name: "all-MiniLM-L6-v2"
  dimensions: 384

llm:
  provider: "ollama"
  base_url: "http://localhost:11434"
  model: "llama3.2"
  max_tokens: 500
```

## Setup Steps

### 1. Create Your Archives Config

Edit `archives_config.yaml` with your actual database connections and directories:

```bash
cd /workspace/image_utils
nano archives_config.yaml
```

Update the connection strings and paths for your environment.

### 2. Create PostgreSQL Databases

Create separate databases for each archive:

```sql
-- Connect to PostgreSQL
psql -U postgres

-- Create databases for each archive
CREATE DATABASE photo_archive;
CREATE DATABASE photo_test_llama3;
CREATE DATABASE photo_test_gpt4;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE photo_archive TO your_user;
GRANT ALL PRIVILEGES ON DATABASE photo_test_llama3 TO your_user;
GRANT ALL PRIVILEGES ON DATABASE photo_test_gpt4 TO your_user;
```

### 3. Initialize Database Schemas

Initialize each database with the required schema:

```bash
cd /workspace/image_utils

# Production database
python image_database.py init --db "postgresql://user:password@localhost:5432/photo_archive"

# Test database
python image_database.py init --db "postgresql://user:password@localhost:5432/photo_test_llama3"
```

### 4. Ingest Photos into an Archive

Use the new ingestion tool to add photos to a specific archive:

```bash
# Ingest into production archive (default)
python photo_archive_ingest.py --dir ~/Documents/photos1

# Ingest into test archive
python photo_archive_ingest.py --dir ~/Downloads/test3 --archive test_llama3

# Dry run (no database writes)
python photo_archive_ingest.py --dir ~/Documents/photos1 --dry-run

# Skip AI features for faster ingestion
python photo_archive_ingest.py --dir ~/Documents/photos1 --no-captions --no-bw-detection
```

### 5. View Archives in Streamlit

Start the Streamlit app:

```bash
cd /workspace/image_utils
streamlit run streamlit_app.py
```

In the sidebar:
1. Select an archive from the dropdown
2. Click "Connect" to connect to that archive's database
3. Browse and search photos in that archive

## Workflow Examples

### Testing New Caption Models

1. Create a new archive entry in `archives_config.yaml`:
```yaml
  - name: "Test New Model"
    id: "test_new_model"
    db_connection: "postgresql://user:password@localhost:5432/photo_test_new"
    root_dir: "~/Downloads/test_new"
    description: "Testing new captioning model"
```

2. Create and initialize the database:
```bash
createdb photo_test_new
python image_database.py init --db "postgresql://user:password@localhost:5432/photo_test_new"
```

3. Ingest photos with the new model:
```bash
python photo_archive_ingest.py --dir ~/Downloads/test_new --archive test_new_model
```

4. Compare results in Streamlit by switching between archives

### Migrating Between Archives

To move photos from one archive to another:

1. Export from source archive (custom query)
2. Update config if needed
3. Re-ingest into target archive

## Command Reference

### Archive Config Loader

```bash
# Show current configuration
python archive_config_loader.py --show

# Create sample config
python archive_config_loader.py --create-sample ./my_config.yaml

# Use custom config path
python archive_config_loader.py --show --config /path/to/config.yaml
```

### Photo Archive Ingest

```bash
# Basic usage
python photo_archive_ingest.py --dir /path/to/photos

# Specify archive
python photo_archive_ingest.py --dir /path/to/photos --archive test_llama3

# Options
python photo_archive_ingest.py --help
```

### Streamlit App

```bash
streamlit run streamlit_app.py
```

## Best Practices

1. **Naming**: Use descriptive names and IDs for archives (e.g., `test_llama3_2024`)
2. **Backup**: Regularly backup your production database
3. **Testing**: Always test new captioning models on a small test archive first
4. **Documentation**: Keep descriptions up-to-date in the config file
5. **Security**: Use environment variables for sensitive connection strings

## Troubleshooting

### Config Not Loading

Ensure `pyyaml` is installed:
```bash
pip install pyyaml
```

### Database Connection Failed

Check your connection string format:
```
postgresql://username:password@host:port/database_name
```

### Schema Not Initialized

Run the init command for the specific database:
```bash
python image_database.py init --db "your-connection-string"
```

## File Locations

- Config file: `/workspace/image_utils/archives_config.yaml`
- Config loader: `/workspace/image_utils/archive_config_loader.py`
- Ingest tool: `/workspace/image_utils/photo_archive_ingest.py`
- Streamlit app: `/workspace/image_utils/streamlit_app.py`
