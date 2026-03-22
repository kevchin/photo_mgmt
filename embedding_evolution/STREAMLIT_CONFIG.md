# Streamlit App Configuration Guide

## Database Configuration

The Streamlit app now supports configurable database connections via environment variables or a `.env` file.

### Quick Start

1. **Create a `.env` file** in the `embedding_evolution` directory:
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** to match your setup:
   ```env
   # Your original working database (384-dim Florence embeddings)
   LEGACY_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/image_archive
   
   # New evolution database (multi-model support)
   EVOLUTION_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/image_archive_evolution
   
   # Which database should the Streamlit app use? Options: "legacy" or "evolution"
   ACTIVE_DATABASE=evolution
   ```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LEGACY_DATABASE_URL` | Your original database with 384-dim embeddings | `postgresql://postgres:postgres@localhost:5432/image_archive` |
| `EVOLUTION_DATABASE_URL` | New forward-compatible database | `postgresql://postgres:postgres@localhost:5432/image_archive_evolution` |
| `ACTIVE_DATABASE` | Which database to use: `"legacy"` or `"evolution"` | `"evolution"` |
| `DATABASE_URL` | Custom database URL (overrides both above) | _none_ |

### Running the Streamlit App

```bash
cd /workspace/embedding_evolution

# Option 1: Use .env file (recommended)
streamlit run search/streamlit_app.py

# Option 2: Override via environment variable
ACTIVE_DATABASE=legacy streamlit run search/streamlit_app.py

# Option 3: Use custom database URL
DATABASE_URL=postgresql://user:pass@host:5432/mydb streamlit run search/streamlit_app.py
```

### How It Works

1. **On startup**, the app reads the `.env` file (if present) or environment variables
2. **Based on `ACTIVE_DATABASE`**, it connects to either:
   - `legacy`: Your original database (unchanged, still works with your existing system)
   - `evolution`: The new multi-model database
3. **The app displays** which database it's using in the console output

### Migration Workflow

1. **Keep your legacy system running** with the original database
2. **Migrate data** to the evolution database:
   ```bash
   python migrations/migrate_legacy.py
   ```
3. **Test with evolution database**:
   ```bash
   ACTIVE_DATABASE=evolution streamlit run search/streamlit_app.py
   ```
4. **Verify your migrated data** appears correctly
5. **Switch back to legacy** anytime to verify original system still works:
   ```bash
   ACTIVE_DATABASE=legacy streamlit run search/streamlit_app.py
   ```

### Troubleshooting

**App won't start or shows no data:**
- Check that PostgreSQL is running: `pg_isready`
- Verify the database exists: `psql -U postgres -l`
- Check the `.env` file syntax (no spaces around `=`)

**Wrong database showing:**
- Add debug output to see which DB is active:
  ```bash
  python -c "from config.database import get_active_database_url; print(get_active_database_url())"
  ```

**Connection refused:**
- Ensure PostgreSQL is running on localhost:5432
- Check credentials in the DATABASE_URL
- Verify the database user has permissions

### Testing Both Databases Side-by-Side

You can run two Streamlit instances on different ports:

```bash
# Terminal 1: Legacy database on port 8501
cd /workspace/embedding_evolution
ACTIVE_DATABASE=legacy streamlit run search/streamlit_app.py --server.port 8501

# Terminal 2: Evolution database on port 8502
cd /workspace/embedding_evolution
ACTIVE_DATABASE=evolution streamlit run search/streamlit_app.py --server.port 8502
```

Then open:
- http://localhost:8501 (legacy data)
- http://localhost:8502 (evolution data)
