# Quick Start: Add a New Photo Archive

This guide walks you through creating a new PostgreSQL database, ingesting photos with AI captions, and configuring it for use in the Streamlit app.

## Prerequisites

- PostgreSQL installed and running
- Python environment with required dependencies installed
- A directory of photos ready to ingest (e.g., `~/Documents/trip_photos`)

---

## Step 1: Create a New PostgreSQL Database

Create a dedicated database for your new photo archive. Replace `photo_trip_archive` with your preferred database name.

```bash
psql -U postgres -c "CREATE DATABASE photo_trip_archive;"
```

**Note:** If your PostgreSQL user is not `postgres`, replace `-U postgres` with your actual username (e.g., `-U $USER`).

---

## Step 2: Initialize the Database Schema

Set up the required tables (`images`, etc.) in your new database.

First, define your database connection string (replace `YOUR_PASSWORD` with your actual password):

```bash
export NEW_DB_URL="postgresql://postgres:YOUR_PASSWORD@localhost:5432/photo_trip_archive"
```

Then run the initialization command:

```bash
python image_database.py init --db "$NEW_DB_URL"
```

You should see output confirming the schema was created successfully.

---

## Step 3: Ingest Photos with AI Captions

Run the ingestion tool to process your photos, extract metadata, generate captions using your local LLM, and create embeddings.

```bash
python photo_archive_ingest.py \
  --dir ~/Documents/trip_photos \
  --archive trip_test \
  --db "$NEW_DB_URL" \
  --batch-size 10 \
  --dry-run false
```

**Parameters explained:**
- `--dir`: Path to your photo directory
- `--archive`: Unique ID for this archive (used in config)
- `--db`: Database connection string
- `--batch-size`: Number of photos to process in each batch (adjust based on your system's memory)
- `--dry-run false`: Actually write to the database (use `true` to test without writing)

**What happens during ingestion:**
1. Scans directory for image files
2. Extracts EXIF metadata (date, GPS, orientation, camera info)
3. Detects black & white photos using LLM
4. Generates descriptive captions using LLM
5. Creates embeddings for semantic search
6. Stores everything in the PostgreSQL database

This process may take some time depending on the number of photos and your LLM speed.

---

## Step 4: Update the Configuration File

Edit `archives_config.yaml` to register your new archive. Add the following block under the `archives:` list:

```yaml
  - name: "Trip Photos Test"
    id: "trip_test"
    db_path: "postgresql://postgres:YOUR_PASSWORD@localhost:5432/photo_trip_archive"
    root_dir: "/home/kc/Documents/trip_photos"
    description: "Test dataset for new captioning model on recent trip."
    embedding_model: "all-MiniLM-L6-v2"
    llm_model: "local-llama3"
```

**Important:** 
- Replace `YOUR_PASSWORD` with your actual PostgreSQL password
- Ensure `root_dir` matches the actual path to your photos
- The `id` field must match the `--archive` value used in Step 3

---

## Step 5: Verify in Streamlit App

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

In the sidebar:
1. Select **"Trip Photos Test"** from the archive dropdown
2. The connection string should auto-populate
3. Browse your photos, test searches, and verify captions

---

## Troubleshooting

### Connection Errors
- Verify PostgreSQL is running: `pg_isready`
- Check username/password in connection string
- Ensure database exists: `psql -U postgres -l | grep photo_trip_archive`

### Ingestion Fails
- Check that the photo directory exists and contains valid images
- Verify LLM service is running if using a local model
- Try a smaller `--batch-size` if you encounter memory issues

### Streamlit Doesn't Show Archive
- Confirm `archives_config.yaml` is in the correct location
- Check YAML syntax (indentation matters)
- Restart Streamlit after editing config

---

## Next Steps

Now that you have a working archive:

1. **Test different captioning models**: Create another archive with a different `llm_model` setting
2. **Compare results**: Switch between archives in Streamlit to compare captions
3. **Add more photos**: Re-run ingestion on the same archive to add new photos
4. **Experiment with embeddings**: Try different `embedding_model` values for semantic search

---

## Command Reference

| Task | Command |
|------|---------|
| Create database | `psql -U postgres -c "CREATE DATABASE dbname;"` |
| Initialize schema | `python image_database.py init --db "connection_string"` |
| Ingest photos | `python photo_archive_ingest.py --dir /path --archive id --db "connection_string"` |
| Launch Streamlit | `streamlit run streamlit_app.py` |
| List databases | `psql -U postgres -l` |
| Drop database (caution!) | `psql -U postgres -c "DROP DATABASE dbname;"` |

---

## Security Note

Never commit `archives_config.yaml` with real passwords to version control. Use environment variables or a secrets manager for production deployments.
