# Migrating Your Existing PostgreSQL Database to the Multi-Archive System

This guide helps you migrate your existing PostgreSQL photo database to the new multi-archive configuration system.

## Quick Start (3 Steps)

### Step 1: Run the Migration Helper Script

The migration script will:
- Connect to your existing database
- Add any missing columns (is_black_and_white, caption_model_version, etc.)
- Generate the YAML configuration snippet for you

```bash
python migrate_to_archives.py --db "postgresql://youruser:yourpassword@localhost:5432/yourdbname" --root "/path/to/your/photos" --name "My Existing Archive"
```

**Example:**
```bash
python migrate_to_archives.py \
  --db "postgresql://postgres:mypass@localhost:5432/photo_archive" \
  --root "~/Documents/photos1" \
  --name "Production Archive"
```

### Step 2: Copy the Generated YAML

The script will output a YAML block like this:

```yaml
  - name: "Production Archive"
    id: "prod_existing"
    db_path: "postgresql://postgres:YOUR_PASSWORD@localhost:5432/photo_archive"
    root_dir: "/home/user/Documents/photos1"
    description: "Migrated existing PostgreSQL database. Update password in YAML."
    embedding_model: "all-MiniLM-L6-v2"
    llm_model: "local-llama3"
```

**Important:** Replace `YOUR_PASSWORD` with your actual database password.

### Step 3: Add to Configuration File

Open `archives_config.yaml` and paste the generated block under the `archives:` list:

```yaml
archives:
  # Your existing archive (from migration script)
  - name: "Production Archive"
    id: "prod_existing"
    db_path: "postgresql://postgres:actualpassword@localhost:5432/photo_archive"
    root_dir: "/home/user/Documents/photos1"
    description: "Migrated existing PostgreSQL database"
    embedding_model: "all-MiniLM-L6-v2"
    llm_model: "local-llama3"

  # Keep other test archives as needed...
```

### Step 4: Test in Streamlit

```bash
streamlit run streamlit_app.py
```

1. In the sidebar, select your archive from the dropdown
2. The connection string should auto-populate
3. Click "Connect" to browse your existing photos

## What the Migration Does

### Schema Updates

The migration script adds these columns to your existing `photos` table if they don't exist:

| Column | Type | Purpose |
|--------|------|---------|
| `is_black_and_white` | BOOLEAN | Marks B&W photos detected by LLM |
| `caption_model_version` | VARCHAR(50) | Tracks which AI model generated the caption |
| `embedding_model_version` | VARCHAR(50) | Tracks which model created embeddings |

**Note:** Your existing data (filename, path, caption, lat, long, orientation, date_taken, etc.) is preserved unchanged.

### No Data Movement

- Your photos stay in their current location
- Your database stays in PostgreSQL
- No data is copied or moved
- Only configuration changes

## Manual Configuration (Alternative)

If you prefer to configure manually without the script:

1. **Edit `archives_config.yaml`:**

```yaml
archives:
  - name: "My Archive"
    id: "my_archive_id"
    db_path: "postgresql://user:password@host:port/database_name"
    root_dir: "/absolute/path/to/photos"
    description: "My existing photo collection"
    embedding_model: "all-MiniLM-L6-v2"
    llm_model: "local-llama3"
```

2. **Manually update schema** (if needed):

```sql
ALTER TABLE photos ADD COLUMN IF NOT EXISTS is_black_and_white BOOLEAN DEFAULT FALSE;
ALTER TABLE photos ADD COLUMN IF NOT EXISTS caption_model_version VARCHAR(50);
ALTER TABLE photos ADD COLUMN IF NOT EXISTS embedding_model_version VARCHAR(50);
```

## Testing Before Adding New Photos

Once configured, you can:

1. **Browse existing photos** in Streamlit with your current captions
2. **Test search functionality** (filename, path, semantic caption search)
3. **Verify metadata display** (GPS, orientation, dates)

## Adding New Test Archives

To test new captioning models without affecting your production data:

1. **Create a new database:**
```bash
createdb photo_test_llama3
python image_database.py init --db "postgresql://postgres:pass@localhost:5432/photo_test_llama3"
```

2. **Add to YAML config:**
```yaml
  - name: "Test Llama 3"
    id: "test_llama3"
    db_path: "postgresql://postgres:pass@localhost:5432/photo_test_llama3"
    root_dir: "~/Downloads/test_photos"
    description: "Testing new captioning model"
```

3. **Ingest photos with new model:**
```bash
python photo_archive_ingest.py --dir ~/Downloads/test_photos --archive test_llama3
```

4. **Compare in Streamlit:** Switch between archives using the dropdown

## Troubleshooting

### Connection Errors
- Verify username/password in YAML
- Check PostgreSQL is running: `pg_isready`
- Ensure database exists: `psql -U postgres -l`

### Schema Errors
- Run migration script again: `python migrate_to_archives.py --db "..." --root "..."`
- Or manually add columns (see SQL above)

### Streamlit Doesn't Show Archive
- Check YAML syntax (indentation matters)
- Ensure archive has unique `id`
- Restart Streamlit after editing YAML

### Photos Not Displaying
- Verify `root_dir` path is absolute and correct
- Check file permissions
- Ensure paths in database match actual directory structure

## Command Reference

```bash
# Migrate existing database
python migrate_to_archives.py --db "postgresql://..." --root "/path/to/photos"

# View available archives
python archive_config_loader.py list

# Ingest new photos into specific archive
python photo_archive_ingest.py --dir /path/to/new/photos --archive prod_existing

# Run Streamlit app
streamlit run streamlit_app.py
```

## Next Steps

After verifying your existing archive works:

1. **Test semantic search** with your existing captions
2. **Create a test archive** with a subset of photos
3. **Try different captioning models** on the test archive
4. **Compare results** by switching archives in Streamlit
5. **Bulk ingest** new photo collections as needed
