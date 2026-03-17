# PostgreSQL Query Examples for Image Archive

This document provides SQL query examples for searching and analyzing images in the `image_archive` database.

## Table Schema

The `images` table contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Auto-increment ID |
| `file_path` | TEXT UNIQUE NOT NULL | Full path to the image file |
| `file_name` | TEXT NOT NULL | Filename only |
| `file_size` | BIGINT NOT NULL | File size in bytes |
| `sha256` | CHAR(64) NOT NULL | SHA256 checksum |
| `perceptual_hash` | TEXT NOT NULL | Perceptual hash for deduplication |
| `width` | INTEGER NOT NULL | Image width in pixels |
| `height` | INTEGER NOT NULL | Image height in pixels |
| `format` | TEXT NOT NULL | Image format (JPEG, PNG, HEIC, etc.) |
| `date_created` | TIMESTAMP | EXIF date taken (if available) |
| `date_modified` | TIMESTAMP | File modification timestamp |
| `gps_latitude` | DOUBLE PRECISION | GPS latitude (if available) |
| `gps_longitude` | DOUBLE PRECISION | GPS longitude (if available) |
| `caption` | TEXT | AI-generated image caption |
| `caption_embedding` | vector(N) | Embedding vector (384 or 1536 dimensions) |
| `tags` | TEXT[] | Array of auto-generated tags |
| `created_at` | TIMESTAMP | When record was created in DB |
| `updated_at` | TIMESTAMP | When record was last updated |

---

## 1. Text Search Queries

### Basic Caption Search
Search for images containing specific keywords in their captions:

```sql
SELECT file_path, caption, created_at
FROM images
WHERE caption ILIKE '%sunset%' 
   OR caption ILIKE '%beach%'
ORDER BY created_at DESC
LIMIT 20;
```

### Search with Multiple Keywords
Find images matching multiple terms:

```sql
SELECT file_name, caption, tags
FROM images
WHERE caption ILIKE '%mountain%' 
  AND caption ILIKE '%lake%'
ORDER BY date_created DESC;
```

### Tag-Based Search
Search using the auto-generated tags array:

```sql
SELECT file_path, tags, caption
FROM images
WHERE 'family' = ANY(tags) 
   OR 'vacation' = ANY(tags)
ORDER BY created_at DESC
LIMIT 20;
```

### Combined Caption and Tag Search
```sql
SELECT file_name, caption, tags
FROM images
WHERE caption ILIKE '%birthday%' 
   OR 'celebration' = ANY(tags)
ORDER BY date_created DESC;
```

---

## 2. GPS Proximity Queries

### Find Images Near Coordinates (Haversine Formula)
Calculate distance from a specific point without PostGIS:

```sql
SELECT 
    file_name, 
    gps_latitude, 
    gps_longitude, 
    caption,
    6371 * acos(
        cos(radians(37.7749)) * cos(radians(gps_latitude)) *
        cos(radians(gps_longitude) - radians(-122.4194)) +
        sin(radians(37.7749)) * sin(radians(gps_latitude))
    ) AS distance_km
FROM images
WHERE gps_latitude IS NOT NULL 
  AND gps_longitude IS NOT NULL
HAVING distance_km < 50
ORDER BY distance_km ASC
LIMIT 20;
```

*Replace `37.7749, -122.4194` with your target latitude/longitude.*

### Images Within Distance Range
Find images between two distances from a point:

```sql
SELECT 
    file_name, 
    caption,
    6371 * acos(
        cos(radians(48.8566)) * cos(radians(gps_latitude)) *
        cos(radians(gps_longitude) - radians(2.3522)) +
        sin(radians(48.8566)) * sin(radians(gps_latitude))
    ) AS distance_km
FROM images
WHERE gps_latitude IS NOT NULL 
  AND gps_longitude IS NOT NULL
HAVING distance_km BETWEEN 10 AND 100
ORDER BY distance_km ASC;
```

### Photos Taken at Specific Location
```sql
SELECT file_name, caption, date_created
FROM images
WHERE gps_latitude BETWEEN 40.70 AND 40.72
  AND gps_longitude BETWEEN -74.01 AND -73.99
ORDER BY date_created DESC;
```

---

## 3. Semantic Search (Vector Similarity)

Find images with similar meanings using embedding vectors:

```sql
-- Replace the vector values with your query embedding
SELECT 
    file_path, 
    caption, 
    caption_embedding <=> '[0.05, -0.12, 0.33, ..., 0.89]'::vector AS similarity_score
FROM images
WHERE caption_embedding IS NOT NULL
ORDER BY similarity_score ASC
LIMIT 10;
```

**Note:** 
- Lower scores indicate higher similarity
- Vector dimensions must match your model (384 for all-MiniLM-L6-v2, 1536 for OpenAI)
- Generate query embeddings using the same model used for image captions

---

## 4. Metadata Queries

### Filter by File Format and Size
```sql
SELECT 
    file_name, 
    file_size / 1024 / 1024 AS size_mb, 
    format,
    width,
    height
FROM images
WHERE format = 'HEIC' 
  AND file_size > 5242880  -- Files larger than 5MB
ORDER BY file_size DESC;
```

### Find Duplicate Images (by Perceptual Hash)
```sql
SELECT 
    perceptual_hash, 
    count(*) as duplicate_count, 
    array_agg(file_name) as files
FROM images
GROUP BY perceptual_hash
HAVING count(*) > 1
ORDER BY duplicate_count DESC;
```

### Images by Date Range
```sql
SELECT file_name, date_created, caption
FROM images
WHERE date_created BETWEEN '2020-01-01' AND '2024-12-31'
ORDER BY date_created DESC;
```

### Images Without Captions
```sql
SELECT file_name, file_path
FROM images
WHERE caption IS NULL
ORDER BY date_modified DESC
LIMIT 100;
```

### Images Without GPS Data
```sql
SELECT file_name, date_created
FROM images
WHERE gps_latitude IS NULL 
  OR gps_longitude IS NULL
ORDER BY date_created DESC
LIMIT 50;
```

### Image Statistics
```sql
-- Count by format
SELECT format, count(*) as count
FROM images
GROUP BY format
ORDER BY count DESC;

-- Average file size by format
SELECT format, 
       avg(file_size) / 1024 / 1024 AS avg_size_mb,
       count(*) as total_images
FROM images
GROUP BY format;

-- Images per year
SELECT EXTRACT(YEAR FROM date_created) as year, count(*)
FROM images
WHERE date_created IS NOT NULL
GROUP BY year
ORDER BY year DESC;
```

---

## 5. Combined Queries

### Caption + GPS + Date Range
Find beach photos within 100km of San Francisco taken between 2020-2024:

```sql
SELECT 
    file_name, 
    caption, 
    date_created,
    6371 * acos(
        cos(radians(37.7749)) * cos(radians(gps_latitude)) *
        cos(radians(gps_longitude) - radians(-122.4194)) +
        sin(radians(37.7749)) * sin(radians(gps_latitude))
    ) AS distance_km
FROM images
WHERE (caption ILIKE '%beach%' OR caption ILIKE '%ocean%')
  AND gps_latitude IS NOT NULL
  AND gps_longitude IS NOT NULL
  AND date_created BETWEEN '2020-01-01' AND '2024-12-31'
HAVING distance_km < 100
ORDER BY date_created DESC
LIMIT 20;
```

### Tag + Format + Size
```sql
SELECT file_name, format, file_size / 1024 / 1024 AS size_mb, tags
FROM images
WHERE 'portrait' = ANY(tags)
  AND format IN ('JPEG', 'PNG')
  AND file_size < 1048576  -- Less than 1MB
ORDER BY file_size DESC;
```

---

## 6. Using with PostGIS (If Installed)

If you have PostGIS installed (`CREATE EXTENSION postgis;`), you can use more efficient spatial queries:

### Enable PostGIS
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

### Proximity Search with PostGIS
```sql
SELECT 
    file_name, 
    caption,
    ST_Distance(
        ST_MakePoint(gps_longitude, gps_latitude)::geography,
        ST_MakePoint(-122.4194, 37.7749)::geography
    ) / 1000 AS distance_km
FROM images
WHERE gps_latitude IS NOT NULL 
  AND gps_longitude IS NOT NULL
  AND ST_DWithin(
        ST_MakePoint(gps_longitude, gps_latitude)::geography,
        ST_MakePoint(-122.4194, 37.7749)::geography,
        50000  -- 50km in meters
      )
ORDER BY distance_km ASC
LIMIT 20;
```

### Create Spatial Index
```sql
CREATE INDEX idx_images_gps ON images 
USING GIST (ST_MakePoint(gps_longitude, gps_latitude));
```

---

## Tips

1. **Performance**: Add indexes on frequently queried columns:
   ```sql
   CREATE INDEX idx_caption ON images USING gin(to_tsvector('english', caption));
   CREATE INDEX idx_date_created ON images(date_created);
   CREATE INDEX idx_tags ON images USING gin(tags);
   ```

2. **Case-Insensitive Search**: Use `ILIKE` instead of `LIKE` for case-insensitive matching.

3. **NULL Handling**: Always check for `IS NOT NULL` on GPS coordinates before calculations.

4. **Vector Search**: Ensure your query embedding uses the same model and dimensions as stored embeddings.

5. **Distance Units**: Haversine formula returns kilometers; multiply by 0.621371 for miles.
