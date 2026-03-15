# GPS Location and Photo Search Guide

This guide explains how to extract GPS coordinates from your photos (including HEIC files) and search for photos by location.

## Overview

The image utilities now support:
- **GPS extraction** from EXIF data in JPEG, PNG, and HEIC files
- **Location-based search** using GPS coordinates and radius
- **Hierarchical location tagging** (country, state, city) via reverse geocoding
- **Database storage** of GPS coordinates for fast spatial queries

## GPS Extraction

### Automatic GPS Extraction

When you run any of these commands, GPS coordinates are automatically extracted from images:

```bash
# Generate captions with GPS metadata
python generate_captions_local.py \
    --source ./photos \
    --output captions.csv \
    --model microsoft/Florence-2-base

# Organize photos (extracts GPS automatically)
python image_organizer.py organize \
    --source ./photos \
    --dest ./organized_photos \
    --postgres-db "postgresql://user:pass@localhost/image_archive"
```

The extracted GPS data includes:
- `gps_latitude`: Decimal latitude (e.g., 21.3099 for Hawaii)
- `gps_longitude`: Decimal longitude (e.g., -157.8581 for Hawaii)

### Supported Formats

GPS extraction works with:
- **JPEG/JPG** - Full EXIF support
- **HEIC/HEIF** - Apple's format (requires `pillow-heif`)
- **PNG** - If GPS data is present in EXIF

### CSV Output

The CSV file will include GPS columns:
```csv
file_path,file_name,caption,gps_latitude,gps_longitude,date_created,...
./photos/beach.jpg,beach.jpg,"A beautiful beach...",21.3099,-157.8581,2024:12:15...
```

## Finding Photos by Location

### Method 1: Search by Coordinates and Radius

If you know the coordinates of a location, search within a radius:

```bash
# Find photos within 50km of Honolulu, Hawaii
python image_database.py search-meta \
    --db "postgresql://user:pass@localhost/image_archive" \
    --gps-lat 21.3099 \
    --gps-lon -157.8581 \
    --radius-km 50
```

**Example coordinates:**
- Honolulu, Hawaii: 21.3099° N, -157.8581° W
- San Francisco, CA: 37.7749° N, -122.4194° W
- New York, NY: 40.7128° N, -74.0060° W
- Tokyo, Japan: 35.6762° N, 139.6503° E

### Method 2: Get Coordinates from Google Maps

1. Go to [Google Maps](https://maps.google.com)
2. Right-click on a location
3. Click the coordinates at the top of the popup
4. Use those coordinates in your search

### Method 3: Search Using Known Photo Locations

If you have one photo from a location and want to find others nearby:

```bash
# First, find the coordinates of your known photo
python image_database.py search-meta \
    --db "postgresql://user:pass@localhost/image_archive" \
    --has-caption | grep -A5 "your_photo_name.jpg"

# Then search around those coordinates
python image_database.py search-meta \
    --db "postgresql://user:pass@localhost/image_archive" \
    --gps-lat <latitude_from_result> \
    --gps-lon <longitude_from_result> \
    --radius-km 25
```

## Reverse Geocoding (Country, State, City)

To add location names (country, state, city) to your photos, you can use a reverse geocoding service. Here's a Python script to do this:

### Install Required Package

```bash
pip install geopy
```

### Reverse Geocoding Script

Create a file `add_location_tags.py`:

```python
#!/usr/bin/env python3
"""Add location tags (country, state, city) to images based on GPS coordinates"""

import argparse
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from image_database import ImageDatabase

def reverse_geocode(lat, lon, geolocator):
    """Get location name from coordinates"""
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
        if location:
            address = location.raw.get('address', {})
            return {
                'city': address.get('city') or address.get('town') or address.get('village'),
                'state': address.get('state') or address.get('region'),
                'country': address.get('country'),
                'country_code': address.get('country_code')
            }
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description='Add location tags to images')
    parser.add_argument('--db', required=True, help='PostgreSQL connection string')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    args = parser.parse_args()
    
    db = ImageDatabase(args.db)
    geolocator = Nominatim(user_agent="photo_manager", timeout=10)
    
    # Get images with GPS but no location tags
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, file_name, gps_latitude, gps_longitude, tags
                FROM images
                WHERE gps_latitude IS NOT NULL 
                  AND gps_longitude IS NOT NULL
                  AND (tags IS NULL OR NOT (tags && ARRAY['country:', 'state:', 'city:']))
                LIMIT %s
            """, (args.batch_size,))
            
            images = cur.fetchall()
            
    print(f"Processing {len(images)} images...")
    
    updates = []
    for img_id, file_name, lat, lon, existing_tags in images:
        location = reverse_geocode(lat, lon, geolocator)
        if location:
            tags = existing_tags or []
            if location.get('city') and not any('city:' in t for t in tags):
                tags.append(f"city:{location['city']}")
            if location.get('state') and not any('state:' in t for t in tags):
                tags.append(f"state:{location['state']}")
            if location.get('country') and not any('country:' in t for t in tags):
                tags.append(f"country:{location['country']}")
            
            updates.append((tags, img_id))
            print(f"{file_name}: {location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')}")
    
    # Batch update
    if updates:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                from psycopg2.extras import execute_batch
                execute_batch(cur, """
                    UPDATE images SET tags = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s
                """, updates)
                conn.commit()
    
    print(f"Updated {len(updates)} images with location tags")
    db.close()

if __name__ == '__main__':
    main()
```

### Usage

```bash
# Add location tags to all images with GPS data
python add_location_tags.py \
    --db "postgresql://user:pass@localhost/image_archive" \
    --batch-size 200

# Then search by location names
python image_database.py search-meta \
    --db "postgresql://user:pass@localhost/image_archive" \
    --tags country:USA state:Hawaii
```

## Example Workflows

### Workflow 1: Find All Hawaii Photos

```bash
# Step 1: Get coordinates for Hawaii (approximately)
HAWAII_LAT=21.3099
HAWAII_LON=-157.8581

# Step 2: Search within 100km radius
python image_database.py search-meta \
    --db "postgresql://user:pass@localhost/image_archive" \
    --gps-lat $HAWAII_LAT \
    --gps-lon $HAWAII_LON \
    --radius-km 100 \
    --limit 500
```

### Workflow 2: Organize Trip Photos by Location

```bash
# Step 1: Generate captions and extract GPS
python generate_captions_local.py \
    --source ./trip_photos \
    --db "postgresql://user:pass@localhost/image_archive" \
    --model microsoft/Florence-2-base

# Step 2: Add location tags
python add_location_tags.py \
    --db "postgresql://user:pass@localhost/image_archive"

# Step 3: Search by city
python image_database.py search-meta \
    --db "postgresql://user:pass@localhost/image_archive" \
    --tags city:Honolulu
```

### Workflow 3: Find Photos Near Multiple Locations

Create a script `search_multiple_locations.py`:

```python
#!/usr/bin/env python3
"""Search for photos near multiple locations"""

from image_database import ImageDatabase
import argparse

LOCATIONS = {
    'Hawaii': (21.3099, -157.8581),
    'California': (36.7783, -119.4179),
    'New_York': (40.7128, -74.0060),
    'Tokyo': (35.6762, 139.6503),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True)
    parser.add_argument('--radius-km', type=float, default=50)
    args = parser.parse_args()
    
    db = ImageDatabase(args.db)
    
    for location_name, (lat, lon) in LOCATIONS.items():
        print(f"\n=== {location_name} (within {args.radius_km}km) ===")
        results = db.search_by_metadata(
            gps_lat=lat,
            gps_lon=lon,
            radius_km=args.radius_km,
            limit=50
        )
        
        if results:
            print(f"Found {len(results)} photos")
            for img in results[:5]:
                print(f"  - {img['file_name']}")
                if img.get('gps_latitude'):
                    print(f"    Location: {img['gps_latitude']:.4f}, {img['gps_longitude']:.4f}")
        else:
            print("No photos found")
    
    db.close()

if __name__ == '__main__':
    main()
```

Usage:
```bash
python search_multiple_locations.py \
    --db "postgresql://user:pass@localhost/image_archive" \
    --radius-km 75
```

## Tips

1. **GPS Accuracy**: Consumer cameras and phones typically have GPS accuracy of 5-20 meters
2. **Privacy**: Consider removing GPS data before sharing photos publicly
3. **Storage**: GPS coordinates use minimal storage (two float values per image)
4. **Performance**: The database uses a composite index on (gps_latitude, gps_longitude) for fast location queries
5. **Missing GPS**: Not all photos have GPS data (older cameras, privacy settings, screenshots)

## Troubleshooting

### No GPS Data Extracted

Check if your images have GPS EXIF data:
```bash
# On Linux/Mac
exiftool -gps:all your_photo.jpg

# Or use Python
python -c "
from PIL import Image
img = Image.open('your_photo.jpg')
exif = img._getexif()
if exif:
    for tag, value in exif.items():
        if 'GPS' in str(tag) or tag > 34853:
            print(f'{tag}: {value}')
"
```

### Coordinates Seem Wrong

GPS coordinates in EXIF are stored as degrees/minutes/seconds. The utility automatically converts them to decimal degrees. If coordinates seem incorrect, check:
- Hemisphere indicators (N/S/E/W) are correctly applied
- The conversion formula handles fractional seconds properly

### Database Query Slow

For large databases (>100k images), ensure the location index exists:
```sql
CREATE INDEX IF NOT EXISTS idx_images_location 
ON images(gps_latitude, gps_longitude)
WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL;
```

## Additional Resources

- [EXIF GPS Specification](https://exiftool.org/TagNames/GPS.html)
- [Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula) for distance calculations
- [Nominatim](https://nominatim.org/) for reverse geocoding
- [Google Maps Coordinates](https://www.google.com/maps) to find coordinates
