from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pillow_heif import register_heif_opener
import sys

# Enable HEIF support in Pillow
register_heif_opener()

def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    if ref in ['S', 'W']:
        return -(degrees + minutes + seconds)
    return degrees + minutes + seconds

def get_heic_gps_data(file_path):
    with Image.open(file_path) as img:
        exif_data = img.getexif()
        if not exif_data:
            return None
        
        # GPS info is typically stored under tag ID 34853
        gps_info = exif_data.get_ifd(0x8825) 
        
        if gps_info:
            # Decode the tags
            gps_data = {GPSTAGS.get(t, t): v for t, v in gps_info.items()}
            
            lat = get_decimal_from_dms(gps_data['GPSLatitude'], gps_data['GPSLatitudeRef'])
            lon = get_decimal_from_dms(gps_data['GPSLongitude'], gps_data['GPSLongitudeRef'])
            
            return lat, lon
    return None

# Usage
print(f"Script name: {sys.argv[0]}")
if len(sys.argv) > 1:
    coords = get_heic_gps_data(sys.argv[1])
    print(f"First argument: {sys.argv[1]}")
    print(f"Latitude: {coords[0]}, Longitude: {coords[1]}" if coords else "No GPS data found.")

