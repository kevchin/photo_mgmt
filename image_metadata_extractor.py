#!/usr/bin/env python3
"""
Image Metadata Extractor

Extracts metadata (GPS coordinates, creation date, camera info, etc.) from 
HEIC, PNG, and JPEG image files.

Usage:
    python image_metadata_extractor.py <image_file>
    python image_metadata_extractor.py <directory>  # Process all images in directory
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pillow_heif import register_heif_opener
import exifread


# Register HEIF opener to enable Pillow to read HEIC files
register_heif_opener()


def _parse_gps_info(gps_info: dict) -> Dict[str, Any]:
    """Convert GPS EXIF data to decimal latitude/longitude."""
    gps_data = {}
    
    if not gps_info:
        return gps_data
    
    def convert_to_degrees(value):
        """Convert GPS coordinates to decimal degrees."""
        try:
            d = float(value[0][0]) / float(value[0][1])
            m = float(value[1][0]) / float(value[1][1])
            s = float(value[2][0]) / float(value[2][1])
            return d + (m / 60.0) + (s / 3600.0)
        except (ZeroDivisionError, IndexError, TypeError):
            return None
    
    latitude = None
    longitude = None
    lat_ref = None
    lon_ref = None
    
    for key, value in gps_info.items():
        tag_name = GPSTAGS.get(key, key)
        
        if tag_name == 'GPSLatitude':
            latitude = convert_to_degrees(value)
        elif tag_name == 'GPSLongitude':
            longitude = convert_to_degrees(value)
        elif tag_name == 'GPSLatitudeRef':
            lat_ref = str(value)
        elif tag_name == 'GPSLongitudeRef':
            lon_ref = str(value)
    
    # Apply reference direction (N/S, E/W)
    if latitude is not None and lat_ref == 'S':
        latitude = -latitude
    if longitude is not None and lon_ref == 'W':
        longitude = -longitude
    
    if latitude is not None:
        gps_data['latitude'] = round(latitude, 6)
    if longitude is not None:
        gps_data['longitude'] = round(longitude, 6)
    if lat_ref:
        gps_data['latitude_ref'] = lat_ref
    if lon_ref:
        gps_data['longitude_ref'] = lon_ref
    
    # Include other GPS info
    for key, value in gps_info.items():
        tag_name = GPSTAGS.get(key, key)
        if tag_name not in ['GPSLatitude', 'GPSLongitude', 'GPSLatitudeRef', 'GPSLongitudeRef']:
            gps_data[tag_name.lower()] = str(value)
    
    return gps_data


def _format_datetime(dt_str: str) -> Optional[str]:
    """Try to parse and format various datetime strings."""
    formats = [
        '%Y:%m:%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y:%m:%d',
        '%Y-%m-%d',
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str.strip(), fmt)
            return dt.isoformat()
        except ValueError:
            continue
    return dt_str


def extract_metadata_pillow(filepath: str) -> Dict[str, Any]:
    """Extract metadata using Pillow library."""
    metadata = {
        'file': filepath,
        'format': None,
        'width': None,
        'height': None,
        'mode': None,
        'exif': {},
        'gps': {},
        'creation_date': None,
        'modification_date': None,
    }
    
    try:
        with Image.open(filepath) as img:
            metadata['format'] = img.format
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['mode'] = img.mode
            
            # Get file timestamps
            stat = os.stat(filepath)
            metadata['modification_date'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            metadata['creation_date'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            
            # Extract EXIF data (not available for all formats like HEIC)
            if hasattr(img, '_getexif'):
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        
                        # Handle GPS data specially
                        if tag_name == 'GPSInfo':
                            metadata['gps'] = _parse_gps_info(value)
                        # Handle datetime fields
                        elif tag_name in ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']:
                            formatted = _format_datetime(str(value))
                            if tag_name == 'DateTimeOriginal' and not metadata.get('creation_date'):
                                metadata['creation_date'] = formatted
                            metadata['exif'][tag_name] = formatted
                        else:
                            # Convert bytes to string if needed
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8', errors='ignore')
                                except Exception:
                                    value = str(value)
                            metadata['exif'][tag_name] = str(value)
                        
    except Exception as e:
        metadata['error'] = f"Pillow error: {str(e)}"
    
    return metadata


def extract_metadata_exifread(filepath: str) -> Dict[str, Any]:
    """Extract metadata using exifread library (alternative method)."""
    metadata = {
        'file': filepath,
        'exif_tags': {},
        'gps': {},
    }
    
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f, stop_tag='UNDEF', details=False)
            
            for tag, value in tags.items():
                tag_str = str(tag)
                value_str = str(value)
                
                # Handle GPS data
                if tag_str.startswith('GPS '):
                    gps_tag = tag_str.replace('GPS ', '').lower()
                    metadata['gps'][gps_tag] = value_str
                else:
                    metadata['exif_tags'][tag_str] = value_str
                    
    except Exception as e:
        metadata['error'] = f"ExifRead error: {str(e)}"
    
    return metadata


def extract_metadata(filepath: str, use_exifread: bool = False) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from an image file.
    
    Args:
        filepath: Path to the image file
        use_exifread: If True, also use exifread library for additional data
        
    Returns:
        Dictionary containing extracted metadata
    """
    filepath = os.path.abspath(filepath)
    
    if not os.path.exists(filepath):
        return {'error': f'File not found: {filepath}'}
    
    supported_extensions = {'.heic', '.heif', '.jpg', '.jpeg', '.png'}
    ext = Path(filepath).suffix.lower()
    
    if ext not in supported_extensions:
        return {'error': f'Unsupported file type: {ext}. Supported: {supported_extensions}'}
    
    # Primary extraction using Pillow
    metadata = extract_metadata_pillow(filepath)
    
    # Optional secondary extraction using exifread
    if use_exifread and 'error' not in metadata:
        try:
            exifread_data = extract_metadata_exifread(filepath)
            if 'exif_tags' in exifread_data:
                # Merge exifread data, preferring Pillow's formatted data
                for key, value in exifread_data['exif_tags'].items():
                    if key not in metadata['exif']:
                        metadata['exif'][key] = value
            if 'gps' in exifread_data and exifread_data['gps']:
                # Merge GPS data if Pillow didn't capture it
                if not metadata.get('gps'):
                    metadata['gps'] = exifread_data['gps']
        except Exception:
            pass  # Silently ignore exifread errors if Pillow succeeded
    
    return metadata


def process_directory(directory: str) -> list:
    """Process all supported image files in a directory."""
    results = []
    supported_extensions = {'.heic', '.heif', '.jpg', '.jpeg', '.png'}
    
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return [{'error': f'Directory not found: {directory}'}]
    
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in supported_extensions:
                filepath = os.path.join(root, filename)
                metadata = extract_metadata(filepath)
                results.append(metadata)
    
    return results


def print_metadata(metadata: Dict[str, Any], indent: int = 0) -> None:
    """Pretty print metadata to console."""
    prefix = "  " * indent
    
    if 'error' in metadata and len(metadata) == 1:
        print(f"{prefix}ERROR: {metadata['error']}")
        return
    
    for key, value in metadata.items():
        if key == 'exif' and isinstance(value, dict):
            print(f"{prefix}{key}:")
            for exif_key, exif_val in sorted(value.items()):
                print(f"{prefix}  {exif_key}: {exif_val}")
        elif key == 'gps' and isinstance(value, dict) and value:
            print(f"{prefix}{key}:")
            for gps_key, gps_val in sorted(value.items()):
                print(f"{prefix}  {gps_key}: {gps_val}")
        elif isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_metadata(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample usage:")
        print("  python image_metadata_extractor.py photo.jpg")
        print("  python image_metadata_extractor.py ./photos/")
        print("  python image_metadata_extractor.py photo.heic --json")
        sys.exit(1)
    
    target = sys.argv[1]
    output_json = '--json' in sys.argv
    
    if os.path.isfile(target):
        metadata = extract_metadata(target)
        if output_json:
            print(json.dumps(metadata, indent=2))
        else:
            print(f"\nMetadata for: {target}\n{'='*50}")
            print_metadata(metadata)
    elif os.path.isdir(target):
        results = process_directory(target)
        if output_json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nFound {len(results)} image(s) in {target}\n{'='*50}")
            for i, metadata in enumerate(results):
                if i > 0:
                    print("\n" + "-"*50 + "\n")
                print(f"File: {metadata.get('file', 'Unknown')}")
                # Print summary only for directory mode
                if 'error' in metadata:
                    print(f"  ERROR: {metadata['error']}")
                else:
                    print(f"  Format: {metadata.get('format', 'Unknown')}")
                    print(f"  Size: {metadata.get('width', '?')}x{metadata.get('height', '?')}")
                    if metadata.get('creation_date'):
                        print(f"  Created: {metadata['creation_date']}")
                    if metadata.get('gps', {}).get('latitude'):
                        gps = metadata['gps']
                        print(f"  GPS: {gps.get('latitude')}, {gps.get('longitude')}")
    else:
        print(f"Error: '{target}' is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
