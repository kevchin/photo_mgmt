#!/usr/bin/env python3
"""
HEIC Metadata Extractor Utility

A standalone utility to extract and display comprehensive metadata from HEIC files.
This is useful for testing and debugging HEIC files before ingestion.

Features:
- Extract EXIF data (camera info, settings, etc.)
- Extract GPS coordinates with map link
- Extract creation/modification dates
- Extract image dimensions and format info
- Output in human-readable or JSON format
- Process single files or entire directories

Usage:
    python heic_metadata_extractor.py <file.heic>
    python heic_metadata_extractor.py <file.heic> --json
    python heic_metadata_extractor.py <directory>
    python heic_metadata_extractor.py <directory> --json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pillow_heif import register_heif_opener


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
            # Handle both nested tuples ((d, m, s) where each is (num, den)) 
            # and flat tuples with IFDRational objects (d, m, s)
            if len(value) == 3:
                # Check if it's a flat tuple of IFDRational or float values
                d_val = value[0]
                m_val = value[1]
                s_val = value[2]
                
                # Convert IFDRational to float if needed
                if hasattr(d_val, 'numerator'):
                    d = float(d_val)
                elif isinstance(d_val, tuple):
                    d = float(d_val[0]) / float(d_val[1])
                else:
                    d = float(d_val)
                    
                if hasattr(m_val, 'numerator'):
                    m = float(m_val)
                elif isinstance(m_val, tuple):
                    m = float(m_val[0]) / float(m_val[1])
                else:
                    m = float(m_val)
                    
                if hasattr(s_val, 'numerator'):
                    s = float(s_val)
                elif isinstance(s_val, tuple):
                    s = float(s_val[0]) / float(s_val[1])
                else:
                    s = float(s_val)
                    
                return d + (m / 60.0) + (s / 3600.0)
            else:
                # Fallback for nested tuple format
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
    altitude = None
    altitude_ref = None
    
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
        elif tag_name == 'GPSAltitude':
            try:
                altitude = float(value[0][0]) / float(value[0][1])
            except (ZeroDivisionError, IndexError, TypeError):
                pass
        elif tag_name == 'GPSAltitudeRef':
            altitude_ref = str(value)
    
    # Apply reference direction (N/S, E/W)
    if latitude is not None and lat_ref == 'S':
        latitude = -latitude
    if longitude is not None and lon_ref == 'W':
        longitude = -longitude
    
    # Apply altitude reference (0=above sea level, 1=below)
    if altitude is not None and altitude_ref == '1':
        altitude = -altitude
    
    if latitude is not None:
        gps_data['latitude'] = round(latitude, 6)
    if longitude is not None:
        gps_data['longitude'] = round(longitude, 6)
    if altitude is not None:
        gps_data['altitude'] = round(altitude, 2)
    if lat_ref:
        gps_data['latitude_ref'] = lat_ref
    if lon_ref:
        gps_data['longitude_ref'] = lon_ref
    if altitude_ref:
        gps_data['altitude_ref'] = altitude_ref
    
    # Include other GPS info
    for key, value in gps_info.items():
        tag_name = GPSTAGS.get(key, key)
        if tag_name not in ['GPSLatitude', 'GPSLongitude', 'GPSLatitudeRef', 
                            'GPSLongitudeRef', 'GPSAltitude', 'GPSAltitudeRef']:
            gps_data[tag_name.lower()] = str(value)
    
    # Add Google Maps link if coordinates available
    if gps_data.get('latitude') and gps_data.get('longitude'):
        gps_data['google_maps_link'] = (
            f"https://www.google.com/maps?q={gps_data['latitude']},{gps_data['longitude']}"
        )
    
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


def _get_camera_info(exif_data: dict) -> Dict[str, Any]:
    """Extract camera information from EXIF data."""
    camera_info = {}
    
    camera_fields = {
        'Make': 'make',
        'Model': 'model',
        'LensModel': 'lens_model',
        'LensMake': 'lens_make',
    }
    
    for exif_tag, field_name in camera_fields.items():
        if exif_tag in exif_data:
            camera_info[field_name] = exif_data[exif_tag]
    
    # Combine make and model if both present
    if camera_info.get('make') and camera_info.get('model'):
        camera_info['camera'] = f"{camera_info['make']} {camera_info['model']}"
    
    return camera_info


def _get_photo_settings(exif_data: dict) -> Dict[str, Any]:
    """Extract photo settings from EXIF data."""
    settings = {}
    
    setting_fields = {
        'ExposureTime': 'exposure_time',
        'FNumber': 'f_number',
        'ISOSpeedRatings': 'iso',
        'FocalLength': 'focal_length',
        'Flash': 'flash',
        'WhiteBalance': 'white_balance',
        'ExposureProgram': 'exposure_program',
        'MeteringMode': 'metering_mode',
    }
    
    for exif_tag, field_name in setting_fields.items():
        if exif_tag in exif_data:
            settings[field_name] = exif_data[exif_tag]
    
    # Format exposure time as fraction
    if 'exposure_time' in settings:
        try:
            val = float(settings['exposure_time'])
            if val < 1:
                settings['exposure_time_formatted'] = f"1/{int(1/val)}"
            else:
                settings['exposure_time_formatted'] = str(val)
        except (ValueError, TypeError):
            pass
    
    return settings


def extract_heic_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a HEIC file.
    
    Args:
        filepath: Path to the HEIC file
        
    Returns:
        Dictionary containing extracted metadata
    """
    filepath = os.path.abspath(filepath)
    
    metadata = {
        'file': filepath,
        'filename': os.path.basename(filepath),
        'format': None,
        'width': None,
        'height': None,
        'mode': None,
        'file_size': None,
        'exif': {},
        'gps': {},
        'camera': {},
        'settings': {},
        'creation_date': None,
        'modification_date': None,
    }
    
    if not os.path.exists(filepath):
        return {'error': f'File not found: {filepath}'}
    
    ext = Path(filepath).suffix.lower()
    supported_extensions = {'.heic', '.heif', '.jpg', '.jpeg', '.png'}
    
    if ext not in supported_extensions:
        return {'error': f'Unsupported file type: {ext}. Supported: {supported_extensions}'}
    
    try:
        # Get file size
        stat = os.stat(filepath)
        metadata['file_size'] = stat.st_size
        metadata['file_size_formatted'] = f"{stat.st_size / (1024*1024):.2f} MB"
        metadata['modification_date'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        with Image.open(filepath) as img:
            metadata['format'] = img.format
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['mode'] = img.mode
            metadata['orientation'] = 'Portrait' if img.height > img.width else 'Landscape'
            
            # Try to get EXIF data using the modern getexif() method
            try:
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        
                        # Handle GPS data specially - access via GPS IFD
                        if tag_id == 0x8825:  # GPSInfo IFD tag
                            gps_info = exif.get_ifd(0x8825)
                            metadata['gps'] = _parse_gps_info(gps_info)
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
                    
                    # Extract camera info and settings
                    metadata['camera'] = _get_camera_info(metadata['exif'])
                    metadata['settings'] = _get_photo_settings(metadata['exif'])
                    
            except Exception as e:
                metadata['exif_error'] = f"EXIF extraction error: {str(e)}"
                    
    except Exception as e:
        metadata['error'] = f"Error processing file: {str(e)}"
    
    return metadata


def process_directory(directory: str) -> List[Dict[str, Any]]:
    """Process all supported image files in a directory."""
    results = []
    supported_extensions = {'.heic', '.heif', '.jpg', '.jpeg', '.png'}
    
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return [{'error': f'Directory not found: {directory}'}]
    
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            ext = Path(filename).suffix.lower()
            if ext in supported_extensions:
                filepath = os.path.join(root, filename)
                metadata = extract_heic_metadata(filepath)
                results.append(metadata)
    
    return results


def print_metadata_detailed(metadata: Dict[str, Any]) -> None:
    """Print detailed metadata in a human-readable format."""
    if 'error' in metadata and len(metadata) == 1:
        print(f"ERROR: {metadata['error']}")
        return
    
    print("\n" + "="*70)
    print(f"FILE: {metadata.get('filename', 'Unknown')}")
    print("="*70)
    
    # Basic info
    print("\n📁 BASIC INFORMATION:")
    print(f"   Full Path: {metadata.get('file', 'Unknown')}")
    print(f"   Format: {metadata.get('format', 'Unknown')}")
    print(f"   Dimensions: {metadata.get('width', '?')} x {metadata.get('height', '?')} pixels")
    print(f"   Orientation: {metadata.get('orientation', 'Unknown')}")
    print(f"   Color Mode: {metadata.get('mode', 'Unknown')}")
    print(f"   File Size: {metadata.get('file_size_formatted', 'Unknown')}")
    
    # Dates
    print("\n📅 DATES:")
    print(f"   Creation Date: {metadata.get('creation_date', 'Not available')}")
    print(f"   Modified Date: {metadata.get('modification_date', 'Not available')}")
    
    # Camera info
    if metadata.get('camera'):
        print("\n📷 CAMERA INFORMATION:")
        camera = metadata['camera']
        if camera.get('camera'):
            print(f"   Camera: {camera['camera']}")
        if camera.get('lens_model'):
            print(f"   Lens: {camera['lens_model']}")
    
    # Photo settings
    if metadata.get('settings'):
        print("\n⚙️ PHOTO SETTINGS:")
        settings = metadata['settings']
        if settings.get('exposure_time_formatted'):
            print(f"   Exposure: {settings['exposure_time_formatted']} sec")
        if settings.get('f_number'):
            print(f"   Aperture: f/{settings['f_number']}")
        if settings.get('iso'):
            print(f"   ISO: {settings['iso']}")
        if settings.get('focal_length'):
            print(f"   Focal Length: {settings['focal_length']} mm")
        if settings.get('flash'):
            print(f"   Flash: {settings['flash']}")
    
    # GPS info
    if metadata.get('gps'):
        print("\n🌍 GPS LOCATION:")
        gps = metadata['gps']
        if gps.get('latitude') and gps.get('longitude'):
            print(f"   Latitude: {gps['latitude']}")
            print(f"   Longitude: {gps['longitude']}")
            if gps.get('altitude'):
                print(f"   Altitude: {gps['altitude']} meters")
            if gps.get('google_maps_link'):
                print(f"   Google Maps: {gps['google_maps_link']}")
        else:
            print("   No GPS coordinates available")
    
    # Additional EXIF data
    if metadata.get('exif'):
        other_exif = {k: v for k, v in metadata['exif'].items() 
                     if k not in ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']}
        if other_exif:
            print("\n📋 OTHER EXIF DATA:")
            for key, value in sorted(other_exif.items())[:10]:  # Limit to first 10
                print(f"   {key}: {value}")
            if len(other_exif) > 10:
                print(f"   ... and {len(other_exif) - 10} more fields")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract metadata from HEIC files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.heic                    # Display metadata for single file
  %(prog)s photo.heic --json             # Output as JSON
  %(prog)s ./photos/                     # Process all images in directory
  %(prog)s ./photos/ --json              # Directory output as JSON
  %(prog)s photo.heic --summary          # Show brief summary only
        """
    )
    parser.add_argument('target', help='HEIC file or directory to process')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    parser.add_argument('--summary', action='store_true', help='Show brief summary only')
    
    args = parser.parse_args()
    target = args.target
    
    if not os.path.exists(target):
        print(f"Error: '{target}' does not exist")
        sys.exit(1)
    
    if os.path.isfile(target):
        metadata = extract_heic_metadata(target)
        
        if args.json:
            print(json.dumps(metadata, indent=2))
        elif args.summary:
            # Brief summary
            if 'error' in metadata:
                print(f"ERROR: {metadata['error']}")
            else:
                print(f"{metadata.get('filename', 'Unknown')}")
                print(f"  {metadata.get('width', '?')}x{metadata.get('height', '?')} | "
                      f"{metadata.get('format', '?')} | "
                      f"Created: {metadata.get('creation_date', 'N/A')}")
                if metadata.get('gps', {}).get('latitude'):
                    gps = metadata['gps']
                    print(f"  GPS: {gps['latitude']}, {gps['longitude']}")
        else:
            print_metadata_detailed(metadata)
            
    elif os.path.isdir(target):
        results = process_directory(target)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nFound {len(results)} image(s) in {target}\n")
            
            if args.summary:
                # Table-style summary
                print(f"{'Filename':<30} {'Size':<12} {'Created':<25} {'GPS':<10}")
                print("-" * 80)
                for meta in results:
                    if 'error' not in meta:
                        filename = meta.get('filename', 'Unknown')[:28]
                        size = meta.get('file_size_formatted', 'Unknown')
                        created = (meta.get('creation_date', 'N/A') or 'N/A')[:23]
                        has_gps = '✓' if meta.get('gps', {}).get('latitude') else ''
                        print(f"{filename:<30} {size:<12} {created:<25} {has_gps:<10}")
            else:
                # Detailed output for each file
                for i, metadata in enumerate(results):
                    if i > 0:
                        print("\n")
                    print_metadata_detailed(metadata)
    else:
        print(f"Error: '{target}' is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
