"""
EXIF metadata extraction for photos.
Extracts date, GPS location, B&W detection, and orientation.
"""
import os
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from PIL import Image, ExifTags
import numpy as np


class ExifReader:
    """Extract EXIF metadata from image files."""
    
    @staticmethod
    def read_exif(image_path: str) -> Dict[str, Any]:
        """
        Extract all available EXIF metadata from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted metadata
        """
        result = {
            'file_path': image_path,
            'file_name': os.path.basename(image_path),
            'directory_path': os.path.dirname(image_path),
            'file_size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else None,
            'capture_date': None,
            'year': None,
            'month': None,
            'day': None,
            'latitude': None,
            'longitude': None,
            'altitude': None,
            'is_black_white': None,
            'orientation': 1,
            'camera_make': None,
            'camera_model': None,
            'lens_model': None,
            'focal_length': None,
            'aperture': None,
            'shutter_speed': None,
            'iso': None,
            'flash': None,
        }
        
        try:
            with Image.open(image_path) as img:
                # Get basic image info
                result['width'] = img.width
                result['height'] = img.height
                result['mode'] = img.mode
                result['format'] = img.format
                
                # Check if black and white
                result['is_black_white'] = ExifReader._detect_black_and_white(img)
                
                # Extract EXIF data
                exif_data = img._getexif()
                
                if exif_data:
                    # Parse EXIF tags
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        
                        # Date/Time
                        if tag == "DateTimeOriginal" or tag == "DateTime":
                            result['capture_date'] = ExifReader._parse_datetime(value)
                            if result['capture_date']:
                                result['year'] = result['capture_date'].year
                                result['month'] = result['capture_date'].month
                                result['day'] = result['capture_date'].day
                        
                        # GPS coordinates
                        elif tag == "GPSInfo":
                            gps_info = ExifReader._parse_gps_info(value)
                            result.update(gps_info)
                        
                        # Orientation
                        elif tag == "Orientation":
                            result['orientation'] = value
                        
                        # Camera info
                        elif tag == "Make":
                            result['camera_make'] = value
                        elif tag == "Model":
                            result['camera_model'] = value
                        elif tag == "LensModel":
                            result['lens_model'] = value
                        
                        # Exposure settings
                        elif tag == "FocalLength":
                            result['focal_length'] = float(value) if isinstance(value, tuple) else value
                        elif tag == "FNumber":
                            result['aperture'] = float(value) if isinstance(value, tuple) else value
                        elif tag == "ExposureTime":
                            result['shutter_speed'] = float(value) if isinstance(value, tuple) else value
                        elif tag == "ISOSpeedRatings":
                            result['iso'] = value
                        elif tag == "Flash":
                            result['flash'] = value
                
                # If no EXIF date, try file modification time
                if not result['capture_date']:
                    file_stat = os.stat(image_path)
                    # Try creation time first (Windows), fall back to modification time
                    try:
                        ctime = file_stat.st_birthtime  # macOS/Windows
                        result['capture_date'] = datetime.fromtimestamp(ctime)
                    except AttributeError:
                        mtime = file_stat.st_mtime  # Linux
                        result['capture_date'] = datetime.fromtimestamp(mtime)
                    
                    if result['capture_date']:
                        result['year'] = result['capture_date'].year
                        result['month'] = result['capture_date'].month
                        result['day'] = result['capture_date'].day
                        
        except Exception as e:
            print(f"Error reading EXIF from {image_path}: {e}")
        
        return result
    
    @staticmethod
    def _parse_datetime(date_str: str) -> Optional[datetime]:
        """Parse EXIF datetime string to datetime object."""
        if not date_str:
            return None
        
        # Common EXIF date formats
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y:%m:%d",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def _parse_gps_info(gps_info: dict) -> Dict[str, Optional[float]]:
        """Parse GPSInfo EXIF tag to latitude, longitude, altitude."""
        result = {
            'latitude': None,
            'longitude': None,
            'altitude': None
        }
        
        if not gps_info:
            return result
        
        # GPS latitude
        if 2 in gps_info and 1 in gps_info:  # Lat ref and lat
            lat_ref = gps_info[1]
            lat = gps_info[2]
            if isinstance(lat, tuple) and len(lat) == 3:
                degrees = float(lat[0])
                minutes = float(lat[1])
                seconds = float(lat[2]) / 60.0
                decimal_degrees = degrees + minutes + seconds
                if lat_ref == 'S':
                    decimal_degrees *= -1
                result['latitude'] = decimal_degrees
        
        # GPS longitude
        if 3 in gps_info and 4 in gps_info:  # Lon ref and lon
            lon_ref = gps_info[3]
            lon = gps_info[4]
            if isinstance(lon, tuple) and len(lon) == 3:
                degrees = float(lon[0])
                minutes = float(lon[1])
                seconds = float(lon[2]) / 60.0
                decimal_degrees = degrees + minutes + seconds
                if lon_ref == 'W':
                    decimal_degrees *= -1
                result['longitude'] = decimal_degrees
        
        # GPS altitude
        if 6 in gps_info:
            alt = gps_info[6]
            if isinstance(alt, tuple) and len(alt) == 2:
                result['altitude'] = float(alt[0]) / float(alt[1])
            elif isinstance(alt, (int, float)):
                result['altitude'] = float(alt)
        
        return result
    
    @staticmethod
    def _detect_black_and_white(img: Image.Image) -> bool:
        """Detect if an image is black and white."""
        # Convert to numpy array
        img_array = np.array(img)
        
        # If already grayscale
        if img.mode == 'L':
            return True
        
        # If RGB, check if all channels are equal
        if img.mode == 'RGB':
            # Sample pixels to avoid processing entire large image
            h, w = img_array.shape[:2]
            sample_step = max(1, min(h, w) // 50)  # Sample ~50x50 pixels
            
            sampled = img_array[::sample_step, ::sample_step]
            
            # Check if R=G=B for all sampled pixels
            if sampled.ndim == 3 and sampled.shape[2] == 3:
                r, g, b = sampled[:, :, 0], sampled[:, :, 1], sampled[:, :, 2]
                is_bw = np.all((r == g) & (g == b))
                return is_bw
        
        # Check color saturation as alternative method
        if img.mode == 'RGB':
            img_hsv = img.convert('HSV')
            hsv_array = np.array(img_hsv)
            saturation = hsv_array[:, :, 1]
            
            # If average saturation is very low, consider it B&W
            avg_saturation = np.mean(saturation)
            if avg_saturation < 10:  # Threshold for B&W
                return True
        
        return False
    
    @staticmethod
    def extract_from_directory(directory: str, recursive: bool = True) -> list:
        """
        Extract EXIF metadata from all images in a directory.
        
        Args:
            directory: Path to directory containing images
            recursive: Whether to search subdirectories
            
        Returns:
            List of metadata dictionaries
        """
        results = []
        
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp', '.heic', '.heif'}
        
        if recursive:
            walker = os.walk(directory)
        else:
            walker = [(directory, [], os.listdir(directory))]
        
        for root, dirs, files in walker:
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}")
                    metadata = ExifReader.read_exif(file_path)
                    results.append(metadata)
        
        return results


def test_exif_reader():
    """Test EXIF reader with a sample image."""
    # Create a test image
    test_image = Image.new('RGB', (800, 600), color='red')
    test_path = "/tmp/test_photo.jpg"
    test_image.save(test_path)
    
    metadata = ExifReader.read_exif(test_path)
    
    print("Extracted metadata:")
    for key, value in metadata.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    return metadata


if __name__ == "__main__":
    test_exif_reader()
