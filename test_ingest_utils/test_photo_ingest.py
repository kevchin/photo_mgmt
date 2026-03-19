#!/usr/bin/env python3
"""
Photo Ingestion Test Utility

This utility tests a single photo or directory of photos to determine:
1. If the image is a duplicate compared to an archive directory
2. The target directory it would be moved to (YYYY/MM/DD format)
3. The caption it would generate
4. If it is a black and white photo
5. If it needs rotation/orientation correction

This is a DRY-RUN tool - it does NOT actually move, copy, or modify any files.
Use this to test the ingestion process before actually ingesting photos.

Usage:
    # Test a single photo
    python test_photo_ingest.py /path/to/photo.jpg --archive /path/to/archive
    
    # Test a directory of photos
    python test_photo_ingest.py /path/to/photos/ --archive /path/to/archive --batch
    
    # With caption generation (requires Florence-2 model)
    python test_photo_ingest.py /path/to/photo.jpg --archive /path/to/archive --generate-caption
    
    # Save results to JSON
    python test_photo_ingest.py /path/to/photo.jpg --archive /path/to/archive --output results.json
"""

# Set environment variables BEFORE importing numpy/scipy to avoid OpenBLAS threading issues
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

# Import existing utilities
try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    HAS_PILLOW = True
except ImportError:
    print("Warning: Pillow not installed. Install with: pip install pillow")
    HAS_PILLOW = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False
    print("Note: pillow-heif not installed. HEIC/HEIF support disabled.")

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Note: ImageHash not installed. Perceptual hashing disabled.")

# Import our utility modules
try:
    from detect_black_white import get_image_color_type, is_black_and_white_fast
    BW_DETECTION_AVAILABLE = True
except ImportError:
    BW_DETECTION_AVAILABLE = False
    print("Note: detect_black_white module not found. B&W detection disabled.")

try:
    from image_orientation import get_exif_rotation, prepare_image_for_processing
    ORIENTATION_AVAILABLE = True
except ImportError:
    ORIENTATION_AVAILABLE = False
    print("Note: image_orientation module not found. Orientation detection disabled.")

try:
    from generate_captions_local import FlorenceCaptionGenerator, EmbeddingGenerator
    CAPTION_GENERATION_AVAILABLE = True
except ImportError:
    CAPTION_GENERATION_AVAILABLE = False
    print("Note: generate_captions_local module not found. Caption generation disabled.")


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tiff', '.tif'}


@dataclass
class PhotoTestResult:
    """Stores test results for a single photo"""
    file_path: str
    file_name: str
    file_size: int
    checksum_md5: str
    checksum_sha256: str
    
    # Duplicate detection
    is_duplicate: bool
    duplicate_of: Optional[str]  # Path to duplicate in archive if found
    similarity_score: Optional[int]  # For near-duplicates
    
    # Date and organization
    exif_date: Optional[str]
    parsed_date: Optional[str]
    year: Optional[int]
    month: Optional[int]
    day: Optional[int]
    target_directory: Optional[str]  # YYYY/MM/DD format
    
    # Image properties
    dimensions: Tuple[int, int]
    format: str
    is_black_and_white: Optional[bool]
    color_type: Optional[str]  # 'grayscale', 'color', or 'unknown'
    
    # Orientation
    needs_rotation: bool
    rotation_angle: int
    orientation_status: str
    
    # Caption
    generated_caption: Optional[str]
    caption_model: Optional[str]
    
    # Summary
    should_ingest: bool  # True if not a duplicate
    ingest_action: str  # 'ingest', 'skip_duplicate', 'error'
    notes: List[str]
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['dimensions'] = list(d['dimensions']) if d['dimensions'] else None
        return d


def calculate_file_checksums(file_path: Path) -> Tuple[str, str]:
    """Calculate MD5 and SHA256 checksums of a file"""
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    
    return md5_hash.hexdigest(), sha256_hash.hexdigest()


def calculate_perceptual_hash(file_path: Path) -> Optional[str]:
    """Calculate perceptual hash for image comparison"""
    if not HAS_IMAGEHASH or not HAS_PILLOW:
        return None
    
    try:
        with Image.open(file_path) as img:
            # Convert to grayscale and resize to small size for efficient hashing
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Resize to a small size before hashing (much faster and less memory)
            img_small = img.resize((32, 32), Image.Resampling.LANCZOS)
            return str(imagehash.phash(img_small))
    except Exception as e:
        print(f"  Warning: Could not calculate perceptual hash: {e}")
        return None


def parse_exif_date(date_str: str) -> Optional[datetime]:
    """Parse EXIF date string to datetime"""
    if not date_str:
        return None
    
    for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', 
                '%Y:%m:%d', '%Y-%m-%d']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def extract_metadata(file_path: Path) -> dict:
    """Extract EXIF and other metadata from image"""
    metadata = {
        'exif_date': None, 
        'parsed_date': None, 
        'year': None, 
        'month': None, 
        'day': None,
        'camera_make': None,
        'camera_model': None,
        'gps_latitude': None,
        'gps_longitude': None
    }
    
    if not HAS_PILLOW:
        # Fallback to file modification time
        mtime = os.path.getmtime(file_path)
        dt = datetime.fromtimestamp(mtime)
        metadata['parsed_date'] = dt.isoformat()
        metadata['year'], metadata['month'], metadata['day'] = dt.year, dt.month, dt.day
        return metadata
    
    try:
        with Image.open(file_path) as img:
            metadata['format'] = img.format
            
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Date tags
                    if tag in ['DateTimeOriginal', 'DateTime'] and value:
                        if not metadata['exif_date']:
                            metadata['exif_date'] = value
                    
                    # Camera info
                    elif tag == 'Make':
                        metadata['camera_make'] = value
                    elif tag == 'Model':
                        metadata['camera_model'] = value
                    
                    # GPS info
                    elif tag == 'GPSInfo':
                        gps_info = value
                        if gps_info:
                            lat_ref = gps_info.get(1)
                            lat = gps_info.get(2)
                            lon_ref = gps_info.get(3)
                            lon = gps_info.get(4)
                            
                            if lat and lat_ref:
                                try:
                                    degrees = float(lat[0])
                                    minutes = float(lat[1])
                                    seconds = float(lat[2])
                                    latitude = degrees + (minutes / 60.0) + (seconds / 3600.0)
                                    if lat_ref == 'S':
                                        latitude = -latitude
                                    metadata['gps_latitude'] = latitude
                                except:
                                    pass
                            
                            if lon and lon_ref:
                                try:
                                    degrees = float(lon[0])
                                    minutes = float(lon[1])
                                    seconds = float(lon[2])
                                    longitude = degrees + (minutes / 60.0) + (seconds / 3600.0)
                                    if lon_ref == 'W':
                                        longitude = -longitude
                                    metadata['gps_longitude'] = longitude
                                except:
                                    pass
            
            # Parse the date
            if metadata['exif_date']:
                parsed = parse_exif_date(metadata['exif_date'])
                if parsed:
                    metadata['parsed_date'] = parsed.isoformat()
                    metadata['year'] = parsed.year
                    metadata['month'] = parsed.month
                    metadata['day'] = parsed.day
            
            # Fallback to file modification time if no EXIF date
            if not metadata['parsed_date']:
                mtime = os.path.getmtime(file_path)
                dt = datetime.fromtimestamp(mtime)
                metadata['parsed_date'] = dt.isoformat()
                metadata['year'], metadata['month'], metadata['day'] = dt.year, dt.month, dt.day
                
    except Exception as e:
        print(f"  Warning: Could not extract metadata: {e}")
        # Fallback to file modification time
        try:
            mtime = os.path.getmtime(file_path)
            dt = datetime.fromtimestamp(mtime)
            metadata['parsed_date'] = dt.isoformat()
            metadata['year'], metadata['month'], metadata['day'] = dt.year, dt.month, dt.day
        except:
            pass
    
    return metadata


def get_target_directory(year: Optional[int], month: Optional[int], 
                         day: Optional[int]) -> Optional[str]:
    """Generate target directory path in YYYY/MM/DD format"""
    if not year:
        return None
    
    if year and month and day:
        return f"{year}/{month:02d}/{day:02d}"
    elif year and month:
        return f"{year}/{month:02d}/unknown_day"
    elif year:
        return f"{year}/unknown_month"
    else:
        return "unknown_date"


def check_duplicate(photo_path: Path, archive_dir: Path, 
                    phash_threshold: int = 5) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Check if a photo is a duplicate of any photo in the archive.
    
    Returns:
        Tuple of (is_duplicate, duplicate_path, similarity_score)
        - is_duplicate: True if exact or near-duplicate found
        - duplicate_path: Path to the duplicate in archive (if found)
        - similarity_score: Hash distance for near-duplicates (lower = more similar)
    """
    if not archive_dir.exists():
        return False, None, None
    
    # Calculate checksums for the new photo
    md5_new, sha256_new = calculate_file_checksums(photo_path)
    phash_new = calculate_perceptual_hash(photo_path)
    
    # Scan archive for potential duplicates
    archive_files = [f for f in archive_dir.glob('**/*') 
                     if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    # First check for exact duplicates (by SHA256)
    for archive_file in archive_files:
        try:
            _, sha256_archive = calculate_file_checksums(archive_file)
            if sha256_new == sha256_archive:
                return True, str(archive_file), 0
        except:
            continue
    
    # Then check for near-duplicates (by perceptual hash)
    if phash_new and HAS_IMAGEHASH:
        try:
            hash1 = imagehash.hex_to_hash(phash_new)
            
            for archive_file in archive_files:
                try:
                    phash_archive = calculate_perceptual_hash(archive_file)
                    if phash_archive:
                        hash2 = imagehash.hex_to_hash(phash_archive)
                        distance = hash1 - hash2
                        
                        if distance <= phash_threshold:
                            return True, str(archive_file), distance
                except:
                    continue
        except:
            pass
    
    return False, None, None


def detect_orientation(photo_path: Path) -> Tuple[bool, int, str]:
    """
    Detect if a photo needs rotation.
    
    Returns:
        Tuple of (needs_rotation, rotation_angle, status_message)
    """
    if not HAS_PILLOW:
        return False, 0, "Orientation detection not available"
    
    try:
        with Image.open(photo_path) as img:
            exif_data = img._getexif()
            if exif_data is None:
                return False, 0, "No EXIF orientation data found"
            
            # EXIF orientation tag is 274
            orientation = exif_data.get(274)
            
            if orientation is None or orientation == 1:
                return False, 0, "No rotation needed (orientation is correct)"
            
            # Map EXIF orientation to rotation angle
            orientation_map = {
                1: 0,      # Normal
                2: 0,      # Mirrored (we don't handle mirroring)
                3: 180,    # Upside down
                4: 0,      # Mirrored + upside down
                5: 270,    # Mirrored + 90° CW
                6: 270,    # 90° CW
                7: 90,     # Mirrored + 90° CCW
                8: 90,     # 90° CCW
            }
            
            rotation_angle = orientation_map.get(orientation, 0)
            
            if rotation_angle == 0:
                return False, 0, "No rotation needed based on EXIF"
            else:
                return True, rotation_angle, f"Needs {rotation_angle}° rotation based on EXIF"
    except Exception as e:
        return False, 0, f"Error detecting orientation: {e}"


def detect_black_and_white(photo_path: Path) -> Tuple[Optional[bool], Optional[str]]:
    """
    Detect if a photo is black and white.
    
    Returns:
        Tuple of (is_black_and_white, color_type)
    """
    if not HAS_PILLOW:
        return None, None
    
    try:
        with Image.open(photo_path) as img:
            # Check if image mode is already grayscale
            if img.mode in ('L', 'LA'):  # L = grayscale, LA = grayscale with alpha
                return True, 'grayscale'
            
            # Convert to RGB and check a sample of pixels
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get pixel data - sample every 10th pixel for speed on large images
            width, height = img.size
            total_pixels = width * height
            
            # For large images, just check a sample
            if total_pixels > 10000:
                step = max(1, total_pixels // 10000)
                pixels = list(img.getdata())[::step]
            else:
                pixels = list(img.getdata())
            
            # Check if all sampled pixels have equal RGB values (with small tolerance)
            tolerance = 3
            for r, g, b in pixels:
                if abs(r - g) > tolerance or abs(r - b) > tolerance or abs(g - b) > tolerance:
                    return False, 'color'
            
            return True, 'grayscale'
    except Exception as e:
        print(f"  Warning: B&W detection error: {e}")
        return None, None


def generate_caption(photo_path: Path, caption_generator=None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a caption for the photo.
    
    Returns:
        Tuple of (caption, model_used)
    """
    if not CAPTION_GENERATION_AVAILABLE or caption_generator is None:
        return None, None
    
    try:
        caption = caption_generator.generate_detailed_caption(str(photo_path))
        if caption:
            return caption, "Florence-2"
        return None, "Florence-2"
    except Exception as e:
        print(f"  Warning: Caption generation error: {e}")
        return None, None


def test_photo(photo_path: Path, archive_dir: Path,
               generate_caption_flag: bool = False,
               caption_generator=None,
               phash_threshold: int = 5) -> PhotoTestResult:
    """
    Test a single photo and return comprehensive results.
    """
    notes = []
    
    # Basic file info
    file_name = photo_path.name
    file_size = photo_path.stat().st_size
    md5, sha256 = calculate_file_checksums(photo_path)
    
    # Extract metadata
    meta = extract_metadata(photo_path)
    target_dir = get_target_directory(meta['year'], meta['month'], meta['day'])
    
    # Get dimensions and format
    dimensions = (0, 0)
    img_format = 'UNKNOWN'
    if HAS_PILLOW:
        try:
            with Image.open(photo_path) as img:
                dimensions = img.size
                img_format = img.format or 'UNKNOWN'
        except:
            pass
    
    # Check for duplicates
    is_dup, dup_of, sim_score = check_duplicate(photo_path, archive_dir, phash_threshold)
    if is_dup:
        if sim_score == 0:
            notes.append(f"EXACT DUPLICATE of: {dup_of}")
        else:
            notes.append(f"NEAR DUPLICATE of: {dup_of} (similarity score: {sim_score})")
    
    # Detect orientation
    needs_rot, rot_angle, orient_status = detect_orientation(photo_path)
    if needs_rot:
        notes.append(orient_status)
    
    # Detect black and white
    is_bw, color_type = detect_black_and_white(photo_path)
    if is_bw:
        notes.append("Image is black and white (grayscale)")
    elif color_type == 'color':
        notes.append("Image is color")
    
    # Generate caption if requested
    caption = None
    caption_model = None
    if generate_caption_flag and CAPTION_GENERATION_AVAILABLE and caption_generator:
        print(f"  Generating caption...")
        caption, caption_model = generate_caption(photo_path, caption_generator)
        if caption:
            notes.append(f"Caption generated: {caption[:80]}..." if len(caption) > 80 else f"Caption: {caption}")
    
    # Determine if should ingest
    should_ingest = not is_dup
    if is_dup:
        ingest_action = 'skip_duplicate'
    elif not target_dir:
        ingest_action = 'error'
        notes.append("ERROR: Could not determine target directory (no date information)")
    else:
        ingest_action = 'ingest'
    
    return PhotoTestResult(
        file_path=str(photo_path.absolute()),
        file_name=file_name,
        file_size=file_size,
        checksum_md5=md5,
        checksum_sha256=sha256,
        is_duplicate=is_dup,
        duplicate_of=dup_of,
        similarity_score=sim_score,
        exif_date=meta['exif_date'],
        parsed_date=meta['parsed_date'],
        year=meta['year'],
        month=meta['month'],
        day=meta['day'],
        target_directory=target_dir,
        dimensions=dimensions,
        format=img_format,
        is_black_and_white=is_bw,
        color_type=color_type,
        needs_rotation=needs_rot,
        rotation_angle=rot_angle,
        orientation_status=orient_status,
        generated_caption=caption,
        caption_model=caption_model,
        should_ingest=should_ingest,
        ingest_action=ingest_action,
        notes=notes
    )


def print_result(result: PhotoTestResult, verbose: bool = True):
    """Print test results in a human-readable format"""
    print("\n" + "=" * 80)
    print(f"PHOTO TEST RESULTS: {result.file_name}")
    print("=" * 80)
    
    # Basic info
    print(f"\n📁 File: {result.file_path}")
    print(f"   Size: {result.file_size:,} bytes | Format: {result.format} | Dimensions: {result.dimensions[0]}x{result.dimensions[1]}")
    print(f"   MD5: {result.checksum_md5}")
    print(f"   SHA256: {result.checksum_sha256}")
    
    # Duplicate status
    print(f"\n🔄 DUPLICATE CHECK:")
    if result.is_duplicate:
        print(f"   ❌ IS DUPLICATE")
        if result.similarity_score == 0:
            print(f"   Exact duplicate of: {result.duplicate_of}")
        else:
            print(f"   Near duplicate of: {result.duplicate_of} (similarity: {result.similarity_score})")
    else:
        print(f"   ✅ NOT A DUPLICATE - Safe to ingest")
    
    # Target directory
    print(f"\n📂 ORGANIZATION:")
    if result.target_directory:
        print(f"   Target directory: {result.target_directory}")
        print(f"   Full path would be: <archive>/{result.target_directory}/{result.file_name}")
    else:
        print(f"   ⚠️  Could not determine target directory")
    
    # Date info
    if result.exif_date or result.parsed_date:
        date_str = result.exif_date or result.parsed_date
        print(f"   Date: {date_str}")
        if result.year and result.month and result.day:
            print(f"   Parsed: {result.year}-{result.month:02d}-{result.day:02d}")
    
    # Image properties
    print(f"\n🖼️  IMAGE PROPERTIES:")
    if result.color_type:
        bw_status = "Black & White" if result.is_black_and_white else "Color"
        print(f"   Type: {bw_status} ({result.color_type})")
    else:
        print(f"   Type: Unknown (B&W detection not available)")
    
    # Orientation
    print(f"\n🔄 ORIENTATION:")
    if result.needs_rotation:
        print(f"   ⚠️  NEEDS ROTATION: {result.rotation_angle}°")
        print(f"   Status: {result.orientation_status}")
    else:
        print(f"   ✅ No rotation needed")
        print(f"   Status: {result.orientation_status}")
    
    # Caption
    if result.generated_caption:
        print(f"\n📝 GENERATED CAPTION ({result.caption_model}):")
        print(f"   \"{result.generated_caption}\"")
    elif CAPTION_GENERATION_AVAILABLE:
        print(f"\n📝 CAPTION: Not generated (use --generate-caption to enable)")
    else:
        print(f"\n📝 CAPTION: Caption generation not available")
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    if result.should_ingest:
        print(f"✅ RECOMMENDATION: INGEST this photo")
        print(f"   Action: Move/copy to <archive>/{result.target_directory}/")
        if result.needs_rotation:
            print(f"   Note: Apply {result.rotation_angle}° rotation before/after ingest")
    else:
        print(f"❌ RECOMMENDATION: SKIP this photo")
        print(f"   Reason: Duplicate detected")
    
    if result.notes:
        print(f"\n📋 NOTES:")
        for note in result.notes:
            print(f"   • {note}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Test photo ingestion process (DRY-RUN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single photo
  python test_photo_ingest.py photo.jpg --archive /path/to/archive
  
  # Test with caption generation
  python test_photo_ingest.py photo.jpg --archive /path/to/archive --generate-caption
  
  # Test a directory of photos
  python test_photo_ingest.py /path/to/photos/ --archive /path/to/archive --batch
  
  # Save results to JSON
  python test_photo_ingest.py photo.jpg --archive /path/to/archive --output results.json
        """
    )
    
    parser.add_argument('photo_paths', nargs='+', 
                       help='Photo file(s) or directory to test')
    parser.add_argument('--archive', required=True,
                       help='Path to archive directory for duplicate checking')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of photos')
    parser.add_argument('--generate-caption', action='store_true',
                       help='Generate AI caption (requires Florence-2 model)')
    parser.add_argument('--caption-model', default='microsoft/Florence-2-base',
                       help='Caption model to use (default: microsoft/Florence-2-base)')
    parser.add_argument('--phash-threshold', type=int, default=5,
                       help='Perceptual hash threshold for near-duplicate detection (default: 5)')
    parser.add_argument('--output', 
                       help='Output file for JSON results')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed output (only show summary)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    
    args = parser.parse_args()
    
    # Validate archive directory
    archive_dir = Path(args.archive)
    if not archive_dir.exists():
        print(f"Error: Archive directory does not exist: {archive_dir}")
        sys.exit(1)
    
    # Collect photos to test
    photo_paths = []
    for path_str in args.photo_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Path does not exist: {path}")
            continue
        
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                photo_paths.append(path)
            else:
                print(f"Warning: Unsupported file type: {path}")
        elif path.is_dir():
            args.batch = True
            files = list(path.glob('**/*'))
            for f in files:
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    photo_paths.append(f)
            print(f"Found {len(photo_paths)} photos in {path}")
    
    if not photo_paths:
        print("Error: No valid photos to test")
        sys.exit(1)
    
    # Initialize caption generator if requested
    caption_generator = None
    if args.generate_caption:
        if not CAPTION_GENERATION_AVAILABLE:
            print("Error: Caption generation module not available")
            print("Install requirements: pip install transformers torch pillow")
            sys.exit(1)
        
        try:
            print(f"\nLoading caption model: {args.caption_model}")
            print("(This may take a few minutes on first run)")
            caption_generator = FlorenceCaptionGenerator(model_name=args.caption_model)
        except Exception as e:
            print(f"Error loading caption model: {e}")
            print("Continuing without caption generation...")
            args.generate_caption = False
    
    # Process photos
    results = []
    
    print(f"\n{'=' * 80}")
    print(f"PHOTO INGESTION TEST")
    print(f"{'=' * 80}")
    print(f"Archive directory: {archive_dir}")
    print(f"Photos to test: {len(photo_paths)}")
    print(f"Near-duplicate threshold: {args.phash_threshold}")
    if args.generate_caption:
        print(f"Caption generation: ENABLED ({args.caption_model})")
    else:
        print(f"Caption generation: DISABLED")
    print(f"{'=' * 80}")
    
    for i, photo_path in enumerate(photo_paths, 1):
        if len(photo_paths) > 1:
            print(f"\n[{i}/{len(photo_paths)}] Testing: {photo_path}")
        
        try:
            result = test_photo(
                photo_path, 
                archive_dir,
                generate_caption_flag=args.generate_caption,
                caption_generator=caption_generator,
                phash_threshold=args.phash_threshold
            )
            results.append(result)
            
            if not args.quiet:
                print_result(result, verbose=args.verbose)
                
        except Exception as e:
            print(f"Error testing {photo_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    if len(results) > 1:
        ingest_count = sum(1 for r in results if r.should_ingest)
        duplicate_count = sum(1 for r in results if r.is_duplicate)
        bw_count = sum(1 for r in results if r.is_black_and_white)
        rotate_count = sum(1 for r in results if r.needs_rotation)
        
        print(f"\n{'=' * 80}")
        print(f"BATCH SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total photos tested: {len(results)}")
        print(f"Ready to ingest: {ingest_count}")
        print(f"Duplicates (skip): {duplicate_count}")
        print(f"Black & white photos: {bw_count}")
        print(f"Need rotation: {rotate_count}")
        print(f"{'=' * 80}")
    
    # Save results to JSON if requested
    if args.output:
        output_data = {
            'test_timestamp': datetime.now().isoformat(),
            'archive_directory': str(archive_dir.absolute()),
            'phash_threshold': args.phash_threshold,
            'caption_generation': args.generate_caption,
            'caption_model': args.caption_model if args.generate_caption else None,
            'total_photos': len(results),
            'results': [r.to_dict() for r in results]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    # Return exit code based on results
    all_duplicates = all(r.is_duplicate for r in results)
    sys.exit(0 if not all_duplicates else 1)


if __name__ == '__main__':
    main()
