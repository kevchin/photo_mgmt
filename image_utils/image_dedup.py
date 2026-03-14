#!/usr/bin/env python3
"""
Image Deduplication Utility

This utility helps deduplicate images across folders using checksums.
It supports HEIC, JPG, and PNG formats.

Features:
- Calculate perceptual hashes (pHash) and file checksums
- Compare new images against an archive
- Identify duplicates and unique images
- Report missing photos that should be added to archive
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict

try:
    import imagehash
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_IMAGE_SUPPORT = True
except ImportError as e:
    print(f"Warning: Some image libraries not installed: {e}")
    print("Run: pip install Pillow pillow-heif ImageHash")
    HAS_IMAGE_SUPPORT = False

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}


@dataclass
class ImageInfo:
    """Stores metadata and identification info for an image"""
    file_path: str
    file_size: int
    checksum_md5: str
    checksum_sha256: str
    perceptual_hash: str
    avg_hash: str
    color_hash: str
    dimensions: Tuple[int, int]
    format: str
    exif_date: Optional[str]
    creation_date: Optional[str]
    gps_info: Optional[Dict]
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['dimensions'] = list(d['dimensions'])
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ImageInfo':
        if 'dimensions' in data and isinstance(data['dimensions'], list):
            data['dimensions'] = tuple(data['dimensions'])
        return cls(**data)


def calculate_file_checksums(file_path: Path) -> Tuple[str, str]:
    """Calculate MD5 and SHA256 checksums of a file"""
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    
    return md5_hash.hexdigest(), sha256_hash.hexdigest()


def calculate_perceptual_hashes(file_path: Path) -> Tuple[str, str, str]:
    """Calculate perceptual hashes for image comparison."""
    if not HAS_IMAGE_SUPPORT:
        return "", "", ""
    try:
        with Image.open(file_path) as img:
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            phash = str(imagehash.phash(img))
            ahash = str(imagehash.average_hash(img))
            dhash = str(imagehash.dhash(img))
            return phash, ahash, dhash
    except Exception as e:
        print(f"Warning: Could not calculate perceptual hash for {file_path}: {e}")
        return "", "", ""


def extract_metadata(file_path: Path) -> dict:
    """Extract EXIF and other metadata from image"""
    metadata = {'exif_date': None, 'creation_date': None, 'gps_info': None}
    
    if not HAS_IMAGE_SUPPORT:
        mtime = os.path.getmtime(file_path)
        metadata['creation_date'] = datetime.fromtimestamp(mtime).isoformat()
        return metadata
    
    try:
        with Image.open(file_path) as img:
            metadata['format'] = img.format
            
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                TAGS = {306: 'datetime', 36867: 'datetime_original', 34853: 'gps_info'}
                
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'datetime_original' and value:
                        metadata['exif_date'] = value
                    elif tag == 'datetime' and value and not metadata['exif_date']:
                        metadata['exif_date'] = value
                    if tag == 'gps_info' and value:
                        metadata['gps_info'] = str(value)
            
            if not metadata['exif_date']:
                mtime = os.path.getmtime(file_path)
                metadata['creation_date'] = datetime.fromtimestamp(mtime).isoformat()
                
    except Exception as e:
        print(f"Warning: Could not extract metadata from {file_path}: {e}")
        metadata['format'] = 'UNKNOWN'
    
    return metadata


def process_image(file_path: Path) -> Optional[ImageInfo]:
    """Process a single image file and extract all information"""
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None
    
    try:
        file_size = file_path.stat().st_size
        md5, sha256 = calculate_file_checksums(file_path)
        phash, ahash, dhash = calculate_perceptual_hashes(file_path)
        meta = extract_metadata(file_path)
        
        if HAS_IMAGE_SUPPORT:
            with Image.open(file_path) as img:
                dimensions = img.size
                img_format = img.format or 'UNKNOWN'
        else:
            dimensions = (0, 0)
            img_format = 'UNKNOWN'
        
        return ImageInfo(
            file_path=str(file_path.absolute()),
            file_size=file_size,
            checksum_md5=md5,
            checksum_sha256=sha256,
            perceptual_hash=phash,
            avg_hash=ahash,
            color_hash=dhash,
            dimensions=dimensions,
            format=img_format,
            exif_date=meta.get('exif_date'),
            creation_date=meta.get('creation_date'),
            gps_info=meta.get('gps_info')
        )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def scan_directory(directory: Path, recursive: bool = True) -> List[ImageInfo]:
    """Scan a directory for images and process them"""
    images = []
    files = list(directory.glob('**/*' if recursive else '*'))
    image_files = [f for f in files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    print(f"Found {len(image_files)} image files to process in {directory}")
    
    for i, file_path in enumerate(image_files, 1):
        img_info = process_image(file_path)
        if img_info:
            images.append(img_info)
        if i % 10 == 0:
            print(f"  Processed {i}/{len(image_files)} images...")
    
    print(f"Successfully processed {len(images)} images")
    return images


def find_duplicates_by_checksum(images: List[ImageInfo]) -> Dict[str, List[ImageInfo]]:
    """Find exact duplicates using SHA256 checksums"""
    checksum_groups: Dict[str, List[ImageInfo]] = {}
    for img in images:
        key = img.checksum_sha256
        if key not in checksum_groups:
            checksum_groups[key] = []
        checksum_groups[key].append(img)
    return {k: v for k, v in checksum_groups.items() if len(v) > 1}


def find_similar_by_phash(images: List[ImageInfo], threshold: int = 5) -> List[Tuple[ImageInfo, ImageInfo, int]]:
    """Find similar images using perceptual hash comparison."""
    if not HAS_IMAGE_SUPPORT:
        return []
    
    similar_pairs = []
    for i, img1 in enumerate(images):
        if not img1.perceptual_hash:
            continue
        for img2 in images[i+1:]:
            if not img2.perceptual_hash:
                continue
            try:
                hash1 = imagehash.hex_to_hash(img1.perceptual_hash)
                hash2 = imagehash.hex_to_hash(img2.perceptual_hash)
                distance = hash1 - hash2
                if distance <= threshold:
                    similar_pairs.append((img1, img2, distance))
            except:
                continue
    return similar_pairs


def compare_folders(archive_dir: Path, new_dir: Path) -> dict:
    """Compare a new folder against an archive"""
    print(f"\n=== Scanning Archive: {archive_dir} ===")
    archive_images = scan_directory(archive_dir)
    
    print(f"\n=== Scanning New Folder: {new_dir} ===")
    new_images = scan_directory(new_dir)
    
    archive_checksums = {img.checksum_sha256 for img in archive_images}
    
    exact_duplicates = [img for img in new_images if img.checksum_sha256 in archive_checksums]
    unique_images = [img for img in new_images if img.checksum_sha256 not in archive_checksums]
    
    all_images = archive_images + unique_images
    similar_pairs = find_similar_by_phash(all_images)
    
    new_file_paths = {img.file_path for img in new_images}
    similar_new_pairs = [(i1, i2, d) for i1, i2, d in similar_pairs 
                         if i1.file_path in new_file_paths or i2.file_path in new_file_paths]
    
    return {
        'archive_count': len(archive_images),
        'new_count': len(new_images),
        'exact_duplicates': exact_duplicates,
        'unique_images': unique_images,
        'similar_pairs': similar_new_pairs
    }


def generate_report(results: dict, output_file: Optional[Path] = None):
    """Generate a human-readable report"""
    lines = ["=" * 80, "IMAGE DEDUPLICATION REPORT", "=" * 80, ""]
    lines.append(f"Archive images: {results['archive_count']}")
    lines.append(f"New images scanned: {results['new_count']}")
    lines.append(f"Exact duplicates found: {len(results['exact_duplicates'])}")
    lines.append(f"Unique images to add: {len(results['unique_images'])}")
    lines.append(f"Similar images found: {len(results['similar_pairs'])}")
    lines.append("")
    
    if results['exact_duplicates']:
        lines.extend(["-" * 80, "EXACT DUPLICATES (already in archive):", "-" * 80])
        for img in results['exact_duplicates']:
            lines.append(f"  {img.file_path}")
            lines.append(f"    Size: {img.file_size:,} bytes | Dimensions: {img.dimensions}")
        lines.append("")
    
    if results['unique_images']:
        lines.extend(["-" * 80, "UNIQUE IMAGES (not in archive - candidates for adding):", "-" * 80])
        for img in results['unique_images']:
            lines.append(f"  {img.file_path}")
            lines.append(f"    Size: {img.file_size:,} bytes | Date: {img.exif_date or img.creation_date or 'Unknown'}")
        lines.append("")
    
    if results['similar_pairs']:
        lines.extend(["-" * 80, "SIMILAR IMAGES (possible near-duplicates):", "-" * 80])
        for img1, img2, dist in results['similar_pairs'][:20]:
            lines.append(f"  Pair (distance: {dist}):")
            lines.append(f"    - {img1.file_path}")
            lines.append(f"    - {img2.file_path}")
        if len(results['similar_pairs']) > 20:
            lines.append(f"  ... and {len(results['similar_pairs']) - 20} more similar pairs")
        lines.append("")
    
    report_text = "\n".join(lines)
    print(report_text)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")
    
    return report_text


def save_image_database(images: List[ImageInfo], db_file: Path):
    """Save image information to a JSON database file"""
    db_data = {
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'image_count': len(images),
        'images': [img.to_dict() for img in images]
    }
    with open(db_file, 'w') as f:
        json.dump(db_data, f, indent=2)
    print(f"Database saved to: {db_file} ({len(images)} images)")


def load_image_database(db_file: Path) -> List[ImageInfo]:
    """Load image information from a JSON database file"""
    with open(db_file, 'r') as f:
        db_data = json.load(f)
    images = [ImageInfo.from_dict(img) for img in db_data['images']]
    print(f"Loaded {len(images)} images from database")
    return images


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Deduplication Utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compare command
    cmp = subparsers.add_parser('compare', help='Compare new folder against archive')
    cmp.add_argument('--archive', required=True, help='Path to archive directory')
    cmp.add_argument('--new', required=True, help='Path to new folder')
    cmp.add_argument('--output', help='Output file for report')
    
    # Index command
    idx = subparsers.add_parser('index', help='Create index/database of images')
    idx.add_argument('--dir', required=True, help='Directory to index')
    idx.add_argument('--output', required=True, help='Output database file')
    
    # Find duplicates command
    fd = subparsers.add_parser('find-dups', help='Find duplicates within a folder')
    fd.add_argument('--dir', required=True, help='Directory to scan')
    fd.add_argument('--output', help='Output file for report')
    fd.add_argument('--similar-threshold', type=int, default=5)
    
    args = parser.parse_args()
    
    if args.command == 'compare':
        archive_dir, new_dir = Path(args.archive), Path(args.new)
        if not archive_dir.exists():
            print(f"Error: Archive directory does not exist: {archive_dir}")
            sys.exit(1)
        if not new_dir.exists():
            print(f"Error: New directory does not exist: {new_dir}")
            sys.exit(1)
        results = compare_folders(archive_dir, new_dir)
        generate_report(results, Path(args.output) if args.output else None)
    
    elif args.command == 'index':
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory does not exist: {dir_path}")
            sys.exit(1)
        images = scan_directory(dir_path)
        save_image_database(images, Path(args.output))
    
    elif args.command == 'find-dups':
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory does not exist: {dir_path}")
            sys.exit(1)
        print(f"Scanning {dir_path} for duplicates...")
        images = scan_directory(dir_path)
        duplicates = find_duplicates_by_checksum(images)
        similar = find_similar_by_phash(images, threshold=args.similar_threshold)
        report = {
            'archive_count': 0, 'new_count': len(images),
            'exact_duplicates': [img for group in duplicates.values() for img in group[1:]],
            'unique_images': [], 'similar_pairs': similar
        }
        generate_report(report, Path(args.output) if args.output else None)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
