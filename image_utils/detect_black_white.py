#!/usr/bin/env python3
"""
Black and White Image Detection Utility

Detects whether images are black and white (grayscale) or color.
An image is considered black and white if all pixels have equal R, G, and B values.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


def is_black_and_white(image_path: str, tolerance: int = 0, sample_size: Optional[Tuple[int, int]] = None) -> bool:
    """
    Detect if an image is black and white (grayscale).
    
    An image is considered black and white if all pixels have equal R, G, and B values
    (within an optional tolerance for compression artifacts).
    
    Args:
        image_path: Path to the image file
        tolerance: Maximum difference allowed between RGB channels (default: 0).
                   Use tolerance > 0 for JPEG images that may have compression artifacts.
        sample_size: Optional tuple (width, height) to resize image for faster processing.
                     If None, processes full resolution image.
    
    Returns:
        True if the image is black and white, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Optionally resize for faster processing on large images
            if sample_size:
                img = img.resize(sample_size, Image.Resampling.LANCZOS)
            
            # Get pixel data
            pixels = list(img.getdata())
            
            # Check if all pixels have equal RGB values (within tolerance)
            for r, g, b in pixels:
                if abs(r - g) > tolerance or abs(r - b) > tolerance or abs(g - b) > tolerance:
                    return False
            
            return True
            
    except Exception as e:
        print(f"Warning: Could not analyze image {image_path}: {e}")
        return False


def is_black_and_white_fast(image_path: str, tolerance: int = 3) -> bool:
    """
    Fast detection of black and white images using numpy for better performance.
    
    This method converts the image to a numpy array and checks if RGB channels
    are equal across all pixels, with a default tolerance for JPEG compression artifacts.
    
    Args:
        image_path: Path to the image file
        tolerance: Maximum difference allowed between RGB channels (default: 3)
    
    Returns:
        True if the image is black and white, False otherwise
    """
    try:
        import numpy as np
        
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Get RGB channels
            r = img_array[:, :, 0]
            g = img_array[:, :, 1]
            b = img_array[:, :, 2]
            
            # Check if all channels are approximately equal
            rg_diff = np.abs(r.astype(np.int16) - g.astype(np.int16))
            rb_diff = np.abs(r.astype(np.int16) - b.astype(np.int16))
            gb_diff = np.abs(g.astype(np.int16) - b.astype(np.int16))
            
            # If any pixel exceeds tolerance, it's not black and white
            if np.any(rg_diff > tolerance) or np.any(rb_diff > tolerance) or np.any(gb_diff > tolerance):
                return False
            
            return True
            
    except ImportError:
        # Fallback to pure Python implementation if numpy not available
        print("Note: NumPy not available, using slower detection method")
        return is_black_and_white(image_path, tolerance=tolerance, sample_size=(800, 600))
    except Exception as e:
        print(f"Warning: Could not analyze image {image_path}: {e}")
        return False


def get_image_color_type(image_path: str, tolerance: int = 3) -> str:
    """
    Determine the color type of an image.
    
    Args:
        image_path: Path to the image file
        tolerance: Maximum difference allowed between RGB channels
    
    Returns:
        One of: 'grayscale', 'color', or 'unknown' (if error occurred)
    """
    try:
        with Image.open(image_path) as img:
            # Check if image mode is already grayscale
            if img.mode in ('L', 'LA'):  # L = grayscale, LA = grayscale with alpha
                return 'grayscale'
            
            # Convert to RGB and check pixel values
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Use numpy for fast detection if available
            try:
                import numpy as np
                img_array = np.array(img)
                
                if len(img_array.shape) == 2:
                    return 'grayscale'
                
                r = img_array[:, :, 0]
                g = img_array[:, :, 1]
                b = img_array[:, :, 2]
                
                rg_diff = np.abs(r.astype(np.int16) - g.astype(np.int16))
                rb_diff = np.abs(r.astype(np.int16) - b.astype(np.int16))
                gb_diff = np.abs(g.astype(np.int16) - b.astype(np.int16))
                
                if np.all(rg_diff <= tolerance) and np.all(rb_diff <= tolerance) and np.all(gb_diff <= tolerance):
                    return 'grayscale'
                else:
                    return 'color'
                    
            except ImportError:
                # Fallback without numpy
                if is_black_and_white(image_path, tolerance=tolerance, sample_size=(800, 600)):
                    return 'grayscale'
                else:
                    return 'color'
                    
    except Exception as e:
        print(f"Warning: Could not determine color type for {image_path}: {e}")
        return 'unknown'


def batch_detect_black_and_white(image_paths: list, use_fast_method: bool = True, 
                                  tolerance: int = 3, verbose: bool = True) -> dict:
    """
    Batch detect black and white images from a list of paths.
    
    Args:
        image_paths: List of image file paths
        use_fast_method: Use numpy-based fast detection (requires numpy)
        tolerance: Tolerance for RGB channel differences
        verbose: Print progress information
    
    Returns:
        Dictionary with results:
        {
            'grayscale': [list of grayscale image paths],
            'color': [list of color image paths],
            'errors': [list of paths that failed to process]
        }
    """
    results = {
        'grayscale': [],
        'color': [],
        'errors': []
    }
    
    total = len(image_paths)
    
    for i, path in enumerate(image_paths, 1):
        if verbose and i % 10 == 0:
            print(f"Processing {i}/{total}...")
        
        try:
            if use_fast_method:
                is_bw = is_black_and_white_fast(path, tolerance=tolerance)
            else:
                is_bw = is_black_and_white(path, tolerance=tolerance, sample_size=(800, 600))
            
            if is_bw:
                results['grayscale'].append(path)
            else:
                results['color'].append(path)
                
        except Exception as e:
            if verbose:
                print(f"Error processing {path}: {e}")
            results['errors'].append(path)
    
    if verbose:
        print(f"\nBatch detection complete:")
        print(f"  Grayscale: {len(results['grayscale'])}")
        print(f"  Color: {len(results['color'])}")
        print(f"  Errors: {len(results['errors'])}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect black and white images')
    parser.add_argument('image_paths', nargs='+', help='Image file paths to analyze')
    parser.add_argument('--tolerance', type=int, default=3, 
                       help='Tolerance for RGB channel differences (default: 3)')
    parser.add_argument('--fast', action='store_true', 
                       help='Use fast numpy-based detection')
    parser.add_argument('--batch', action='store_true',
                       help='Process as batch and show summary')
    
    args = parser.parse_args()
    
    if args.batch:
        results = batch_detect_black_and_white(
            args.image_paths, 
            use_fast_method=args.fast,
            tolerance=args.tolerance
        )
        print("\nGrayscale images:")
        for path in results['grayscale']:
            print(f"  {path}")
    else:
        for path in args.image_paths:
            is_bw = is_black_and_white_fast(path, tolerance=args.tolerance) if args.fast \
                    else is_black_and_white(path, tolerance=args.tolerance, sample_size=(800, 600))
            color_type = get_image_color_type(path, tolerance=args.tolerance)
            print(f"{path}: {'Grayscale' if is_bw else 'Color'} ({color_type})")
