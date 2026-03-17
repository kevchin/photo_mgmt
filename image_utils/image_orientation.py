#!/usr/bin/env python3
"""
Image Orientation Utility

Provides functions to automatically detect and correct image orientation
using EXIF data and AI-based scene analysis.

This module can be imported by other image processing tools to ensure
images are properly oriented before further processing.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


def get_exif_rotation(image_path: str) -> int:
    """
    Extract rotation angle from EXIF orientation tag.
    
    Returns:
        Rotation angle in degrees (0, 90, 180, or 270)
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is None:
                return 0
            
            # EXIF orientation tag is 274
            orientation = exif_data.get(274)
            
            if orientation is None:
                return 0
            
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
            
            return orientation_map.get(orientation, 0)
            
    except Exception as e:
        print(f"  Warning: Could not read EXIF data: {e}")
        return 0


def apply_orientation_from_exif(image_path: str, output_path: Optional[str] = None, 
                                 overwrite: bool = False) -> Tuple[bool, str]:
    """
    Apply EXIF orientation to an image and save it.
    
    Args:
        image_path: Path to input image
        output_path: Path for output image (default: modify in place if overwrite=True)
        overwrite: If True, overwrite original file
        
    Returns:
        Tuple of (was_rotated, output_path)
    """
    try:
        rotation_angle = get_exif_rotation(image_path)
        
        if rotation_angle == 0:
            return False, image_path
        
        with Image.open(image_path) as img:
            # Rotate the image
            # PIL rotates counter-clockwise, so we negate the angle
            rotated_img = img.rotate(-rotation_angle, expand=True)
            
            # Determine output path
            if output_path is None:
                if overwrite:
                    output_path = image_path
                else:
                    # Create output filename
                    base = Path(image_path)
                    output_path = str(base.parent / f"{base.stem}_oriented{base.suffix}")
            
            # Save the rotated image without the orientation tag
            # We need to preserve other EXIF data but remove orientation
            exif_data = img.info.get('exif', b'')
            
            # Convert back to RGB if necessary for saving
            if rotated_img.mode in ('RGBA', 'P'):
                rotated_img = rotated_img.convert('RGB')
            
            # Save with quality setting
            save_kwargs = {'quality': 95}
            if exif_data and output_path.lower().endswith(('.jpg', '.jpeg')):
                # Try to preserve EXIF but without orientation tag
                try:
                    from PIL.ExifTags import TAGS
                    # Reload to get fresh exif
                    with Image.open(image_path) as orig:
                        exif_dict = orig._getexif()
                        if exif_dict:
                            # Remove orientation tag (274) since we've applied the rotation
                            exif_dict.pop(274, None)
                            # Note: Re-encoding EXIF is complex, simplified approach
                except:
                    pass
            
            rotated_img.save(output_path, **save_kwargs)
            return True, output_path
            
    except Exception as e:
        print(f"  Error applying orientation: {e}")
        return False, image_path


def correct_image_orientation(image_path: str, output_path: Optional[str] = None,
                               overwrite: bool = False, 
                               use_exif: bool = True) -> Tuple[bool, str, str]:
    """
    Correct image orientation using EXIF data.
    
    This is a lightweight method that only uses EXIF orientation tags.
    For more robust detection, use the auto_rotate.py script with AI analysis.
    
    Args:
        image_path: Path to input image
        output_path: Path for output image (default: overwrite original)
        overwrite: If True, overwrite original file
        use_exif: If True, use EXIF orientation tag
        
    Returns:
        Tuple of (was_corrected, output_path, reason)
    """
    if not use_exif:
        return False, image_path, "EXIF correction disabled"
    
    was_rotated, final_path = apply_orientation_from_exif(
        image_path, output_path, overwrite
    )
    
    if was_rotated:
        return True, final_path, "Applied EXIF orientation correction"
    else:
        return False, image_path, "No EXIF orientation correction needed"


def prepare_image_for_processing(image_path: str, 
                                  temp_dir: Optional[str] = None,
                                  correct_orientation: bool = True) -> Tuple[str, bool, str]:
    """
    Prepare an image for processing by correcting orientation if needed.
    
    This function is designed to be called before caption generation or other
    image analysis to ensure the image is properly oriented.
    
    Args:
        image_path: Path to the input image
        temp_dir: Directory for temporary files (if rotation creates new file)
        correct_orientation: Whether to correct orientation
        
    Returns:
        Tuple of (image_path_to_use, was_modified, reason)
        - If orientation was corrected, returns path to corrected image
        - If no correction needed, returns original path
    """
    if not correct_orientation:
        return image_path, False, "Orientation correction disabled"
    
    # Check if file exists
    if not os.path.isfile(image_path):
        return image_path, False, "File not found"
    
    # Check if it's a supported image format
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'}
    if Path(image_path).suffix.lower() not in valid_extensions:
        return image_path, False, "Unsupported format"
    
    try:
        # Try EXIF-based correction first (fast)
        was_corrected, corrected_path, reason = correct_image_orientation(
            image_path, 
            output_path=None,  # Will create temp file if needed
            overwrite=False,   # Don't overwrite original during processing
            use_exif=True
        )
        
        if was_corrected:
            return corrected_path, True, reason
        else:
            return image_path, False, reason
            
    except Exception as e:
        print(f"  Warning: Could not prepare image: {e}")
        return image_path, False, f"Error: {e}"


def cleanup_temp_image(temp_image_path: str, original_path: str) -> None:
    """
    Clean up temporary image file created during processing.
    
    Args:
        temp_image_path: Path to temporary file
        original_path: Path to original file (to verify we should delete temp)
    """
    try:
        if temp_image_path != original_path and os.path.exists(temp_image_path):
            # Only delete if it's a different file and contains "_oriented" in name
            if "_oriented" in Path(temp_image_path).stem:
                os.remove(temp_image_path)
    except Exception as e:
        print(f"  Warning: Could not clean up temp file {temp_image_path}: {e}")
