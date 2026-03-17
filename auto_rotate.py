#!/usr/bin/env python3
"""
Auto-rotate images based on AI scene analysis.

This script uses Florence-2 to analyze images and determine if they need rotation
to be upright. It detects common orientation issues like upside-down or sideways photos.

Usage:
    python auto_rotate.py <image_path> [--output <output_path>] [--model <model_name>]
    python auto_rotate.py <directory> --recursive [--model <model_name>]
    
Examples:
    python auto_rotate.py photo.jpg
    python auto_rotate.py photo.jpg --output rotated_photo.jpg
    python auto_rotate.py ./photos --recursive
    python auto_rotate.py ./photos --model microsoft/Florence-2-base
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow library required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: PyTorch required. Install with: pip install torch")
    sys.exit(1)

# Register HEIF/HEIC opener for Pillow
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files will be skipped.")

# Global model variables
_model = None
_processor = None
_device = None


def load_model(model_name: str = "microsoft/Florence-2-base"):
    """Load the Florence-2 model for image analysis."""
    global _model, _processor, _device
    
    if _model is not None:
        return _model, _processor, _device
    
    print(f"Loading Florence-2 model: {model_name}...")
    print("(This may take a few minutes on first run)")
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        print("Error: transformers library required. Install with: pip install transformers")
        sys.exit(1)
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {_device.upper()}")
    
    _processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32
    ).to(_device)
    
    print("Model loaded successfully!")
    return _model, _processor, _device


def analyze_orientation(image_path: str, model_name: str = "microsoft/Florence-2-base") -> Tuple[int, str, float]:
    """
    Analyze an image to determine if it needs rotation.
    
    Returns:
        Tuple of (rotation_angle, reason, confidence)
        rotation_angle: 0, 90, 180, or 270 degrees clockwise
        reason: Explanation of why this rotation was chosen
        confidence: Confidence score (0.0 to 1.0)
    """
    model, processor, device = load_model(model_name)
    
    # Open image
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        
        # Check EXIF orientation first
        exif_rotation = get_exif_rotation(image_path)
        if exif_rotation != 0:
            print(f"  EXIF orientation tag suggests {exif_rotation}° rotation")
            # If EXIF says it needs rotation, we'll still verify with AI
            # but give it priority
        
        # Prepare prompts for orientation detection
        prompts = [
            "Is this image upside down? Answer yes or no, then explain.",
            "Is this image rotated 90 degrees clockwise? Answer yes or no, then explain.",
            "Is this image rotated 90 degrees counterclockwise? Answer yes or no, then explain.",
            "Is this image oriented correctly (upright)? Answer yes or no, then explain."
        ]
        
        results = []
        
        for prompt in prompts:
            try:
                inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=50,
                        num_beams=1,
                        do_sample=False
                    )
                
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results.append(response)
                
            except Exception as e:
                print(f"  Warning: Error analyzing with prompt '{prompt[:30]}...': {e}")
                results.append(f"Error: {e}")
        
        # Analyze results to determine rotation
        rotation_angle, reason, confidence = determine_rotation_from_responses(results, exif_rotation)
        
        return rotation_angle, reason, confidence


def get_exif_rotation(image_path: str) -> int:
    """Extract rotation angle from EXIF orientation tag."""
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


def determine_rotation_from_responses(responses: list, exif_rotation: int) -> Tuple[int, str, float]:
    """Determine the best rotation angle from AI responses."""
    
    # Parse responses
    upside_down_yes = 0
    cw_90_yes = 0
    ccw_90_yes = 0
    upright_yes = 0
    
    reasons = []
    
    for i, response in enumerate(responses):
        response_lower = response.lower()
        
        if i == 0:  # Upside down question
            if 'yes' in response_lower.split('.')[0]:
                upside_down_yes += 1
            reasons.append(f"Upside down check: {response[:100]}")
            
        elif i == 1:  # 90° CW question
            if 'yes' in response_lower.split('.')[0]:
                cw_90_yes += 1
            reasons.append(f"90° CW check: {response[:100]}")
            
        elif i == 2:  # 90° CCW question
            if 'yes' in response_lower.split('.')[0]:
                ccw_90_yes += 1
            reasons.append(f"90° CCW check: {response[:100]}")
            
        elif i == 3:  # Upright question
            if 'yes' in response_lower.split('.')[0]:
                upright_yes += 1
            reasons.append(f"Upright check: {response[:100]}")
    
    # Determine rotation based on votes
    if upright_yes > 0 and upright_yes >= upside_down_yes and upright_yes >= cw_90_yes and upright_yes >= ccw_90_yes:
        rotation = 0
        reason = "Image appears to be correctly oriented"
        confidence = 0.9 if upright_yes == 1 else 0.95
        
    elif upside_down_yes > 0:
        rotation = 180
        reason = "Image appears to be upside down"
        confidence = 0.85
        
    elif cw_90_yes > 0:
        rotation = 90  # Rotate 90° CW to fix
        reason = "Image appears to be rotated 90° counterclockwise (needs 90° CW rotation)"
        confidence = 0.85
        
    elif ccw_90_yes > 0:
        rotation = 270  # Rotate 270° CW (= 90° CCW) to fix
        reason = "Image appears to be rotated 90° clockwise (needs 270° CW rotation)"
        confidence = 0.85
        
    else:
        # Default to EXIF if available, otherwise no rotation
        if exif_rotation != 0:
            rotation = exif_rotation
            reason = f"Using EXIF orientation tag ({exif_rotation}°)"
            confidence = 0.7
        else:
            rotation = 0
            reason = "No orientation issues detected"
            confidence = 0.5
    
    return rotation, reason, confidence


def rotate_image(image_path: str, output_path: Optional[str] = None, 
                 rotation_angle: int = 0, overwrite: bool = False) -> str:
    """
    Rotate an image by the specified angle.
    
    Args:
        image_path: Path to input image
        output_path: Path for output image (default: modify in place)
        rotation_angle: Angle to rotate clockwise (0, 90, 180, 270)
        overwrite: If True, overwrite original file
    
    Returns:
        Path to the rotated image
    """
    if rotation_angle == 0:
        print("  No rotation needed")
        return image_path
    
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
                output_path = str(base.parent / f"{base.stem}_rotated_{rotation_angle}{base.suffix}")
        
        # Save the rotated image
        # Preserve EXIF data except orientation tag
        exif_data = img.info.get('exif', b'')
        if exif_data:
            try:
                exif_dict = img._getexif()
                if exif_dict:
                    # Remove orientation tag (274) since we've applied the rotation
                    exif_dict.pop(274, None)
                    # Note: Re-encoding EXIF is complex, so we might lose some data
            except:
                pass
        
        rotated_img.save(output_path, quality=95)
        print(f"  Saved rotated image to: {output_path}")
    
    return output_path


def process_image(image_path: str, output_path: Optional[str] = None,
                  model_name: str = "microsoft/Florence-2-base",
                  dry_run: bool = False,
                  overwrite: bool = False) -> bool:
    """
    Process a single image: analyze orientation and rotate if needed.
    
    Returns:
        True if rotation was applied, False otherwise
    """
    print(f"\nProcessing: {image_path}")
    
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"  Error: File not found: {image_path}")
        return False
    
    # Check if it's an image
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if HEIF_SUPPORT:
        valid_extensions.update({'.heic', '.heif'})
    
    if Path(image_path).suffix.lower() not in valid_extensions:
        print(f"  Skipping: Not a supported image format")
        return False
    
    try:
        # Analyze orientation
        rotation_angle, reason, confidence = analyze_orientation(image_path, model_name)
        
        print(f"  Analysis result:")
        print(f"    Rotation needed: {rotation_angle}°")
        print(f"    Reason: {reason}")
        print(f"    Confidence: {confidence:.0%}")
        
        if dry_run:
            print("  [DRY RUN] Would rotate but not saving")
            return rotation_angle != 0
        
        if rotation_angle != 0:
            # Rotate the image
            rotate_image(image_path, output_path, rotation_angle, overwrite)
            return True
        else:
            print("  Image is correctly oriented")
            return False
            
    except Exception as e:
        print(f"  Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(directory: str, recursive: bool = False,
                      model_name: str = "microsoft/Florence-2-base",
                      dry_run: bool = False,
                      overwrite: bool = False) -> Tuple[int, int]:
    """
    Process all images in a directory.
    
    Returns:
        Tuple of (processed_count, rotated_count)
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return 0, 0
    
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        return 0, 0
    
    # Find all images
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if HEIF_SUPPORT:
        valid_extensions.update({'.heic', '.heif'})
    
    if recursive:
        image_files = [f for f in directory.rglob('*') 
                      if f.suffix.lower() in valid_extensions and f.is_file()]
    else:
        image_files = [f for f in directory.glob('*') 
                      if f.suffix.lower() in valid_extensions and f.is_file()]
    
    total_images = len(image_files)
    if total_images == 0:
        print(f"No images found in {directory}")
        return 0, 0
    
    print(f"Found {total_images} images to process")
    print(f"Recursive: {recursive}, Dry run: {dry_run}")
    
    processed = 0
    rotated = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{total_images}]")
        
        if process_image(str(image_file), None, model_name, dry_run, overwrite):
            rotated += 1
        processed += 1
    
    return processed, rotated


def main():
    parser = argparse.ArgumentParser(
        description="Auto-rotate images based on AI scene analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg                          # Analyze and rotate a single image
  %(prog)s photo.jpg --output rotated.jpg     # Save to specific output file
  %(prog)s ./photos --recursive               # Process all images in directory
  %(prog)s ./photos --dry-run                 # Analyze without rotating
  %(prog)s photo.jpg --model microsoft/Florence-2-large  # Use larger model
        """
    )
    
    parser.add_argument("input", help="Image file or directory to process")
    parser.add_argument("--output", "-o", help="Output file path (for single image)")
    parser.add_argument("--model", "-m", default="microsoft/Florence-2-base",
                       help="Model to use (default: microsoft/Florence-2-base)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Process subdirectories recursively")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Analyze only, don't rotate images")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite original files (use with caution!)")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    if args.overwrite and not args.dry_run:
        confirm = input("WARNING: Overwriting original files! Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted")
            sys.exit(0)
    
    # Process
    if input_path.is_file():
        if args.recursive:
            print("Warning: --recursive ignored for single file")
        
        success = process_image(
            str(input_path),
            args.output,
            args.model,
            args.dry_run,
            args.overwrite
        )
        
        if success and not args.dry_run:
            print("\n✓ Image rotated successfully")
        elif not args.dry_run:
            print("\n✓ No rotation needed")
        else:
            print("\n✓ Analysis complete")
            
    elif input_path.is_dir():
        processed, rotated = process_directory(
            str(input_path),
            args.recursive,
            args.model,
            args.dry_run,
            args.overwrite
        )
        
        print(f"\n{'='*50}")
        print(f"Summary:")
        print(f"  Total images processed: {processed}")
        print(f"  Images rotated: {rotated}")
        print(f"  Images unchanged: {processed - rotated}")
        if args.dry_run:
            print(f"  (Dry run - no files were modified)")
    
    else:
        print(f"Error: Invalid input path: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
