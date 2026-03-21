#!/usr/bin/env python3
"""
Simple Caption Generator for Single Image

Generates a caption for a single image using the local Florence-2 model.
No database, no file modifications - just outputs the caption to console.

Usage:
    python simple_caption.py /path/to/image.jpg
    python simple_caption.py /path/to/image.jpg --detailed
    python simple_caption.py /path/to/image.jpg --model microsoft/Florence-2-large

Requirements:
    pip install transformers torch pillow
"""

import sys
import argparse
from pathlib import Path

# Check for required packages
try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Run: pip install pillow")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)


def generate_caption(image_path: str, 
                     model_name: str = "microsoft/Florence-2-base",
                     detailed: bool = False,
                     very_detailed: bool = False):
    """
    Generate a caption for a single image using Florence-2
    
    Args:
        image_path: Path to the image file
        model_name: HuggingFace model name (default: microsoft/Florence-2-base)
        detailed: Use detailed caption mode
        very_detailed: Use very detailed caption mode
    
    Returns:
        Generated caption string
    """
    # Determine the task based on detail level
    if very_detailed:
        task = "<MORE_DETAILED_CAPTION>"
    elif detailed:
        task = "<DETAILED_CAPTION>"
    else:
        task = "<CAPTION>"
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS GPU", file=sys.stderr)
    else:
        device = "cpu"
        print("Using CPU", file=sys.stderr)
    
    # Load model and processor
    print(f"Loading model: {model_name}...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)
    print("Model loaded successfully", file=sys.stderr)
    
    # Open and process image
    print(f"Processing image: {image_path}", file=sys.stderr)
    image = Image.open(image_path).convert("RGB")
    
    # Prepare input
    inputs = processor(
        text=task,
        images=image,
        return_tensors="pt"
    ).to(device, torch.float16 if device != "cpu" else torch.float32)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=1
        )
    
    # Decode result
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]
    
    # Post-process to extract caption
    caption = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height)
    )
    
    # Extract the actual caption text from the result dict
    if isinstance(caption, dict):
        caption = list(caption.values())[0]
    
    return str(caption).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate a caption for a single image using local Florence-2 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic caption (fast)
    python simple_caption.py photo.jpg
    
    # Detailed caption
    python simple_caption.py photo.jpg --detailed
    
    # Very detailed caption
    python simple_caption.py photo.jpg --very-detailed
    
    # Use larger model for better quality
    python simple_caption.py photo.jpg --model microsoft/Florence-2-large
    
    # Specify CPU only
    python simple_caption.py photo.jpg --device cpu
        """
    )
    
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='microsoft/Florence-2-base',
                        help='Model name or path (default: microsoft/Florence-2-base)')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed caption (2-3 sentences)')
    parser.add_argument('--very-detailed', action='store_true',
                        help='Generate very detailed caption')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to run on (default: auto)')
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp']:
        print(f"Warning: Unusual file extension: {image_path.suffix}", file=sys.stderr)
    
    try:
        caption = generate_caption(
            str(image_path),
            model_name=args.model,
            detailed=args.detailed,
            very_detailed=args.very_detailed
        )
        
        # Output just the caption to stdout
        print("\n" + "="*50)
        print("CAPTION:")
        print("="*50)
        print(caption)
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error generating caption: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
