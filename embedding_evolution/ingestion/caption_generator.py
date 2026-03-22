"""
Caption generation using various LLM models.
Supports Florence-2, LLaVA, BLIP-2, and other vision-language models.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from config.models import ModelConfig, ModelType, get_model_config


class CaptionGenerator:
    """Generate captions for images using specified LLM models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize caption generator with specified model.
        
        Args:
            model_name: Name of the model (e.g., "florence-2-base", "llava-1.6-34b")
            device: Device to run model on ("cuda", "cpu", or None for auto-detect)
        """
        self.config = get_model_config(model_name)
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Warn if GPU memory might be insufficient
                if gpu_memory < self.config.gpu_memory_required_gb:
                    print(f"Warning: Model requires {self.config.gpu_memory_required_gb} GB, "
                          f"but GPU has only {gpu_memory:.1f} GB")
            else:
                self.device = "cpu"
                print("Using CPU (slow - consider upgrading to GPU)")
        else:
            self.device = device
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and processor."""
        print(f"Loading model: {self.config.model_id}")
        
        try:
            if "florence" in self.model_name.lower():
                self._load_florence()
            elif "llava" in self.model_name.lower():
                self._load_llava()
            elif "blip" in self.model_name.lower():
                self._load_blip()
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_florence(self):
        """Load Florence-2 model."""
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    def _load_llava(self):
        """Load LLaVA model with quantization if needed."""
        if self.device == "cuda" and self.config.gpu_memory_required_gb > 8:
            # Use 4-bit quantization for large models
            print("Using 4-bit quantization for LLaVA model")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.device = "cuda"  # Quantized model handles device mapping
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True
        )
    
    def _load_blip(self):
        """Load BLIP-2 model."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        self.processor = Blip2Processor.from_pretrained(self.config.model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    def generate_caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt (uses model default if None)
            
        Returns:
            Generated caption text
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Open image
        image = Image.open(image_path).convert("RGB")
        
        try:
            if "florence" in self.model_name.lower():
                return self._generate_florence(image, prompt)
            elif "llava" in self.model_name.lower():
                return self._generate_llava(image, prompt)
            elif "blip" in self.model_name.lower():
                return self._generate_blip(image, prompt)
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
                
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            raise
    
    def _generate_florence(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Generate caption using Florence-2."""
        # Use detailed caption preset if no custom prompt
        if prompt is None:
            prompt = "<DETAILED_CAPTION>" if self.config.caption_preset == "detailed" else "<CAPTION>"
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                do_sample=False,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        caption = self.processor.post_process_generation(
            generated_text, 
            task=prompt.replace("<", "").replace(">", "").lower(), 
            image_size=image.size
        )
        
        # Extract caption from result
        if isinstance(caption, dict):
            caption = list(caption.values())[0]
        
        return caption.strip()
    
    def _generate_llava(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Generate caption using LLaVA."""
        if prompt is None:
            prompt = "Describe this image in detail."
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        
        input_tensor = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            [prompt], 
            [image], 
            return_tensors='pt'
        ).to(self.device if not hasattr(self.model, 'device') else self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        # Remove the prompt from the output
        caption = caption.replace(prompt, "").strip()
        
        return caption
    
    def _generate_blip(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Generate caption using BLIP-2."""
        if prompt is None:
            prompt = "Describe this image in detail."
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption
    
    def batch_generate(self, image_paths: list, prompt: Optional[str] = None) -> Dict[str, str]:
        """
        Generate captions for multiple images.
        
        Args:
            image_paths: List of image file paths
            prompt: Optional custom prompt
            
        Returns:
            Dictionary mapping image paths to captions
        """
        results = {}
        for i, path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {path}")
            try:
                caption = self.generate_caption(path, prompt)
                results[path] = caption
            except Exception as e:
                print(f"Failed to process {path}: {e}")
                results[path] = None
        
        return results


def test_caption_generator():
    """Test the caption generator with a sample image."""
    # Test with Florence-2-base (should work on 4GB GPU)
    print("Testing Florence-2-base...")
    generator = CaptionGenerator("florence-2-base")
    
    # Create a test image if none available
    test_image = Image.new('RGB', (224, 224), color='red')
    test_path = "/tmp/test_image.jpg"
    test_image.save(test_path)
    
    caption = generator.generate_caption(test_path)
    print(f"Generated caption: {caption}")
    
    return caption


if __name__ == "__main__":
    test_caption_generator()
