"""
Stage 1: Vision-Language Model (VLM) Caption Generator

Supports multiple VLM architectures:
- Florence-2 (Microsoft)
- LLaVA (Large Language-and-Vision Assistant)
- Qwen2.5-VL
- BLIP-2

Generates text captions from images that are then passed to Stage 2 for embedding.
"""

import torch
from PIL import Image
from pathlib import Path
from typing import Optional, List, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models"""
    
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model(**kwargs)
    
    @abstractmethod
    def _load_model(self, **kwargs):
        """Load the specific VLM model and processor"""
        pass
    
    @abstractmethod
    def generate_caption(self, image: Image.Image, prompt: str = "") -> str:
        """Generate a caption for the given image"""
        pass
    
    def process_image_path(self, image_path: Union[str, Path], prompt: str = "") -> str:
        """Process an image file path and return caption"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            return self.generate_caption(image, prompt)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise


class Florence2VLM(BaseVLM):
    """Microsoft Florence-2 Vision-Language Model"""
    
    def _load_model(self, trust_remote_code: bool = True, **kwargs):
        """Load Florence-2 model"""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            logger.info(f"Loading Florence-2 model: {self.model_id}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=trust_remote_code
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            logger.info("Florence-2 model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            raise
    
    def generate_caption(self, image: Image.Image, prompt: str = "<DETAILED_CAPTION>") -> str:
        """Generate caption using Florence-2"""
        
        # Prepare input
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                do_sample=False,
                num_beams=1
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=False
        )[0]
        
        # Clean up response
        caption = self.processor.post_process_generation(
            generated_text, 
            task=prompt.strip("<>"),
            image_size=image.size
        )
        
        return caption.strip()


class LLaVAVLM(BaseVLM):
    """LLaVA Vision-Language Model"""
    
    def _load_model(self, **kwargs):
        """Load LLaVA model"""
        try:
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            
            logger.info(f"Loading LLaVA model: {self.model_id}")
            
            self.processor = LlavaProcessor.from_pretrained(self.model_id)
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            logger.info("LLaVA model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            raise
    
    def generate_caption(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption using LLaVA"""
        
        # Prepare input
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # Decode
        caption = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return caption.strip()


class Qwen2VLVLM(BaseVLM):
    """Qwen2.5-VL Vision-Language Model"""
    
    def _load_model(self, trust_remote_code: bool = True, **kwargs):
        """Load Qwen2.5-VL model"""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            logger.info(f"Loading Qwen2.5-VL model: {self.model_id}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=trust_remote_code
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            logger.info("Qwen2.5-VL model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            raise
    
    def generate_caption(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption using Qwen2.5-VL"""
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Prepare input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512
            )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        caption = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return caption.strip()


def create_vlm(model_type: str, model_id: str, device: str = "cuda", **kwargs) -> BaseVLM:
    """Factory function to create appropriate VLM instance"""
    
    vlm_classes = {
        "florence2": Florence2VLM,
        "florence": Florence2VLM,
        "llava": LLaVAVLM,
        "qwen2vl": Qwen2VLVLM,
        "qwen": Qwen2VLVLM,
    }
    
    if model_type.lower() not in vlm_classes:
        raise ValueError(f"Unsupported VLM type: {model_type}. Supported: {list(vlm_classes.keys())}")
    
    return vlm_classes[model_type.lower()](model_id=model_id, device=device, **kwargs)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test with Florence-2
    print("Testing Florence-2 VLM...")
    vlm = create_vlm("florence2", "microsoft/Florence-2-base")
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        caption = vlm.process_image_path(image_path, "<DETAILED_CAPTION>")
        print(f"\nGenerated Caption:\n{caption}")
    else:
        print("Provide an image path to test: python stage1_vlm_captioner.py <image_path>")
