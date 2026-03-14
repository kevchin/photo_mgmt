#!/usr/bin/env python3
"""
Caption Generator and Embedding Utility

Generates captions for images using an LLM API and creates embeddings
for semantic search with pgvector.

Supports:
- OpenAI-compatible APIs for caption generation
- Any embedding model via OpenAI API or local models
- Batch processing with progress tracking
- Resume capability for interrupted runs
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Import our database module
from image_database import ImageDatabase, ImageMetadata


class CaptionGenerator:
    """Generate captions for images using LLM"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, 
                 model: str = "gpt-4o", max_tokens: int = 200):
        """
        Initialize the caption generator
        
        Args:
            api_key: API key for the LLM service
            base_url: Optional custom base URL (for compatible APIs)
            model: Model to use for caption generation
            max_tokens: Maximum tokens in the caption
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
    
    def generate_caption(self, image_path: str, include_details: bool = True) -> str:
        """
        Generate a descriptive caption for an image
        
        Args:
            image_path: Path to the image file
            include_details: Whether to request detailed information
            
        Returns:
            Generated caption text
        """
        try:
            # Read and encode the image
            with open(image_path, 'rb') as f:
                import base64
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Build the prompt
            if include_details:
                prompt = """Describe this image in detail. Include:
- Main subjects and people (approximate count, ages if visible)
- Setting/location type (beach, mountain, indoor, etc.)
- Activities happening
- Time of day or season if apparent
- Notable objects or elements
- Overall mood or atmosphere

Keep the description to 2-3 sentences. Be specific but concise."""
            else:
                prompt = "Describe this image briefly in one sentence."
            
            # Call the vision API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return None


class EmbeddingGenerator:
    """Generate embeddings for text using embedding models"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 model: str = "text-embedding-3-small", dimensions: int = 1536):
        """
        Initialize the embedding generator
        
        Args:
            api_key: API key for the embedding service
            base_url: Optional custom base URL
            model: Embedding model to use
            dimensions: Number of dimensions for the embedding
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = dimensions
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None


def process_image(db: ImageDatabase, img_record: Dict, 
                  caption_gen: Optional[CaptionGenerator],
                  embed_gen: Optional[EmbeddingGenerator],
                  regenerate: bool = False) -> bool:
    """
    Process a single image: generate caption and embedding
    
    Args:
        db: Database instance
        img_record: Image record from database
        caption_gen: Caption generator instance
        embed_gen: Embedding generator instance
        regenerate: Whether to regenerate existing captions
        
    Returns:
        True if successfully processed
    """
    # Check if already has caption and we're not regenerating
    if img_record.get('caption') and not regenerate:
        # Still might need embedding
        if img_record.get('caption_embedding') is None and embed_gen:
            embedding = embed_gen.generate_embedding(img_record['caption'])
            if embedding:
                metadata = ImageMetadata(**{k: v for k, v in img_record.items() 
                                           if k in ImageMetadata.__dataclass_fields__})
                metadata.caption_embedding = embedding
                db.insert_image(metadata)
                print(f"  Added embedding for: {img_record['file_name']}")
                return True
        return False
    
    # Need to generate caption
    if not caption_gen:
        print(f"  Skipping {img_record['file_name']}: no caption generator configured")
        return False
    
    print(f"  Processing: {img_record['file_name']}")
    
    # Generate caption
    caption = caption_gen.generate_caption(img_record['file_path'])
    if not caption:
        print(f"    Failed to generate caption")
        return False
    
    print(f"    Caption: {caption[:80]}...")
    
    # Generate embedding
    embedding = None
    if embed_gen:
        embedding = embed_gen.generate_embedding(caption)
        if not embedding:
            print(f"    Warning: Failed to generate embedding")
    
    # Update database
    metadata = ImageMetadata(**{k: v for k, v in img_record.items() 
                               if k in ImageMetadata.__dataclass_fields__})
    metadata.caption = caption
    metadata.caption_embedding = embedding
    
    db.insert_image(metadata)
    print(f"    Updated database")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions and embeddings for images in the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate captions and embeddings for all images without them
  %(prog)s --db "postgresql://user:pass@localhost/image_archive" \\
      --openai-key $OPENAI_API_KEY
  
  # Regenerate captions for all images
  %(prog)s --db "postgresql://user:pass@localhost/image_archive" \\
      --openai-key $OPENAI_API_KEY --regenerate
  
  # Only generate embeddings for existing captions
  %(prog)s --db "postgresql://user:pass@localhost/image_archive" \\
      --openai-key $OPENAI_API_KEY --embeddings-only
  
  # Use a different model
  %(prog)s --db "postgresql://user:pass@localhost/image_archive" \\
      --openai-key $OPENAI_API_KEY --caption-model gpt-4-turbo
  
  # Use a custom API endpoint (e.g., local Ollama, vLLM, etc.)
  %(prog)s --db "postgresql://user:pass@localhost/image_archive" \\
      --openai-key ollama --openai-base-url http://localhost:11434/v1 \\
      --caption-model llava --embedding-model nomic-embed-text
        """
    )
    
    parser.add_argument('--db', required=True, help='PostgreSQL connection string')
    parser.add_argument('--openai-key', required=True, 
                       help='OpenAI API key (or key for compatible API)')
    parser.add_argument('--openai-base-url', help='Custom OpenAI-compatible API base URL')
    parser.add_argument('--caption-model', default='gpt-4o',
                       help='Model for caption generation (default: gpt-4o)')
    parser.add_argument('--embedding-model', default='text-embedding-3-small',
                       help='Model for embeddings (default: text-embedding-3-small)')
    parser.add_argument('--regenerate', action='store_true',
                       help='Regenerate captions even if they exist')
    parser.add_argument('--embeddings-only', action='store_true',
                       help='Only generate embeddings for existing captions')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of images to process (0 = all)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed captions (default: brief)')
    
    args = parser.parse_args()
    
    # Initialize database
    print(f"Connecting to database...")
    db = ImageDatabase(args.db)
    
    # Get images to process
    if args.embeddings_only:
        # Only images with captions but no embeddings
        print("Finding images with captions but no embeddings...")
        # We'll filter in Python since we need to check both fields
        stats = db.get_statistics()
        print(f"Total images in database: {stats['total_images']}")
        print(f"Images with captions: {stats['with_captions']}")
        
        # Get all images and filter (inefficient but works)
        # For large databases, add a dedicated query method
        from psycopg2.extras import RealDictCursor
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM images 
                    WHERE caption IS NOT NULL AND caption != '' 
                    AND caption_embedding IS NULL
                    ORDER BY id
                    %s
                """, ("LIMIT " + str(args.limit) if args.limit > 0 else ""))
                images_to_process = [dict(row) for row in cur.fetchall()]
    elif args.regenerate:
        # All images
        print("Getting all images for regeneration...")
        from psycopg2.extras import RealDictCursor
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM images 
                    ORDER BY id
                    %s
                """, ("LIMIT " + str(args.limit) if args.limit > 0 else ""))
                images_to_process = [dict(row) for row in cur.fetchall()]
    else:
        # Images without captions
        print("Finding images without captions...")
        from psycopg2.extras import RealDictCursor
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM images 
                    WHERE caption IS NULL OR caption = ''
                    ORDER BY id
                    %s
                """, ("LIMIT " + str(args.limit) if args.limit > 0 else ""))
                images_to_process = [dict(row) for row in cur.fetchall()]
    
    print(f"Found {len(images_to_process)} images to process\n")
    
    if not images_to_process:
        print("No images to process!")
        db.close()
        return
    
    # Initialize generators
    caption_gen = None
    embed_gen = None
    
    if not args.embeddings_only:
        print(f"Initializing caption generator with model: {args.caption_model}")
        caption_gen = CaptionGenerator(
            api_key=args.openai_key,
            base_url=args.openai_base_url,
            model=args.caption_model
        )
    
    print(f"Initializing embedding generator with model: {args.embedding_model}")
    embed_gen = EmbeddingGenerator(
        api_key=args.openai_key,
        base_url=args.openai_base_url,
        model=args.embedding_model
    )
    
    # Process images
    print(f"\nProcessing {len(images_to_process)} images with {args.workers} workers...\n")
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_image, db, img, caption_gen, embed_gen, args.regenerate
            ): img for img in images_to_process
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            img = futures[future]
            try:
                if future.result():
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"Error processing {img['file_name']}: {e}")
                error_count += 1
            
            # Progress indicator
            if i % 10 == 0 or i == len(images_to_process):
                print(f"Progress: {i}/{len(images_to_process)} "
                      f"(success: {success_count}, errors: {error_count})")
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {success_count}")
    print(f"Errors/Skipped: {error_count}")
    
    # Show updated stats
    stats = db.get_statistics()
    print(f"\nUpdated database statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"With captions: {stats['with_captions']}")
    
    db.close()


if __name__ == '__main__':
    main()
