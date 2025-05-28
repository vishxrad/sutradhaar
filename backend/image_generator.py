import os
import re
from typing import List, Optional, Dict, Any
from google.cloud import aiplatform
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import base64
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexImageGenerator:
    def __init__(self, project_id: str = "gen-lang-client-0276400412", location: str = "us-central1"):
        """
        Initialize Vertex AI Imagen client
        
        Args:
            project_id: Google Cloud Project ID
            location: Region for Vertex AI (default: us-central1)
        """
        self.project_id = project_id
        self.location = location
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        try:
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=location)
            
            # Initialize the ImageGenerationModel
            self.model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            logger.info(f"Successfully initialized Vertex AI Image Generator with project {project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI Image Generator: {str(e)}")
            raise
        
    def sanitize_filename(self, text: str) -> str:
        """
        Convert text to a safe filename
        """
        # Remove special characters and replace spaces with underscores
        filename = re.sub(r'[^\w\s-]', '', text)
        filename = re.sub(r'[-\s]+', '_', filename)
        return filename.lower()[:50]  # Limit length to 50 chars
    
    def generate_image(self, prompt: str, output_dir: str, filename_prefix: str) -> Optional[str]:
        """
        Generate an image using Vertex AI Imagen with ImageGenerationModel
        
        Args:
            prompt: Text prompt for image generation
            output_dir: Directory to save the image
            filename_prefix: Prefix for the filename
            
        Returns:
            Path to the generated image file or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Generating image for prompt: '{prompt}' (attempt {attempt + 1}/{self.max_retries})")
                
                # Generate image using the correct method
                response = self.model.generate_images(
                    prompt=prompt,
                    number_of_images=1,
                    aspect_ratio="4:3",
                    safety_filter_level="block_some",
                    person_generation="allow_adult"
                )
                
                # Extract image data
                if response.images:
                    image_object = response.images[0]
                    image_data = image_object._image_bytes
                    
                    # Create filename
                    safe_prompt = self.sanitize_filename(prompt)
                    filename = f"{filename_prefix}_{safe_prompt}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    
                    logger.info(f"Successfully generated image: {filepath}")
                    return filepath
                else:
                    logger.warning(f"No images returned from Vertex AI for prompt: '{prompt}'")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    return None
                    
            except Exception as e:
                logger.error(f"Error generating image for prompt '{prompt}' (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to generate image after {self.max_retries} attempts")
                    return None
        
        return None
    
    def generate_images_for_slides(self, segments_data: List[dict], base_output_dir: str = "generated_images") -> Dict[str, Any]:
        """
        Generate images for all slides in the segments data with comprehensive error handling
        
        Args:
            segments_data: Parsed script data with segments and slides
            base_output_dir: Base directory for saving images
            
        Returns:
            Dictionary with image paths, success/failure stats, and error details
        """
        image_paths = {}
        stats = {
            'total_requested': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            for segment_idx, segment in enumerate(segments_data, 1):
                segment_title = segment.get('segment_title', f'segment_{segment_idx}')
                segment_dir = os.path.join(base_output_dir, f"segment_{segment_idx}_{self.sanitize_filename(segment_title)}")
                
                for slide_idx, slide in enumerate(segment.get('slides', []), 1):
                    slide_title = slide.get('title', f'slide_{slide_idx}')
                    image_prompt = slide.get('image_prompt', '')
                    
                    if image_prompt:
                        stats['total_requested'] += 1
                        filename_prefix = f"seg{segment_idx}_slide{slide_idx}"
                        
                        try:
                            image_path = self.generate_image(
                                prompt=image_prompt,
                                output_dir=segment_dir,
                                filename_prefix=filename_prefix
                            )
                            
                            # Store the mapping
                            slide_key = f"segment_{segment_idx}_slide_{slide_idx}"
                            
                            if image_path:
                                stats['successful'] += 1
                                image_paths[slide_key] = {
                                    'segment_title': segment_title,
                                    'slide_title': slide_title,
                                    'image_prompt': image_prompt,
                                    'image_path': image_path,
                                    'status': 'success'
                                }
                            else:
                                stats['failed'] += 1
                                error_msg = f"Failed to generate image for segment {segment_idx}, slide {slide_idx}"
                                stats['errors'].append(error_msg)
                                image_paths[slide_key] = {
                                    'segment_title': segment_title,
                                    'slide_title': slide_title,
                                    'image_prompt': image_prompt,
                                    'image_path': None,
                                    'status': 'failed',
                                    'error': error_msg
                                }
                                
                        except Exception as e:
                            stats['failed'] += 1
                            error_msg = f"Exception generating image for segment {segment_idx}, slide {slide_idx}: {str(e)}"
                            stats['errors'].append(error_msg)
                            logger.error(error_msg)
                            
                            slide_key = f"segment_{segment_idx}_slide_{slide_idx}"
                            image_paths[slide_key] = {
                                'segment_title': segment_title,
                                'slide_title': slide_title,
                                'image_prompt': image_prompt,
                                'image_path': None,
                                'status': 'error',
                                'error': error_msg
                            }
        
        except Exception as e:
            error_msg = f"Critical error in generate_images_for_slides: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(error_msg)
        
        # Log summary
        logger.info(f"Image generation complete. Success: {stats['successful']}/{stats['total_requested']}, Failed: {stats['failed']}")
        if stats['errors']:
            logger.warning(f"Errors encountered: {len(stats['errors'])}")
        
        return {
            'image_paths': image_paths,
            'stats': stats
        }
