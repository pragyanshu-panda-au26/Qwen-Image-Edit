import os
from huggingface_hub import InferenceClient
from PIL import Image
import io
import streamlit as st
import time
from typing import Dict


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'webp']



class ImageProcessor:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the HuggingFace client with error handling"""
        token = os.environ.get("HF_TOKEN")
        if not token:
            st.error("❌ HF_TOKEN not found! Please create a `.env` file with your HuggingFace token.")
            st.info("""
            **Setup Instructions:**
            1. Create a `.env` file in your project directory
            2. Add: `HF_TOKEN=your_huggingface_token_here`
            3. Get your token from: https://huggingface.co/settings/tokens
            """)
            st.stop()
        
        try:
            self.client = InferenceClient(
                provider="fal-ai",
                api_key=token,
            )
            # Test the client with a simple call
            st.success("✅ HuggingFace client initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize HuggingFace client: {str(e)}")
            st.error("Please check your HF_TOKEN and internet connection.")
            st.stop()
    
    def validate_image(self, file) -> Dict[str, any]:
        """Validate uploaded image file with comprehensive checks"""
        try:
            # Check file size
            if file.size > MAX_FILE_SIZE:
                return {
                    'valid': False, 
                    'error': f'File too large: {file.size/1024/1024:.1f}MB (max: {MAX_FILE_SIZE/1024/1024}MB)'
                }
            
            # Check file format by extension
            file_extension = file.name.split('.')[-1].lower() if '.' in file.name else ''
            if file_extension not in SUPPORTED_FORMATS:
                return {
                    'valid': False,
                    'error': f'Unsupported format: {file_extension} (supported: {", ".join(SUPPORTED_FORMATS)})'
                }
            
            # Get file bytes and validate as image
            file_bytes = file.getvalue()
            if len(file_bytes) == 0:
                return {'valid': False, 'error': 'Empty file'}
            
            # Try to open and validate as image
            image = Image.open(io.BytesIO(file_bytes))
            
            # Check image dimensions (reasonable limits)
            if image.width > 4000 or image.height > 4000:
                return {
                    'valid': False,
                    'error': f'Image too large: {image.width}x{image.height} (max: 4000x4000)'
                }
            
            if image.width < 50 or image.height < 50:
                return {
                    'valid': False,
                    'error': f'Image too small: {image.width}x{image.height} (min: 50x50)'
                }
            
            return {
                'valid': True,
                'size': file.size,
                'dimensions': f"{image.width}x{image.height}",
                'format': image.format or 'Unknown',
                'mode': image.mode,
                'bytes': file_bytes
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Invalid image file: {str(e)}'
            }
    
    def process_single_image(self, image_bytes: bytes, prompt: str, filename: str) -> Dict[str, any]:
        """Process a single image with comprehensive error handling"""
        try:
            # Validate inputs
            if not isinstance(image_bytes, bytes) or len(image_bytes) == 0:
                return {
                    'success': False,
                    'filename': filename,
                    'error': 'Invalid image data format'
                }
            
            if not prompt or not prompt.strip():
                return {
                    'success': False,
                    'filename': filename,
                    'error': 'Empty prompt'
                }
            
            # Prepare image - convert to standard format
            try:
                # Open image and ensure it's in RGB mode for consistency
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                if image.mode not in ['RGB', 'L']:
                    if image.mode == 'RGBA':
                        # Create white background for RGBA images
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                        image = background
                    else:
                        image = image.convert('RGB')
                
                # Save as PNG to ensure compatibility
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG', optimize=True)
                processed_bytes = img_buffer.getvalue()
                
            except Exception as e:
                return {
                    'success': False,
                    'filename': filename,
                    'error': f'Image preprocessing failed: {str(e)}'
                }
            
            # Make API call with timeout and retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    edited_image = self.client.image_to_image(
                        processed_bytes,
                        prompt=prompt.strip(),
                        model="Qwen/Qwen-Image-Edit",
                    )
                    
                    # Convert result to bytes for storage
                    result_buffer = io.BytesIO()
                    edited_image.save(result_buffer, format='PNG', optimize=True)
                    result_bytes = result_buffer.getvalue()
                    
                    return {
                        'success': True,
                        'filename': filename,
                        'image_bytes': result_bytes,
                        'image_obj': edited_image,
                        'original_size': len(image_bytes),
                        'result_size': len(result_bytes)
                    }
                    
                except Exception as api_error:
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    else:
                        # Handle specific API errors
                        error_msg = str(api_error).lower()
                        
                        if 'images' in error_msg or 'parameter' in error_msg:
                            final_error = "API parameter error - image format may be incompatible"
                        elif 'timeout' in error_msg:
                            final_error = "Request timeout - try with a smaller image"
                        elif 'quota' in error_msg or 'rate limit' in error_msg:
                            final_error = "API quota exceeded - please wait before trying again"
                        elif 'unauthorized' in error_msg or 'token' in error_msg:
                            final_error = "Authentication error - check your HF_TOKEN"
                        elif 'model' in error_msg:
                            final_error = "Model not available - try again later"
                        else:
                            final_error = f"API error: {str(api_error)}"
                        
                        return {
                            'success': False,
                            'filename': filename,
                            'error': final_error
                        }
            
        except Exception as e:
            return {
                'success': False,
                'filename': filename,
                'error': f'Unexpected error: {str(e)}'
            }
