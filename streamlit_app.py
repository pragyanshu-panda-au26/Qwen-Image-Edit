import streamlit as st
import os
from huggingface_hub import InferenceClient
from PIL import Image
import io
import zipfile
from typing import List, Dict, Optional
import time
import hashlib
import traceback
from image_processor import ImageProcessor

# Load environment variables
load_dotenv()

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
MAX_FILES = 15  # Maximum number of files
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'webp']


def initialize_session_state():
    """Initialize all session state variables"""
    if 'uploaded_files_data' not in st.session_state:
        st.session_state.uploaded_files_data = {}
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = ""
    if 'processing_in_progress' not in st.session_state:
        st.session_state.processing_in_progress = False

def create_file_hash(file_content: bytes, filename: str) -> str:
    """Create unique hash for file identification"""
    hash_input = f"{filename}_{len(file_content)}_{hashlib.md5(file_content).hexdigest()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

def create_zip_download(results: List[Dict]) -> bytes:
    """Create ZIP file containing all successful results"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, result in enumerate(results):
            if result['success']:
                # Create filename
                base_name = os.path.splitext(result['filename'])[0]
                safe_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
                zip_filename = f"{i+1:02d}_edited_{safe_name}.png"
                
                # Add file to ZIP
                zip_file.writestr(zip_filename, result['image_bytes'])
    
    return zip_buffer.getvalue()

def main():
    st.set_page_config(
        page_title="Batch Image Editor with Qwen",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header { 
        font-size: 3rem; 
        color: #FF6B6B; 
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-metric { color: #28a745; }
    .error-metric { color: #dc3545; }
    .info-box { 
        padding: 1rem; 
        border-radius: 0.5rem; 
        background-color: #f8f9fa; 
        border-left: 4px solid #007bff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown('<h1 class="main-header">🎨 Batch Image Editor with Qwen</h1>', unsafe_allow_html=True)
    st.markdown("Transform multiple images simultaneously using AI-powered editing!")
    
    # Initialize processor
    try:
        processor = ImageProcessor()
    except:
        st.stop()  # Stop if processor initialization fails
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ℹ️ Information")
        st.markdown(f"""
        <div class="info-box">
        <strong>Limits & Requirements:</strong><br>
        • Max files: {MAX_FILES}<br>
        • Max size per file: {MAX_FILE_SIZE/1024/1024:.0f}MB<br>
        • Supported: {', '.join(SUPPORTED_FORMATS)}<br>
        • Min size: 50x50 pixels<br>
        • Max size: 4000x4000 pixels
        </div>
        """, unsafe_allow_html=True)
        
        # Current session info
        if st.session_state.uploaded_files_data:
            st.markdown("### 📊 Current Session")
            files_count = len(st.session_state.uploaded_files_data)
            total_size = sum(data['size'] for data in st.session_state.uploaded_files_data.values())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files", files_count)
            with col2:
                st.metric("Total Size", f"{total_size/1024/1024:.1f}MB")
        
        # Control buttons
        st.markdown("### 🎛️ Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Files", help="Remove all uploaded files"):
                st.session_state.uploaded_files_data = {}
                st.rerun()
        
        with col2:
            if st.button("🔄 Clear Results", help="Clear processing results"):
                st.session_state.processing_results = []
                st.rerun()
        
        if st.button("🆕 New Session", help="Start fresh", type="primary"):
            for key in ['uploaded_files_data', 'processing_results', 'current_prompt']:
                st.session_state[key] = {} if 'data' in key or 'results' in key else ""
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.markdown("## 📁 Upload Images")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose images to edit...",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help=f"Select up to {MAX_FILES} images (max {MAX_FILE_SIZE/1024/1024:.0f}MB each)",
            disabled=st.session_state.processing_in_progress
        )
        
        # Process uploaded files
        if uploaded_files and len(uploaded_files) > 0:
            # Limit number of files
            if len(uploaded_files) > MAX_FILES:
                st.warning(f"⚠️ Too many files! Only processing first {MAX_FILES} files.")
                uploaded_files = uploaded_files[:MAX_FILES]
            
            # Validate and process new files
            new_files = 0
            invalid_files = []
            
            progress_placeholder = st.empty()
            
            for i, file in enumerate(uploaded_files):
                progress_placeholder.text(f"Validating files... ({i+1}/{len(uploaded_files)})")
                
                file_hash = create_file_hash(file.getvalue(), file.name)
                
                # Only process if not already in session
                if file_hash not in st.session_state.uploaded_files_data:
                    validation = processor.validate_image(file)
                    
                    if validation['valid']:
                        st.session_state.uploaded_files_data[file_hash] = {
                            'filename': file.name,
                            'size': validation['size'],
                            'dimensions': validation['dimensions'],
                            'format': validation['format'],
                            'mode': validation['mode'],
                            'bytes': validation['bytes'],
                            'hash': file_hash
                        }
                        new_files += 1
                    else:
                        invalid_files.append({
                            'filename': file.name,
                            'error': validation['error']
                        })
            
            progress_placeholder.empty()
            
            # Show results
            if new_files > 0:
                st.success(f"✅ Successfully added {new_files} new images!")
            
            if invalid_files:
                st.error("❌ Some files were rejected:")
                for invalid in invalid_files:
                    st.write(f"• **{invalid['filename']}**: {invalid['error']}")
        
        # Display current files
        if st.session_state.uploaded_files_data:
            st.markdown(f"## 📋 Loaded Files ({len(st.session_state.uploaded_files_data)})")
            
            with st.expander("📂 View and manage loaded files", expanded=len(st.session_state.uploaded_files_data) <= 5):
                for file_hash, data in list(st.session_state.uploaded_files_data.items()):
                    col_info, col_preview, col_actions = st.columns([3, 1, 1])
                    
                    with col_info:
                        st.markdown(f"**{data['filename']}**")
                        st.text(f"📏 {data['dimensions']} • 💾 {data['size']/1024:.1f}KB • 🎨 {data['format']}")
                    
                    with col_preview:
                        try:
                            image = Image.open(io.BytesIO(data['bytes']))
                            st.image(image, width=80)
                        except:
                            st.text("Preview error")
                    
                    with col_actions:
                        if st.button("🗑️", key=f"remove_{file_hash}", help="Remove this file"):
                            del st.session_state.uploaded_files_data[file_hash]
                            st.rerun()
    
    with col2:
        st.markdown("## ✨ Edit Configuration")
        
        # Prompt input
        prompt = st.text_area(
            "Describe your edit:",
            value=st.session_state.current_prompt,
            placeholder="Examples:\n• Turn into watercolor painting\n• Add sunglasses\n• Make it look vintage\n• Change background to sunset",
            height=120,
            help="This prompt will be applied to all images",
            disabled=st.session_state.processing_in_progress
        )
        
        # Update session state
        st.session_state.current_prompt = prompt
        
        # Processing options
        st.markdown("### ⚙️ Options")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_preview = st.checkbox("Show previews", value=True)
            max_preview = st.selectbox("Max previews", [1, 3, 5, 8, "All"], index=0)
        
        with col_opt2:
            show_errors = st.checkbox("Show error details", value=True)
            auto_download = st.checkbox("Auto-download results", value=False)
        
        # Processing status
        can_process = (
            bool(st.session_state.uploaded_files_data) and 
            bool(prompt.strip()) and 
            not st.session_state.processing_in_progress
        )
        
        # Process button
        if st.button(
            "🚀 Start Batch Processing",
            type="primary",
            disabled=not can_process,
            use_container_width=True
        ):
            if can_process:
                st.session_state.processing_in_progress = True
                st.session_state.processing_results = []
                
                # Processing UI
                st.markdown("### 🔄 Processing Status")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                files_data = list(st.session_state.uploaded_files_data.items())
                total_files = len(files_data)
                
                # Process each image
                for i, (file_hash, file_data) in enumerate(files_data):
                    current_progress = (i + 1) / total_files
                    status_text.text(f"Processing: {file_data['filename']} ({i+1}/{total_files})")
                    
                    result = processor.process_single_image(
                        file_data['bytes'],
                        prompt,
                        file_data['filename']
                    )
                    
                    st.session_state.processing_results.append(result)
                    progress_bar.progress(current_progress)
                    
                    # Brief pause to show progress
                    time.sleep(0.2)
                
                # Complete processing
                status_text.text("✅ Processing completed!")
                time.sleep(1)
                
                st.session_state.processing_in_progress = False
                status_text.empty()
                progress_bar.empty()
                
                st.rerun()
        
        # Show processing status
        if st.session_state.processing_in_progress:
            st.info("🔄 Processing in progress... Please wait.")
    
    # Results Section
    if st.session_state.processing_results:
        st.markdown("---")
        st.markdown("## 📸 Processing Results")
        
        # Calculate statistics
        successful = [r for r in st.session_state.processing_results if r['success']]
        failed = [r for r in st.session_state.processing_results if not r['success']]
        
        # Display statistics
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            st.metric("✅ Successful", len(successful), delta=None)
        with col_stats2:
            st.metric("❌ Failed", len(failed), delta=None)
        with col_stats3:
            st.metric("📊 Success Rate", f"{len(successful)/len(st.session_state.processing_results)*100:.0f}%")
        with col_stats4:
            if successful:
                total_size = sum(r.get('result_size', 0) for r in successful) / 1024 / 1024
                st.metric("💾 Output Size", f"{total_size:.1f}MB")
        
        # Show failed files if any
        if failed and show_errors:
            with st.expander("❌ Failed Files", expanded=False):
                for fail in failed:
                    st.error(f"**{fail['filename']}**: {fail['error']}")
        
        # Show successful results
        if successful:
            # Download section
            st.markdown("### 📥 Download Results")
            
            # Create download data
            zip_data = create_zip_download(successful)
            
            col_dl1, col_dl2 = st.columns([1, 1])
            
            with col_dl1:
                download_btn = st.download_button(
                    label=f"📦 Download All ({len(successful)} images)",
                    data=zip_data,
                    file_name=f"batch_edited_{int(time.time())}.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )
            
            with col_dl2:
                if st.button("📋 Copy Results Summary", use_container_width=True):
                    summary = f"Batch Edit Results:\n"
                    summary += f"✅ Successful: {len(successful)}\n"
                    summary += f"❌ Failed: {len(failed)}\n"
                    summary += f"📝 Prompt: {prompt[:100]}...\n"
                    st.code(summary)
            
            # Preview results
            if show_preview and successful:
                st.markdown("### 🖼️ Result Preview")
                
                # Determine preview count
                if max_preview == "All":
                    preview_count = len(successful)
                else:
                    preview_count = min(int(max_preview), len(successful))
                
                for i, result in enumerate(successful[:preview_count]):
                    with st.expander(f"🎨 Result {i+1}: {result['filename']}", expanded=i < 2):
                        col_before, col_after = st.columns(2)
                        
                        with col_before:
                            st.markdown("**📷 Original**")
                            # Find original image
                            original_data = None
                            for file_data in st.session_state.uploaded_files_data.values():
                                if file_data['filename'] == result['filename']:
                                    original_data = file_data
                                    break
                            
                            if original_data:
                                try:
                                    original_image = Image.open(io.BytesIO(original_data['bytes']))
                                    st.image(original_image, use_container_width=True)
                                    st.text(f"Size: {original_data['dimensions']}")
                                except:
                                    st.error("Could not display original image")
                        
                        with col_after:
                            st.markdown("**✨ Edited**")
                            try:
                                st.image(result['image_obj'], use_container_width =True)
                                edited_img = result['image_obj']
                                st.text(f"Size: {edited_img.width}x{edited_img.height}")
                                
                                # Individual download
                                img_buffer = io.BytesIO()
                                edited_img.save(img_buffer, format='PNG')
                                st.download_button(
                                    f"💾 Download",
                                    data=img_buffer.getvalue(),
                                    file_name=f"edited_{result['filename'].split('.')[0]}.png",
                                    mime="image/png",
                                    key=f"download_{i}"
                                )
                            except:
                                st.error("Could not display edited image")
                
                if len(successful) > preview_count:
                    st.info(f"Showing {preview_count} of {len(successful)} results. Download ZIP to get all images.")

if __name__ == "__main__":
    main()
