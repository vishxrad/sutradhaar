# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import openai
import os
import re
import requests
import asyncio
import aiohttp
from dotenv import load_dotenv
from image_generator import VertexImageGenerator
import time
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from fastapi.staticfiles import StaticFiles
import mimetypes
from jinja2 import Template
import base64
from pathlib import Path
from PIL import Image
import io
from presentation_templates import generate_html_template, get_theme_colors
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import texttospeech
from pdf2image import convert_from_path
import subprocess
import tempfile
import shutil
import glob

# Import all database functions from the new file
from database import (
    init_database,
    save_script_to_db,
    get_script_from_db,
    get_all_scripts_from_db,
    save_images_to_db,
    get_images_from_db,
    save_audio_to_db,
    get_audio_from_db,
    save_presentation_to_db,
    get_presentation_from_db,
    save_pdf_images_to_db,
    get_pdf_images_from_db,
    get_all_assets_from_db,
)


load_dotenv()

app = FastAPI()


@app.on_event("startup")
def on_startup():
    """Initialize the database when the application starts."""
    print("Application starting up...")
    try:
        init_database()
    except Exception as e:
        print(f"FATAL: Could not initialize database on startup: {e}")
        # In a real production app, you might want to exit if the DB is not available
        # import sys
        # sys.exit(1)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add static file serving for images
app.mount("/images", StaticFiles(directory="generated_images"), name="images")
app.mount(
    "/presentations",
    StaticFiles(directory="generated_presentations"),
    name="presentations",
)
app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")
app.mount(
    "/pdf-images", StaticFiles(directory="generated_pdf_images"), name="pdf-images"
)
app.mount("/chunks", StaticFiles(directory="generated_chunks"), name="chunks")
app.mount(
    "/generated_final_videos",
    StaticFiles(directory="generated_final_videos"),
    name="generated_final_videos",
)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models for request bodies
class ScriptRequest(BaseModel):
    topic: str


class ImageRequest(BaseModel):
    script_id: str
    use_unsplash_fallback: bool = True


class PresentationRequest(BaseModel):
    script_id: str
    template: str = "modern"


class HTMLGenerationRequest(BaseModel):
    script_id: str
    template: str = "modern"


class AudioRequest(BaseModel):
    script_id: str
    speaker: str = "female"  # "male" or "female"


class VideoChunkRequest(BaseModel):
    script_id: str


class CombineVideosRequest(BaseModel):
    script_id: str
    transition_type: str = Field(
        "fade",
        description="FFmpeg xfade transition type (e.g., fade, wipeleft, slideup, dissolve).",
    )
    transition_duration: float = Field(
        1.0, gt=0, description="Duration of each transition in seconds (must be > 0)."
    )
    output_filename: Optional[str] = Field(
        "final_presentation.mp4", description="Filename for the combined video."
    )


# Initialize OpenAI client
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Vertex AI Image Generator
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")

if GOOGLE_CLOUD_PROJECT:
    image_generator = VertexImageGenerator(
        project_id=GOOGLE_CLOUD_PROJECT, location=VERTEX_AI_LOCATION
    )
else:
    print(
        "Warning: GOOGLE_CLOUD_PROJECT environment variable not set. Image generation will be disabled."
    )
    image_generator = None


# --- APPLICATION HELPER FUNCTIONS ---
# (These are NOT database functions)

def parse_script_data(script_text):
    """
    Parses the script text from an LLM into a structured format.
    """
    segments_data = []
    
    segment_pattern = re.compile(
        r"Segment\s*\d+\s*:\s*(?P<title>[^\n]+?)\s*"
        r"(?:Summary:\s*(?P<summary>.*?)\s*)?"
        r"\s*(?P<slides_block>Slide\s*\d+:.*?)"
        r"(?=(Segment\s*\d+\s*:|$))",
        re.DOTALL | re.IGNORECASE
    )
    
    slide_pattern = re.compile(
        r"Slide\s*\d+\s*:\s*Title:\s*(?P<slide_title>.*?)\s*"
        r"Narration:\s*(?P<narration>.*?)\s*"
        r"(?:Image prompt:\s*(?P<image_prompt>.*?)\s*)?"
        r"(?=(Slide\s*\d+\s*:|Segment\s*\d+\s*:|$))",
        re.DOTALL | re.IGNORECASE
    )

    for segment_match in segment_pattern.finditer(script_text):
        title = segment_match.group("title").strip()
        title = title.strip('*').strip()  # Remove leading/trailing asterisks and whitespace
        
        summary_text = segment_match.group("summary")
        current_summary = summary_text.strip() if summary_text else ""
        
        slides_block_text = segment_match.group("slides_block")
        
        slides_list = []
        if slides_block_text:
            for slide_match in slide_pattern.finditer(slides_block_text):
                slide_title = slide_match.group("slide_title").strip()
                slide_narration = slide_match.group("narration").strip()
                
                image_prompt_text = slide_match.group("image_prompt")
                current_image_prompt = image_prompt_text.strip() if image_prompt_text else ""
                
                slides_list.append({
                    "title": slide_title,
                    "narration": slide_narration,
                    "image_prompt": current_image_prompt
                })
        
        segments_data.append({
            "segment_title": title,
            "summary": current_summary,
            "slides": slides_list
        })
    return segments_data

async def generate_vertex_ai_image_async(image_generator, prompt: str, output_dir: str, filename_prefix: str, max_retries: int = 2) -> Optional[str]:
    """
    Async wrapper for Vertex AI image generation with retries
    """
    if not image_generator:
        return None
    
    loop = asyncio.get_event_loop()
    
    for attempt in range(max_retries):
        try:
            with ThreadPoolExecutor() as executor:
                image_path = await loop.run_in_executor(
                    executor, 
                    image_generator.generate_image,
                    prompt,
                    output_dir,
                    filename_prefix
                )
                if image_path:
                    return image_path
        except Exception as e:
            print(f"Vertex AI attempt {attempt + 1} failed for '{prompt}': {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    
    return None

async def search_unsplash_image_async(query: str, unsplash_access_key: Optional[str] = None) -> Optional[str]:
    """
    Async version: Search for an image on Unsplash and return the URL
    """
    if not unsplash_access_key:
        unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    
    if not unsplash_access_key:
        print("Warning: UNSPLASH_ACCESS_KEY not set, cannot fallback to Unsplash")
        return None
    
    try:
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {unsplash_access_key}"}
        params = {
            "query": query,
            "per_page": 5,
            "orientation": "landscape",
            "order_by": "popular"
        }
        
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data["results"]:
                    top_results = data["results"][:3]
                    import random
                    selected = random.choice(top_results)
                    return selected["urls"]["regular"]
                else:
                    return None
                    
    except Exception as e:
        print(f"Error searching Unsplash for '{query}': {e}")
        return None

async def download_unsplash_image_async(image_url: str, output_dir: str, filename: str) -> Optional[str]:
    """
    Download an image from Unsplash URL and save it locally
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        parsed_url = urlparse(image_url)
        file_ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
        file_path = os.path.join(output_dir, f"{filename}{file_ext}")
        
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(image_url) as response:
                response.raise_for_status()
                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                
                print(f"Downloaded Unsplash image: {file_path}")
                return file_path
                
    except Exception as e:
        print(f"Error downloading Unsplash image from '{image_url}': {e}")
        return None

async def generate_single_image_with_fallback(
    image_generator,
    slide_info: dict,
    segment_idx: int,
    slide_idx: int,
    script_id: str,
    use_unsplash_fallback: bool = True
) -> dict:
    """
    Generate a single image with Vertex AI and fallback to Unsplash if needed
    """
    slide_key = f"segment_{segment_idx}_slide_{slide_idx}"
    segment_title = slide_info.get('segment_title', f'segment_{segment_idx}')
    segment_summary = slide_info.get('segment_summary', '')
    slide_title = slide_info.get('title', f'slide_{slide_idx}')
    slide_narration = slide_info.get('narration', '')
    image_prompt = slide_info.get('image_prompt', '')
    
    result = {
        "slide_key": slide_key,
        "segment_title": segment_title,
        "segment_summary": segment_summary,
        "slide_title": slide_title,
        "slide_narration": slide_narration,
        "image_prompt": image_prompt,
        "image_path": None,
        "unsplash_url": None,
        "source": "failed",
        "error": None
    }
    
    if not image_prompt:
        result["error"] = "No image prompt provided"
        return result
    
    output_dir = f"generated_images/{script_id}"
    filename_prefix = f"seg{segment_idx}_slide{slide_idx}"
    
    try:
        vertex_ai_path = await generate_vertex_ai_image_async(
            image_generator,
            image_prompt,
            output_dir,
            filename_prefix,
            max_retries=2
        )
        
        if vertex_ai_path:
            result["image_path"] = vertex_ai_path
            result["source"] = "vertex_ai"
            return result
            
    except Exception as e:
        result["error"] = f"Vertex AI failed: {str(e)}"
        print(f"Vertex AI failed for '{image_prompt}': {str(e)}")
    
    if use_unsplash_fallback:
        try:
            unsplash_url = await search_unsplash_image_async(image_prompt)
            if unsplash_url:
                result["unsplash_url"] = unsplash_url
                
                unsplash_filename = f"{filename_prefix}_unsplash"
                unsplash_local_path = await download_unsplash_image_async(
                    unsplash_url, 
                    output_dir, 
                    unsplash_filename
                )
                
                if unsplash_local_path:
                    result["image_path"] = unsplash_local_path
                    result["source"] = "unsplash"
                    return result
                else:
                    result["error"] = "Unsplash URL found but download failed"
            else:
                result["error"] = "Both Vertex AI and Unsplash failed"
        except Exception as e:
            result["error"] = f"Both Vertex AI and Unsplash failed. Unsplash error: {str(e)}"
    else:
        result["error"] = "Vertex AI failed and Unsplash fallback disabled"
    
    return result

async def synthesize_text_async(text: str, speaker: str, output_path: str) -> bool:
    """Async wrapper for Google Text-to-Speech synthesis"""
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            success = await loop.run_in_executor(
                executor,
                synthesize_text_sync,
                text,
                speaker,
                output_path
            )
            return success
    except Exception as e:
        print(f"Error in async TTS for '{output_path}': {e}")
        return False

def synthesize_text_sync(text: str, speaker: str, output_path: str) -> bool:
    """Synchronous Google Text-to-Speech synthesis"""
    try:
        client = texttospeech.TextToSpeechClient()
        cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
        if not cleaned_text:
            print(f"Empty text provided for {output_path}")
            return False
        
        ssml_text = f'<speak><prosody rate="100%">{cleaned_text}</prosody></speak>'
        input_text = texttospeech.SynthesisInput(ssml=ssml_text)
        
        if speaker.lower() == "male":
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                name="en-IN-Wavenet-F",
            )
        else:
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                name="en-IN-Wavenet-E",
            )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = client.synthesize_speech(
            request={
                "input": input_text,
                "voice": voice,
                "audio_config": audio_config
            }
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        
        print(f"Audio content written to '{output_path}'")
        return True
        
    except Exception as e:
        print(f"Error synthesizing text for '{output_path}': {e}")
        return False

def convert_pdf_to_images(script_id: str, pdf_path: str) -> dict:
    """
    Convert PDF to individual slide images using pdf2image
    """
    try:
        images_base_dir = "generated_pdf_images"
        script_images_dir = os.path.join(images_base_dir, script_id)
        os.makedirs(script_images_dir, exist_ok=True)
        
        images = convert_from_path(pdf_path, dpi=150, fmt='jpeg')
        
        images_data = []
        conversion_results = {
            "script_id": script_id,
            "pdf_path": pdf_path,
            "images_folder": script_images_dir,
            "total_slides": len(images),
            "image_paths": [],
            "file_sizes": [],
            "errors": []
        }
        
        for i, image in enumerate(images, 1):
            image_filename = f"slide_{i:03d}.jpg"
            image_path = os.path.join(script_images_dir, image_filename)
            
            try:
                image.save(image_path, 'JPEG', quality=85, optimize=True)
                file_size = os.path.getsize(image_path)
                conversion_results["image_paths"].append(image_path)
                conversion_results["file_sizes"].append(file_size)
                images_data.append({
                    "slide_number": i,
                    "image_path": image_path,
                    "filename": image_filename,
                    "file_size": file_size
                })
            except Exception as e:
                error_msg = f"Failed to save slide {i}: {str(e)}"
                conversion_results["errors"].append(error_msg)
        
        total_size = sum(conversion_results["file_sizes"])
        conversion_results["total_size_bytes"] = total_size
        conversion_results["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        if images_data:
            if save_pdf_images_to_db(script_id, images_data):
                conversion_results["database_saved"] = True
            else:
                conversion_results["database_saved"] = False
                conversion_results["errors"].append("Failed to save image paths to database")
        
        return conversion_results
        
    except Exception as e:
        error_msg = f"Error converting PDF to images: {str(e)}"
        return {
            "script_id": script_id,
            "pdf_path": pdf_path,
            "errors": [error_msg],
        }

def compress_pdf_ghostscript(input_path: str, output_path: str, quality: str = "ebook") -> dict:
    """
    Compress PDF using Ghostscript with detailed results
    """
    try:
        original_size = os.path.getsize(input_path)
        cmd = [
            "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS=/{quality}", "-dNOPAUSE", "-dQUIET", "-dBATCH",
            f"-sOutputFile={output_path}", input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_path):
            compressed_size = os.path.getsize(output_path)
            compression_ratio = (1 - compressed_size / original_size) * 100
            return {
                "success": True, "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": round(compression_ratio, 1),
                "size_reduction_mb": round((original_size - compressed_size) / (1024 * 1024), 2)
            }
        else:
            return {"success": False, "error": result.stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}

def _check_ffmpeg_tools():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH. Please install ffmpeg (which includes ffprobe).")

async def get_media_duration(media_path: str) -> Optional[float]:
    """
    Get the duration of a media file (audio or video) using ffprobe.
    """
    if not os.path.exists(media_path):
        return None
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", media_path
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            return float(stdout.decode().strip())
        else:
            print(f"ffprobe error for {media_path}: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"Error getting media duration for {media_path}: {e}")
        return None

async def create_single_video_chunk(
    chunk_order: int, image_path: str, audio_path: str, audio_duration: float,
    output_dir: str, script_id: str
) -> Dict[str, Any]:
    """
    Creates a single video chunk from an image and an audio file.
    """
    if not os.path.exists(image_path) or not os.path.exists(audio_path):
        raise FileNotFoundError("Image or audio file not found.")

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"chunk_{chunk_order:03d}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    total_video_duration = audio_duration + 2.0

    cmd = [
        "ffmpeg", "-loop", "1", "-framerate", "25", "-i", image_path,
        "-i", audio_path, "-vf", "fps=25,scale='trunc(iw/2)*2':-2,format=yuv420p",
        "-af", "adelay=1000ms:all=1,apad", "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage",
        "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
        "-t", str(total_video_duration), "-y", output_path
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_message = (
            f"FFmpeg error for chunk {chunk_order}:\n"
            f"Command: {' '.join(cmd)}\nStderr: {stderr.decode()}"
        )
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(error_message)

    return {
        "script_id": script_id, "chunk_order": chunk_order,
        "video_path": output_path, "image_source": image_path,
        "audio_source": audio_path, "original_audio_duration": audio_duration,
        "total_video_duration": total_video_duration, "filename": output_filename
    }


# --- API ENDPOINTS ---

@app.get("/")
def fetch_frontend():
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/generate-script")
def generate_script(request: ScriptRequest):
    """
    Generate a script for the given topic using OpenAI and save to database
    """
    topic = request.topic
    
    prompt = f"""You are a scriptwriter for an educational explainer video.
    The video will cover the topic: "{topic}" and should be structured into 5 distinct educational segments.
    The total narration for the entire video should be approximately 5 minutes. Make sure not to use any special characters like inverted commas, apostrophes, etc. or anything with a full stop that is not a full stop (like using e.g. or etc., instead use the full words like example or etectra)

    For each of the 5 segments:
    - The total narration for the segment should be approximately 1 minute (around 150 words).
    - Provide a segment 
    - Provide a short summary of the segment.
    - Divide the segment into 4 slides.
        For each of the 4 slides:
        - Provide a short title (max 5 words).
        - Write a narration script of approximately 30 words (so that 4 slides total ~150 words for the segment).
        - Generate multiple shorter sentences instead of paragraphs.
        - Suggest a visual description (image prompt for an AI image generator or Unsplash search). Make sure the prompt is not very complex and easy to understand by text to image models. Put in extra effort to make sure the images generated are as realistic and don't make the model hallucinate. Explain all the characterstics like lighting, number of objects etc in detail.

    Format the output cleanly, following this structure for each segment:

    Segment 1: [Segment Title Here]
    Summary: [Segment Summary Here]
    Slide 1:
    Title: [Slide 1 Title]
    Narration: [Slide 1 Narration - approx 30 words]
    Image prompt: [Slide 1 Image Prompt]
    Slide 2:
    Title: [Slide 2 Title]
    Narration: [Slide 2 Narration - approx 30 words]
    Image prompt: [Slide 2 Image Prompt]
    Slide 3:
    Title: [Slide 3 Title]
    Narration: [Slide 3 Narration - approx 30 words]
    Image prompt: [Slide 3 Image Prompt]
    Slide 4:
    Title: [Slide 4 Title]
    Narration: [Slide 4 Narration - approx 30 words]
    Image prompt: [Slide 4 Image Prompt]

    Ensure this structure is repeated for all 5 segments.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful scriptwriting assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_script_data = response.choices[0].message.content
        parsed_script_data = parse_script_data(raw_script_data)
        
        script_id = f"script_{int(time.time())}"
        
        # This now calls the function from database.py
        if save_script_to_db(script_id, topic, raw_script_data, parsed_script_data):
            return {
                "script_id": script_id,
                "topic": topic, 
                "parsed_script": parsed_script_data,
                "message": "Script generated and saved successfully."
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save script to database")
        
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating script: {e}")


@app.post("/generate-images")
async def generate_images(request: ImageRequest):
    """
    Generate images for a previously generated script in parallel and save to database.
    """
    script_id = request.script_id
    use_unsplash_fallback = request.use_unsplash_fallback
    
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found.")
    
    segments_data = script_data["parsed_script"]
    tasks = []
    
    overview_prompt = f"Create a professional, educational overview image representing the topic: {script_data['topic']}. Style: professional, educational, high-quality."
    overview_slide_info = {
        "title": f"Overview: {script_data['topic']}",
        "narration": f"Complete educational presentation covering {script_data['topic']}",
        "image_prompt": overview_prompt,
        "segment_title": "Presentation Overview",
        "segment_summary": f"Comprehensive content about {script_data['topic']}"
    }
    tasks.append(generate_single_image_with_fallback(
        image_generator, overview_slide_info, 0, 0, script_id, use_unsplash_fallback
    ))
    
    for segment_idx, segment in enumerate(segments_data, 1):
        for slide_idx, slide in enumerate(segment.get('slides', []), 1):
            if slide.get('image_prompt', ''):
                slide_info = {
                    **slide,
                    'segment_title': segment.get('segment_title', f'segment_{segment_idx}'),
                    'segment_summary': segment.get('summary', '')
                }
                tasks.append(generate_single_image_with_fallback(
                    image_generator, slide_info, segment_idx, slide_idx, script_id, use_unsplash_fallback
                ))
    
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
    end_time = asyncio.get_event_loop().time()
    
    image_results = {
        "script_id": script_id, "topic": script_data["topic"], "images": {},
        "stats": {
            "total_requested": len(tasks), "vertex_ai_success": 0,
            "unsplash_fallback": 0, "failed": 0, "errors": [],
            "generation_time_seconds": round(end_time - start_time, 2)
        }
    }
    
    for result in results:
        if isinstance(result, Exception):
            image_results["stats"]["failed"] += 1
            image_results["stats"]["errors"].append(f"Task failed: {str(result)}")
            continue
        
        if result["source"] == "vertex_ai":
            image_results["stats"]["vertex_ai_success"] += 1
        elif result["source"] == "unsplash":
            image_results["stats"]["unsplash_fallback"] += 1
        else:
            image_results["stats"]["failed"] += 1
            if result.get("error"):
                image_results["stats"]["errors"].append(f"{result['slide_key']}: {result['error']}")
        
        image_results["images"][result["slide_key"]] = result
    
    if image_results["images"]:
        if not save_images_to_db(script_id, image_results["images"]):
            image_results["warning"] = "Images generated but failed to save to database"
    
    return image_results


@app.get("/script/{script_id}")
def get_script(script_id: str):
    """
    Retrieve a previously generated script from database
    """
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    return script_data


@app.get("/scripts")
def list_scripts():
    """
    List all generated scripts from database
    """
    scripts = get_all_scripts_from_db()
    return {"scripts": scripts}


@app.get("/script/{script_id}/images")
def get_script_images(script_id: str):
    """
    Retrieve images for a specific script from database
    """
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    
    images = get_images_from_db(script_id)
    return {"script_id": script_id, "topic": script_data["topic"], "images": images}



# PASTE THIS CODE INTO YOUR main.py FILE

def generate_presentation_html(script_id: str):
    """
    Generate a complete standalone HTML file for the presentation.
    This is a helper function and does not return a response.
    """
    # Get script data
    script_data = get_script_from_db(script_id)
    if not script_data:
        # This will be caught by the calling endpoint
        raise HTTPException(status_code=404, detail="Script not found")

    # Get images data
    images_data = get_images_from_db(script_id)

    # Build slides data
    slides = []

    # 1. Title Slide
    title_slide = {"type": "title", "title": script_data["topic"], "order": 1}

    overview_key = "segment_0_slide_0"
    if overview_key in images_data:
        overview_image_info = images_data[overview_key]
        if overview_image_info.get("image_path") and os.path.exists(
            overview_image_info["image_path"]
        ):
            try:
                with open(overview_image_info["image_path"], "rb") as img_file:
                    image_data = img_file.read()
                    file_ext = Path(overview_image_info["image_path"]).suffix.lower()
                    mime_type = (
                        "image/jpeg" if file_ext in [".jpg", ".jpeg"] else "image/png"
                    )
                    image_base64 = f"data:{mime_type};base64,{base64.b64encode(image_data).decode()}"
                    title_slide["image_base64"] = image_base64
                    title_slide["image_alt"] = overview_image_info.get(
                        "image_prompt", f'Overview of {script_data["topic"]}'
                    )
            except Exception as e:
                print(f"Error encoding overview image for title slide: {e}")

    slides.append(title_slide)

    # 2-6. Process 5 Segments (Section + 4 Main slides each)
    segments_data = script_data["parsed_script"]
    slide_order = 2
    main_slide_layouts = [
        "main",
        "main-image-dominant",
        "main-image-dominant-2",
        "main-text-focus",
    ]
    layout_index = 0

    for segment_idx, segment in enumerate(segments_data, 1):
        # Section Slide
        slides.append(
            {
                "type": "section",
                "title": segment.get("segment_title", f"Segment {segment_idx}"),
                "body": segment.get("summary", ""),
                "order": slide_order,
            }
        )
        slide_order += 1

        # 4 Main slides for this segment
        for slide_idx, slide in enumerate(segment.get("slides", []), 1):
            slide_key = f"segment_{segment_idx}_slide_{slide_idx}"
            image_info = images_data.get(slide_key, {})
            image_base64 = None
            if image_info.get("image_path") and os.path.exists(
                image_info["image_path"]
            ):
                try:
                    with open(image_info["image_path"], "rb") as img_file:
                        image_data = img_file.read()
                        file_ext = Path(image_info["image_path"]).suffix.lower()
                        mime_type = (
                            "image/jpeg"
                            if file_ext in [".jpg", ".jpeg"]
                            else "image/png"
                        )
                        image_base64 = f"data:{mime_type};base64,{base64.b64encode(image_data).decode()}"
                except Exception as e:
                    print(f"Error encoding image {image_info['image_path']}: {e}")

            chosen_layout = main_slide_layouts[layout_index % len(main_slide_layouts)]
            layout_index += 1

            slides.append(
                {
                    "type": chosen_layout,
                    "title": slide.get("title", f"Slide {slide_idx}"),
                    "body": slide.get("narration", ""),
                    "image_base64": image_base64,
                    "image_alt": slide.get("image_prompt", ""),
                    "order": slide_order,
                }
            )
            slide_order += 1

    # 27. Thank You Slide
    slides.append(
        {
            "type": "thankyou",
            "title": "Thank You",
            "subtitle": "Made using Sutradhaar",
            "order": slide_order,
        }
    )

    # Generate theme colors
    theme_colors = get_theme_colors(images_data)

    # Generate HTML using template
    html_content = generate_html_template(
        topic=script_data["topic"],
        slides=slides,
        theme=theme_colors,
        script_id=script_id,
    )

    return {
        "script_id": script_id,
        "topic": script_data["topic"],
        "html_content": html_content,
        "filename": f"{script_id}_presentation.html",
    }


@app.get("/presentation/{script_id}/pdf")
async def generate_presentation_pdf(script_id: str):
    """
    Generate and return a PDF presentation using Decktape.
    Also converts PDF to individual slide images and stores them.
    """
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")

    html_result = generate_presentation_html(script_id)
    html_content = html_result["html_content"]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_html:
        temp_html.write(html_content)
        temp_html_path = temp_html.name
    
    presentations_dir = "generated_presentations"
    os.makedirs(presentations_dir, exist_ok=True)
    filename = f"{script_id}_presentation.pdf"
    pdf_path = os.path.join(presentations_dir, filename)
    
    try:
        cmd = ["decktape", "reveal", temp_html_path, pdf_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {result.stderr}")
        
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            raise HTTPException(status_code=500, detail="PDF file was not created")
        
        # Compress the PDF
        temp_compressed = pdf_path.replace('.pdf', '_temp_compressed.pdf')
        compression_result = compress_pdf_ghostscript(pdf_path, temp_compressed, "ebook")
        
        if compression_result["success"]:
            os.replace(temp_compressed, pdf_path)
            file_size = compression_result["compressed_size"]
        else:
            file_size = os.path.getsize(pdf_path)
        
        save_presentation_to_db(script_id, pdf_path, filename, file_size)
        
        # Convert PDF to images
        convert_pdf_to_images(script_id, pdf_path)
        
        return FileResponse(
            path=pdf_path, media_type="application/pdf", filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")
    finally:
        if 'temp_html_path' in locals() and os.path.exists(temp_html_path):
            os.unlink(temp_html_path)


@app.get("/presentation/{script_id}/images")
def get_presentation_images(script_id: str):
    """
    Get information about the PDF slide images for a presentation.
    """
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    
    pdf_images = get_pdf_images_from_db(script_id)
    if not pdf_images:
        raise HTTPException(status_code=404, detail="PDF images not found. Generate PDF first.")
    
    valid_images = [img for img in pdf_images if os.path.exists(img["image_path"])]
    total_size = sum(img["file_size"] for img in valid_images)
    
    return {
        "script_id": script_id, "topic": script_data["topic"],
        "total_images": len(valid_images),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "images": valid_images
    }


@app.post("/generate-audio")
async def generate_audio(request: AudioRequest):
    """
    Generate audio files for a script using Google Text-to-Speech.
    """
    script_id = request.script_id
    speaker = request.speaker.lower()
    
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found.")
    
    images_data = get_images_from_db(script_id)
    segments_data = script_data["parsed_script"]
    audio_dir = f"generated_audio/{script_id}"
    os.makedirs(audio_dir, exist_ok=True)
    
    tasks_with_info = []
    audio_counter = 1
    
    # Title slide audio
    title_text = f"Welcome to our presentation on {script_data['topic']}"
    tasks_with_info.append({
        "content": title_text, "speaker": speaker,
        "path": os.path.join(audio_dir, f"audio_{audio_counter}.mp3"),
        "key": "title_slide", "type": "title", "seg_idx": 0, "slide_idx": 0
    })
    audio_counter += 1
    
    # Segment and slide audio
    for seg_idx, segment in enumerate(segments_data, 1):
        if segment.get('summary', '').strip():
            tasks_with_info.append({
                "content": segment['summary'], "speaker": speaker,
                "path": os.path.join(audio_dir, f"audio_{audio_counter}.mp3"),
                "key": f"summary_seg{seg_idx}", "type": "summary", "seg_idx": seg_idx, "slide_idx": None
            })
            audio_counter += 1
        
        for slide_idx, slide in enumerate(segment.get('slides', []), 1):
            narration = images_data.get(f"segment_{seg_idx}_slide_{slide_idx}", {}).get('slide_narration', slide.get('narration', ''))
            if narration.strip():
                tasks_with_info.append({
                    "content": narration, "speaker": speaker,
                    "path": os.path.join(audio_dir, f"audio_{audio_counter}.mp3"),
                    "key": f"narration_seg{seg_idx}_slide{slide_idx}", "type": "narration", "seg_idx": seg_idx, "slide_idx": slide_idx
                })
                audio_counter += 1
    
    # Thank you slide audio
    thank_you_text = "Thank you for your attention. This presentation was made using Sutradhaar."
    tasks_with_info.append({
        "content": thank_you_text, "speaker": speaker,
        "path": os.path.join(audio_dir, f"audio_{audio_counter}.mp3"),
        "key": "thank_you_slide", "type": "thank_you", "seg_idx": 99, "slide_idx": 99
    })
    
    tts_tasks = [synthesize_text_async(t["content"], t["speaker"], t["path"]) for t in tasks_with_info]
    results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    
    audio_files_to_save = {}
    successful_count = 0
    for i, result in enumerate(results):
        task_info = tasks_with_info[i]
        if result is True:
            successful_count += 1
            audio_files_to_save[task_info["key"]] = {
                "audio_type": task_info["type"], "segment_idx": task_info["seg_idx"],
                "slide_idx": task_info["slide_idx"], "content": task_info["content"],
                "audio_path": task_info["path"], "speaker": task_info["speaker"]
            }
    
    if audio_files_to_save:
        save_audio_to_db(script_id, audio_files_to_save)
    
    return {
        "script_id": script_id, "topic": script_data["topic"],
        "stats": {"successful": successful_count, "failed": len(results) - successful_count}
    }


@app.post("/generate-video-chunks", summary="Generate Individual Video Chunks")
async def generate_video_chunks_endpoint(request: VideoChunkRequest):
    """
    Generates individual video chunks for a given script_id.
    """
    script_id = request.script_id
    _check_ffmpeg_tools()

    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found.")

    pdf_images = get_pdf_images_from_db(script_id)
    audio_map = get_audio_from_db(script_id)
    if not pdf_images or not audio_map:
        raise HTTPException(status_code=404, detail="PDF images or audio not found.")
    
    ordered_audio_files = list(audio_map.values())
    if len(pdf_images) != len(ordered_audio_files):
        raise HTTPException(status_code=500, detail="Mismatch in number of images and audio files.")

    script_output_dir = os.path.join("generated_chunks", script_id)
    
    duration_tasks = [get_media_duration(af.get("audio_path")) for af in ordered_audio_files]
    audio_durations = await asyncio.gather(*duration_tasks)
    
    video_creation_tasks = []
    for i, pdf_image_info in enumerate(pdf_images):
        if audio_durations[i] is not None:
            video_creation_tasks.append(
                create_single_video_chunk(
                    chunk_order=i + 1, image_path=pdf_image_info["image_path"],
                    audio_path=ordered_audio_files[i]["audio_path"],
                    audio_duration=audio_durations[i],
                    output_dir=script_output_dir, script_id=script_id
                )
            )
    
    generation_results = await asyncio.gather(*video_creation_tasks, return_exceptions=True)
    
    successful_chunks = [res for res in generation_results if not isinstance(res, Exception)]
    failed_chunks = [str(res) for res in generation_results if isinstance(res, Exception)]
    
    return {
        "script_id": script_id, "topic": script_data["topic"],
        "chunks_successfully_generated": len(successful_chunks),
        "chunks_failed_generation": len(failed_chunks),
        "generated_chunks_details": successful_chunks,
        "generation_errors": failed_chunks
    }


@app.post("/combine-video-chunks-fast", summary="Fast Video Concatenation (No Transitions)")
async def combine_video_chunks_fast_endpoint(request: VideoChunkRequest):
    """
    Combines individual video chunks using FFmpeg's concat demuxer.
    """
    script_id = request.script_id
    chunks_input_dir = os.path.join("generated_chunks", script_id)
    if not os.path.isdir(chunks_input_dir):
        raise HTTPException(status_code=404, detail=f"Chunk directory not found for script_id: {script_id}")

    chunk_files = sorted(glob.glob(os.path.join(chunks_input_dir, "chunk_*.mp4")))
    if not chunk_files:
        raise HTTPException(status_code=404, detail=f"No video chunk files found in {chunks_input_dir}")

    final_output_dir = os.path.join("generated_final_videos", script_id)
    os.makedirs(final_output_dir, exist_ok=True)
    
    concat_file_path = os.path.join(final_output_dir, f"{script_id}_concat_list.txt")
    output_filename = f"final_presentation_{script_id}.mp4"
    final_video_path = os.path.join(final_output_dir, output_filename)
    
    try:
        with open(concat_file_path, 'w') as f:
            for chunk_file in chunk_files:
                f.write(f"file '{os.path.abspath(chunk_file)}'\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating concat file: {e}")

    ffmpeg_cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file_path,
        "-c", "copy", "-y", final_video_path
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg fast concat failed: {stderr.decode()}")

        return {
            "script_id": script_id,
            "final_video_path": final_video_path,
            "video_url": f"/generated_final_videos/{script_id}/{output_filename}",
            "message": "Video combined successfully",
            "file_exists": os.path.exists(final_video_path)
        }
    finally:
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)


@app.get("/assets/all")
def get_all_assets():
    """
    Retrieve all generated scripts, PDFs, and videos from the database.
    """
    try:
        db_assets = get_all_assets_from_db()
        assets = []

        for row in db_assets:
            script_id = row["script_id"]
            video_path, video_url, video_file_size = None, None, 0
            
            final_video_dir = f"generated_final_videos/{script_id}"
            if os.path.exists(final_video_dir):
                video_files = glob.glob(os.path.join(final_video_dir, "*.mp4"))
                if video_files:
                    video_path = video_files[0]
                    video_filename = os.path.basename(video_path)
                    video_url = f"/generated_final_videos/{script_id}/{video_filename}"
                    video_file_size = os.path.getsize(video_path)

            def format_file_size(size_bytes):
                if not size_bytes or size_bytes == 0: return "0 B"
                if size_bytes < 1024: return f"{size_bytes} B"
                if size_bytes < 1024**2: return f"{size_bytes/1024:.1f} KB"
                if size_bytes < 1024**3: return f"{size_bytes/(1024**2):.1f} MB"
                return f"{size_bytes/(1024**3):.1f} GB"

            assets.append({
                "script_id": script_id, "topic": row["topic"],
                "script_created_at": row["script_created_at"],
                "script_url": f"/script/{script_id}",
                "pdf": {
                    "available": bool(row["pdf_path"] and os.path.exists(row["pdf_path"])),
                    "path": row["pdf_path"], "filename": row["pdf_filename"],
                    "file_size": format_file_size(row.get("pdf_file_size")),
                    "download_url": f"/presentation/{script_id}/pdf" if row["pdf_path"] else None,
                    "created_at": row["pdf_created_at"],
                },
                "video": {
                    "available": bool(video_path), "path": video_path,
                    "file_size": format_file_size(video_file_size),
                    "video_url": video_url,
                    "filename": os.path.basename(video_path) if video_path else None,
                },
            })
        
        return {"total_assets": len(assets), "assets": assets}
            
    except Exception as e:
        print(f"Error retrieving all assets: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving assets: {str(e)}")

# The /combine-video-chunks endpoint was omitted for brevity but should be included
# if you need transitions. The fast version is included above.