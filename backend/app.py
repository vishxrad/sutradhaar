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
from presentation_templates import create_presentation
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import texttospeech
from pdf2image import convert_from_path
import subprocess
import tempfile
import shutil
import glob
from enum import Enum

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

class VideoType(str, Enum):
    SUMMARY = "summary"
    RECAP = "recap"
    EXPLAINER = "explainer"

DURATION_MAPPING = {
    VideoType.SUMMARY: 1,
    VideoType.RECAP: 2,
    VideoType.EXPLAINER: 5,
}


class ScriptRequest(BaseModel):
    topic: str
    video_type: VideoType = Field(
        default=VideoType.EXPLAINER,
        description="The type of video to generate.",
    )


class TemplateName(str, Enum):
    MODERN = "modern"
    CRIMSON = "crimson"

class PresentationPDFRequest(BaseModel):
    script_id: str
    template: TemplateName = Field(
        default=TemplateName.MODERN,
        description="The name of the presentation template to use."
    )


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


class ProcessingMode(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class VideoGenerationRequest(BaseModel):
    script_id: str
    processing_mode: ProcessingMode = Field(
        default=ProcessingMode.PARALLEL,
        description="Choose 'parallel' for faster processing on multi-core systems, or 'sequential' for lower resource usage.",
    )


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

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_Original"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.studio.nebius.com/v1/"),
)

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

import re

# In main.py, replace the parse_script_data function

# In main.py, update the parse_script_data function

def parse_script_data(script_text):
    """
    Parses script text, handling and CLEANING conditional visual types.
    """
    slides_data = []
    slide_pattern = re.compile(
        r"Slide\s*\d+\s*:\s*"
        r"Title:\s*(?P<title>.*?)\s*"
        r"Narration:\s*(?P<narration>.*?)\s*"
        r"Slide content:\s*(?P<slide_content>.*?)\s*"
        r"visual_type:\s*(?P<visual_type>\w+)\s*"
        r"(?P<visual_data>.*?)"
        r"(?=(Slide\s*\d+\s*:|$))",
        re.DOTALL | re.IGNORECASE,
    )

    for match in slide_pattern.finditer(script_text):
        groups = match.groupdict()
        
        raw_content = groups.get("slide_content", "").strip()
        content_points = [
            line.strip().lstrip("- ").strip()
            for line in raw_content.split("\n")
            if line.strip()
        ]

        slide_info = {
            "title": groups.get("title", "").strip(),
            "narration": groups.get("narration", "").strip(),
            "slide_content": content_points,
            "visual_type": groups.get("visual_type", "ai_image").strip(),
        }

        visual_data = groups.get("visual_data", "").strip()
        if slide_info["visual_type"] == "ai_image":
            prompt_match = re.search(r"image_prompt:\s*(.*)", visual_data, re.DOTALL)
            slide_info["image_prompt"] = prompt_match.group(1).strip() if prompt_match else ""
        elif slide_info["visual_type"] == "chart":
            data_match = re.search(r"chart_data:\s*({.*})", visual_data, re.DOTALL)
            slide_info["chart_data"] = data_match.group(1).strip() if data_match else "{}"
        elif slide_info["visual_type"] == "flowchart":
            code_match = re.search(r"flowchart_code:\s*(.*)", visual_data, re.DOTALL)
            if code_match:
                # --- THIS IS THE FIX ---
                # Clean the mermaid code: remove trailing junk like '---'
                mermaid_code = code_match.group(1).strip()
                slide_info["flowchart_code"] = mermaid_code.strip().rstrip(';').strip().rstrip('-').strip()
            else:
                slide_info["flowchart_code"] = ""
            
        slides_data.append(slide_info)

    return slides_data




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

async def generate_visual_for_slide(
    slide_info: dict, slide_idx: int, script_id: str, use_unsplash_fallback: bool
) -> dict:
    """
    Dispatcher function that generates a visual based on the slide's 'visual_type'.
    """
    visual_type = slide_info.get("visual_type", "ai_image")
    output_dir = f"generated_images/{script_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a unique filename for each slide
    filename_prefix = f"slide_{slide_idx:02d}"
    
    result = {
        "slide_key": f"slide_{slide_idx}",
        "slide_title": slide_info.get("title"),
        "visual_type": visual_type,
        "image_path": None,
        "source": "failed",
        "error": None
    }

    try:
        if visual_type == "chart":
            chart_data = slide_info.get("chart_data")
            if not chart_data:
                raise ValueError("Chart data is missing.")
            output_path = os.path.join(output_dir, f"{filename_prefix}_chart.png")
            result["image_path"] = await generate_plotly_chart_image(chart_data, output_path)
            result["source"] = "plotly"

        elif visual_type == "flowchart":
            mermaid_code = slide_info.get("flowchart_code")
            if not mermaid_code:
                raise ValueError("Flowchart code is missing.")
            output_path = os.path.join(output_dir, f"{filename_prefix}_flowchart.png")
            result["image_path"] = await generate_mermaid_diagram_image(mermaid_code, output_path)
            result["source"] = "mermaid"

        else: # Default to 'ai_image'
            image_prompt = slide_info.get("image_prompt")
            if not image_prompt:
                raise ValueError("AI image prompt is missing.")
            
            # Reuse the existing Vertex AI + Unsplash logic
            vertex_path = await generate_vertex_ai_image_async(image_generator, image_prompt, output_dir, filename_prefix)
            if vertex_path:
                result["image_path"] = vertex_path
                result["source"] = "vertex_ai"
            elif use_unsplash_fallback:
                unsplash_url = await search_unsplash_image_async(image_prompt)
                if unsplash_url:
                    unsplash_path = await download_unsplash_image_async(unsplash_url, output_dir, f"{filename_prefix}_unsplash")
                    if unsplash_path:
                        result["image_path"] = unsplash_path
                        result["source"] = "unsplash"
                    else:
                        raise Exception("Unsplash download failed.")
                else:
                    raise Exception("Vertex AI and Unsplash search both failed.")
            else:
                raise Exception("Vertex AI failed and Unsplash fallback is disabled.")

    except Exception as e:
        result["error"] = str(e)
        print(f"Failed to generate visual for slide {slide_idx}: {e}")

    return result

# Add these helper functions near your other helpers in main.py

import plotly.graph_objects as go
import json

async def generate_plotly_chart_image(chart_data: str, output_path: str) -> str:
    """
    Generates a chart image from JSON data using Plotly.
    """
    try:
        data = json.loads(chart_data)
        fig_type = data.get("type", "bar")
        
        if fig_type == "bar":
            fig = go.Figure(data=[go.Bar(x=data.get("x"), y=data.get("y"))])
        elif fig_type == "pie":
            fig = go.Figure(data=[go.Pie(labels=data.get("labels"), values=data.get("values"))])
        elif fig_type == "line":
            fig = go.Figure(data=[go.Scatter(x=data.get("x"), y=data.get("y"), mode='lines+markers')])
        else: # Default to bar chart if type is unknown
            fig = go.Figure(data=[go.Bar(x=data.get("x"), y=data.get("y"))])

        fig.update_layout(
            title_text=data.get("title", "Chart"),
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=18, color="black")
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use asyncio.to_thread to run the blocking I/O operation
        await asyncio.to_thread(fig.write_image, output_path, width=1280, height=720, scale=1)
        
        print(f"Plotly chart saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating Plotly chart: {e}")
        raise  # Re-raise the exception to be caught by the caller

async def generate_mermaid_diagram_image(mermaid_code: str, output_path: str) -> str:
    """
    Generates a diagram image from Mermaid syntax using mermaid-cli.
    """
    if shutil.which("mmdc") is None:
        raise RuntimeError("mermaid-cli (mmdc) not found. Please run 'npm install -g @mermaid-js/mermaid-cli'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as temp_mmd:
        temp_mmd.write(mermaid_code)
        temp_mmd_path = temp_mmd.name

    cmd = ["mmdc", "-i", temp_mmd_path, "-o", output_path, "-w", "1280", "-H", "720"]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"Mermaid-cli failed: {stderr.decode()}")
        
        print(f"Mermaid diagram saved to {output_path}")
        return output_path
    finally:
        if os.path.exists(temp_mmd_path):
            os.remove(temp_mmd_path)


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
            # For empty text, synthesize a short pause to create a silent audio file.
            # This is crucial to keep audio and video tracks in sync.
            cleaned_text = " "
        
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
    Generate a script with narration and on-screen content for each slide.
    """
    topic = request.topic
    video_type = request.video_type
    duration_in_minutes = DURATION_MAPPING[video_type]

    slides_per_minute = 4
    total_slides = duration_in_minutes * slides_per_minute
    total_words = duration_in_minutes * 200
    words_per_slide_narration = round(total_words / total_slides)

    # --- Updated Prompt ---
    # In main.py, replace the prompt in the /generate-script endpoint

    prompt = f"""You are an expert scriptwriter and data visualizer for educational videos.
    The video will cover the topic: "{topic}" and have {total_slides} slides.

    For each of the {total_slides} slides, you must decide the best visual representation.
    You have three choices for the 'visual_type':
    1.  'ai_image': For conceptual, abstract, or general topics.
    2.  'chart': For slides with clear numerical data, comparisons, or statistics.
    3.  'flowchart': For processes, decision trees, or workflows.

    Based on your choice, provide the corresponding data with these STRICT rules:
    - If 'visual_type' is 'ai_image', provide an 'image_prompt'.
    - If 'visual_type' is 'chart', provide 'chart_data' as a valid JSON object. You MUST generate plausible sample data. The 'x', 'y', 'labels', and 'values' fields must ALWAYS be JSON arrays (lists), NOT single strings.
    - If 'visual_type' is 'flowchart', provide clean 'flowchart_code' using Mermaid syntax. Do NOT add any extra characters like '---' at the end.

    Format the output for each slide EXACTLY as shown in the examples below.

    ---
    EXAMPLE FOR AI IMAGE:
    Slide 1:
    Title: The Concept of Gravity
    Narration: Gravity is the invisible force that pulls objects together. It's what keeps the planets in orbit around the sun and what keeps you on the ground.
    Slide content:
    - Invisible force pulling objects together
    - Keeps planets in orbit
    - Holds us on the ground
    visual_type: ai_image
    image_prompt: A majestic view of the solar system with glowing orbital lines around the sun, showing planets held in their paths, cosmic, educational style.

    ---
    EXAMPLE FOR CHART (BAR CHART):
    Slide 2:
    Title: Poverty Rates in Key Regions
    Narration: Poverty remains a significant challenge, with rates varying across different regions. Some areas show higher concentrations of poverty than others.
    Slide content:
    - Region A: 25.4%
    - Region B: 18.2%
    - Region C: 32.5%
    visual_type: chart
    chart_data: {{"type": "bar", "title": "Poverty Rate by Region", "x": ["Region A", "Region B", "Region C"], "y": [25.4, 18.2, 32.5]}}

    ---
    EXAMPLE FOR CHART (PIE CHART):
    Slide 3:
    Title: Earth's Atmospheric Composition
    Narration: Our atmosphere is a mix of gases. It's primarily composed of nitrogen and oxygen, with small amounts of other gases that are crucial for life.
    Slide content:
    - Nitrogen: ~78%
    - Oxygen: ~21%
    - Other Gases: ~1%
    visual_type: chart
    chart_data: {{"type": "pie", "title": "Atmospheric Composition", "labels": ["Nitrogen", "Oxygen", "Other"], "values": [78, 21, 1]}}

    ---
    EXAMPLE FOR FLOWCHART:
    Slide 4:
    Title: The Scientific Method
    Narration: The scientific method is a structured process for inquiry. It starts with an observation, leads to a hypothesis, followed by experimentation, and finally, a conclusion.
    Slide content:
    - Start with an Observation
    - Form a Hypothesis
    - Conduct Experiment
    - Analyze Data & Conclude
    visual_type: flowchart
    flowchart_code: graph TD; A[Observation] --> B(Form Hypothesis); B --> C{{Experiment}}; C --> D((Analyze Data)); D --> E[Conclusion];

    ---
    Now, generate the complete script for the topic "{topic}" following these strict rules and formats for all {total_slides} slides.
    """

    try:
        # The rest of your endpoint logic remains the same...
        response = client.chat.completions.create(
            model="Qwen/Qwen3-14B",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful scriptwriting assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw_script_data = response.choices[0].message.content
        parsed_script_data = parse_script_data(raw_script_data) # This now needs the updated parser
        script_id = f"script_{int(time.time())}"

        if save_script_to_db(
            script_id, topic, raw_script_data, parsed_script_data
        ):
            return {
                "script_id": script_id,
                "topic": topic,
                "parsed_script": parsed_script_data,
                "message": f"Script for '{video_type.value}' generated successfully.",
            }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to save script to database"
            )

    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating script: {e}"
        )

@app.post("/generate-images")
async def generate_images(request: ImageRequest):
    """
    Generate visuals for a script, dispatching to the appropriate generator.
    """
    script_id = request.script_id
    use_unsplash_fallback = request.use_unsplash_fallback
    
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found.")
    
    # The new parser returns a flat list of slides
    slides_data = script_data["parsed_script"]
    tasks = []
    
    for idx, slide in enumerate(slides_data):
        tasks.append(
            generate_visual_for_slide(slide, idx + 1, script_id, use_unsplash_fallback)
        )
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    image_results = {
        "script_id": script_id,
        "topic": script_data["topic"],
        "images": {},
        "stats": {
            "total_requested": len(tasks),
            "sources": {"vertex_ai": 0, "unsplash": 0, "plotly": 0, "mermaid": 0, "failed": 0},
            "errors": [],
            "generation_time_seconds": round(end_time - start_time, 2)
        }
    }
    
    for res in results:
        source = res["source"]
        image_results["stats"]["sources"][source] += 1
        if source == "failed" and res.get("error"):
            image_results["stats"]["errors"].append(f"Slide {res['slide_key']}: {res['error']}")
        
        # Use the slide_key from the result for the dictionary
        image_results["images"][res["slide_key"]] = res
    
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



@app.post("/generate-presentation/pdf", summary="Generate Presentation as PDF")
async def generate_presentation_pdf(request: PresentationPDFRequest):
    """
    Generate and return a PDF presentation using a selected template.
    """
    script_id = request.script_id
    template_name = request.template.value

    # 1. Get all the necessary data from the database
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")

    images_data = get_images_from_db(script_id) or {}
    parsed_slides = script_data.get("parsed_script", [])

    # 2. Transform the flat slides structure into the format expected by templates
    try:
        formatted_slides = []
        
        # Title slide
        formatted_slides.append({
            "type": "title",
            "title": script_data["topic"],
            "order": 1
        })
        
        # Main content slides
        for idx, slide in enumerate(parsed_slides, 2):
            slide_key = f"slide_{idx-1}"  # slides are 1-indexed in images_data
            image_info = images_data.get(slide_key, {})
            
            # Convert image to base64 if available
            image_base64 = None
            if image_info.get("image_path") and os.path.exists(image_info["image_path"]):
                try:
                    with open(image_info["image_path"], "rb") as img_file:
                        image_data = img_file.read()
                        file_ext = Path(image_info["image_path"]).suffix.lower()
                        mime_type = "image/jpeg" if file_ext in [".jpg", ".jpeg"] else "image/png"
                        image_base64 = f"data:{mime_type};base64,{base64.b64encode(image_data).decode()}"
                except Exception as e:
                    print(f"Error encoding image {image_info['image_path']}: {e}")
            
            formatted_slides.append({
                "type": "main",
                "title": slide.get("title", f"Slide {idx-1}"),
                "body": slide.get("slide_content", []),  # This will be formatted by the template filter
                "image_base64": image_base64,
                "image_alt": slide.get("image_prompt", ""),
                "order": idx
            })
        
        # Thank you slide
        formatted_slides.append({
            "type": "thankyou",
            "title": "Thank You",
            "subtitle": "Made using Sutradhaar",
            "order": len(formatted_slides) + 1
        })

        html_content = create_presentation(
            topic=script_data["topic"],
            slides=formatted_slides,
            template_name=template_name,
            images_data=images_data,
            script_id=script_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate HTML: {e}")

    # 3. The rest of the PDF generation logic remains the same
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_html:
        temp_html.write(html_content)
        temp_html_path = temp_html.name
    
    presentations_dir = "generated_presentations"
    os.makedirs(presentations_dir, exist_ok=True)
    filename = f"{script_id}_{template_name}_presentation.pdf"
    pdf_path = os.path.join(presentations_dir, filename)
    
    try:
        cmd = ["decktape", "reveal", temp_html_path, pdf_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {result.stderr}")
        
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            raise HTTPException(status_code=500, detail="PDF file was not created")
        
        save_presentation_to_db(script_id, pdf_path, filename, os.path.getsize(pdf_path))
        convert_pdf_to_images(script_id, pdf_path)
        
        return FileResponse(
            path=pdf_path, media_type="application/pdf", filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
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


# REPLACE the existing /generate-audio endpoint in main.py with this

@app.post("/generate-audio")
async def generate_audio(request: AudioRequest):
    """
    Generate audio files for a script using Google Text-to-Speech.
    This version ensures an audio file is generated for every slide to maintain
    sync with the video presentation.
    """
    script_id = request.script_id
    speaker = request.speaker.lower()
    
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found.")
    
    parsed_slides = script_data.get("parsed_script", [])
    audio_dir = f"generated_audio/{script_id}"
    os.makedirs(audio_dir, exist_ok=True)
    
    # Create a list of all narrations in order to ensure a 1:1 match with PDF slides
    narrations_to_process = []

    # 1. Title slide
    narrations_to_process.append({
        "content": f"Welcome to our presentation on {script_data['topic']}",
        "key": "title_slide",
        "type": "title"
    })

    # 2. Main content slides
    for idx, slide in enumerate(parsed_slides, 1):
        narrations_to_process.append({
            "content": slide.get('narration', '').strip(),
            "key": f"slide_{idx}",
            "type": "narration"
        })

    # 3. Thank you slide
    narrations_to_process.append({
        "content": "Thank you for your attention. This presentation was made using Sutradhaar.",
        "key": "thank_you_slide",
        "type": "thank_you"
    })
    
    tasks_with_info = []
    # Use a sequential, 3-digit padded index for audio files to match PDF slide images
    for idx, narration_info in enumerate(narrations_to_process, 1):
        tasks_with_info.append({
            "content": narration_info["content"],
            "speaker": speaker,
            "path": os.path.join(audio_dir, f"audio_{idx:03d}.mp3"),
            "key": narration_info["key"],
            "type": narration_info["type"]
        })
    
    # The rest of the logic is the same
    tts_tasks = [synthesize_text_async(t["content"], t["speaker"], t["path"]) for t in tasks_with_info]
    results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    
    audio_files_to_save = {}
    successful_count = 0
    for i, result in enumerate(results):
        task_info = tasks_with_info[i]
        if result is True:
            successful_count += 1
            audio_files_to_save[task_info["key"]] = {
                "audio_type": task_info["type"],
                "content": task_info["content"],
                "audio_path": task_info["path"],
                "speaker": task_info["speaker"]
            }
    
    if audio_files_to_save:
        save_audio_to_db(script_id, audio_files_to_save)
    
    return {
        "script_id": script_id, "topic": script_data["topic"],
        "stats": {"successful": successful_count, "failed": len(results) - successful_count}
    }


@app.post("/generate-video", summary="Generate and Combine Full Video")
async def generate_full_video(request: VideoGenerationRequest):
    """
    Generates individual video chunks and combines them into a final video.
    Allows choosing between parallel and sequential processing for chunk generation.
    """
    script_id = request.script_id
    mode = request.processing_mode

    # --- Stage 1: Setup and Data Validation ---
    _check_ffmpeg_tools()

    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(
            status_code=404, detail=f"Script '{script_id}' not found."
        )

    pdf_images = get_pdf_images_from_db(script_id)
    audio_map = get_audio_from_db(script_id)
    if not pdf_images or not audio_map:
        raise HTTPException(
            status_code=404,
            detail="PDF images or audio not found. Please generate them first.",
        )

    # Sort audio files based on their filename to ensure correct order.
    # The filenames are padded (e.g., audio_001.mp3), so a simple string sort is reliable.
    ordered_audio_files = sorted(list(audio_map.values()), key=lambda x: x.get("audio_path", ""))

    if len(pdf_images) != len(ordered_audio_files):
        raise HTTPException(
            status_code=500,
            detail=f"Mismatch in number of images ({len(pdf_images)}) and audio files ({len(ordered_audio_files)}).",
        )

    # --- Stage 2: Video Chunk Generation ---
    chunks_output_dir = os.path.join("generated_chunks", script_id)

    duration_tasks = [
        get_media_duration(af.get("audio_path")) for af in ordered_audio_files
    ]
    audio_durations = await asyncio.gather(*duration_tasks)

    video_creation_tasks = []
    for i, pdf_image_info in enumerate(pdf_images):
        audio_duration = audio_durations[i]
        if audio_duration is not None:
            task = create_single_video_chunk(
                chunk_order=i + 1,
                image_path=pdf_image_info["image_path"],
                audio_path=ordered_audio_files[i]["audio_path"],
                audio_duration=audio_duration,
                output_dir=chunks_output_dir,
                script_id=script_id,
            )
            video_creation_tasks.append(task)

    start_time = time.time()

    if mode == ProcessingMode.PARALLEL:
        print(
            f"Generating {len(video_creation_tasks)} video chunks in parallel..."
        )
        generation_results = await asyncio.gather(
            *video_creation_tasks, return_exceptions=True
        )
    else:  # SEQUENTIAL
        print(
            f"Generating {len(video_creation_tasks)} video chunks sequentially..."
        )
        generation_results = []
        for i, task in enumerate(video_creation_tasks):
            print(f"  - Generating chunk {i+1}/{len(video_creation_tasks)}...")
            try:
                result = await task
                generation_results.append(result)
            except Exception as e:
                generation_results.append(e)
                print(f"  - FAILED chunk {i+1}: {e}")

    chunk_gen_duration = time.time() - start_time

    successful_chunks = [
        res for res in generation_results if not isinstance(res, Exception)
    ]
    failed_chunks = [
        str(res) for res in generation_results if isinstance(res, Exception)
    ]

    if not successful_chunks:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "All video chunk generations failed.",
                "errors": failed_chunks,
            },
        )

    # --- Stage 3: Combine Video Chunks ---
    print("Combining video chunks...")
    chunk_files = sorted(
        glob.glob(os.path.join(chunks_output_dir, "chunk_*.mp4"))
    )
    if not chunk_files:
        raise HTTPException(
            status_code=404,
            detail=f"No video chunk files found in {chunks_output_dir} after generation.",
        )

    final_output_dir = os.path.join("generated_final_videos", script_id)
    os.makedirs(final_output_dir, exist_ok=True)

    concat_file_path = os.path.join(
        final_output_dir, f"{script_id}_concat_list.txt"
    )
    output_filename = (
        f"final_presentation_{script_id}_{int(time.time())}.mp4"
    )
    final_video_path = os.path.join(final_output_dir, output_filename)

    try:
        with open(concat_file_path, "w") as f:
            for chunk_file in chunk_files:
                f.write(f"file '{os.path.abspath(chunk_file)}'\n")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating concat file: {e}"
        )

    ffmpeg_cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file_path,
        "-c",
        "copy",
        "-y",
        final_video_path,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg fast concat failed: {stderr.decode()}",
            )

        final_video_size = (
            os.path.getsize(final_video_path)
            if os.path.exists(final_video_path)
            else 0
        )

        return {
            "script_id": script_id,
            "topic": script_data["topic"],
            "message": "Video generated and combined successfully.",
            "final_video_path": final_video_path,
            "video_url": f"/generated_final_videos/{script_id}/{output_filename}",
            "file_size_bytes": final_video_size,
            "stats": {
                "processing_mode": mode,
                "chunks_requested": len(video_creation_tasks),
                "chunks_succeeded": len(successful_chunks),
                "chunks_failed": len(failed_chunks),
                "chunk_generation_time_seconds": round(chunk_gen_duration, 2),
                "failed_chunk_errors": failed_chunks,
            },
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