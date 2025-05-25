from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
import re
import requests
from typing import Dict, Any, Optional, List
import asyncio
import aiohttp
from dotenv import load_dotenv
from image_generator import VertexImageGenerator
import sqlite3
import json
import time
from contextlib import contextmanager
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from fastapi.staticfiles import StaticFiles
import mimetypes
from jinja2 import Template
import base64
from pathlib import Path
from presentation_templates import generate_html_template, get_theme_colors
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import texttospeech  # Add this import
from pdf2image import convert_from_path  # Add this import at the top


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add static file serving for images
app.mount("/images", StaticFiles(directory="generated_images"), name="images")
app.mount("/presentations", StaticFiles(directory="generated_presentations"), name="presentations")
app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")
app.mount("/pdf-images", StaticFiles(directory="generated_pdf_images"), name="pdf-images")

# Database configuration
DATABASE_PATH = "sutradhaar.db"

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

# Initialize OpenAI client
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")

client = openai.OpenAI()

# Initialize Vertex AI Image Generator
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")

if GOOGLE_CLOUD_PROJECT:
    image_generator = VertexImageGenerator(
        project_id=GOOGLE_CLOUD_PROJECT,
        location=VERTEX_AI_LOCATION
    )
else:
    print("Warning: GOOGLE_CLOUD_PROJECT environment variable not set. Image generation will be disabled.")
    image_generator = None

# Database functions
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize the database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create scripts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scripts (
                script_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                raw_script TEXT NOT NULL,
                parsed_script TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        ''')
        
        # Create presentations table - Updated with pdf_images_path column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS presentations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                script_id TEXT NOT NULL,
                pdf_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at REAL NOT NULL,
                file_size INTEGER,
                pdf_images_path TEXT,
                FOREIGN KEY (script_id) REFERENCES scripts (script_id),
                UNIQUE(script_id)
            )
        ''')
        
        # Check and migrate presentations table if needed
        try:
            cursor.execute("PRAGMA table_info(presentations)")
            columns = [column[1] for column in cursor.fetchall()]
            print(f"Presentations table columns: {columns}")
            
            # Add missing columns if they don't exist
            if 'filename' not in columns:
                print("Adding filename column to presentations table...")
                cursor.execute('ALTER TABLE presentations ADD COLUMN filename TEXT')
                
            if 'file_size' not in columns:
                print("Adding file_size column to presentations table...")
                cursor.execute('ALTER TABLE presentations ADD COLUMN file_size INTEGER')
            
            if 'pdf_images_path' not in columns:
                print("Adding pdf_images_path column to presentations table...")
                cursor.execute('ALTER TABLE presentations ADD COLUMN pdf_images_path TEXT')
                
        except Exception as e:
            print(f"Presentations table migration error: {e}")
        
        # Create images table - Updated schema with additional fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                script_id TEXT NOT NULL,
                segment_idx INTEGER NOT NULL,
                slide_idx INTEGER NOT NULL,
                slide_key TEXT NOT NULL,
                segment_title TEXT,
                segment_summary TEXT,
                slide_title TEXT,
                slide_narration TEXT,
                image_prompt TEXT,
                image_path TEXT,
                unsplash_url TEXT,
                source TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (script_id) REFERENCES scripts (script_id),
                UNIQUE(script_id, slide_key)
            )
        ''')
        
        # Migrate existing images data if needed
        try:
            cursor.execute("PRAGMA table_info(images)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Check if we need to add new columns
            if 'segment_summary' not in columns:
                print("Adding segment_summary column...")
                cursor.execute('ALTER TABLE images ADD COLUMN segment_summary TEXT')
                
            if 'slide_narration' not in columns:
                print("Adding slide_narration column...")
                cursor.execute('ALTER TABLE images ADD COLUMN slide_narration TEXT')
            
            # Handle old schema migration
            if 'vertex_ai_path' in columns and 'image_path' not in columns:
                print("Migrating database schema...")
                cursor.execute('ALTER TABLE images ADD COLUMN image_path TEXT')
                
                cursor.execute('''
                    UPDATE images 
                    SET image_path = COALESCE(vertex_ai_path, unsplash_path)
                    WHERE image_path IS NULL
                ''')
                
                # Create new table with updated schema
                cursor.execute('''
                    CREATE TABLE images_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        script_id TEXT NOT NULL,
                        segment_idx INTEGER NOT NULL,
                        slide_idx INTEGER NOT NULL,
                        slide_key TEXT NOT NULL,
                        segment_title TEXT,
                        segment_summary TEXT,
                        slide_title TEXT,
                        slide_narration TEXT,
                        image_prompt TEXT,
                        image_path TEXT,
                        unsplash_url TEXT,
                        source TEXT,
                        created_at REAL NOT NULL,
                        FOREIGN KEY (script_id) REFERENCES scripts (script_id),
                        UNIQUE(script_id, slide_key)
                    )
                ''')
                
                # Copy data to new table
                cursor.execute('''
                    INSERT INTO images_new 
                    (script_id, segment_idx, slide_idx, slide_key, segment_title, segment_summary,
                     slide_title, slide_narration, image_prompt, image_path, unsplash_url, source, created_at)
                    SELECT script_id, segment_idx, slide_idx, slide_key, segment_title, NULL,
                           slide_title, NULL, image_prompt, image_path, unsplash_url, source, created_at
                    FROM images
                ''')
                
                cursor.execute('DROP TABLE images')
                cursor.execute('ALTER TABLE images_new RENAME TO images')
                
                print("Database migration completed successfully")
                
        except Exception as e:
            print(f"Migration error (this is normal for new databases): {e}")
        
        # Create audio table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                script_id TEXT NOT NULL,
                audio_type TEXT NOT NULL,
                segment_idx INTEGER,
                slide_idx INTEGER,
                content TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                speaker TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (script_id) REFERENCES scripts (script_id),
                UNIQUE(script_id, audio_type, segment_idx, slide_idx)
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully with all tables and columns")

def save_script_to_db(script_id: str, topic: str, raw_script: str, parsed_script: List[dict]) -> bool:
    """Save script data to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = time.time()
            
            cursor.execute('''
                INSERT OR REPLACE INTO scripts 
                (script_id, topic, raw_script, parsed_script, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                script_id,
                topic,
                raw_script,
                json.dumps(parsed_script),
                current_time,
                current_time
            ))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving script to database: {e}")
        return False

def get_script_from_db(script_id: str) -> Optional[dict]:
    """Retrieve script data from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT script_id, topic, raw_script, parsed_script, created_at, updated_at
                FROM scripts WHERE script_id = ?
            ''', (script_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "script_id": row["script_id"],
                    "topic": row["topic"],
                    "raw_script": row["raw_script"],
                    "parsed_script": json.loads(row["parsed_script"]),
                    "timestamp": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None
    except Exception as e:
        print(f"Error retrieving script from database: {e}")
        return None

def get_all_scripts_from_db() -> List[dict]:
    """Retrieve all scripts metadata from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT script_id, topic, created_at, updated_at
                FROM scripts ORDER BY created_at DESC
            ''')
            
            rows = cursor.fetchall()
            return [
                {
                    "script_id": row["script_id"],
                    "topic": row["topic"],
                    "timestamp": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                for row in rows
            ]
    except Exception as e:
        print(f"Error retrieving scripts from database: {e}")
        return []

def save_images_to_db(script_id: str, images_data: dict) -> bool:
    """Save image data to database with all script information"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = time.time()
            
            # Clear existing images for this script
            cursor.execute('DELETE FROM images WHERE script_id = ?', (script_id,))
            
            # Insert new image records with all information
            for slide_key, image_info in images_data.items():
                # Extract segment and slide indices from slide_key
                parts = slide_key.split('_')
                segment_idx = int(parts[1]) if len(parts) > 1 else 0
                slide_idx = int(parts[3]) if len(parts) > 3 else 0
                
                cursor.execute('''
                    INSERT INTO images 
                    (script_id, segment_idx, slide_idx, slide_key, segment_title, segment_summary,
                     slide_title, slide_narration, image_prompt, image_path, unsplash_url, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    script_id,
                    segment_idx,
                    slide_idx,
                    slide_key,
                    image_info.get("segment_title"),
                    image_info.get("segment_summary"),  # Added
                    image_info.get("slide_title"),
                    image_info.get("slide_narration"),  # Added
                    image_info.get("image_prompt"),
                    image_info.get("image_path"),
                    image_info.get("unsplash_url"),
                    image_info.get("source"),
                    current_time
                ))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving images to database: {e}")
        return False

def get_images_from_db(script_id: str) -> dict:
    """Retrieve image data from database with all script information"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT slide_key, segment_title, segment_summary, slide_title, slide_narration,
                       image_prompt, image_path, unsplash_url, source, created_at,
                       segment_idx, slide_idx
                FROM images WHERE script_id = ?
                ORDER BY segment_idx, slide_idx
            ''', (script_id,))
            
            rows = cursor.fetchall()
            images = {}
            
            for row in rows:
                images[row["slide_key"]] = {
                    "segment_idx": row["segment_idx"],
                    "slide_idx": row["slide_idx"],
                    "segment_title": row["segment_title"],
                    "segment_summary": row["segment_summary"],  # Added
                    "slide_title": row["slide_title"],
                    "slide_narration": row["slide_narration"],  # Added
                    "image_prompt": row["image_prompt"],
                    "image_path": row["image_path"],
                    "unsplash_url": row["unsplash_url"],
                    "source": row["source"],
                    "created_at": row["created_at"]
                }
            
            return images
    except Exception as e:
        print(f"Error retrieving images from database: {e}")
        return {}

def save_audio_to_db(script_id: str, audio_files: dict) -> bool:
    """Save audio file paths to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = time.time()
            
            # Create audio table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    script_id TEXT NOT NULL,
                    audio_type TEXT NOT NULL,
                    segment_idx INTEGER,
                    slide_idx INTEGER,
                    content TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (script_id) REFERENCES scripts (script_id),
                    UNIQUE(script_id, audio_type, segment_idx, slide_idx)
                )
            ''')
            
            # Clear existing audio for this script
            cursor.execute('DELETE FROM audio WHERE script_id = ?', (script_id,))
            
            # Insert new audio records
            for audio_key, audio_info in audio_files.items():
                cursor.execute('''
                    INSERT INTO audio 
                    (script_id, audio_type, segment_idx, slide_idx, content, audio_path, speaker, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    script_id,
                    audio_info.get("audio_type"),
                    audio_info.get("segment_idx"),
                    audio_info.get("slide_idx"),
                    audio_info.get("content"),
                    audio_info.get("audio_path"),
                    audio_info.get("speaker"),
                    current_time
                ))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving audio to database: {e}")
        return False

def get_audio_from_db(script_id: str) -> dict:
    """Retrieve audio data from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT audio_type, segment_idx, slide_idx, content, audio_path, speaker, created_at
                FROM audio WHERE script_id = ?
                ORDER BY segment_idx, slide_idx
            ''', (script_id,))
            
            rows = cursor.fetchall()
            audio_files = {}
            
            for row in rows:
                audio_key = f"{row['audio_type']}_seg{row['segment_idx']}_slide{row['slide_idx']}" if row['slide_idx'] else f"{row['audio_type']}_seg{row['segment_idx']}"
                audio_files[audio_key] = {
                    "audio_type": row["audio_type"],
                    "segment_idx": row["segment_idx"],
                    "slide_idx": row["slide_idx"],
                    "content": row["content"],
                    "audio_path": row["audio_path"],
                    "speaker": row["speaker"],
                    "created_at": row["created_at"]
                }
            
            return audio_files
    except Exception as e:
        print(f"Error retrieving audio from database: {e}")
        return {}

# Initialize database on startup
init_database()

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
        print(segments_data)
    return segments_data

async def generate_vertex_ai_image_async(image_generator, prompt: str, output_dir: str, filename_prefix: str, max_retries: int = 2) -> Optional[str]:
    """
    Async wrapper for Vertex AI image generation with retries
    """
    if not image_generator:
        return None
    
    # Run the synchronous image generation in a thread pool
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
                await asyncio.sleep(1)  # Brief delay before retry
    
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
            "order_by": "popular"  # or "latest", "oldest", "popular"
        }
        
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data["results"]:
                    # Pick from top 3 most popular
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
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get file extension from URL or default to .jpg
        parsed_url = urlparse(image_url)
        file_ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
        
        # Full file path
        file_path = os.path.join(output_dir, f"{filename}{file_ext}")
        
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout for download
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(image_url) as response:
                response.raise_for_status()
                
                # Download and save the image
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
    segment_summary = slide_info.get('segment_summary', '')  # Added
    slide_title = slide_info.get('title', f'slide_{slide_idx}')
    slide_narration = slide_info.get('narration', '')  # Added
    image_prompt = slide_info.get('image_prompt', '')
    
    result = {
        "slide_key": slide_key,
        "segment_title": segment_title,
        "segment_summary": segment_summary,  # Added
        "slide_title": slide_title,
        "slide_narration": slide_narration,  # Added
        "image_prompt": image_prompt,
        "image_path": None,
        "unsplash_url": None,
        "source": "failed",
        "error": None
    }
    
    if not image_prompt:
        result["error"] = "No image prompt provided"
        return result
    
    # Create output directory based on script_id
    output_dir = f"generated_images/{script_id}"
    filename_prefix = f"seg{segment_idx}_slide{slide_idx}"
    
    # Try Vertex AI first
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
    
    # Fallback to Unsplash if Vertex AI failed
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

@app.post("/generate-script")
def generate_script(request: ScriptRequest):
    """
    Generate a script for the given topic using OpenAI and save to database
    """
    topic = request.topic
    
    prompt = f"""You are a scriptwriter for an educational explainer video.
    The video will cover the topic: "{topic}" and should be structured into 5 distinct educational segments.
    The total narration for the entire video should be approximately 5 minutes.

    For each of the 5 segments:
    - The total narration for the segment should be approximately 1 minute (around 200 words).
    - Provide a segment title.
    - Provide a short summary of the segment.
    - Divide the segment into 4 slides.
        For each of the 4 slides:
        - Provide a short title (max 5 words).
        - Write a narration script of approximately 50 words (so that 4 slides total ~200 words for the segment).
        - Suggest a visual description (image prompt for an AI image generator or Unsplash search). Make sure the prompt is not very complex and easy to understand by text to image models.

    Format the output cleanly, following this structure for each segment:

    Segment 1: [Segment Title Here]
    Summary: [Segment Summary Here]
    Slide 1:
    Title: [Slide 1 Title]
    Narration: [Slide 1 Narration - approx 50 words]
    Image prompt: [Slide 1 Image Prompt]
    Slide 2:
    Title: [Slide 2 Title]
    Narration: [Slide 2 Narration - approx 50 words]
    Image prompt: [Slide 2 Image Prompt]
    Slide 3:
    Title: [Slide 3 Title]
    Narration: [Slide 3 Narration - approx 50 words]
    Image prompt: [Slide 3 Image Prompt]
    Slide 4:
    Title: [Slide 4 Title]
    Narration: [Slide 4 Narration - approx 50 words]
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
        print(parsed_script_data)
        
        # Generate a unique script ID
        script_id = f"script_{int(time.time())}"
        
        # Save to database
        if save_script_to_db(script_id, topic, raw_script_data, parsed_script_data):
            return {
                "script_id": script_id,
                "topic": topic, 
                "parsed_script": parsed_script_data,
                "message": "Script generated and saved successfully. Use the script_id to generate images."
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
    Generate images for a previously generated script in parallel and save to database
    """
    script_id = request.script_id
    use_unsplash_fallback = request.use_unsplash_fallback
    
    # Get script from database
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found. Generate a script first.")
    
    segments_data = script_data["parsed_script"]
    
    # Prepare tasks for parallel execution
    tasks = []
    
    for segment_idx, segment in enumerate(segments_data, 1):
        segment_title = segment.get('segment_title', f'segment_{segment_idx}')
        segment_summary = segment.get('summary', '')  # Added
        
        for slide_idx, slide in enumerate(segment.get('slides', []), 1):
            if slide.get('image_prompt', ''):
                slide_info = {
                    **slide,
                    'segment_title': segment_title,
                    'segment_summary': segment_summary  # Added
                }
                
                # Create async task for this image
                task = generate_single_image_with_fallback(
                    image_generator,
                    slide_info,
                    segment_idx,
                    slide_idx,
                    script_id,
                    use_unsplash_fallback
                )
                tasks.append(task)
    
    print(f"Starting parallel generation of {len(tasks)} images for script {script_id}...")
    start_time = asyncio.get_event_loop().time()
    
    # Execute all image generation tasks in parallel
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        results = []
    
    end_time = asyncio.get_event_loop().time()
    print(f"Parallel image generation completed in {end_time - start_time:.2f} seconds")
    
    # Process results
    image_results = {
        "script_id": script_id,
        "topic": script_data["topic"],
        "images": {},
        "stats": {
            "total_requested": len(tasks),
            "vertex_ai_success": 0,
            "unsplash_fallback": 0,
            "failed": 0,
            "errors": [],
            "generation_time_seconds": round(end_time - start_time, 2)
        }
    }
    
    for result in results:
        if isinstance(result, Exception):
            image_results["stats"]["failed"] += 1
            image_results["stats"]["errors"].append(f"Task failed with exception: {str(result)}")
            continue
            
        slide_key = result["slide_key"]
        
        # Update statistics
        if result["source"] == "vertex_ai":
            image_results["stats"]["vertex_ai_success"] += 1
        elif result["source"] == "unsplash":
            image_results["stats"]["unsplash_fallback"] += 1
        else:
            image_results["stats"]["failed"] += 1
            if result.get("error"):
                image_results["stats"]["errors"].append(f"{slide_key}: {result['error']}")
        
        # Store the result with all script information
        image_data = {
            "segment_title": result["segment_title"],
            "segment_summary": result["segment_summary"],  # Added
            "slide_title": result["slide_title"],
            "slide_narration": result["slide_narration"],  # Added
            "image_prompt": result["image_prompt"],
            "image_path": result["image_path"],
            "unsplash_url": result["unsplash_url"],
            "source": result["source"]
        }
        image_results["images"][slide_key] = image_data
    
    # Save images to database
    if image_results["images"]:
        if save_images_to_db(script_id, image_results["images"]):
            print(f"Successfully saved {len(image_results['images'])} image records to database")
        else:
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
    # Check if script exists
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    
    # Get images from database
    images = get_images_from_db(script_id)
    
    return {
        "script_id": script_id,
        "topic": script_data["topic"],
        "images": images
    }

@app.get("/generation-status/{script_id}")
def get_generation_status(script_id: str):
    """
    Check if images have been generated for a script
    """
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    
    images = get_images_from_db(script_id)
    
    # Count expected vs actual images
    segments_data = script_data["parsed_script"]
    expected_images = 0
    for segment in segments_data:
        for slide in segment.get('slides', []):
            if slide.get('image_prompt', ''):
                expected_images += 1
    
    generated_images = len(images)
    vertex_ai_count = sum(1 for img in images.values() if img['source'] == 'vertex_ai')
    unsplash_count = sum(1 for img in images.values() if img['source'] == 'unsplash')
    failed_count = expected_images - vertex_ai_count - unsplash_count  # Fix: Calculate failed properly
    
    return {
        "script_id": script_id,
        "topic": script_data["topic"],
        "expected_images": expected_images,
        "generated_images": generated_images,
        "completion_percentage": round((generated_images / expected_images * 100) if expected_images > 0 else 0, 1),
        "sources": {
            "vertex_ai": vertex_ai_count,
            "unsplash": unsplash_count,
            "failed": failed_count
        },
        "status": "complete" if generated_images == expected_images and expected_images > 0 else "partial" if generated_images > 0 else "none"
    }


# @app.get("/presentation/{script_id}/html")
def generate_presentation_html(script_id: str):
    """
    Generate a complete standalone HTML file for the presentation
    Returns the HTML content as a string that can be saved to a file
    """
    # Get script data
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    
    # Get images data
    images_data = get_images_from_db(script_id)
    
    # Build slides data
    slides = []
    
    # 1. Title Slide
    slides.append({
        "type": "title",
        "title": script_data["topic"],
        "order": 1
    })
    
    # 2-6. Process 5 Segments (Section + 4 Main slides each)
    segments_data = script_data["parsed_script"]
    slide_order = 2
    
    for segment_idx, segment in enumerate(segments_data, 1):
        segment_title = segment.get('segment_title', f'Segment {segment_idx}')
        segment_summary = segment.get('summary', '')
        
        # Section Slide
        slides.append({
            "type": "section",
            "title": segment_title,
            "body": segment_summary,
            "order": slide_order
        })
        slide_order += 1
        
        # 4 Main slides for this segment
        for slide_idx, slide in enumerate(segment.get('slides', []), 1):
            slide_key = f"segment_{segment_idx}_slide_{slide_idx}"
            image_info = images_data.get(slide_key, {})
            
            # Encode image as base64 if it exists
            image_base64 = None
            if image_info.get('image_path') and os.path.exists(image_info['image_path']):
                try:
                    with open(image_info['image_path'], 'rb') as img_file:
                        image_data = img_file.read()
                        # Determine file extension for MIME type
                        file_ext = Path(image_info['image_path']).suffix.lower()
                        mime_type = "image/jpeg" if file_ext in ['.jpg', '.jpeg'] else "image/png"
                        image_base64 = f"data:{mime_type};base64,{base64.b64encode(image_data).decode()}"
                except Exception as e:
                    print(f"Error encoding image {image_info['image_path']}: {e}")
            
            slides.append({
                "type": "main",
                "title": slide.get('title', f'Slide {slide_idx}'),
                "body": slide.get('narration', ''),
                "image_base64": image_base64,
                "image_alt": slide.get('image_prompt', ''),
                "order": slide_order
            })
            slide_order += 1
    
    # 27. Thank You Slide
    slides.append({
        "type": "thankyou",
        "title": "Thank You",
        "subtitle": "Made using Sutradhaar",
        "order": 27
    })
    
    # Generate theme colors
    theme_colors = get_theme_colors(images_data)
    
    # Generate HTML using template
    html_content = generate_html_template(
        topic=script_data["topic"],
        slides=slides,
        theme=theme_colors,
        script_id=script_id
    )
    
    return {
        "script_id": script_id,
        "topic": script_data["topic"],
        "html_content": html_content,
        "filename": f"{script_id}_presentation.html"
    }

def save_presentation_to_db(script_id: str, pdf_path: str, filename: str, file_size: int) -> bool:
    """Save presentation PDF path to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = time.time()
            
            cursor.execute('''
                INSERT OR REPLACE INTO presentations 
                (script_id, pdf_path, filename, created_at, file_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                script_id,
                pdf_path,
                filename,
                current_time,
                file_size
            ))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving presentation to database: {e}")
        return False

def get_presentation_from_db(script_id: str) -> Optional[dict]:
    """Retrieve presentation data from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT script_id, pdf_path, filename, created_at, file_size
                FROM presentations WHERE script_id = ?
            ''', (script_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "script_id": row["script_id"],
                    "pdf_path": row["pdf_path"],
                    "filename": row["filename"],
                    "created_at": row["created_at"],
                    "file_size": row["file_size"]
                }
            return None
    except Exception as e:
        print(f"Error retrieving presentation from database: {e}")
        return None

async def synthesize_text_async(text: str, speaker: str, output_path: str) -> bool:
    """Async wrapper for Google Text-to-Speech synthesis"""
    try:
        # Run the synchronous TTS in a thread pool
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
        
        # Clean and prepare text for SSML
        cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
        if not cleaned_text:
            print(f"Empty text provided for {output_path}")
            return False
        
        # Wrap the text in SSML with slower speaking rate
        ssml_text = f'<speak><prosody rate="100%">{cleaned_text}</prosody></speak>'
        input_text = texttospeech.SynthesisInput(ssml=ssml_text)
        
        # Select voice based on speaker preference
        if speaker.lower() == "male":
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                name="en-IN-Wavenet-F",  # Male voice
            )
        else:  # female (default)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                name="en-IN-Wavenet-D",  # Female voice
            )
        
        # Configure audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Make the TTS request
        response = client.synthesize_speech(
            request={
                "input": input_text,
                "voice": voice,
                "audio_config": audio_config
            }
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the audio content to file
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        
        print(f"Audio content written to '{output_path}'")
        return True
        
    except Exception as e:
        print(f"Error synthesizing text for '{output_path}': {e}")
        return False

@app.post("/generate-audio")
async def generate_audio(request: AudioRequest):
    """
    Generate audio files for a script using Google Text-to-Speech
    Creates segment summaries and slide narrations as separate audio files
    """
    script_id = request.script_id
    speaker = request.speaker.lower()
    
    if speaker not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Speaker must be 'male' or 'female'")
    
    # Get script data
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found. Generate a script first.")
    
    # Get images data to access slide narrations
    images_data = get_images_from_db(script_id)
    
    segments_data = script_data["parsed_script"]
    
    # Create output directory
    audio_dir = f"generated_audio/{script_id}"
    os.makedirs(audio_dir, exist_ok=True)
    
    # Prepare tasks for parallel execution
    tasks = []
    audio_files = {}
    
    # Generate audio for each segment summary and slide narration
    for segment_idx, segment in enumerate(segments_data, 1):
        segment_title = segment.get('segment_title', f'Segment {segment_idx}')
        segment_summary = segment.get('summary', '')
        
        # Task 1: Segment summary audio
        if segment_summary.strip():
            summary_filename = f"segment_{segment_idx}_summary_{speaker}.mp3"
            summary_path = os.path.join(audio_dir, summary_filename)
            
            task = synthesize_text_async(segment_summary, speaker, summary_path)
            tasks.append((task, {
                "audio_key": f"summary_seg{segment_idx}",
                "audio_type": "summary",
                "segment_idx": segment_idx,
                "slide_idx": None,
                "content": segment_summary,
                "audio_path": summary_path,
                "speaker": speaker
            }))
        
        # Task 2-5: Slide narration audio for each slide in the segment
        for slide_idx, slide in enumerate(segment.get('slides', []), 1):
            slide_key = f"segment_{segment_idx}_slide_{slide_idx}"
            
            # Get slide narration from images_data or fallback to slide data
            slide_narration = ""
            if slide_key in images_data and images_data[slide_key].get('slide_narration'):
                slide_narration = images_data[slide_key]['slide_narration']
            elif slide.get('narration'):
                slide_narration = slide['narration']
            
            if slide_narration.strip():
                narration_filename = f"segment_{segment_idx}_slide_{slide_idx}_{speaker}.mp3"
                narration_path = os.path.join(audio_dir, narration_filename)
                
                task = synthesize_text_async(slide_narration, speaker, narration_path)
                tasks.append((task, {
                    "audio_key": f"narration_seg{segment_idx}_slide{slide_idx}",
                    "audio_type": "narration",
                    "segment_idx": segment_idx,
                    "slide_idx": slide_idx,
                    "content": slide_narration,
                    "audio_path": narration_path,
                    "speaker": speaker
                }))
    
    if not tasks:
        raise HTTPException(status_code=400, detail="No text content found to generate audio")
    
    print(f"Starting parallel generation of {len(tasks)} audio files for script {script_id}...")
    start_time = asyncio.get_event_loop().time()
    
    # Execute all TTS tasks in parallel
    task_list = [task[0] for task in tasks]
    results = await asyncio.gather(*task_list, return_exceptions=True)
    
    end_time = asyncio.get_event_loop().time()
    print(f"Parallel audio generation completed in {end_time - start_time:.2f} seconds")
    
    # Process results
    audio_results = {
        "script_id": script_id,
        "topic": script_data["topic"],
        "speaker": speaker,
        "audio_files": {},
        "stats": {
            "total_requested": len(tasks),
            "successful": 0,
            "failed": 0,
            "errors": [],
            "generation_time_seconds": round(end_time - start_time, 2)
        }
    }
    
    for i, result in enumerate(results):
        task_info = tasks[i][1]
        audio_key = task_info["audio_key"]
        
        if isinstance(result, Exception):
            audio_results["stats"]["failed"] += 1
            audio_results["stats"]["errors"].append(f"{audio_key}: {str(result)}")
            continue
        
        if result:  # TTS was successful
            audio_results["stats"]["successful"] += 1
            audio_files[audio_key] = task_info
            audio_results["audio_files"][audio_key] = {
                "audio_type": task_info["audio_type"],
                "segment_idx": task_info["segment_idx"],
                "slide_idx": task_info["slide_idx"],
                "content_preview": task_info["content"][:100] + "..." if len(task_info["content"]) > 100 else task_info["content"],
                "audio_path": task_info["audio_path"],
                "file_size": os.path.getsize(task_info["audio_path"]) if os.path.exists(task_info["audio_path"]) else 0
            }
        else:  # TTS failed
            audio_results["stats"]["failed"] += 1
            audio_results["stats"]["errors"].append(f"{audio_key}: TTS synthesis failed")
    
    # Save audio file info to database
    if audio_files:
        if save_audio_to_db(script_id, audio_files):
            print(f"Successfully saved {len(audio_files)} audio file records to database")
        else:
            audio_results["warning"] = "Audio generated but failed to save to database"
    
    return audio_results

@app.get("/script/{script_id}/audio")
def get_script_audio(script_id: str):
    """
    Retrieve audio files for a specific script from database
    """
    # Check if script exists
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    
    # Get audio files from database
    audio_files = get_audio_from_db(script_id)
    
    return {
        "script_id": script_id,
        "topic": script_data["topic"],
        "audio_files": audio_files
    }

def save_pdf_images_to_db(script_id: str, images_folder: str) -> bool:
    """Save PDF images folder path to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Add pdf_images_path column if it doesn't exist
            try:
                cursor.execute('ALTER TABLE presentations ADD COLUMN pdf_images_path TEXT')
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Update the presentation record with images folder path
            cursor.execute('''
                UPDATE presentations 
                SET pdf_images_path = ?
                WHERE script_id = ?
            ''', (images_folder, script_id))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving PDF images path to database: {e}")
        return False

def convert_pdf_to_images(script_id: str, pdf_path: str) -> dict:
    """
    Convert PDF to individual slide images using pdf2image
    Returns dictionary with conversion results and image paths
    """
    try:
        # Create images directory structure
        images_base_dir = "generated_pdf_images"
        script_images_dir = os.path.join(images_base_dir, script_id)
        os.makedirs(script_images_dir, exist_ok=True)
        
        print(f"Converting PDF to images: {pdf_path}")
        print(f"Images will be saved to: {script_images_dir}")
        
        # Convert PDF to images
        # DPI controls image quality (150 is good balance of quality/file size)
        images = convert_from_path(
            pdf_path,
            dpi=150,
            # Remove output_folder to prevent automatic saving
            fmt='jpeg'
            # Remove jpegopt since we'll control quality in image.save()
        )
        
        # Save images with consistent naming
        image_paths = []
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
            # Create filename: slide_001.jpg, slide_002.jpg, etc.
            image_filename = f"slide_{i:03d}.jpg"
            image_path = os.path.join(script_images_dir, image_filename)
            
            try:
                # Save the image
                image.save(image_path, 'JPEG', quality=85, optimize=True)
                
                # Get file size
                file_size = os.path.getsize(image_path)
                
                conversion_results["image_paths"].append(image_path)
                conversion_results["file_sizes"].append(file_size)
                
                print(f"Saved slide {i}: {image_path} ({file_size} bytes)")
                
            except Exception as e:
                error_msg = f"Failed to save slide {i}: {str(e)}"
                conversion_results["errors"].append(error_msg)
                print(f"Error: {error_msg}")
        
        # Calculate total size of all images
        total_size = sum(conversion_results["file_sizes"])
        conversion_results["total_size_bytes"] = total_size
        conversion_results["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        # Save images folder path to database
        if save_pdf_images_to_db(script_id, script_images_dir):
            conversion_results["database_saved"] = True
            print(f"PDF images path saved to database: {script_images_dir}")
        else:
            conversion_results["database_saved"] = False
            conversion_results["errors"].append("Failed to save images path to database")
        
        print(f"PDF conversion completed: {len(images)} slides, {conversion_results['total_size_mb']} MB total")
        return conversion_results
        
    except Exception as e:
        error_msg = f"Error converting PDF to images: {str(e)}"
        print(error_msg)
        return {
            "script_id": script_id,
            "pdf_path": pdf_path,
            "images_folder": None,
            "total_slides": 0,
            "image_paths": [],
            "file_sizes": [],
            "errors": [error_msg],
            "total_size_bytes": 0,
            "total_size_mb": 0,
            "database_saved": False
        }

# Update the PDF generation endpoint to include image conversion
@app.get("/presentation/{script_id}/pdf")
async def generate_presentation_pdf(script_id: str):
    """
    Generate and return a PDF presentation using Decktape
    Also converts PDF to individual slide images and stores them
    """
    import subprocess
    import tempfile
    from fastapi.responses import FileResponse
    
    # Get script data
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    
    # Check if PDF already exists
    existing_presentation = get_presentation_from_db(script_id)
    if existing_presentation and os.path.exists(existing_presentation["pdf_path"]):
        # Check if images have already been generated
        if not existing_presentation.get("pdf_images_path") or not os.path.exists(existing_presentation.get("pdf_images_path", "")):
            # Generate images from existing PDF
            print("PDF exists but images missing. Converting PDF to images...")
            conversion_result = convert_pdf_to_images(script_id, existing_presentation["pdf_path"])
            
            if conversion_result["errors"]:
                print(f"Warning: PDF to image conversion had errors: {conversion_result['errors']}")
        
        # Return existing PDF
        return FileResponse(
            path=existing_presentation["pdf_path"],
            media_type="application/pdf",
            filename=existing_presentation["filename"],
            headers={"Content-Disposition": f"attachment; filename={existing_presentation['filename']}"}
        )
    
    # Get the HTML content from the existing endpoint logic
    html_result = generate_presentation_html(script_id)
    html_content = html_result["html_content"]
    
    # Create temporary HTML file (will be deleted after PDF generation)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_html:
        temp_html.write(html_content)
        temp_html_path = temp_html.name
    
    # Create generated_presentations directory if it doesn't exist
    presentations_dir = "generated_presentations"
    os.makedirs(presentations_dir, exist_ok=True)
    
    # Create permanent PDF file path
    filename = f"{script_id}_presentation.pdf"
    pdf_path = os.path.join(presentations_dir, filename)
    
    try:
        # Run Decktape to convert HTML to PDF
        cmd = [
            "decktape",
            "reveal",
            temp_html_path,
            pdf_path,
        ]
        
        # Execute Decktape
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500, 
                detail=f"PDF generation failed: {result.stderr}"
            )
        
        # Check if PDF was created
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            raise HTTPException(status_code=500, detail="PDF file was not created")
        
        # Get file size
        file_size = os.path.getsize(pdf_path)
        
        # Save presentation info to database
        if not save_presentation_to_db(script_id, pdf_path, filename, file_size):
            print("Warning: PDF generated but failed to save to database")
        
        print(f"PDF saved successfully: {pdf_path} ({file_size} bytes)")
        
        # Convert PDF to images
        print("Converting PDF to individual slide images...")
        conversion_result = convert_pdf_to_images(script_id, pdf_path)
        
        if conversion_result["errors"]:
            print(f"Warning: PDF to image conversion had errors: {conversion_result['errors']}")
        else:
            print(f"Successfully converted PDF to {conversion_result['total_slides']} images ({conversion_result['total_size_mb']} MB)")
        
        # Return the PDF file
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except subprocess.TimeoutExpired:
        # Clean up PDF file if it was partially created
        if os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except:
                pass
        raise HTTPException(status_code=500, detail="PDF generation timed out")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, 
            detail="Decktape not found. Please install it with: npm install -g decktape"
        )
    except Exception as e:
        # Clean up PDF file if it was partially created
        if os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")
    finally:
        # Clean up temporary HTML file (we don't store this)
        try:
            os.unlink(temp_html_path)
        except:
            pass

# Add a new endpoint to get PDF images info
@app.get("/presentation/{script_id}/images")
def get_presentation_images(script_id: str):
    """
    Get information about the PDF slide images for a presentation
    """
    # Get presentation data
    presentation_data = get_presentation_from_db(script_id)
    if not presentation_data:
        raise HTTPException(status_code=404, detail="Presentation not found")
    
    images_folder = presentation_data.get("pdf_images_path")
    if not images_folder or not os.path.exists(images_folder):
        raise HTTPException(status_code=404, detail="PDF images not found. Generate PDF first.")
    
    # Get list of image files
    try:
        image_files = []
        for filename in sorted(os.listdir(images_folder)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(images_folder, filename)
                file_size = os.path.getsize(file_path)
                
                image_files.append({
                    "filename": filename,
                    "file_path": file_path,
                    "file_size": file_size,
                    "slide_number": int(filename.split('_')[1].split('.')[0]) if '_' in filename else 0
                })
        
        total_size = sum(img["file_size"] for img in image_files)
        
        return {
            "script_id": script_id,
            "images_folder": images_folder,
            "total_images": len(image_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "images": image_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading images folder: {str(e)}")

