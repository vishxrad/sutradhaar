from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

load_dotenv()

app = FastAPI()

# Add static file serving for images
app.mount("/images", StaticFiles(directory="generated_images"), name="images")

# Database configuration
DATABASE_PATH = "sutradhaar.db"

# Pydantic models for request bodies
class ScriptRequest(BaseModel):
    topic: str

class ImageRequest(BaseModel):
    script_id: str
    use_unsplash_fallback: bool = True

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
        
        # Migrate existing data if needed
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
        
        conn.commit()
        print("Database initialized successfully")

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
            "per_page": 1,
            "orientation": "landscape"
        }
        
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data["results"]:
                    return data["results"][0]["urls"]["regular"]
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
        - Write a narration script of approximately 200 words (so that 4 slides total ~200 words for the segment).
        - Suggest a visual description (image prompt for an AI image generator or Unsplash search). Make sure the prompt is not very complex and easy to understand by text to image models.

    Format the output cleanly, following this structure for each segment:

    Segment 1: [Segment Title Here]
    Summary: [Segment Summary Here]
    Slide 1:
    Title: [Slide 1 Title]
    Narration: [Slide 1 Narration - approx 200 words]
    Image prompt: [Slide 1 Image Prompt]
    Slide 2:
    Title: [Slide 2 Title]
    Narration: [Slide 2 Narration - approx 200 words]
    Image prompt: [Slide 2 Image Prompt]
    Slide 3:
    Title: [Slide 3 Title]
    Narration: [Slide 3 Narration - approx 200 words]
    Image prompt: [Slide 3 Image Prompt]
    Slide 4:
    Title: [Slide 4 Title]
    Narration: [Slide 4 Narration - approx 200 words]
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

@app.get("/presentation/{script_id}")
def generate_presentation_data(script_id: str):
    """
    Generate presentation JSON data for Reveal.js frontend
    Returns 27 slides: 1 title + 5 sections + 20 main + 1 thank you
    """
    # Get script data
    script_data = get_script_from_db(script_id)
    if not script_data:
        raise HTTPException(status_code=404, detail="Script not found")
    
    # Get images data directly from database for better performance
    images_data = get_images_from_db(script_id)
    
    presentation = {
        "script_id": script_id,
        "topic": script_data["topic"],
        "slides": [],
        "metadata": {
            "total_slides": 27,
            "created_at": script_data["timestamp"],
            "has_images": len(images_data) > 0
        }
    }
    
    # 1. Title Slide
    presentation["slides"].append({
        "type": "title",
        "id": "title_slide",
        "title": script_data["topic"],
        "order": 1
    })
    
    # 2-6. Process 5 Segments (Section + 4 Main slides each)
    segments_data = script_data["parsed_script"]
    slide_order = 2
    
    for segment_idx, segment in enumerate(segments_data, 1):
        segment_title = segment.get('segment_title', f'Segment {segment_idx}')
        segment_summary = segment.get('summary', '')
        
        # Section Slide for this segment
        presentation["slides"].append({
            "type": "section",
            "id": f"section_{segment_idx}",
            "title": segment_title,
            "body": segment_summary,
            "segment_number": segment_idx,
            "order": slide_order
        })
        slide_order += 1
        
        # 4 Main slides for this segment
        for slide_idx, slide in enumerate(segment.get('slides', []), 1):
            slide_key = f"segment_{segment_idx}_slide_{slide_idx}"
            
            # Get image data for this slide
            image_info = images_data.get(slide_key, {})
            image_url = None
            if image_info.get('image_path'):
                # Convert local path to served URL
                image_url = f"/images/{script_id}/{image_info['image_path'].split('/')[-1]}"
            
            presentation["slides"].append({
                "type": "main",
                "id": f"main_{segment_idx}_{slide_idx}",
                "title": slide.get('title', f'Slide {slide_idx}'),
                "body": slide.get('narration', ''),
                "image": {
                    "url": image_url,
                    "alt": slide.get('image_prompt', ''),
                    "source": image_info.get('source', 'none')
                },
                "segment_number": segment_idx,
                "slide_number": slide_idx,
                "order": slide_order
            })
            slide_order += 1
    
    # 27. Thank You Slide
    presentation["slides"].append({
        "type": "thankyou",
        "id": "thankyou_slide",
        "title": "Thank You",
        "subtitle": "Made using Sutradhaar",
        "order": 27
    })
    
    return presentation

@app.get("/presentation/{script_id}/theme")
def get_presentation_theme(script_id: str):
    """
    Generate theme colors based on the first available image
    """
    try:
        images_data = get_images_from_db(script_id)
        
        # Find first available image
        first_image_path = None
        for image_info in images_data.values():
            if image_info.get('image_path'):
                first_image_path = image_info['image_path']
                break
        
        if first_image_path and os.path.exists(first_image_path):
            # Use ColorThief to extract dominant color
            from colorthief import ColorThief
            color_thief = ColorThief(first_image_path)
            dominant_color = color_thief.get_color(quality=1)
            
            # Generate complementary colors
            r, g, b = dominant_color
            
            return {
                "primary_color": f"rgb({r}, {g}, {b})",
                "primary_hex": f"#{r:02x}{g:02x}{b:02x}",
                "background_color": f"rgba({r}, {g}, {b}, 0.1)",
                "text_color": "white" if (r + g + b) < 384 else "black",
                "accent_color": f"rgb({min(255, r+50)}, {min(255, g+50)}, {min(255, b+50)})"
            }
        else:
            # Default theme
            return {
                "primary_color": "rgb(74, 144, 226)",
                "primary_hex": "#4a90e2",
                "background_color": "rgba(74, 144, 226, 0.1)",
                "text_color": "white",
                "accent_color": "rgb(124, 194, 255)"
            }
            
    except Exception as e:
        print(f"Error generating theme: {e}")
        # Return default theme on error
        return {
            "primary_color": "rgb(74, 144, 226)",
            "primary_hex": "#4a90e2", 
            "background_color": "rgba(74, 144, 226, 0.1)",
            "text_color": "white",
            "accent_color": "rgb(124, 194, 255)"
        }

@app.get("/presentation/{script_id}/html")
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

@app.get("/presentation/{script_id}/download")
def download_presentation_html(script_id: str):
    """
    Download the presentation as an HTML file
    """
    from fastapi.responses import Response
    
    result = generate_presentation_html(script_id)
    
    # Return as downloadable file
    return Response(
        content=result["html_content"],
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={result['filename']}"}
    )

def get_theme_colors(images_data: dict) -> dict:
    """Extract theme colors from the first available image"""
    try:
        # Find first available image
        first_image_path = None
        for image_info in images_data.values():
            if image_info.get('image_path') and os.path.exists(image_info['image_path']):
                first_image_path = image_info['image_path']
                break
        
        if first_image_path:
            from colorthief import ColorThief
            color_thief = ColorThief(first_image_path)
            dominant_color = color_thief.get_color(quality=1)
            r, g, b = dominant_color
            
            return {
                "primary_color": f"rgb({r}, {g}, {b})",
                "primary_hex": f"#{r:02x}{g:02x}{b:02x}",
                "background_color": f"rgba({r}, {g}, {b}, 0.1)",
                "text_color": "white" if (r + g + b) < 384 else "black",
                "accent_color": f"rgb({min(255, r+50)}, {min(255, g+50)}, {min(255, b+50)})"
            }
    except Exception as e:
        print(f"Error generating theme: {e}")
    
    # Default theme
    return {
        "primary_color": "rgb(74, 144, 226)",
        "primary_hex": "#4a90e2",
        "background_color": "rgba(74, 144, 226, 0.1)",
        "text_color": "white",
        "accent_color": "rgb(124, 194, 255)"
    }

def generate_html_template(topic: str, slides: list, theme: dict, script_id: str) -> str:
    """Generate the complete HTML presentation with 16:9 aspect ratio"""
    
    template_str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ topic }} - Sutradhaar Presentation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
  <!-- Reveal.js core CSS -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js/dist/reveal.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js/dist/theme/black.css">
  
  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">

  <style>
    /* Force 16:9 aspect ratio */
    html, body {
      margin: 0;
      padding: 0;
      overflow: hidden;
    }
    
    .reveal {
      font-family: 'Inter', sans-serif;
      width: 100vw !important;
      height: 100vh !important;
    }
    
    .reveal .slides {
      width: 1920px !important;
      height: 1080px !important;
      left: 50% !important;
      top: 50% !important;
      transform: translate(-50%, -50%) scale(var(--reveal-scale, 1)) !important;
      transform-origin: center center !important;
    }
    
    .reveal .slides section {
      width: 1920px !important;
      height: 1080px !important;
      padding: 60px !important;
      box-sizing: border-box !important;
      display: flex !important;
      flex-direction: column !important;
      justify-content: center !important;
    }
    
    .reveal h1, .reveal h2, .reveal h3 {
      font-family: 'Inter', sans-serif;
      font-weight: 600;
      margin: 0 !important;
    }
    
    /* Title Slide - 16:9 optimized */
    .title-slide {
      background: linear-gradient(135deg, {{ theme.primary_color }} 0%, {{ theme.accent_color }} 100%);
      color: white;
      text-align: center;
    }
    
    .title-slide h1 {
      font-size: 120px;
      font-weight: 700;
      text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
      line-height: 1.2;
      max-width: 1600px;
      margin: 0 auto;
    }
    
    /* Section Slide - 16:9 optimized */
    .section-slide {
      background: linear-gradient(45deg, {{ theme.primary_color }} 0%, {{ theme.accent_color }} 100%);
      color: white;
      text-align: center;
    }
    
    .section-slide h2 {
      font-size: 96px;
      margin-bottom: 60px;
      border-bottom: 6px solid rgba(255,255,255,0.3);
      padding-bottom: 30px;
      max-width: 1600px;
      margin-left: auto;
      margin-right: auto;
    }
    
    .section-slide p {
      font-size: 48px;
      line-height: 1.6;
      max-width: 1400px;
      margin: 0 auto;
    }
    
    /* Main Slide - 16:9 optimized */
    .main-slide {
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      color: #2c3e50;
    }
    
    .main-slide h2 {
      font-size: 84px;
      margin-bottom: 60px;
      text-align: center;
      max-width: 1600px;
      margin-left: auto;
      margin-right: auto;
    }
    
    .main-slide-content {
      display: grid;
      grid-template-columns: 1fr 480px;
      gap: 80px;
      align-items: center;
      max-width: 1600px;
      margin: 0 auto;
      height: auto;
    }
    
    .main-slide-text {
      font-size: 42px;
      line-height: 1.7;
      text-align: left;
    }
    
    .main-slide-image {
      display: flex;
      justify-content: center;
      align-items: center;
    }
    
    .main-slide img {
      width: 480px;
      height: 480px;
      object-fit: cover;
      border-radius: 30px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.15);
      transition: transform 0.3s ease;
    }
    
    .main-slide img:hover {
      transform: scale(1.05);
    }
    
    .no-image-placeholder {
      width: 480px;
      height: 480px;
      background: #ddd;
      border-radius: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #666;
      font-size: 32px;
      text-align: center;
    }
    
    /* Thank You Slide - 16:9 optimized */
    .thankyou-slide {
      background: linear-gradient(135deg, {{ theme.primary_color }} 0%, {{ theme.accent_color }} 100%);
      color: white;
      text-align: center;
    }
    
    .thankyou-slide h3 {
      font-size: 108px;
      margin-bottom: 40px;
    }
    
    .thankyou-slide p {
      font-size: 48px;
      opacity: 0.9;
    }
    
    /* Controls - Scaled for 16:9 */
    .controls {
      position: fixed;
      top: 40px;
      right: 40px;
      z-index: 1000;
    }
    
    .controls button {
      background: rgba(0,0,0,0.7);
      color: white;
      border: none;
      padding: 20px 30px;
      border-radius: 10px;
      cursor: pointer;
      font-family: 'Inter', sans-serif;
      font-size: 18px;
      margin-left: 20px;
    }
    
    .controls button:hover {
      background: rgba(0,0,0,0.9);
    }
    
    /* Responsive scaling for different screen sizes */
    @media screen and (max-width: 1920px) {
      .reveal .slides {
        transform: translate(-50%, -50%) scale(calc(100vw / 1920)) !important;
      }
    }
    
    @media screen and (max-height: 1080px) {
      .reveal .slides {
        transform: translate(-50%, -50%) scale(calc(100vh / 1080)) !important;
      }
    }
    
    @media screen and (max-width: 1920px) and (max-height: 1080px) {
      .reveal .slides {
        transform: translate(-50%, -50%) scale(min(calc(100vw / 1920), calc(100vh / 1080))) !important;
      }
    }
    
    /* Mobile responsive - maintain 16:9 but scale content */
    @media (max-width: 768px) {
      .main-slide-content {
        grid-template-columns: 1fr;
        gap: 40px;
        text-align: center;
      }
      
      .main-slide img,
      .no-image-placeholder {
        width: 360px;
        height: 360px;
      }
      
      .title-slide h1 {
        font-size: 80px;
      }
      
      .section-slide h2 {
        font-size: 64px;
      }
      
      .main-slide h2 {
        font-size: 56px;
      }
      
      .main-slide-text {
        font-size: 32px;
      }
      
      .section-slide p {
        font-size: 36px;
      }
      
      .thankyou-slide h3 {
        font-size: 72px;
      }
      
      .thankyou-slide p {
        font-size: 36px;
      }
    }
    
    /* Plain mode */
    body.plain-mode .reveal section {
      background: white !important;
      color: black !important;
    }
    
    body.plain-mode .title-slide,
    body.plain-mode .section-slide,
    body.plain-mode .thankyou-slide {
      background: white !important;
      color: black !important;
    }
    
    body.plain-mode .section-slide h2 {
      border-bottom-color: rgba(0,0,0,0.3);
    }
    
    body.plain-mode .main-slide {
      background: white !important;
      color: black !important;
    }
    
    /* Print styles for PDF - maintain 16:9 */
    @media print {
      html, body {
        width: 297mm !important;
        height: 167mm !important;
      }
      
      .reveal .slides {
        width: 297mm !important;
        height: 167mm !important;
        transform: none !important;
        left: 0 !important;
        top: 0 !important;
      }
      
      .reveal .slides section {
        width: 297mm !important;
        height: 167mm !important;
        page-break-after: always;
        margin: 0;
        padding: 20mm;
      }
      
      .controls {
        display: none !important;
      }
      
      body.plain-mode .reveal section {
        background: white !important;
        color: black !important;
      }
    }
  </style>
</head>
<body>

<div class="reveal">
  <div class="slides">
    {% for slide in slides %}
      {% if slide.type == 'title' %}
        <section class="title-slide">
          <h1>{{ slide.title }}</h1>
        </section>
      {% elif slide.type == 'section' %}
        <section class="section-slide" data-transition="slide">
          <h2>{{ slide.title }}</h2>
          <p>{{ slide.body }}</p>
        </section>
      {% elif slide.type == 'main' %}
        <section class="main-slide" data-transition="zoom">
          <h2>{{ slide.title }}</h2>
          <div class="main-slide-content">
            <div class="main-slide-text">
              <p>{{ slide.body }}</p>
            </div>
            <div class="main-slide-image">
              {% if slide.image_base64 %}
                <img src="{{ slide.image_base64 }}" alt="{{ slide.image_alt }}" />
              {% else %}
                <div class="no-image-placeholder">
                  No Image Available
                </div>
              {% endif %}
            </div>
          </div>
        </section>
      {% elif slide.type == 'thankyou' %}
        <section class="thankyou-slide" data-transition="slide">
          <h3>{{ slide.title }} 🙏</h3>
          <p style="font-style: italic;">{{ slide.subtitle }}</p>
        </section>
      {% endif %}
    {% endfor %}
  </div>
</div>

<!-- Controls -->
<div class="controls">
  <button onclick="toggleMode()">🎨 Toggle Theme</button>
  <button onclick="window.print()">🖨️ Print/PDF</button>
</div>

<!-- Reveal.js core JS -->
<script src="https://cdn.jsdelivr.net/npm/reveal.js/dist/reveal.js"></script>

<script>
// Calculate scaling for 16:9 aspect ratio
function updateScale() {
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;
  const slideWidth = 1920;
  const slideHeight = 1080;
  
  const scaleX = windowWidth / slideWidth;
  const scaleY = windowHeight / slideHeight;
  const scale = Math.min(scaleX, scaleY);
  
  document.documentElement.style.setProperty('--reveal-scale', scale);
}

// Initialize Reveal.js with 16:9 configuration
Reveal.initialize({
  width: 1920,
  height: 1080,
  margin: 0,
  minScale: 0.1,
  maxScale: 3,
  controls: true,
  progress: true,
  center: false,
  hash: true,
  transition: 'fade',
  transitionSpeed: 'default',
  backgroundTransition: 'fade'
});

// Update scale on window resize
window.addEventListener('resize', updateScale);
updateScale();

// Toggle between themed and plain mode
function toggleMode() {
  document.body.classList.toggle('plain-mode');
}

// Print styles for PDF generation
window.addEventListener('beforeprint', function() {
  document.body.classList.add('plain-mode');
});

window.addEventListener('afterprint', function() {
  document.body.classList.remove('plain-mode');
});
</script>

</body>
</html>"""

    template = Template(template_str)
    return template.render(topic=topic, slides=slides, theme=theme, script_id=script_id)

# Add PDF generation endpoint (optional)
@app.get("/presentation/{script_id}/pdf")
def generate_presentation_pdf(script_id: str):
    """
    Generate a PDF of the presentation in 16:9 aspect ratio
    """
    try:
        import pdfkit
        
        # Get HTML content
        html_result = generate_presentation_html(script_id)
        html_content = html_result["html_content"]
        
        # Configure PDF options for 16:9 aspect ratio
        options = {
            'page-size': 'A4',
            'orientation': 'Landscape',
            'margin-top': '0.5in',
            'margin-right': '0.5in', 
            'margin-bottom': '0.5in',
            'margin-left': '0.5in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None,
            'disable-smart-shrinking': None,
            'print-media-type': None,
            'javascript-delay': 3000,  # Wait for JS to load
            'no-stop-slow-scripts': None,
            'debug-javascript': None
        }
        
        # Generate PDF - this is the key fix
        try:
            pdf_content = pdfkit.from_string(html_content, False, options=options)
            
            # Verify we got actual PDF content
            if not pdf_content or len(pdf_content) < 100:
                raise Exception("PDF generation returned empty or invalid content")
            
            # Check if content starts with PDF signature
            if not pdf_content.startswith(b'%PDF'):
                raise Exception("Generated content is not a valid PDF")
            
        except Exception as pdf_error:
            print(f"pdfkit error: {pdf_error}")
            # Fallback: try with simpler options
            simple_options = {
                'page-size': 'A4',
                'orientation': 'Landscape',
                'encoding': "UTF-8"
            }
            pdf_content = pdfkit.from_string(html_content, False, options=simple_options)
        
        from fastapi.responses import Response
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={script_id}_presentation_16x9.pdf",
                "Content-Type": "application/pdf"
            }
        )
        
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="PDF generation requires 'pdfkit' package. Install with: pip install pdfkit"
        )
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")