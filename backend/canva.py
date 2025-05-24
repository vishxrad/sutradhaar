from fastapi import FastAPI
# import ollama # No longer needed
import openai # Import the OpenAI library
from dotenv import load_dotenv
import os # To access environment variables
import re 

load_dotenv() # This line loads variables from .env into os.environ

app = FastAPI()

# Ensure your OpenAI API key is set as an environment variable
# For production, consider more robust configuration management
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    # You might want to raise an error or handle this more gracefully
    # For now, the openai client will raise an error if the key is not found.

client = openai.OpenAI() # Initializes the client using the OPENAI_API_KEY env var

def parse_script_data(script_text):
    """
    Parses the script text from Ollama into a structured format.
    """
    segments_data = []
    
    # Regex to find segments: Corrected to stop before the next segment
    segment_pattern = re.compile(
        r"Segment\s*\d+\s*:\s*(.*?)\s*Summary:\s*(.*?)\s*(.*?)(?=(Segment\s*\d+\s*:|$))", 
        re.DOTALL | re.IGNORECASE
    )
    # Regex to find slides within a segment
    slide_pattern = re.compile(
        r"Slide\s*\d+\s*:\s*Title:\s*(.*?)\s*Narration:\s*(.*?)\s*Image prompt:\s*(.*?)(?=(Slide\s*\d+\s*:|Segment\s*\d+\s*:|$))", 
        re.DOTALL | re.IGNORECASE
    )

    for segment_match in segment_pattern.finditer(script_text):
        segment_title = segment_match.group(1).strip()
        segment_summary = segment_match.group(2).strip()
        # group(3) now correctly contains only the slides for the current segment
        slides_text_for_current_segment = segment_match.group(3) 
        
        slides_list = []
        for slide_match in slide_pattern.finditer(slides_text_for_current_segment): # Iterate over the current segment's slide text
            slide_title = slide_match.group(1).strip()
            slide_narration = slide_match.group(2).strip()
            slide_image_prompt = slide_match.group(3).strip()
            slides_list.append({
                "title": slide_title,
                "narration": slide_narration,
                "image_prompt": slide_image_prompt
            })
        
        segments_data.append({
            "segment_title": segment_title,
            "summary": segment_summary,
            "slides": slides_list
        })
        
    return segments_data

@app.get("/")
async def get_content():
    topic = "Drugs are bad"  # Hardcoded topic

    prompt = f"""You are a scriptwriter for an educational explainer video.
    The video will cover the topic: "{topic}" and should be structured into 5 distinct educational segments.
    The total narration for the entire video should be approximately 5 minutes (around 1000 words).

    For each of the 5 segments:
    - The total narration for the segment should be approximately 1 minute (around 200 words).
    - Provide a segment title.
    - Provide a short summary of the segment.
    - Divide the segment into 4 slides.
        For each of the 4 slides:
        - Provide a short title (max 5 words).
        - Write a narration script of approximately 50 words (so that 4 slides total ~200 words for the segment).
        - Suggest a visual description (image prompt for an AI image generator or Unsplash search).

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
        # Replace ollama call with OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Or another model like "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful scriptwriting assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_script_data = response.choices[0].message.content
        parsed_script_data = parse_script_data(raw_script_data)
        
        return {"topic": topic, "parsed_script": parsed_script_data, "raw_script": raw_script_data}
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        return {"error": f"OpenAI API returned an API Error: {e}", "raw_response_on_error": str(e)}
    except Exception as e:
        # It's good to catch a more general exception for unexpected errors
        return {"error": str(e), "raw_response_on_error": "No response object for this type of error"}


@app.get("/async-endpoint")
async def async_endpoint():
    return {"message": "This is an async endpoint!"}