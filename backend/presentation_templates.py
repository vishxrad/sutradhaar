# presentation_templates.py

from jinja2 import Environment, FileSystemLoader
import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import random

# ==============================================================================
# SECTION 1: DYNAMIC THEME GENERATION LOGIC
# All the helpers for creating a theme from images must be here.
# ==============================================================================

# In presentation_templates.py

def extract_pixels_from_image(image_path, max_pixels=1000):
    """Extract a sample of pixels from a single image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img.thumbnail((100, 100), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            pixels = img_array.reshape(-1, 3)
            
            # --- THIS IS THE CORRECTED LIST COMPREHENSION ---
            filtered_pixels = []
            for p in pixels:
                # Use int() to prevent numpy overflow warning
                brightness = (int(p[0]) + int(p[1]) + int(p[2])) / 3
                
                # Calculate saturation
                max_val = max(p)
                saturation = (max_val - min(p)) / max_val if max_val > 0 else 0
                
                # Apply all filters in a single, clear if statement
                if 30 < brightness < 225 and saturation > 0.2:
                    filtered_pixels.append(p)
            
            if len(filtered_pixels) > max_pixels:
                return random.sample(filtered_pixels, max_pixels)
            return filtered_pixels
            
    except Exception as e:
        print(f"Error extracting pixels from {image_path}: {e}")
        return []

def find_dominant_color_kmeans(pixels, k=8):
    """Use K-means clustering to find the best dominant color."""
    try:
        if not pixels: return None
        if len(pixels) < k: k = len(pixels)
        
        pixel_array = np.array(pixels)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(pixel_array)
        
        # Score colors to find the best one (not just the biggest cluster)
        best_color, best_score = None, -1
        for i, color in enumerate(kmeans.cluster_centers_):
            r, g, b = color
            brightness = (r + g + b) / 3
            saturation = (max(color) - min(color)) / max(color) if max(color) > 0 else 0
            
            if 50 < brightness < 200 and saturation > 0.3:
                score = np.count_nonzero(kmeans.labels_ == i) * saturation
                if score > best_score:
                    best_score = score
                    best_color = tuple(map(int, color))
        
        if best_color: return best_color
        
        # Fallback to the largest cluster if no "good" color is found
        dominant_idx = np.argmax(np.bincount(kmeans.labels_))
        return tuple(map(int, kmeans.cluster_centers_[dominant_idx]))
    except Exception as e:
        print(f"K-means clustering failed: {e}")
        return None

def generate_color_scheme(r, g, b):
    """Generate a complete color scheme from a base RGB color."""
    brightness = (r * 0.299 + g * 0.587 + b * 0.114)
    text_color = "#FFFFFF" if brightness < 128 else "#2c3e50"
    return {
        "primary_color": f"rgb({r}, {g}, {b})",
        "primary_hex": f"#{r:02x}{g:02x}{b:02x}",
        "background_color": f"rgba({r}, {g}, {b}, 0.15)",
        "main_slide_bg": f"rgba({r}, {g}, {b}, 0.08)",
        "text_color": text_color,
        "accent_color": f"rgb({min(255, r+40)}, {min(255, g+40)}, {min(255, b+40)})",
        "secondary_color": f"rgb({max(0, r-30)}, {max(0, g-30)}, {max(0, b-30)})",
        "light_accent": f"rgba({min(255, r+40)}, {min(255, g+40)}, {min(255, b+40)}, 0.3)"
    }

def get_dynamic_theme(images_data: dict) -> dict:
    """The main function to generate a theme dynamically from images."""
    print("--- Running Dynamic Theme Generation ---")
    all_pixels = []
    processed_images = 0
    for image_info in images_data.values():
        if image_info.get('image_path') and os.path.exists(image_info['image_path']):
            pixels = extract_pixels_from_image(image_info['image_path'])
            if pixels:
                all_pixels.extend(pixels)
                processed_images += 1
            if processed_images >= 10: break
    
    if all_pixels:
        dominant_color = find_dominant_color_kmeans(all_pixels)
        if dominant_color:
            return generate_color_scheme(*dominant_color)
    
    print("Warning: Dynamic theme generation failed. Falling back.")
    return get_default_theme() # Fallback

# ==============================================================================
# SECTION 2: STATIC THEMES & THEME DISPATCHER
# ==============================================================================

def get_crimson_theme():
    """Returns a fixed dictionary for the red theme."""
    return {
        "primary_color": "rgb(192, 57, 43)", "primary_hex": "#c0392b",
        "background_color": "rgba(192, 57, 43, 0.15)",
        "main_slide_bg": "rgba(192, 57, 43, 0.08)", "text_color": "white",
        "accent_color": "rgb(231, 76, 60)",
        "secondary_color": "rgb(162, 27, 13)",
        "light_accent": "rgba(231, 76, 60, 0.3)"
    }

def get_default_theme():
    """Returns a default slate theme."""
    return {
        "primary_color": "rgb(44, 62, 80)", "primary_hex": "#2c3e50",
        "background_color": "rgba(44, 62, 80, 0.15)",
        "main_slide_bg": "rgba(44, 62, 80, 0.08)", "text_color": "white",
        "accent_color": "rgb(52, 73, 94)",
        "secondary_color": "rgb(34, 49, 63)",
        "light_accent": "rgba(52, 73, 94, 0.3)"
    }

def get_theme_for_template(template_name: str, images_data: dict = None) -> dict:
    """Dispatcher to get the correct theme based on template name."""
    if template_name == "crimson":
        return get_crimson_theme()
    
    if template_name == "modern":
        # The 'modern' template uses dynamic colors
        return get_dynamic_theme(images_data or {})
    
    # Fallback for any other name
    return get_default_theme()

# ==============================================================================
# SECTION 3: TEMPLATE RENDERING LOGIC
# ==============================================================================

def format_text_as_bullets(content) -> str:
    """Convert a list of strings or a single string into an HTML unordered list."""
    if not content: return ""
    if isinstance(content, str): content = [content]
    if not isinstance(content, list): return str(content)
    
    bullets = "".join([f"<li style='margin-bottom: 20px;'>{item}</li>" for item in content if str(item).strip()])
    if not bullets: return ""
    return f"<ul style='margin-top: 0; margin-bottom: 0; text-align: left; list-style-position: outside; padding-left: 40px;'>{bullets}</ul>"

def generate_html_template(topic: str, slides: list, theme: dict, script_id: str, template_name: str) -> str:
    """Loads and renders a presentation from a specified template file."""
    template_dir = 'presentation_templates'
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    env.filters['format_bullets'] = format_text_as_bullets
    
    template_path = f'{template_name}/template.html'
    try:
        template = env.get_template(template_path)
        return template.render(topic=topic, slides=slides, theme=theme, script_id=script_id)
    except Exception as e:
        print(f"CRITICAL: Failed to load or render template '{template_path}'. Error: {e}")
        raise

# ==============================================================================
# SECTION 4: MAIN ORCHESTRATION FUNCTION
# ==============================================================================

def create_presentation(topic: str, slides: list, template_name: str, images_data: dict, script_id: str) -> str:
    """Main function to create a complete presentation."""
    theme = get_theme_for_template(template_name, images_data)
    return generate_html_template(topic, slides, theme, script_id, template_name)