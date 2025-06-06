from jinja2 import Template
import os
from app import *

def get_theme_colors(images_data: dict) -> dict:
    """Extract theme colors from all available images using K-means clustering"""
    try:
        all_pixels = []
        processed_images = 0
        
        # Collect pixels from all available images
        for image_info in images_data.values():
            if image_info.get('image_path') and os.path.exists(image_info['image_path']):
                pixels = extract_pixels_from_image(image_info['image_path'])
                if pixels:
                    all_pixels.extend(pixels)
                    processed_images += 1
                    
                # Limit to prevent memory issues - sample from first 10 images
                if processed_images >= 10:
                    break
        
        if all_pixels:
            print(f"Processed {processed_images} images with {len(all_pixels)} total pixels")
            
            # Use K-means clustering on all collected pixels
            dominant_color = find_dominant_color_kmeans(all_pixels, k=8)  # More clusters for better variety
            
            if dominant_color:
                r, g, b = dominant_color
                return generate_color_scheme(r, g, b)
                
    except Exception as e:
        print(f"Error generating theme from all images: {e}")
    
    # Fallback to default theme
    return get_default_theme()

def extract_pixels_from_image(image_path, max_pixels=1000):
    """Extract a sample of pixels from a single image"""
    try:
        from PIL import Image
        import numpy as np
        
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            
            # Resize for faster processing while maintaining aspect ratio
            img.thumbnail((100, 100), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            pixels = img_array.reshape(-1, 3)
            
            # Filter out very dark and very light pixels
            filtered_pixels = []
            for pixel in pixels:
                r, g, b = pixel
                brightness = (r + g + b) / 3
                
                # Skip nearly black/white pixels and very unsaturated pixels
                if 30 < brightness < 225:
                    max_val = max(r, g, b)
                    min_val = min(r, g, b)
                    saturation = (max_val - min_val) / max_val if max_val > 0 else 0
                    
                    # Only keep pixels with some color saturation
                    if saturation > 0.2:
                        filtered_pixels.append(pixel)
            
            # Sample random pixels if we have too many
            if len(filtered_pixels) > max_pixels:
                import random
                filtered_pixels = random.sample(filtered_pixels, max_pixels)
            
            return filtered_pixels
            
    except Exception as e:
        print(f"Error extracting pixels from {image_path}: {e}")
        return []

def find_dominant_color_kmeans(pixels, k=8):
    """Use K-means clustering to find dominant color from all pixels"""
    try:
        from sklearn.cluster import KMeans
        import numpy as np
        
        if len(pixels) < k:
            # Not enough pixels, fall back to simple average
            pixels = np.array(pixels)
            return tuple(map(int, np.mean(pixels, axis=0)))
        
        pixels = np.array(pixels)
        print(f"Running K-means on {len(pixels)} pixels with {k} clusters")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_
        
        # Count pixels in each cluster
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        
        # Score each color based on cluster size, vibrancy, and brightness
        best_color = None
        best_score = 0
        
        for i, color in enumerate(colors):
            r, g, b = color
            cluster_size = label_counts[i]
            
            # Calculate color properties
            brightness = (r + g + b) / 3
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            saturation = (max_val - min_val) / max_val if max_val > 0 else 0
            
            # Skip colors that are too dark, too light, or too unsaturated
            if brightness < 50 or brightness > 200 or saturation < 0.3:
                continue
            
            # Score: cluster size + vibrancy + good brightness range
            brightness_score = 100 - abs(brightness - 125)  # Prefer mid-range brightness
            vibrancy_score = saturation * 100
            size_score = (cluster_size / len(pixels)) * 50
            
            total_score = size_score + vibrancy_score + brightness_score
            
            if total_score > best_score:
                best_score = total_score
                best_color = tuple(map(int, color))
        
        if best_color:
            print(f"Selected color: rgb{best_color} with score: {best_score:.2f}")
            return best_color
            
        # Fallback to largest cluster if no good color found
        dominant_idx = np.argmax(label_counts)
        fallback_color = tuple(map(int, colors[dominant_idx]))
        print(f"Using fallback color: rgb{fallback_color}")
        return fallback_color
        
    except ImportError:
        print("scikit-learn not available, using simple average")
        import numpy as np
        pixels = np.array(pixels)
        return tuple(map(int, np.mean(pixels, axis=0)))
    except Exception as e:
        print(f"K-means clustering failed: {e}")
        return None

def generate_color_scheme(r, g, b):
    """Generate a complete color scheme from base RGB values"""
    # Ensure color is vibrant enough
    r, g, b = enhance_vibrancy(r, g, b)
    
    # Generate complementary colors
    accent_r = min(255, r + 40)
    accent_g = min(255, g + 40) 
    accent_b = min(255, b + 40)
    
    # Generate lighter version for backgrounds
    bg_alpha = 0.15
    main_slide_alpha = 0.08  # Even lighter for content slides
    
    # Determine text color based on brightness
    brightness = (r * 0.299 + g * 0.587 + b * 0.114)
    text_color = "white" if brightness < 128 else "#2c3e50"
    
    return {
        "primary_color": f"rgb({r}, {g}, {b})",
        "primary_hex": f"#{r:02x}{g:02x}{b:02x}",
        "background_color": f"rgba({r}, {g}, {b}, {bg_alpha})",
        "main_slide_bg": f"rgba({r}, {g}, {b}, {main_slide_alpha})",  # New: lighter for content
        "text_color": text_color,
        "accent_color": f"rgb({accent_r}, {accent_g}, {accent_b})",
        "secondary_color": f"rgb({max(0, r-30)}, {max(0, g-30)}, {max(0, b-30)})",
        "light_accent": f"rgba({accent_r}, {accent_g}, {accent_b}, 0.3)"
    }

def enhance_vibrancy(r, g, b, factor=1.2):
    """Enhance color vibrancy while keeping it realistic"""
    # Convert to HSV for better control
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    
    if max_val == min_val:  # Grayscale
        return r, g, b
    
    # Increase saturation
    delta = max_val - min_val
    new_delta = min(255, delta * factor)
    
    # Scale colors proportionally
    if max_val > 0:
        scale = new_delta / delta
        
        # Apply scaling while preserving ratios
        new_r = min(255, int(min_val + (r - min_val) * scale))
        new_g = min(255, int(min_val + (g - min_val) * scale))
        new_b = min(255, int(min_val + (b - min_val) * scale))
        
        return new_r, new_g, new_b
    
    return r, g, b

def get_default_theme():
    """Return default theme colors"""
    return {
        "primary_color": "rgb(74, 144, 226)",
        "primary_hex": "#4a90e2",
        "background_color": "rgba(74, 144, 226, 0.15)",
        "main_slide_bg": "rgba(74, 144, 226, 0.08)",  # Added missing property
        "text_color": "white",
        "accent_color": "rgb(124, 194, 255)",
        "secondary_color": "rgb(44, 114, 196)",
        "light_accent": "rgba(124, 194, 255, 0.3)"
    }
def format_text_as_bullets(text: str) -> str:
    """Convert text into HTML bullet points with spacing"""
    if not text:
        return ""
    
    # Split by sentences (basic approach)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if len(sentences) <= 1:
        # If only one sentence or less, return as single bullet
        return f"<ul><li>{text}</li></ul>"
    
    # Create bullet points for multiple sentences with spacing
    bullets = "".join([f"<li style='margin-bottom: 20px;'>{sentence.strip()}.</li>" for sentence in sentences if sentence.strip()])
    return f"<ul style='margin-top: 0; margin-bottom: 0;'>{bullets}</ul>"

def generate_html_template(topic: str, slides: list, theme: dict, script_id: str) -> str:
    """Generate the complete HTML presentation with 16:9 aspect ratio"""
    
    # Add the format_text_as_bullets function to Jinja2 filters
    from jinja2 import Environment, BaseLoader
    
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
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">

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
      /* Remove the global margin: 0 !important; */
    }
    
    .section-slide h2 {
      font-size: 90px;
      font-weight: 700;
      line-height: 1.25;
      color: white;
      text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
      margin-top: 0 !important;
      margin-bottom: 30px !important; /* This should now work */
      padding-bottom: 0;
      position: relative;
      max-width: 85%;
      margin-left: auto;
      margin-right: auto;
    }
    
    /* Update the ::after pseudo-element */
    .section-slide h2::after {
      content: '';
      display: block;
      width: 120px;
      height: 6px;
      background: rgba(255,255,255,0.45);
      border-radius: 3px;
      margin-top: 30px; /* Space between H2 text and the line */
      margin-bottom: 0; /* Remove this since the line doesn't need bottom margin */
      margin-left: auto;
      margin-right: auto;
    }

    .section-slide p {
      font-size: 40px;
      font-weight: 300;
      line-height: 1.6;
      max-width: 70%;
      margin-top: 40px !important; /* Space after the h2::after line */
      margin-bottom: 0 !important;
      margin-left: auto;
      margin-right: auto;
      opacity: 0.9;
      text-align: center;
    }
    
    /* Enhanced Title Slide with Image Background and Overlay */
    .title-slide {
      position: relative;
      color: white;
      text-align: center;
      overflow: hidden;
      padding: 0 !important;
      background: linear-gradient(135deg, {{ theme.primary_color }} 0%, {{ theme.accent_color }} 100%);
    }
    
    /* Background image for title slide */
    .title-slide-background {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
    }
    
    .title-slide-background img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      filter: blur(2px) brightness(0.8);
    }
    
    /* Gradient overlay for better text readability */
    .title-slide-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(
        135deg,
        rgba({{ theme.primary_color | replace('rgb(', '') | replace(')', '') }}, 0.85) 0%,
        rgba({{ theme.accent_color | replace('rgb(', '') | replace(')', '') }}, 0.75) 50%,
        rgba(0, 0, 0, 0.6) 100%
      );
      z-index: 2;
    }
    
    /* Decorative elements */
    .title-slide-overlay::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: 
        radial-gradient(circle at 20% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(255,255,255,0.05) 0%, transparent 50%);
      z-index: 1;
    }
    
    /* Content container */
    .title-slide-content {
      position: relative;
      z-index: 3;
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      padding: 80px;
      text-align: center;
    }
    
    /* Title styling */
    .title-slide h1 {
      font-size: 120px;
      font-weight: 800;
      text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
      line-height: 1.1;
      max-width: 1600px;
      margin: 0 auto 40px auto;
      background: linear-gradient(45deg, #ffffff, rgba(255,255,255,0.9));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
      0% {
        filter: drop-shadow(0 0 20px rgba(255,255,255,0.3));
      }
      100% {
        filter: drop-shadow(0 0 30px rgba(255,255,255,0.5));
      }
    }
    
    /* Subtitle for presentations */
    .title-slide-subtitle {
      font-size: 36px;
      font-weight: 300;
      opacity: 0.9;
      margin-top: 20px;
      text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
      letter-spacing: 2px;
    }
    
    /* Fallback styling when no image is available */
    .title-slide.no-image {
      background: linear-gradient(135deg, {{ theme.primary_color }} 0%, {{ theme.accent_color }} 100%);
    }
    
    .title-slide.no-image .title-slide-content {
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(10px);
      border-radius: 30px;
      border: 1px solid rgba(255,255,255,0.1);
      margin: 100px;
      box-shadow: 0 25px 50px rgba(0,0,0,0.2);
    }
    
    /* Section Slide - 16:9 optimized */
    .section-slide {
      background: linear-gradient(145deg, {{ theme.primary_color }} 10%, {{ theme.accent_color }} 90%); /* Updated gradient */
      color: white;
      /* Override general section alignment and padding to position content at the top */
      justify-content: flex-start !important; 
      align-items: center !important; /* Center content horizontally */
      padding-top: 80px !important; /* More space from the top edge */
      padding-left: 60px !important; /* Standard horizontal padding */
      padding-right: 60px !important;
      padding-bottom: 60px !important; /* Standard bottom padding */
      text-align: center; /* Default text alignment for children */
    }

    .section-slide h2 {
    font-size: 90px; /* Prominent but balanced size */
    font-weight: 700; /* Bolder than default h2 */
    line-height: 1.25;
    color: white;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.3); /* Subtle depth */
    margin-top: 0 !important; /* Override global margin:0!important */
    margin-bottom: 0 !important; /* Let ::after handle spacing below */
    padding-bottom: 0; /* No extra padding needed here */
    position: relative; /* For ::after pseudo-element positioning */
    max-width: 85%; /* Prevent title from being too wide */
    margin-left: auto; /* Center the title horizontally */
    margin-right: auto; /* Center the title horizontally */
  }
    
    /* Decorative separator line for section title */
    .section-slide h2::after {
      content: '';
      display: block; /* Makes margin work as expected */
      width: 120px; /* Width of the line */
      height: 6px;  /* Thickness of the line */
      background: rgba(255,255,255,0.45); /* Semi-transparent white, good contrast */
      border-radius: 3px; /* Rounded ends */
      margin-top: 30px; /* Space between H2 text and the line */
      margin-bottom: 0; /* Remove this since the line doesn't need bottom margin */
      margin-left: auto; /* Centers the line */
      margin-right: auto; /* Centers the line */
    }

    .section-slide p {
      font-size: 40px; /* Readable size for section description */
      font-weight: 300; /* Lighter weight for descriptive text */
      line-height: 1.6;
      max-width: 70%; /* Constrain width for better readability */
      margin-top: 40px !important; /* Space after the h2::after line */
      margin-bottom: 0 !important; /* Override global margin if any on P by mistake */
      margin-left: auto;
      margin-right: auto;
      opacity: 0.9; /* Slightly subdued */
      text-align: center; /* Ensure paragraph text is centered */
    }
    
    /* Main Slide Layout 1 - Original (Text left, Image right) */
    .main-slide {
      background: #e6e1d8 !important;
      color: #2c3e50 !important;
      justify-content: flex-start !important; /* Ensure headings start at top */
      padding-top: 80px !important; 
    }
    
    .main-slide h2 {
      font-size: 72px;
      margin-bottom: 50px;
      text-align: center;
      margin-top: 0 !important; 
      color: #2c3e50 !important;
      text-shadow: none;
    }
    
    .main-slide-content {
      display: grid;
      grid-template-columns: 1fr 480px;
      gap: 80px;
      align-items: center;
      max-width: 1600px;
      margin: 0 auto;
      height: auto;
      flex: 1;
    }
    
    .main-slide-text {
      font-size: 42px;
      line-height: 1.7;
      text-align: justify;
      color: #2c3e50 !important;
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

    /* Layout 2 - Image Dominant Left (70% image, 30% text) */
    .main-image-dominant {
      background: #e6e1d8 !important;
      color: #2c3e50 !important;
      justify-content: flex-start !important; /* Add this */
      padding-top: 80px !important; /* Add this */
    }
    
    .main-image-dominant h2 {
      font-size: 72px;
      margin-bottom: 50px;
      margin-top: 0 !important; /* Add this */
      text-align: center;
      color: #2c3e50 !important;
      text-shadow: none;
    }
    
    .main-image-dominant .content-container {
      display: grid;
      grid-template-columns: 60% 40%;
      gap: 60px;
      max-width: 1700px;
      margin: 0 auto;
      height: auto; /* Changed from 100% to auto */
      align-items: center;
      flex: 1; /* Add this to take remaining space */
    }
    
    .main-image-dominant .image-section {
      position: relative;
      height: 600px;
      overflow: hidden;
      border-radius: 25px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.2);
    }
    
    .main-image-dominant .image-section img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.4s ease;
    }
    
    .main-image-dominant .image-section:hover img {
      transform: scale(1.03);
    }
    
    .main-image-dominant .text-section {
      padding: 40px 20px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    
    .main-image-dominant .text-section p {
      font-size: 38px;
      line-height: 1.6;
      text-align: left;
      color: #2c3e50 !important;
    }
    
    .main-image-dominant .no-image-placeholder {
      width: 100%;
      height: 600px;
      background: linear-gradient(135deg, #e0e0e0, #f5f5f5);
      border-radius: 25px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #888;
      font-size: 36px;
    }

    /* Layout 2B - Image Dominant Right (30% text, 70% image) */
    .main-image-dominant-2 {
      background: #e6e1d8 !important;
      color: #2c3e50 !important;
      justify-content: flex-start !important; /* Add this */
      padding-top: 80px !important; /* Add this */
    }
    
    .main-image-dominant-2 h2 {
      font-size: 72px;
      margin-bottom: 50px;
      margin-top: 0 !important; /* Add this */
      text-align: center;
      color: #2c3e50 !important;
      text-shadow: none;
    }
   
    .main-image-dominant-2 .content-container {
      display: grid;
      grid-template-columns: 30% 70%;
      gap: 60px;
      max-width: 1700px;
      height: auto; /* Changed from 100% to auto */
      align-items: center;
      flex: 1; /* Add this to take remaining space */
    }
    
    .main-image-dominant-2 .text-section {
      padding: 40px 20px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    
    .main-image-dominant-2 .text-section p {
      font-size: 38px;
      line-height: 1.6;
      text-align: right;
      color: #2c3e50 !important;
    }
    
    .main-image-dominant-2 .image-section {
      position: relative;
      height: 600px;
      overflow: hidden;
      border-radius: 25px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.2);
    }
    
    .main-image-dominant-2 .image-section img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.4s ease;
    }
    
    .main-image-dominant-2 .image-section:hover img {
      transform: scale(1.03);
    }
    
    .main-image-dominant-2 .no-image-placeholder {
      width: 100%;
      height: 600px;
      background: linear-gradient(135deg, #e0e0e0, #f5f5f5);
      border-radius: 25px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #888;
      font-size: 36px;
    }

    /* Layout 3 - Image Top Full Width */
    .main-image-top {
      background: #e6e1d8 !important;
      color: #2c3e50 !important;
      display: flex !important;
      flex-direction: column !important;
      justify-content: flex-start !important;
      padding: 0 !important;
      padding-top: 80px !important; /* Add consistent top padding */
    }
    
    .main-image-top h2 {
      font-size: 72px;
      margin-bottom: 50px;
      margin-top: 0 !important; /* Add this */
      text-align: center;
      color: #2c3e50 !important;
      text-shadow: none;
      margin-left: 60px;
      margin-right: 60px;
      background: rgba(255,255,255,0.9);
      padding: 20px;
      border-radius: 15px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    
    .main-image-top .top-image-container {
      width: 100%;
      height: 500px;
      overflow: hidden;
      position: relative;
    }
    
    .main-image-top .top-image-container img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      filter: brightness(0.9);
    }
    
    .main-image-top .top-image-container::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 100px;
      background: linear-gradient(transparent, rgba(255,255,255,0.3));
    }
    
    .main-image-top .bottom-text-container {
      flex: 1;
      padding: 50px 80px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #e6e1d8 !important;
    }
    
    .main-image-top .bottom-text-container p {
      font-size: 40px;
      line-height: 1.65;
      text-align: justify;
      max-width: 1400px;
      color: #2c3e50 !important;
    }
    
    .main-image-top .no-image-placeholder {
      height: 500px;
      background: linear-gradient(135deg, #e8f4f8, #d1ecf1);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #7f8c8d;
      font-size: 32px;
    }

    /* Layout 4 - Text Focus with Small Accent Image */
    .main-text-focus {
      background: #e6e1d8 !important;
      color: #2c3e50 !important;
      position: relative;
      overflow: hidden;
      justify-content: flex-start !important; /* Add this */
      padding-top: 80px !important; /* Add this */
    }
    
    .main-text-focus h2 {
      font-size: 72px;
      margin-bottom: 50px; /* Changed from 100px to 50px for consistency */
      margin-top: 0 !important; /* Add this */
      text-align: center;
      color: #2c3e50 !important;
      text-shadow: none;
      position: relative;
      z-index: 2;
    }
    
    .main-text-focus .focus-content {
      display: grid;
      grid-template-columns: 1fr 700px;
      gap: 25px;
      max-width: 1600px;
      align-items: center;
      position: relative;
      z-index: 2;
      flex: 1; /* Add this to take remaining space */
    }

    /* Thank You Slide - Keep centered */
    .thankyou-slide {
      background: linear-gradient(135deg, {{ theme.primary_color }} 0%, {{ theme.accent_color }} 100%);
      color: white;
      text-align: center;
      justify-content: center !important; /* Keep centered for thank you slides */
    }
    
    .thankyou-slide h3 {
      font-size: 108px;
      margin-bottom: 40px;
    }
    
    .thankyou-slide p {
      font-size: 48px;
      opacity: 0.9;
      text-align: center;
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
        <section class="title-slide{% if not slide.image_base64 %} no-image{% endif %}">
          {% if slide.image_base64 %}
            <div class="title-slide-background">
              <img src="{{ slide.image_base64 }}" alt="{{ slide.image_alt or 'Presentation background' }}" />
            </div>
            <div class="title-slide-overlay"></div>
          {% endif %}
          <div class="title-slide-content">
            <h1>{{ slide.title }}</h1>
          </div>
        </section>
      {% elif slide.type == 'section' %}
        <section class="section-slide">
          <h2>{{ slide.title }}</h2>
          <p>{{ slide.body }}</p>
        </section>
      {% elif slide.type == 'main' %}
        <section class="main-slide">
          <h2>{{ slide.title }}</h2>
          <div class="main-slide-content">
            <div class="main-slide-text">
              {{ slide.body | format_bullets | safe }}
            </div>
            <div class="main-slide-image">
              {% if slide.image_base64 %}
                <img src="{{ slide.image_base64 }}" alt="{{ slide.image_alt }}" />
              {% else %}
                <div class="no-image-placeholder">No Image Available</div>
              {% endif %}
            </div>
          </div>
        </section>
      {% elif slide.type == 'main-image-dominant' %}
        <section class="main-image-dominant">
          <h2>{{ slide.title }}</h2>
          <div class="content-container">
            <div class="image-section">
              {% if slide.image_base64 %}
                <img src="{{ slide.image_base64 }}" alt="{{ slide.image_alt }}" />
              {% else %}
                <div class="no-image-placeholder">No Image Available</div>
              {% endif %}
            </div>
            <div class="text-section">
              {{ slide.body | format_bullets | safe }}
            </div>
          </div>
        </section>
      {% elif slide.type == 'main-image-dominant-2' %}
        <section class="main-image-dominant-2">
          <h2>{{ slide.title }}</h2>
          <div class="content-container">
            <div class="text-section">
              {{ slide.body | format_bullets | safe }}
            </div>
            <div class="image-section">
              {% if slide.image_base64 %}
                <img src="{{ slide.image_base64 }}" alt="{{ slide.image_alt }}" />
              {% else %}
                <div class="no-image-placeholder">No Image Available</div>
              {% endif %}
            </div>
          </div>
        </section>
      {% elif slide.type == 'main-image-top' %}
        <section class="main-image-top">
          <div class="top-image-container">
            {% if slide.image_base64 %}
              <img src="{{ slide.image_base64 }}" alt="{{ slide.image_alt }}" />
            {% else %}
              <div class="no-image-placeholder">No Image Available</div>
            {% endif %}
          </div>
          <h2>{{ slide.title }}</h2>
          <div class="bottom-text-container">
            {{ slide.body | format_bullets | safe }}
          </div>
        </section>
      {% elif slide.type == 'main-text-focus' %}
        <section class="main-text-focus">
          <h2>{{ slide.title }}</h2>
          <div class="focus-content">
            <div class="main-text">
              {{ slide.body | format_bullets | safe }}
            </div>
            <div class="accent-image">
              {% if slide.image_base64 %}
                <img src="{{ slide.image_base64 }}" alt="{{ slide.image_alt }}" />
              {% else %}
                <div class="no-image-placeholder">No Image Available</div>
              {% endif %}
            </div>
          </div>
        </section>
      {% elif slide.type == 'thankyou' %}
        <section class="thankyou-slide">
          <h3>{{ slide.title }} 🙏</h3>
          <p style="font-style: italic;">{{ slide.subtitle }}</p>
        </section>
      {% endif %}
    {% endfor %}
  </div>
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
  transition: 'none',
  transitionSpeed: 'fast',
});

// Update scale on window resize
window.addEventListener('resize', updateScale);
updateScale();
</script>

</body>
</html>"""

    # Create Jinja2 environment with custom filter
    env = Environment(loader=BaseLoader())
    env.filters['format_bullets'] = format_text_as_bullets
    template = env.from_string(template_str)
    
    return template.render(topic=topic, slides=slides, theme=theme, script_id=script_id)