from jinja2 import Template
import os
from app import *
def get_theme_colors(images_data: dict) -> dict:
    """Extract theme colors from the first available image with enhanced methods"""
    try:
        # Find first available image
        first_image_path = None
        for image_info in images_data.values():
            if image_info.get('image_path') and os.path.exists(image_info['image_path']):
                first_image_path = image_info['image_path']
                break
        
        if first_image_path:
            # Method 1: ColorThief with better quality settings
            try:
                from colorthief import ColorThief
                color_thief = ColorThief(first_image_path)
                
                # Get dominant color with high quality
                dominant_color = color_thief.get_color(quality=1)
                
                # Get color palette for better color selection
                palette = color_thief.get_palette(color_count=6, quality=1)
                
                # Select the most vibrant color from palette
                best_color = select_best_color(palette)
                if best_color:
                    dominant_color = best_color
                    
            except Exception as e:
                print(f"ColorThief failed: {e}")
                # Fallback to PIL-based extraction
                dominant_color = extract_color_with_pil(first_image_path)
            
            if dominant_color:
                r, g, b = dominant_color
                
                # Enhanced color processing
                return generate_color_scheme(r, g, b)
                
    except Exception as e:
        print(f"Error generating theme: {e}")
    
    # Default theme
    return get_default_theme()

def select_best_color(palette):
    """Select the most vibrant and suitable color from palette"""
    best_color = None
    best_score = 0
    
    for color in palette:
        r, g, b = color
        
        # Calculate color vibrancy (saturation + brightness)
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0
        brightness = (r + g + b) / 3
        
        # Avoid very dark or very light colors
        if brightness < 50 or brightness > 200:
            continue
            
        # Calculate score (higher is better)
        score = saturation * 100 + brightness * 0.5
        
        if score > best_score:
            best_score = score
            best_color = color
    
    return best_color

def extract_color_with_pil(image_path):
    """Fallback color extraction using PIL"""
    try:
        from PIL import Image
        import numpy as np
        
        # Open and resize image for faster processing
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize((150, 150))  # Smaller size for faster processing
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Reshape to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Remove very dark and very light pixels
            filtered_pixels = []
            for pixel in pixels:
                brightness = sum(pixel) / 3
                if 30 < brightness < 225:  # Filter out too dark/light pixels
                    filtered_pixels.append(pixel)
            
            if filtered_pixels:
                # Find most common color using simple clustering
                return find_dominant_color_kmeans(filtered_pixels)
            
    except Exception as e:
        print(f"PIL extraction failed: {e}")
        
    return None

def find_dominant_color_kmeans(pixels, k=5):
    """Use K-means clustering to find dominant color"""
    try:
        from sklearn.cluster import KMeans
        import numpy as np
        
        pixels = np.array(pixels)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_
        
        # Count pixels in each cluster
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        
        # Select color from largest cluster that's not too dark/light
        for i in np.argsort(label_counts)[::-1]:  # Sort by cluster size
            color = colors[i]
            brightness = sum(color) / 3
            if 50 < brightness < 200:  # Good brightness range
                return tuple(map(int, color))
                
        # Fallback to largest cluster
        dominant_idx = np.argmax(label_counts)
        return tuple(map(int, colors[dominant_idx]))
        
    except ImportError:
        print("scikit-learn not available, using simple average")
        # Simple fallback: average of all pixels
        pixels = np.array(pixels)
        return tuple(map(int, np.mean(pixels, axis=0)))
    except Exception as e:
        print(f"K-means failed: {e}")
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
        "text_color": "white",
        "accent_color": "rgb(124, 194, 255)",
        "secondary_color": "rgb(44, 114, 196)",
        "light_accent": "rgba(124, 194, 255, 0.3)"
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
      background: linear-gradient(135deg, {{ theme.main_slide_bg }} 0%, rgba(255,255,255,0.98) 100%);
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
        <section class="section-slide">
          <h2>{{ slide.title }}</h2>
          <p>{{ slide.body }}</p>
        </section>
      {% elif slide.type == 'main' %}
        <section class="main-slide">
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

    template = Template(template_str)
    return template.render(topic=topic, slides=slides, theme=theme, script_id=script_id)