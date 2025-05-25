# from weasyprint import HTML
# HTML('/home/visharad/Downloads/script_1748164758_presentation.html').write_pdf('output.pdf')


from playwright.sync_api import sync_playwright
import os

def html_to_pdf(input_html_path, output_pdf_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Set viewport to 16:9 ratio to match your presentation
        page.set_viewport_size({"width": 1920, "height": 1080})
        
        # Convert to absolute path
        abs_path = os.path.abspath(input_html_path)
        file_url = f'file://{abs_path}'
        
        page.goto(file_url)
        
        # Wait for Reveal.js to load
        page.wait_for_selector('.reveal')
        page.wait_for_timeout(3000)  # Longer wait for your complex styling
        
        # Add print-specific CSS to ensure 16:9 format
        page.add_style_tag(content="""
            @page {
                size: 297mm 167mm !important;
                margin: 0 !important;
            }
            
            body {
                width: 297mm !important;
                height: 167mm !important;
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
            }
            
            .reveal, .reveal .slides {
                width: 297mm !important;
                height: 167mm !important;
                transform: none !important;
                left: 0 !important;
                top: 0 !important;
            }
            
            .reveal .slides section {
                width: 297mm !important;
                height: 167mm !important;
                page-break-after: always !important;
                margin: 0 !important;
                padding: 5mm !important;
                box-sizing: border-box !important;
            }
            
            .controls {
                display: none !important;
            }
        """)
        
        # Generate PDF with 16:9 dimensions (A4 landscape is close to 16:9)
        page.pdf(
            path=output_pdf_path,
            format="A4",
            landscape=True,  # This gives us 16:9-ish ratio
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            prefer_css_page_size=True
        )
        browser.close()

# Use the correct filename
html_to_pdf('/home/visharad/Downloads/script_1748164758_presentation.html', 'output3.pdf')