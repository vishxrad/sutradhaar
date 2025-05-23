import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

class CanvaAPI:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = "https://api.canva.com/rest/v1"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    def create_presentation(self, title="My Presentation"):
        """Create a new presentation design"""
        url = f"{self.base_url}/designs"
        payload = {
            "design_type": "presentation",
            "title": title
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create design: {response.text}")
    

    def upload_image(self, image_path):
        """Upload an image to Canva"""
        url = f"{self.base_url}/assets"
        
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.post(url, headers=headers, files=files)
            if response.status_code == 201:
                return response.json()['asset']['id']
            else:
                raise Exception(f"Failed to upload image: {response.text}")
        

    def add_slide_content(self, design_id, title, body, image_asset_id=None):
        """Add title, body text, and optional image to a slide"""
        url = f"{self.base_url}/designs/{design_id}/pages"
        
        elements = [
            {
                "type": "text",
                "text": title,
                "position": {"x": 50, "y": 50},
                "font_size": 32,
                "font_weight": "bold"
            },
            {
                "type": "text",
                "text": body,
                "position": {"x": 50, "y": 150},
                "font_size": 16
            }
        ]
        
        if image_asset_id:
            elements.append({
                "type": "image",
                "asset_id": image_asset_id,
                "position": {"x": 50, "y": 300},
                "width": 400,
                "height": 200
            })
        
        payload = {"elements": elements}
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to add content: {response.text}")
    

def create_presentation_with_content(image_path, title, body):
    # Initialize API client
    canva = CanvaAPI(os.getenv('CANVA_ACCESS_TOKEN'))
    
    try:
        # Create new presentation
        design = canva.create_presentation(title="My PPT")
        design_id = design['design']['id']
        print(f"Created design: {design_id}")
        
        # Upload image
        image_asset_id = canva.upload_image(image_path)
        print(f"Uploaded image: {image_asset_id}")
        
        # Add content to slide
        slide = canva.add_slide_content(design_id, title, body, image_asset_id)
        print("Added content to slide")
        
        # Get shareable URL
        share_url = canva.get_design_url(design_id)
        print(f"Presentation URL: {share_url}")
        
        return design_id, share_url
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Usage
if __name__ == "__main__":
    image_path = "/home/visharad/Downloads/liana-s-ird6OOE2LXI-unsplash.jpg"
    title = "Greenhouse Gases Explained"
    body = "Greenhouse gases trap heat in the atmosphere and keep Earth warm enough to sustain life. However, increased emissions from human activities like burning fossil fuels are enhancing this effect, leading to global warming."
    
    design_id, url = create_presentation_with_content(image_path, title, body)