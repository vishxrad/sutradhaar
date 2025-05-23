import os.path
import uuid

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these SCOPES, delete token.json.
SCOPES = ['https://www.googleapis.com/auth/presentations']
CLIENT_SECRET_FILE = 'client_secret.json'
TOKEN_FILE = 'token.json'

class GoogleSlidesAPI:
    def __init__(self):
        self.creds = self._get_credentials()
        self.service = build('slides', 'v1', credentials=self.creds)

    def _get_credentials(self):
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing token: {e}. Please re-authenticate.")
                    creds = None # Force re-authentication
            if not creds: # creds might be None if refresh failed
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        return creds

    def create_presentation(self, title="My Google Slides Presentation"):
        """Creates a new Google Slides presentation."""
        try:
            body = {'title': title}
            presentation = self.service.presentations().create(body=body).execute()
            presentation_id = presentation.get('presentationId')
            print(f"Created presentation with ID: {presentation_id}")
            return presentation_id
        except HttpError as error:
            print(f"An API error occurred while creating presentation: {error}")
            print(f"Details: {error.resp.status}, {error._get_reason()}")
            return None

    def add_slide_with_content(self, presentation_id, slide_title, body_text, image_url=None):
        """Adds a new slide with title, body text, and an optional image."""
        requests_batch = []
        
        page_id = f"slide_{uuid.uuid4().hex}"
        title_shape_id = f"title_{uuid.uuid4().hex}"
        body_shape_id = f"body_{uuid.uuid4().hex}"
        image_element_id = f"image_{uuid.uuid4().hex}"


        # 1. Create a new slide
        requests_batch.append({
            'createSlide': {
                'objectId': page_id,
                'insertionIndex': '1', 
                'slideLayoutReference': {'predefinedLayout': 'BLANK'}
            }
        })

        # 2. Add Title Text Box and Text (EMU units for size and position)
        # 1 inch = 914400 EMUs. 1 pt = 12700 EMUs.
        # Let's use PT for easier understanding and convert.
        emu_per_pt = 12700
        title_pt_x, title_pt_y = 50, 50
        title_width_pt, title_height_pt = 600, 50

        requests_batch.extend([
            {'createShape': {
                'objectId': title_shape_id, 'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': page_id,
                    'size': {
                        'width': {'magnitude': title_width_pt * emu_per_pt, 'unit': 'EMU'},
                        'height': {'magnitude': title_height_pt * emu_per_pt, 'unit': 'EMU'}
                    },
                    'transform': {
                        'scaleX': 1, 'scaleY': 1,
                        'translateX': title_pt_x * emu_per_pt, 'translateY': title_pt_y * emu_per_pt, 'unit': 'EMU'
                    }}}},
            {'insertText': {'objectId': title_shape_id, 'insertionIndex': 0, 'text': slide_title}},
            {'updateTextStyle': {
                'objectId': title_shape_id, 'textRange': {'type': 'ALL'},
                'style': {'bold': True, 'fontSize': {'magnitude': 24, 'unit': 'PT'}},
                'fields': 'bold,fontSize'}}
        ])

        # 3. Add Body Text Box and Text
        body_pt_x, body_pt_y = 50, 120
        body_width_pt, body_height_pt = 600, 150

        requests_batch.extend([
            {'createShape': {
                'objectId': body_shape_id, 'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': page_id,
                    'size': {
                        'width': {'magnitude': body_width_pt * emu_per_pt, 'unit': 'EMU'},
                        'height': {'magnitude': body_height_pt * emu_per_pt, 'unit': 'EMU'}
                    },
                    'transform': {
                        'scaleX': 1, 'scaleY': 1,
                        'translateX': body_pt_x * emu_per_pt, 'translateY': body_pt_y * emu_per_pt, 'unit': 'EMU'
                    }}}},
            {'insertText': {'objectId': body_shape_id, 'insertionIndex': 0, 'text': body_text}},
            {'updateTextStyle': {
                'objectId': body_shape_id, 'textRange': {'type': 'ALL'},
                'style': {'fontSize': {'magnitude': 14, 'unit': 'PT'}},
                'fields': 'fontSize'}}
        ])
        
        if image_url:
            img_pt_x, img_pt_y = 50, 300
            img_width_pt, img_height_pt = 400, 225 # Example size

            requests_batch.append({'createImage': {
                'objectId': image_element_id, 'url': image_url,
                'elementProperties': {
                    'pageObjectId': page_id,
                    'size': {
                        'width': {'magnitude': img_width_pt * emu_per_pt, 'unit': 'EMU'},
                        'height': {'magnitude': img_height_pt * emu_per_pt, 'unit': 'EMU'}
                    },
                    'transform': {
                        'scaleX': 1, 'scaleY': 1,
                        'translateX': img_pt_x * emu_per_pt, 'translateY': img_pt_y * emu_per_pt, 'unit': 'EMU'
                    }}}})
        try:
            body = {'requests': requests_batch}
            response = self.service.presentations().batchUpdate(
                presentationId=presentation_id, body=body).execute()
            print(f"Added content to slide on presentation {presentation_id}.")
            return response
        except HttpError as error:
            print(f"An API error occurred while adding slide content: {error}")
            print(f"Details: {error.resp.status}, {error._get_reason()}")
            return None

    def get_presentation_url(self, presentation_id):
        return f"https://docs.google.com/presentation/d/{presentation_id}/edit"

def create_google_slides_presentation_with_content(main_title, slide_title, slide_body, image_url=None):
    """
    Orchestrates creating a Google Slides presentation and adding content.
    """
    if not os.path.exists(CLIENT_SECRET_FILE):
        print(f"ERROR: {CLIENT_SECRET_FILE} not found. Please download it from Google Cloud Console.")
        return None, None
        
    try:
        slides_api = GoogleSlidesAPI() # Handles authentication
        
        presentation_id = slides_api.create_presentation(title=main_title)
        
        if not presentation_id:
            print("Failed to create presentation.")
            return None, None
            
        slide_creation_response = slides_api.add_slide_with_content(
            presentation_id,
            slide_title=slide_title,
            body_text=slide_body,
            image_url=image_url
        )

        if not slide_creation_response:
            print("Failed to add content to the slide.")
            # Optionally, you might want to return the URL even if content adding failed partially
            return presentation_id, slides_api.get_presentation_url(presentation_id) 

        presentation_url = slides_api.get_presentation_url(presentation_id)
        print(f"Successfully created presentation and added content.")
        
        return presentation_id, presentation_url
        
    except HttpError as error:
        print(f"A Google API HttpError occurred: {error}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

if __name__ == "__main__":
    # IMPORTANT: 
    # 1. Ensure 'client_secret.json' is in the same directory.
    # 2. You have enabled the Google Slides API in your Google Cloud Project.
    # 3. The first run will open a browser window for authentication. A 'token.json' file will be created.
    
    # --- User Inputs ---
    pres_title = "My Automated Presentation"
    sl_title = "Greenhouse Gases Explained (Google Slides)"
    sl_body = ("Greenhouse gases trap heat in the atmosphere, keeping Earth warm. "
               "Increased emissions from human activities like burning fossil fuels "
               "are enhancing this effect, leading to global warming.")
    # Replace with a publicly accessible image URL or set to None
    # Example: A publicly accessible image URL of a plant
    img_url = "https://images.unsplash.com/photo-1452800185063-6db5e12b8e2e?q=80&w=800" 
    # Or use a generic placeholder if you don't have one:
    # img_url = "https://via.placeholder.com/400x225.png?text=My+Image"


    print("Attempting to create Google Slides presentation...")
    new_presentation_id, url = create_google_slides_presentation_with_content(
        main_title=pres_title,
        slide_title=sl_title,
        slide_body=sl_body,
        image_url=img_url 
    )

    if url:
        print(f"\nAccess your presentation at: {url}")
        print(f"Presentation ID: {new_presentation_id}")
    else:
        print("\nCould not create the presentation or add content fully.")

# ```

# **How to Use `google_slides_manager.py`:**

# 1.  Save the code above as `google_slides_manager.py`.
# 2.  Ensure `client_secret.json` (downloaded from Google Cloud Console) is in the same directory.
# 3.  Run the script from your terminal: `python google_slides_manager.py`
# 4.  **First Run:** Your web browser will open, asking you to log in with your Google account and authorize the application to access your Google Slides.
# 5.  After authorization, a `token.json` file will be created in the same directory. This stores your access tokens so you don't have to re-authorize every time (unless the token expires or you change scopes).
# 6.  The script will then attempt to create a new presentation and add a slide with the specified content. The URL of the new presentation will be printed.

# This script provides a foundation. The Google Slides API is very powerful, allowing for detailed control over layouts, styling, animations, and more. You can expand upon this class to add more features as needed. Remember that element positioning and sizing in the Slides API often use EMUs (English Metric Units) or Points.# filepath: google_slides_manager.py
# import os.path
# import uuid

# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError

# # If modifying these SCOPES, delete token.json.
# SCOPES = ['https://www.googleapis.com/auth/presentations']
# CLIENT_SECRET_FILE = 'client_secret.json'
# TOKEN_FILE = 'token.json'

# class GoogleSlidesAPI:
#     def __init__(self):
#         self.creds = self._get_credentials()
#         self.service = build('slides', 'v1', credentials=self.creds)

#     def _get_credentials(self):
#         creds = None
#         if os.path.exists(TOKEN_FILE):
#             creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
#         if not creds or not creds.valid:
#             if creds and creds.expired and creds.refresh_token:
#                 try:
#                     creds.refresh(Request())
#                 except Exception as e:
#                     print(f"Error refreshing token: {e}. Please re-authenticate.")
#                     creds = None # Force re-authentication
#             if not creds: # creds might be None if refresh failed
#                 flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
#                 creds = flow.run_local_server(port=0)
#             with open(TOKEN_FILE, 'w') as token:
#                 token.write(creds.to_json())
#         return creds

#     def create_presentation(self, title="My Google Slides Presentation"):
#         """Creates a new Google Slides presentation."""
#         try:
#             body = {'title': title}
#             presentation = self.service.presentations().create(body=body).execute()
#             presentation_id = presentation.get('presentationId')
#             print(f"Created presentation with ID: {presentation_id}")
#             return presentation_id
#         except HttpError as error:
#             print(f"An API error occurred while creating presentation: {error}")
#             print(f"Details: {error.resp.status}, {error._get_reason()}")
#             return None

#     def add_slide_with_content(self, presentation_id, slide_title, body_text, image_url=None):
#         """Adds a new slide with title, body text, and an optional image."""
#         requests_batch = []
        
#         page_id = f"slide_{uuid.uuid4().hex}"
#         title_shape_id = f"title_{uuid.uuid4().hex}"
#         body_shape_id = f"body_{uuid.uuid4().hex}"
#         image_element_id = f"image_{uuid.uuid4().hex}"


#         # 1. Create a new slide
#         requests_batch.append({
#             'createSlide': {
#                 'objectId': page_id,
#                 'insertionIndex': '1', 
#                 'slideLayoutReference': {'predefinedLayout': 'BLANK'}
#             }
#         })

#         # 2. Add Title Text Box and Text (EMU units for size and position)
#         # 1 inch = 914400 EMUs. 1 pt = 12700 EMUs.
#         # Let's use PT for easier understanding and convert.
#         emu_per_pt = 12700
#         title_pt_x, title_pt_y = 50, 50
#         title_width_pt, title_height_pt = 600, 50

#         requests_batch.extend([
#             {'createShape': {
#                 'objectId': title_shape_id, 'shapeType': 'TEXT_BOX',
#                 'elementProperties': {
#                     'pageObjectId': page_id,
#                     'size': {
#                         'width': {'magnitude': title_width_pt * emu_per_pt, 'unit': 'EMU'},
#                         'height': {'magnitude': title_height_pt * emu_per_pt, 'unit': 'EMU'}
#                     },
#                     'transform': {
#                         'scaleX': 1, 'scaleY': 1,
#                         'translateX': title_pt_x * emu_per_pt, 'translateY': title_pt_y * emu_per_pt, 'unit': 'EMU'
#                     }}}},
#             {'insertText': {'objectId': title_shape_id, 'insertionIndex': 0, 'text': slide_title}},
#             {'updateTextStyle': {
#                 'objectId': title_shape_id, 'textRange': {'type': 'ALL'},
#                 'style': {'bold': True, 'fontSize': {'magnitude': 24, 'unit': 'PT'}},
#                 'fields': 'bold,fontSize'}}
#         ])

#         # 3. Add Body Text Box and Text
#         body_pt_x, body_pt_y = 50, 120
#         body_width_pt, body_height_pt = 600, 150

#         requests_batch.extend([
#             {'createShape': {
#                 'objectId': body_shape_id, 'shapeType': 'TEXT_BOX',
#                 'elementProperties': {
#                     'pageObjectId': page_id,
#                     'size': {
#                         'width': {'magnitude': body_width_pt * emu_per_pt, 'unit': 'EMU'},
#                         'height': {'magnitude': body_height_pt * emu_per_pt, 'unit': 'EMU'}
#                     },
#                     'transform': {
#                         'scaleX': 1, 'scaleY': 1,
#                         'translateX': body_pt_x * emu_per_pt, 'translateY': body_pt_y * emu_per_pt, 'unit': 'EMU'
#                     }}}},
#             {'insertText': {'objectId': body_shape_id, 'insertionIndex': 0, 'text': body_text}},
#             {'updateTextStyle': {
#                 'objectId': body_shape_id, 'textRange': {'type': 'ALL'},
#                 'style': {'fontSize': {'magnitude': 14, 'unit': 'PT'}},
#                 'fields': 'fontSize'}}
#         ])
        
#         if image_url:
#             img_pt_x, img_pt_y = 50, 300
#             img_width_pt, img_height_pt = 400, 225 # Example size

#             requests_batch.append({'createImage': {
#                 'objectId': image_element_id, 'url': image_url,
#                 'elementProperties': {
#                     'pageObjectId': page_id,
#                     'size': {
#                         'width': {'magnitude': img_width_pt * emu_per_pt, 'unit': 'EMU'},
#                         'height': {'magnitude': img_height_pt * emu_per_pt, 'unit': 'EMU'}
#                     },
#                     'transform': {
#                         'scaleX': 1, 'scaleY': 1,
#                         'translateX': img_pt_x * emu_per_pt, 'translateY': img_pt_y * emu_per_pt, 'unit': 'EMU'
#                     }}}})
#         try:
#             body = {'requests': requests_batch}
#             response = self.service.presentations().batchUpdate(
#                 presentationId=presentation_id, body=body).execute()
#             print(f"Added content to slide on presentation {presentation_id}.")
#             return response
#         except HttpError as error:
#             print(f"An API error occurred while adding slide content: {error}")
#             print(f"Details: {error.resp.status}, {error._get_reason()}")
#             return None

#     def get_presentation_url(self, presentation_id):
#         return f"https://docs.google.com/presentation/d/{presentation_id}/edit"

# def create_google_slides_presentation_with_content(main_title, slide_title, slide_body, image_url=None):
#     """
#     Orchestrates creating a Google Slides presentation and adding content.
#     """
#     if not os.path.exists(CLIENT_SECRET_FILE):
#         print(f"ERROR: {CLIENT_SECRET_FILE} not found. Please download it from Google Cloud Console.")
#         return None, None
        
#     try:
#         slides_api = GoogleSlidesAPI() # Handles authentication
        
#         presentation_id = slides_api.create_presentation(title=main_title)
        
#         if not presentation_id:
#             print("Failed to create presentation.")
#             return None, None
            
#         slide_creation_response = slides_api.add_slide_with_content(
#             presentation_id,
#             slide_title=slide_title,
#             body_text=slide_body,
#             image_url=image_url
#         )

#         if not slide_creation_response:
#             print("Failed to add content to the slide.")
#             # Optionally, you might want to return the URL even if content adding failed partially
#             return presentation_id, slides_api.get_presentation_url(presentation_id) 

#         presentation_url = slides_api.get_presentation_url(presentation_id)
#         print(f"Successfully created presentation and added content.")
        
#         return presentation_id, presentation_url
        
#     except HttpError as error:
#         print(f"A Google API HttpError occurred: {error}")
#         return None, None
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return None, None

# if __name__ == "__main__":
#     # IMPORTANT: 
#     # 1. Ensure 'client_secret.json' is in the same directory.
#     # 2. You have enabled the Google Slides API in your Google Cloud Project.
#     # 3. The first run will open a browser window for authentication. A 'token.json' file will be created.
    
#     # --- User Inputs ---
#     pres_title = "My Automated Presentation"
#     sl_title = "Greenhouse Gases Explained (Google Slides)"
#     sl_body = ("Greenhouse gases trap heat in the atmosphere, keeping Earth warm. "
#                "Increased emissions from human activities like burning fossil fuels "
#                "are enhancing this effect, leading to global warming.")
#     # Replace with a publicly accessible image URL or set to None
#     # Example: A publicly accessible image URL of a plant
#     img_url = "https://images.unsplash.com/photo-1452800185063-6db5e12b8e2e?q=80&w=800" 
#     # Or use a generic placeholder if you don't have one:
#     # img_url = "https://via.placeholder.com/400x225.png?text=My+Image"


#     print("Attempting to create Google Slides presentation...")
#     new_presentation_id, url = create_google_slides_presentation_with_content(
#         main_title=pres_title,
#         slide_title=sl_title,
#         slide_body=sl_body,
#         image_url=img_url 
#     )

#     if url:
#         print(f"\nAccess your presentation at: {url}")
#         print(f"Presentation ID: {new_presentation_id}")
#     else:
#         print("\nCould not create the presentation or add content fully.")

# ```

# **How to Use `google_slides_manager.py`:**

# 1.  Save the code above as `google_slides_manager.py`.
# 2.  Ensure `client_secret.json` (downloaded from Google Cloud Console) is in the same directory.
# 3.  Run the script from your terminal: `python google_slides_manager.py`
# 4.  **First Run:** Your web browser will open, asking you to log in with your Google account and authorize the application to access your Google Slides.
# 5.  After authorization, a `token.json` file will be created in the same directory. This stores your access tokens so you don't have to re-authorize every time (unless the token expires or you change scopes).
# 6.  The script will then attempt to create a new presentation and add a slide with the specified content. The URL of the new presentation will be printed.

# This script provides a foundation. The Google Slides API is very powerful, allowing for detailed control over layouts, styling, animations, and more. You can expand upon this class to add more features as needed. Remember that element positioning and sizing in the Slides API often use EMUs (English Metric Units) or Points.