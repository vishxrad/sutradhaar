import os
import uuid
import flask

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

app = flask.Flask(__name__)
app.secret_key = os.urandom(24) # Needed for session management

# --- Configuration ---
CLIENT_SECRET_FILE = 'client_secret.json'
SCOPES = ['https://www.googleapis.com/auth/presentations']
# This must be one of the Authorized redirect URIs configured in Google Cloud Console
REDIRECT_URI = 'http://127.0.0.1:5000/oauth2callback' 

# --- GoogleSlidesAPI Class (slightly modified to accept credentials) ---
class GoogleSlidesAPI:
    def __init__(self, credentials):
        self.creds = credentials
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                    # In a real app, you'd save the refreshed creds back to the session
                    flask.session['credentials'] = self._credentials_to_dict(self.creds)
                except Exception as e:
                    # Log the error and potentially clear session credentials
                    print(f"Error refreshing token: {e}")
                    flask.session.pop('credentials', None) # Force re-login
                    raise Exception(f"Failed to refresh token: {e}. Please log in again.")
            else:
                # If no valid credentials, force re-login by clearing session
                flask.session.pop('credentials', None)
                raise Exception("Invalid or missing credentials. Please log in.")
        
        self.service = build('slides', 'v1', credentials=self.creds)

    @staticmethod  # Make this a static method
    def _credentials_to_dict(credentials): # No 'self'
        return {'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes}

    def create_presentation(self, title="My Google Slides Presentation"):
        try:
            body = {'title': title}
            presentation = self.service.presentations().create(body=body).execute()
            presentation_id = presentation.get('presentationId')
            print(f"Created presentation with ID: {presentation_id}")
            return presentation_id
        except HttpError as error:
            print(f"An API error occurred while creating presentation: {error}")
            raise # Re-raise to be caught by the route

    def add_slide_with_content(self, presentation_id, slide_title, body_text, image_url=None):
        requests_batch = []
        page_id = f"slide_{uuid.uuid4().hex}"
        title_shape_id = f"title_{uuid.uuid4().hex}"
        body_shape_id = f"body_{uuid.uuid4().hex}"
        image_element_id = f"image_{uuid.uuid4().hex}"
        emu_per_pt = 12700

        requests_batch.append({
            'createSlide': {
                'objectId': page_id, 'insertionIndex': '1',
                'slideLayoutReference': {'predefinedLayout': 'BLANK'}
            }
        })
        title_pt_x, title_pt_y, title_width_pt, title_height_pt = 50, 50, 600, 50
        requests_batch.extend([
            {'createShape': {'objectId': title_shape_id, 'shapeType': 'TEXT_BOX', 'elementProperties': {'pageObjectId': page_id, 'size': {'width': {'magnitude': title_width_pt * emu_per_pt, 'unit': 'EMU'}, 'height': {'magnitude': title_height_pt * emu_per_pt, 'unit': 'EMU'}}, 'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': title_pt_x * emu_per_pt, 'translateY': title_pt_y * emu_per_pt, 'unit': 'EMU'}}}},
            {'insertText': {'objectId': title_shape_id, 'insertionIndex': 0, 'text': slide_title}},
            {'updateTextStyle': {'objectId': title_shape_id, 'textRange': {'type': 'ALL'}, 'style': {'bold': True, 'fontSize': {'magnitude': 24, 'unit': 'PT'}}, 'fields': 'bold,fontSize'}}
        ])
        body_pt_x, body_pt_y, body_width_pt, body_height_pt = 50, 120, 600, 150
        requests_batch.extend([
            {'createShape': {'objectId': body_shape_id, 'shapeType': 'TEXT_BOX', 'elementProperties': {'pageObjectId': page_id, 'size': {'width': {'magnitude': body_width_pt * emu_per_pt, 'unit': 'EMU'}, 'height': {'magnitude': body_height_pt * emu_per_pt, 'unit': 'EMU'}}, 'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': body_pt_x * emu_per_pt, 'translateY': body_pt_y * emu_per_pt, 'unit': 'EMU'}}}},
            {'insertText': {'objectId': body_shape_id, 'insertionIndex': 0, 'text': body_text}},
            {'updateTextStyle': {'objectId': body_shape_id, 'textRange': {'type': 'ALL'}, 'style': {'fontSize': {'magnitude': 14, 'unit': 'PT'}}, 'fields': 'fontSize'}}
        ])
        if image_url:
            img_pt_x, img_pt_y, img_width_pt, img_height_pt = 50, 300, 400, 225
            requests_batch.append({'createImage': {'objectId': image_element_id, 'url': image_url, 'elementProperties': {'pageObjectId': page_id, 'size': {'width': {'magnitude': img_width_pt * emu_per_pt, 'unit': 'EMU'}, 'height': {'magnitude': img_height_pt * emu_per_pt, 'unit': 'EMU'}}, 'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': img_pt_x * emu_per_pt, 'translateY': img_pt_y * emu_per_pt, 'unit': 'EMU'}}}})
        try:
            body = {'requests': requests_batch}
            response = self.service.presentations().batchUpdate(presentationId=presentation_id, body=body).execute()
            print(f"Added content to slide on presentation {presentation_id}.")
            return response
        except HttpError as error:
            print(f"An API error occurred while adding slide content: {error}")
            raise # Re-raise

    def get_presentation_url(self, presentation_id):
        return f"https://docs.google.com/presentation/d/{presentation_id}/edit"

# --- Flask Routes ---
@app.route('/')
def index():
    if 'credentials' not in flask.session:
        return flask.redirect(flask.url_for('login'))
    
    # User is logged in, show the form
    return flask.render_template('index.html')

@app.route('/login')
def login():
    if not os.path.exists(CLIENT_SECRET_FILE):
        return "Error: client_secret.json not found. Please configure OAuth 2.0.", 500

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline', # Request a refresh token
        include_granted_scopes='true'
    )
    flask.session['state'] = state 
    print(f"DEBUG: In /login, generated state: {state}")
    print(f"DEBUG: Session before redirect in /login: {flask.session}")
    return flask.redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    print(f"DEBUG: In /oauth2callback, request.args: {flask.request.args}")
    print(f"DEBUG: Session at start of /oauth2callback: {flask.session}")
    
    state_from_session = flask.session.get('state')
    state_from_request = flask.request.args.get('state')
    
    print(f"DEBUG: State from session: {state_from_session}")
    print(f"DEBUG: State from request: {state_from_request}")

    # if not state or state != flask.request.args.get('state'): # Old line
    if not state_from_session or state_from_session != state_from_request: # Use the variables
        print("DEBUG: State mismatch detected!")
        return flask.abort(400, 'State mismatch. Possible CSRF attack.')

    if not os.path.exists(CLIENT_SECRET_FILE):
        return "Error: client_secret.json not found.", 500

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        state=state_from_session, # Use the validated state from session
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(authorization_response=flask.request.url)
    
    credentials = flow.credentials
    # flask.session['credentials'] = GoogleSlidesAPI._credentials_to_dict(None, credentials) # Old line
    flask.session['credentials'] = GoogleSlidesAPI._credentials_to_dict(credentials) # Call static method
    return flask.redirect(flask.url_for('index'))

@app.route('/create_presentation', methods=['POST'])
def create_presentation_route():
    if 'credentials' not in flask.session:
        return flask.redirect(flask.url_for('login'))

    try:
        creds_dict = flask.session['credentials']
        credentials = Credentials(**creds_dict)
        slides_api = GoogleSlidesAPI(credentials)

        pres_title = flask.request.form.get('pres_title', 'My Automated Presentation')
        slide_title = flask.request.form.get('slide_title', 'My Slide')
        slide_body = flask.request.form.get('slide_body', 'Some default content.')
        image_url = flask.request.form.get('image_url') # Can be empty

        presentation_id = slides_api.create_presentation(title=pres_title)
        if not presentation_id:
            return flask.render_template('index.html', error="Failed to create presentation shell.")

        slides_api.add_slide_with_content(
            presentation_id,
            slide_title,
            slide_body,
            image_url if image_url else None
        )
        
        presentation_url = slides_api.get_presentation_url(presentation_id)
        return flask.render_template('index.html', success_url=presentation_url, presentation_id=presentation_id)

    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Error during presentation creation: {e}", exc_info=True)
        # Check if it's an auth error that requires re-login
        if "token" in str(e).lower() or "credentials" in str(e).lower():
             flask.session.pop('credentials', None) # Clear broken credentials
             return flask.render_template('index.html', error=f"An error occurred: {e}. Please try logging in again.", needs_login=True)
        return flask.render_template('index.html', error=f"An error occurred: {e}")


@app.route('/logout')
def logout():
    flask.session.pop('credentials', None)
    flask.session.pop('state', None)
    return flask.redirect(flask.url_for('index'))

if __name__ == '__main__':
    # When running locally, disable OAuthlib's HTTPs verification.
    # DO NOT do this in production!
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.run(host='0.0.0.0', debug=True, port=5000)