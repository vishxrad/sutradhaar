from google.cloud import texttospeech
import os
from dotenv import load_dotenv

load_dotenv()

def synthesize_text_with_rate(text, speaking_rate="75%", output_filename="output_slower.mp3"):
    """Synthesizes speech from the input text with a specified speaking rate using SSML."""

    client = texttospeech.TextToSpeechClient()

    # Wrap the text in SSML with the <prosody> tag to control the rate
    ssml_text = f'<speak><prosody rate="{speaking_rate}">{text}</prosody></speak>'
    input_text = texttospeech.SynthesisInput(ssml=ssml_text)

    # Select the voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN",
        name="en-IN-Wavenet-F",
    )

    # Select the audio config
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # The response's audio_content is binary.
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content with slower rate written to "{output_filename}"')

if __name__ == "__main__":
    text_to_synthesize = """
Personalized learning is also a key trend.
Educational platforms will increasingly use data
analytics to tailor courses to individual student
needs, helping them learn in styles and paces that
work best for them, making education even more
accessible."""
    synthesize_text_with_rate(text_to_synthesize, speaking_rate="75%")
    synthesize_text_with_rate(text_to_synthesize, speaking_rate="50%", output_filename="output_very_slower.mp3")
    synthesize_text_with_rate(text_to_synthesize, speaking_rate="125%", output_filename="output_faster.mp3")