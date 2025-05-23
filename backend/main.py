from moviepy.editor import *
from moviepy.video.tools.drawing import color_gradient
from PIL import Image

# === INPUTS ===
image_path = "/home/visharad/Downloads/liana-s-ird6OOE2LXI-unsplash.jpg"
audio_path = "/home/visharad/Downloads/videoplayback.wav"
slide_title = "Greenhouse Gases Explained"
slide_text = "Greenhouse gases trap heat in the atmosphere and keep Earth warm enough to sustain life. However, increased emissions from human activities like burning fossil fuels are enhancing this effect, leading to global warming."

# Monkey patch PIL.Image.ANTIALIAS
Image.ANTIALIAS = Image.LANCZOS

# === SETUP ===
audio = AudioFileClip(audio_path)
duration = audio.duration

# === IMAGE BACKGROUND ===
bg = ImageClip(image_path).set_duration(duration).resize(height=720).set_position("center")

# === TEXT OVERLAY ===
title_txt = TextClip(slide_title, fontsize=50, font='Liberation-Sans-Bold', color='white', bg_color='blue', size=(1280, None)).set_position(("center", 50)).set_duration(duration)
body_txt = TextClip(slide_text, fontsize=32, font='Liberation-Sans', color='white', size=(1000, None), method='caption').set_position(("center", 150)).set_duration(duration)

# === FINAL CLIP ===
final = CompositeVideoClip([bg, title_txt, body_txt]).set_audio(audio)
final.write_videofile("slide1.mp4", fps=24)