import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import gradio as gr
import yt_dlp
import whisper
from transformers import pipeline

MODEL_NAME = "google/flan-t5-small"
qg_pipeline = pipeline("text2text-generation", model=MODEL_NAME)
whisper_model = whisper.load_model("small")


def download_video(url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': tempfile.gettempdir() + '/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def convert_to_wav(video_path):
    audio_path = video_path.replace('.mp4', '.wav')
    subprocess.run(["ffmpeg", "-i", video_path, audio_path])
    return audio_path


def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']


def generate_questions(transcript):
    questions = qg_pipeline(transcript, max_length=50, num_return_sequences=5)
    return [q['generated_text'] for q in questions]


def process(url):
    download_video(url)
    video_file = tempfile.gettempdir() + '/video.mp4'
    audio_file = convert_to_wav(video_file)
    transcript = transcribe_audio(audio_file)
    questions = generate_questions(transcript)
    return transcript, questions


demo = gr.Interface(
    fn=process,
    inputs=gr.inputs.Textbox(label="Video URL"),
    outputs=["text", "text"],
    title="Video Processing App",
    description="Process video for audio conversion, transcription, and question generation.",
    css=".gradio-container {background: linear-gradient(#7c3aed, #06b6d4); color: white;}"
)

demo.launch()