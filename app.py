import gradio as gr

def process_video(video):
    # Implement your video processing logic here
    return "Video processed successfully!"

iface = gr.Interface(fn=process_video, inputs=gr.Video(label="Upload Video"), outputs="text")
iface.launch()