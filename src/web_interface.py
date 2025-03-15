import gradio as gr
from src.video_processing import process_video

def video_interface(video):
    vehicle_count = process_video(video)
    return f"Total Vehicles Counted: {vehicle_count}"

demo = gr.Interface(
    fn=video_interface,
    inputs=gr.Video(label="Upload MP4 Video"),
    outputs=gr.Text(label="Total Vehicles Counted"),
    title="Vehicle Detection and Counting",
    description="Upload a video to detect and count vehicles using YOLO and Norfair.",
    allow_flagging="never"
)

def launch_app():
    demo.launch()