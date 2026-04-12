import gradio as gr
import pandas as pd
import torch
import os

# Import your existing ML models and processing classes
from src.detect import DetectModel
from src.segment import SegmentModel
from src.object import Fruits

# Configuration
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WEIGHT_MODEL_SEGMENT = './pretrain/yolov8_segment_v1.pt'

# Initialize models globally so they don't reload on every request
print("Loading Detect and Segment models...")
model_detect = DetectModel(model_name='none') # Using threshold for detection as per default in predictor.py
model_segment = SegmentModel(weight_path=WEIGHT_MODEL_SEGMENT, model_name='yolo', device=DEVICE)
print("Models loaded successfully!")

def extract_chili_features(image_path):
    """
    Passes the input image to the chili object detection and segmentation pipeline.
    """
    if image_path is None or not os.path.exists(image_path):
        return pd.DataFrame()
        
    # Run the image through your pipeline
    chilis = Fruits(image_path, model_detect=model_detect, model_segment=model_segment)
    
    rows = []
    for chili in chilis.items:
        row = {
            'ID': chili.id,
            'Name': chili.name,
            'Width BBox': chili.width_bbox,
            'Height BBox': chili.height_bbox,
            'Average Width': chili.width_fruit,
            'Area': chili.area_fruit,
            'Redness': chili.redness,
            'Wrinkle': chili.wrinkle
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Create the Gradio interface
    demo = gr.Interface(
        fn=extract_chili_features,
        inputs=gr.Image(type="filepath", label="Upload Chili Image"),
        outputs=gr.Dataframe(label="Extracted Chili Features", interactive=False),
        title="🌶️ Real Chili Feature Extractor",
        description="Upload an image of a chili pepper. The model will extract and display phenotyping features like area, redness, and wrinkle."
    )
    
    # Launch the application
    demo.launch()