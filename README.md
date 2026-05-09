# 🌶️ Chili Pepper Phenotyping & Feature Extraction

This repository contains a complete pipeline for applying Machine Learning to chili pepper phenotyping and feature extraction. The project uses Computer Vision and Deep Learning (YOLOv8 for segmentation, YOLO for detection) to automatically analyze images of chili peppers and extract morphological features such as area, width, height, redness, and wrinkle.

## ✨ Features
- **Object Detection**: Detect individual chili peppers and seeds using a deep learning model or threshold-based techniques.
- **Instance Segmentation**: Segment the chili pepper components (seed, outer, placenta, endocarp, chili body) using YOLOv8.
- **Phenotypic Feature Extraction**: Automatically calculate bounding box dimensions, average width/length, area, redness index, and a wrinkle index.
- **Web Interface**: Easy-to-use Gradio web app for interactive extraction.
- **Dockerized**: Fully containerized for easy and consistent deployment.

## 📸 Demo

*(Note: Place your demo images in a `docs/` directory and uncomment the lines below to display them)*

<!--
### Original & Detected Bounding Boxes
!Demo Image 1

### Segmented Phenotypic Components & Feature Extraction
!Demo Image 2
-->

## 🚀 Installation

### 1. Local Setup
Clone this repository, then install the required dependencies:
```shell
git clone https://github.com/yourusername/chili-phenotyping.git
cd chili-phenotyping
pip install -r requirements.txt
```
*Note: You may also need to install system libraries for OpenCV and pyzbar (e.g., `sudo apt-get install libgl1 libglib2.0-0 libzbar0` on Ubuntu).*

### 2. Docker Setup
You can run this project easily using Docker without worrying about local dependencies:
```shell
docker build -t chili-phenotyping-app .
docker run -p 7860:7860 chili-phenotyping-app
```
Access the Gradio web UI at `http://localhost:7860`.

## 💻 Usage

### Gradio Web Interface
Run the web application to use the interactive UI locally:
```shell
python app.py
```

### CLI Pipeline
You can run the feature extraction in batch over a directory of images or a single image using `predictor.py`:
```shell
python predictor.py --input_path /path/to/images --weight_seg ./pretrain/yolov8_segment_v1.pt --save_csv
```

**Key CLI Arguments:**
- `--input_path`: Path to an input image or directory of images.
- `--use_model_det`: Flag to use a deep learning model for detection instead of thresholding.
- `--save_csv`: Flag to save the extracted features to a CSV file (`./runs/predict/output.csv`).
*Use `python predictor.py -h` to see all available command line arguments.*

## 📦 Pretrained Weights
Ensure you have the pretrained weights placed inside the `./pretrain/` folder.
- **YOLOv8 Segmentation Weights**: Download link
