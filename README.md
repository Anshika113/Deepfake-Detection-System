# Deepfake Detection System

An end-to-end deepfake detection system supporting both image-level and video-level detection using EfficientNet and temporal modeling.

## Features
- Image deepfake detection using EfficientNet-B0
- Video deepfake detection using frame-level CNN + BiLSTM
- Clean training and evaluation pipeline
- Separate evaluation scripts for image and video
- Reproducible experiments via YAML configs

## Project Structure
Deepfake_detector/
|
├── src/
│ ├── models.py
│ ├── dataset.py
│ ├── train_video.py
│ ├── evaluate_img.py
│ └── evaluate_vid.py
│
├── configs/
│ └── default.yaml
│
├── data/
│ └── processed/
│ └── metadata.csv
│
├── .gitignore
└── README.md
