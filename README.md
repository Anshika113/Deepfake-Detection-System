# Deepfake Detection System

An end-to-end deepfake detection system supporting **image-level** and **video-level** detection using **EfficientNet-B0** and **temporal modeling (Bi-LSTM)**.

This project is designed with a **clean research-to-production pipeline**, separate training and evaluation flows, and reproducible configuration management.

---

## Key Features

* Image deepfake detection using **EfficientNet-B0**
* Video deepfake detection using **frame-level CNN + Bi-LSTM**
* Modular dataset pipeline with video-wise splitting
* Separate training and evaluation scripts
* YAML-based configuration for reproducibility
* Clean GitHub-ready project structure

---

## Project Structure

```text
Deepfake_detector/
├── src/
│   ├── models.py            # Image & video deepfake models
│   ├── dataset.py           # Dataset and dataloaders
│   ├── train_video.py       # Video model training script
│   ├── evaluate_img.py      # Image model evaluation
│   └── evaluate_vid.py      # Video model evaluation
│
├── configs/
│   └── default.yaml         # Training & dataset configuration
│
├── data/
│   └── processed/
│       └── metadata.csv     # Frame-level metadata
│
├── models/                  # Saved checkpoints (gitignored)
├── tools/                   # Utility scripts
├── app/                     # (Optional) App / inference code
│
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── environment.yml
└── README.md
```

---

## Model Architecture

### Image Model

(Image-only model is a baseline and underperforms compared to video model.)
* **Backbone:** EfficientNet-B0
* **Input:** Single RGB image `(3 × 224 × 224)`
* **Output:** Binary deepfake probability

### Video Model

* **Backbone:** EfficientNet-B0 (frame-level)
* **Temporal Head:** Bi-LSTM
* **Input:** Sequence of frames `(T × 3 × 224 × 224)`
* **Output:** Video-level deepfake prediction

---

## Training

Train the **video deepfake detector**:

```bash
python src/train_video.py
```

* Best model is saved automatically to:

  ```
  models/best.pth
  ```

---

## Evaluation

### Image Model Evaluation

```bash
python src/evaluate_img.py
```

### Video Model Evaluation

```bash
python src/evaluate_vid.py
```

Each evaluation reports:

* Accuracy
* F1-score
* ROC-AUC
* Confusion Matrix
* Classification Report

---

## Configuration

All major parameters are controlled via:

```yaml
configs/default.yaml
```

Includes:

* Batch size
* Sequence length
* Number of workers
* Temporal aggregation method

---

## Environment Setup

Using Conda:

```bash
conda env create -f environment.yml
conda activate deepfake
```

Or Docker:

```bash
docker-compose up --build
```

---

## Notes

* Trained models and logs are **excluded from GitHub** via `.gitignore`
* Repository is structured for **research, reproducibility, and scalability**
* Supports easy extension to other backbones or temporal models

---

## Future Improvements

* Attention-based temporal aggregation
* Audio-visual deepfake detection
* Real-time inference pipeline
* Model explainability (Grad-CAM)

---

## Author

**Anshika Mishra**
B.Tech Computer Science & Engineering
Focus: AI, ML, Cloud and Deep Learning
