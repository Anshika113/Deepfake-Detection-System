import sys
import tempfile
from pathlib import Path
import cv2
import yaml
import torch
import streamlit as st
import numpy as np
from PIL import Image
from torchvision import transforms
import importlib.util

# Project setup
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.append(str(PROJECT_ROOT))
IMAGE_CKPT = "models/image_best.pth"
VIDEO_CKPT = "models/best.pth"
SEQ_LEN = 16
FRAME_SIZE = 224

# Load models dynamically
def load_models_module():
    path = PROJECT_ROOT / "src" / "models.py"
    spec = importlib.util.spec_from_file_location("models_dynamic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
models_mod = load_models_module()
DeepfakeDetector = models_mod.DeepfakeDetector

# Face detection (safe)
try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=False, device="cpu")
except Exception:
    _mtcnn = None
def crop_face_safe(pil_img):
    if _mtcnn is None:
        return pil_img
    try:
        boxes, _ = _mtcnn.detect(pil_img)
        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[0])
            return pil_img.crop((x1, y1, x2, y2))
    except Exception:
        pass
    return pil_img

# Image → pseudo video
def image_to_pseudo_video(pil_img, seq_len):
    aug = transforms.Compose([
        transforms.RandomAffine(5, translate=(0.03, 0.03), scale=(0.97, 1.03)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)
    ])
    return [aug(pil_img) for _ in range(seq_len)]

# Tensor helper
def frames_to_tensor(frames):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    x = torch.stack([tfm(f) for f in frames])
    return x.unsqueeze(0)

# Model loader
def load_model(ckpt):
    model = DeepfakeDetector()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    return model

# VIDEO inference (robust)
def predict_video(path, model):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), SEQ_LEN, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = crop_face_safe(pil)
        pil = pil.resize((FRAME_SIZE, FRAME_SIZE))
        frames.append(pil)
    while len(frames) < SEQ_LEN:
        frames.append(frames[-1])
    x = frames_to_tensor(frames)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return prob, frames

# IMAGE inference
def predict_image(path, model):
    pil = Image.open(path).convert("RGB")
    pil = crop_face_safe(pil)
    pil = pil.resize((FRAME_SIZE, FRAME_SIZE))
    frames = image_to_pseudo_video(pil, SEQ_LEN)
    x = frames_to_tensor(frames)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return prob, frames

# UI
st.set_page_config("Deepfake Detector", layout="wide")
st.title("🎬 Deepfake Detector")
uploaded = st.file_uploader(
    "Upload image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mkv"]
)
if st.button("Run prediction"):
    if uploaded is None:
        st.warning("Upload a file first.")
        st.stop()
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    if suffix in [".jpg", ".jpeg", ".png"]:
        model = load_model(IMAGE_CKPT)
        prob, frames = predict_image(tmp_path, model)
        input_type = "IMAGE"
    else:
        model = load_model(VIDEO_CKPT)
        prob, frames = predict_video(tmp_path, model)
        input_type = "VIDEO"
    if prob > 0.55:
        label = "FAKE"
    elif prob < 0.45:
        label = "REAL"
    else:
        label = "UNCERTAIN"
    st.success(f"""
**Input type:** {input_type}  
**Prediction:** {label}  
**Confidence:** {prob:.4f}
""")
    st.image(frames[:8], caption=[f"Frame {i+1}" for i in range(len(frames[:8]))])
