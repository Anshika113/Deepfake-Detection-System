import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

def crop_face_pil(image: Image.Image):
    img = np.array(image)
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return image 
    x1, y1, x2, y2 = boxes[0].astype(int)
    face = img[y1:y2, x1:x2]
    return Image.fromarray(face)
