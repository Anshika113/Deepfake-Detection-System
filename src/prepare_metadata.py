import os
import re
import pandas as pd
ROOT = "data/kaggle"
OUTPUT_DIR = "data/processed"
OUT_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
REAL_KEYWORDS = {"real", "reals", "training_real", "real_images", "real_and_fake_face"}
FAKE_KEYWORDS = {"fake", "fakes", "training_fake", "fake_images"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
rows = []
def infer_label_from_path(path_components):
    label = None
    for comp in reversed(path_components):
        comp_clean = re.sub(r'[^a-z0-9_]', '_', comp) 
        if any(k in comp_clean for k in FAKE_KEYWORDS):
            return 1
        if any(k in comp_clean for k in REAL_KEYWORDS):
            return 0
    return None
def is_image_file(fname):
    return os.path.splitext(fname)[1].lower() in IMG_EXTS
for dirpath, dirnames, filenames in os.walk(ROOT):
    for fname in filenames:
        if not is_image_file(fname):
            continue
        full_path = os.path.join(dirpath, fname)
        rel = os.path.relpath(full_path, ROOT)
        parts = rel.split(os.sep)[:-1]  
        label = infer_label_from_path([p.lower() for p in parts])
        if label is None:
            parent = os.path.basename(os.path.dirname(full_path)).lower()
            label = infer_label_from_path([parent])
        if label is None:
            continue
        rel_from_root = os.path.join("data", "kaggle", rel).replace("\\", "/")
        video_id = os.path.splitext(os.path.basename(full_path))[0]
        rows.append({"frame_path": rel_from_root, "video_id": video_id, "label": label})
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.DataFrame(rows, columns=["frame_path", "video_id", "label"])
df.to_csv(OUT_FILE, index=False)
print("Wrote metadata:", OUT_FILE)
print("Total samples:", len(df))
if len(df) == 0:
    print("\nNo samples found. Directory listing of data/kaggle:")
    for p in sorted(os.listdir(ROOT)):
        print(" -", p)
    print("\nRun these for more detail:")
    print("  dir data\\kaggle /b")
    print("  dir data\\kaggle\\<folder> /s /b | findstr /i .jpg")
else:
    print("\nLabel counts:")
    print(df['label'].value_counts().to_string())
    print("\nSample rows:")
    print(df.head().to_string(index=False))
