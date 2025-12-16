import os
import shutil
from sklearn.model_selection import train_test_split

# CONFIG
SRC = "data/kaggle/real_and_fake_face"
OUT = "data/image_dataset"
SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png")

SPLITS = ["train", "val", "test"]
CLASSES = ["real", "fake"]

# UTILS
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    
def list_images(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ]

# CREATE OUTPUT STRUCTURE
for split in SPLITS:
    for cls in CLASSES:
        ensure_dir(os.path.join(OUT, split, cls))

# LOAD IMAGES
real_dir = os.path.join(SRC, "training_real")
fake_dir = os.path.join(SRC, "training_fake")

real_imgs = list_images(real_dir)
fake_imgs = list_images(fake_dir)

assert len(real_imgs) > 0, "❌ No REAL images found"
assert len(fake_imgs) > 0, "❌ No FAKE images found"

# SPLIT (70 / 15 / 15)
train_real, temp_real = train_test_split(
    real_imgs, test_size=0.3, random_state=SEED
)
val_real, test_real = train_test_split(
    temp_real, test_size=0.5, random_state=SEED
)

train_fake, temp_fake = train_test_split(
    fake_imgs, test_size=0.3, random_state=SEED
)
val_fake, test_fake = train_test_split(
    temp_fake, test_size=0.5, random_state=SEED
)

splits_map = {
    "train": {"real": train_real, "fake": train_fake},
    "val": {"real": val_real, "fake": val_fake},
    "test": {"real": test_real, "fake": test_fake},
}

# COPY FILES (SAFE COPY)
print("📂 Organizing IMAGE dataset...")

for split, cls_data in splits_map.items():
    for cls, files in cls_data.items():
        out_dir = os.path.join(OUT, split, cls)
        for src_path in files:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(out_dir, fname)
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(fname)
                dst_path = os.path.join(out_dir, f"{base}_{SEED}{ext}")

            shutil.copy2(src_path, dst_path)

print("✅ Image dataset prepared successfully!")
