import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import yaml

from dataset import create_dataloaders
from models import DeepfakeDetector

CHECKPOINT_PATH = "models/best.pth"


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    training = cfg.get("training", {})
    dataset = cfg.get("dataset", {})

    dl_kwargs = {
        "batch_size": int(training.get("batch_size", 8)),
        "sequence_length": int(dataset.get("sequence_length", 16)),
        "num_workers": int(training.get("num_workers", 4)),
    }

    return cfg, dl_kwargs


def main():
    cfg, dl_kwargs = load_config()
    metadata_path = cfg["dataset"]["metadata_path"]

    print("Evaluation mode: VIDEO")
    print("Creating dataloaders with:", dl_kwargs)

    _, _, test_loader = create_dataloaders(
        metadata_path=metadata_path,
        **dl_kwargs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # MODEL
    model = DeepfakeDetector(
        aggregator=cfg.get("model", {}).get("aggregator", "lstm")
    ).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print("Loading checkpoint:", CHECKPOINT_PATH)
    state = torch.load(CHECKPOINT_PATH, map_location=device)

    # 🔥 THE ACTUAL FIX
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    print("✅ Model loaded cleanly with strict=True")
    
    model.eval()


    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating VIDEO"):
            frames = batch["frames"].to(device)   # (B, T, C, H, W)
            labels = batch["label"].to(device)

            logits = model(frames)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)
    preds = (all_probs >= 0.5).astype(int)

    print("\n----- VIDEO TEST RESULTS -----")
    print(f"Samples evaluated: {len(all_labels)}")

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC: {auc:.4f}")
    else:
        print("AUC: not computable")

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    cm = confusion_matrix(all_labels, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    print("\nClassification Report:\n")
    print(classification_report(all_labels, preds, digits=4))


if __name__ == "__main__":
    main()
