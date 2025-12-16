import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
import yaml
from dataset import create_dataloaders
from models import DeepfakeDetector

def load_config(path="configs/default.yaml"):
    cfg = yaml.safe_load(open(path))
    t = cfg.get("training", {})
    d = cfg.get("dataset", {})
    dl_kwargs = {
        "batch_size": int(t.get("batch_size", 4)),
        "sequence_length": int(d.get("sequence_length", 8)),
    }
    if "num_workers" in t:
        dl_kwargs["num_workers"] = int(t.get("num_workers", 0))
    return cfg, dl_kwargs

def find_latest_checkpoint(models_root="models"):
    """
    Search recursively for .pth files under models_root and return the newest one by mtime.
    If none found, return None.
    """
    best = None
    best_mtime = 0
    if not os.path.exists(models_root):
        return None
    for root, _, files in os.walk(models_root):
        for f in files:
            if f.endswith(".pth"):
                path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    mtime = 0
                if mtime > best_mtime:
                    best_mtime = mtime
                    best = path
    return best

def to_iterable_probs(probs):
    """
    Ensure posterior probabilities are returned as a 1D Python list,
    whether input is torch.Tensor, numpy array, scalar, or float.
    """
    if isinstance(probs, torch.Tensor):
        arr = probs.detach().cpu().numpy()
    elif isinstance(probs, np.ndarray):
        arr = probs
    else:
        try:
            arr = np.array(probs)
        except Exception:
            arr = np.array([float(probs)])
    arr = np.atleast_1d(arr).astype(float)
    return arr.tolist()

def main():
    cfg, dl_kwargs = load_config()
    metadata_path = cfg.get("dataset", {}).get("metadata_path", "data/processed/metadata.csv")

    print("Creating dataloaders with:", dl_kwargs)
    train_loader, val_loader, test_loader = create_dataloaders(metadata_path, **dl_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = DeepfakeDetector(
        backbone=cfg.get("model", {}).get("backbone", "efficientnet-b0"),
        aggregator=cfg.get("model", {}).get("aggregator", "lstm")
    )

    ckpt = find_latest_checkpoint(cfg.get("logging", {}).get("save_dir", "models"))
    if ckpt is None:
        print("No checkpoint (.pth) found under 'models/' — evaluation will use random weights.")
    else:
        print("Loading checkpoint:", ckpt)
        try:
            loaded = torch.load(ckpt, map_location=device)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                model.load_state_dict(loaded["model_state_dict"])
            elif isinstance(loaded, dict) and "state_dict" in loaded:
                model.load_state_dict(loaded["state_dict"])
            else:
                model.load_state_dict(loaded)
            print("✔ Loaded weights successfully.")
        except Exception as e:
            print("Failed to load checkpoint:", e)
            print("Proceeding with random weights.")

    model.to(device).eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            frames = batch["frames"].to(device)  
            labels = batch["label"].cpu().numpy()
            out = model(frames)
            try:
                probs = torch.sigmoid(out)
            except Exception:
                probs = out

            prob_list = to_iterable_probs(probs)
            if len(prob_list) == len(labels):
                all_probs.extend(prob_list)
            elif len(prob_list) == 1 and len(labels) >= 1:
                all_probs.extend([prob_list[0]] * len(labels))
            else:
                minlen = min(len(prob_list), len(labels))
                all_probs.extend(prob_list[:minlen])
                all_labels = all_labels + labels[:minlen].tolist() if len(all_labels) else labels[:minlen].tolist()
                continue

            all_labels.extend(labels.tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)
    preds = (all_probs > 0.5).astype(int)

    if len(np.unique(all_labels)) < 2:
        print("Not enough label variety in test set to compute AUC/F1 reliably.")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    try:
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        cm = confusion_matrix(all_labels, preds)
    except Exception:
        acc = f1 = float("nan")
        cm = None

    print("\n----- TEST RESULTS -----")
    print(f"Examples evaluated: {len(all_labels)}")
    print(f"AUC: {auc}")
    print(f"Accuracy: {acc}")
    print(f"F1: {f1}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(all_labels, preds, digits=4))

if __name__ == "__main__":
    main()
