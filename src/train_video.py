import os
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
from sklearn.metrics import roc_auc_score
from dataset import create_dataloaders
from models import DeepfakeDetector

# CONFIG
CONFIG_PATH = "configs/default.yaml"
SAVE_DIR = "models"
SAVE_PATH = os.path.join(SAVE_DIR, "best.pth")
EPOCHS = 5
LR = 1e-4

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    training = cfg.get("training", {})
    dataset = cfg.get("dataset", {})

    return {
        "batch_size": int(training.get("batch_size", 8)),
        "sequence_length": int(dataset.get("sequence_length", 4)),
        "num_workers": int(training.get("num_workers", 4)),
        "aggregator": cfg.get("model", {}).get("aggregator", "lstm"),
    }

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cfg = load_config(CONFIG_PATH)
    print("Training VIDEO model with config:", cfg)

    train_loader, val_loader, _ = create_dataloaders(
        metadata_path="data/processed/metadata.csv",
        batch_size=cfg["batch_size"],
        sequence_length=cfg["sequence_length"],
        num_workers=cfg["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = DeepfakeDetector(aggregator=cfg["aggregator"]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):

        # ================= TRAIN =================
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]"):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device).float().view(-1)

            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATE =================
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [VAL]"):
                frames = batch["frames"].to(device)
                labels = batch["label"].to(device).float().view(-1)

                logits = model(frames)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.0

        print(
            f"\nEpoch {epoch}: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val AUC={val_auc:.4f}"
        )
        
        # ================= SAVE BEST =================
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "aggregator": cfg["aggregator"],
                },
                SAVE_PATH
            )
            print("✅ Saved new BEST model (AUC based)")

    print("\n🎯 Training complete. Best Val AUC:", best_val_auc)

if __name__ == "__main__":
    main()
