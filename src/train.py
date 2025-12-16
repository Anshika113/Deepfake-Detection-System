import os
import sys
import yaml
import torch
import argparse
import numpy as np
from datetime import datetime
from torch import optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from dataset import create_dataloaders
from models import DeepfakeDetector, CNN3D

# CONFIG
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    config = {}

    # DATASET (FF++)
    dataset = cfg.get("dataset", {})
    config["metadata_path"] = dataset.get(
        "metadata_path", "data/processed/metadata.csv"
    )
    config["sequence_length"] = dataset.get("sequence_length", 16)
    config["frame_size"] = dataset.get("frame_size", 224)
    
    # MODEL
    config["model_type"] = cfg.get("model_type", "2d+lstm")
    model_cfg = cfg.get("model", {})
    config["backbone"] = model_cfg.get("backbone", "efficientnet-b0")
    config["aggregator"] = model_cfg.get("aggregator", "lstm")

    # TRAINING (FORCED SAFE VALUES)
    config["epochs"] = 3
    config["batch_size"] = 4
    config["lr"] = 1e-4
    config["num_workers"] = 0
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # LOGGING
    save_root = "models"
    os.makedirs(save_root, exist_ok=True)
    exp = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    config["model_dir"] = os.path.join(save_root, exp)
    os.makedirs(config["model_dir"], exist_ok=True)

    config["log_dir"] = os.path.join(config["model_dir"], "logs")
    os.makedirs(config["log_dir"], exist_ok=True)

    return config

# TRAINER
class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config["device"])
        print("Using device:", self.device)

        self.writer = SummaryWriter(config["log_dir"])

        # MODEL
        if config["model_type"] == "2d+lstm":
            self.model = DeepfakeDetector(
                backbone=config["backbone"],
                aggregator=config["aggregator"]
            ).to(self.device)
        else:
            self.model = CNN3D().to(self.device)

        # FORCE FREEZE BACKBONE (CRITICAL)
        if hasattr(self.model, "frame_backbone"):
            for p in self.model.frame_backbone.parameters():
                p.requires_grad = False
            print("Backbone frozen: frame_backbone")
        else:
            for name, module in self.model.named_children():
                if "backbone" in name:
                    for p in module.parameters():
                        p.requires_grad = False
                    print(f"Backbone frozen: {name}")

        # LOSS
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # OPTIMIZER (HEAD ONLY)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=config["lr"])

        # SCHEDULER
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=1
        )

        # DATA
        self.train_loader, self.val_loader, _ = create_dataloaders(
            config["metadata_path"],
            batch_size=config["batch_size"],
            sequence_length=config["sequence_length"],
            frame_sampling="uniform",
            num_workers=config["num_workers"],
            pin_memory=False,
        )

        print("Train batches:", len(self.train_loader))
        print("Val batches:", len(self.val_loader))

    def train(self):
        best_auc = 0.0

        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                frames = batch["frames"].to(self.device)
                labels = batch["label"].float().to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(frames).squeeze()
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(self.train_loader)

            # VALIDATION
            self.model.eval()
            probs_all, labels_all = [], []

            with torch.no_grad():
                for batch in self.val_loader:
                    frames = batch["frames"].to(self.device)
                    labels = batch["label"].float().to(self.device)

                    logits = self.model(frames).squeeze()
                    probs = torch.sigmoid(logits)

                    probs_all.extend(probs.cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())

            try:
                val_auc = roc_auc_score(labels_all, probs_all)
            except Exception:
                val_auc = 0.0

            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("AUC/val", val_auc, epoch)

            self.scheduler.step(val_auc)

            # SAVE BEST
            if val_auc > best_auc:
                best_auc = val_auc
                save_path = os.path.join(self.cfg["model_dir"], "best.pth")
                torch.save(
                    {"model_state_dict": self.model.state_dict()},
                    save_path,
                )
                print("Saved new best model:", save_path)

        self.writer.close()
        
# MAIN
def main():
    config = load_config()
    print("Metadata:", config["metadata_path"])
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
