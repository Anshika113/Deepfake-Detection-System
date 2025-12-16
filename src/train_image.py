import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn, optim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
train_ds = ImageFolder("data/train", transform=train_tfms)
val_ds   = ImageFolder("data/val", transform=val_tfms)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=32)

model = torchvision.models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    for x,y in train_dl:
        x = x.to(DEVICE)
        y = y.float().to(DEVICE)

        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/image_best.pth")
print("✅ IMAGE MODEL SAVED: models/image_best.pth")
