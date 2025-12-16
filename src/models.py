import torch
import torch.nn as nn

# IMAGE MODEL
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ImageEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(-1)

# VIDEO MODEL
from efficientnet_pytorch import EfficientNet


class FrameBackbone(nn.Module):
    def __init__(self, freeze_layers=5):
        super().__init__()

        self.model = EfficientNet.from_pretrained("efficientnet-b0")

        in_features = self.model._fc.in_features
        self.model._fc = nn.Identity()

        for i, (_, param) in enumerate(self.model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False

        self.out_features = in_features

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t = x.shape[:2]
        x = x.view(b * t, *x.shape[2:])
        feats = self.model(x)
        return feats.view(b, t, -1)


class TemporalAggregator(nn.Module):
    def __init__(self, input_dim, mode="lstm", dropout=0.4):
        super().__init__()
        self.mode = mode

        if mode == "lstm":
            self.temporal = nn.LSTM(
                input_size=input_dim,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            self.fc = nn.Linear(512, 1)

        elif mode == "conv1d":
            self.temporal = nn.Sequential(
                nn.Conv1d(input_dim, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.fc = nn.Linear(256, 1)

        else:
            raise ValueError("Invalid temporal mode")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.mode == "lstm":
            out, _ = self.temporal(x)
            out = out[:, -1, :]
        else:
            x = x.transpose(1, 2)
            out = self.temporal(x).squeeze(-1)

        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


class DeepfakeDetector(nn.Module):
    def __init__(self, aggregator="lstm"):
        super().__init__()
        self.frame_backbone = FrameBackbone()
        self.temporal_agg = TemporalAggregator(
            self.frame_backbone.out_features,
            mode=aggregator
        )

    def forward(self, x):
        feats = self.frame_backbone(x)
        return self.temporal_agg(feats)
