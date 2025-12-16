import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
class FrameBackbone(nn.Module):
    def __init__(self, backbone='efficientnet-b0', freeze_layers=5):
        super().__init__()
        if backbone != 'efficientnet-b0':
            raise ValueError("Only 'efficientnet-b0' supported in this simplified version.")
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.model._fc.in_features
        self.model._fc = nn.Identity() 
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False
        self.out_features = in_features
    def forward(self, x):
        batch_size, timesteps = x.shape[:2]
        x = x.view(batch_size * timesteps, *x.shape[2:])
        features = self.model(x)
        features = features.view(batch_size, timesteps, -1)
        return features

class TemporalAggregator(nn.Module):
    def __init__(self, input_dim, aggregator='lstm', dropout_p=0.4):
        super().__init__()
        self.aggregator = aggregator
        self.dropout_p = dropout_p
        if aggregator == 'lstm':
            self.temporal = nn.LSTM(
                input_size=input_dim,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
                batch_first=True)
            self.dropout = nn.Dropout(p=self.dropout_p)
            self.fc = nn.Linear(512, 1) 
        elif aggregator == 'conv1d':
            self.temporal = nn.Sequential(
                nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.dropout = nn.Dropout(p=self.dropout_p)
            self.fc = nn.Linear(256, 1)
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

    def forward(self, x):
        if self.aggregator == 'lstm':
            out, _ = self.temporal(x)  
            out = out[:, -1, :] 
            out = self.dropout(out)
        else:
            x = x.transpose(1, 2)
            out = self.temporal(x).squeeze(-1)
            out = self.dropout(out)

        return self.fc(out).squeeze(-1) 

class DeepfakeDetector(nn.Module):
    def __init__(self, backbone='efficientnet-b0', aggregator='lstm'):
        super().__init__()
        self.frame_backbone = FrameBackbone(backbone)
        self.temporal_agg = TemporalAggregator(
            self.frame_backbone.out_features, aggregator)

    def forward(self, x):
        frame_features = self.frame_backbone(x)
        video_logits = self.temporal_agg(frame_features)
        return video_logits  

class CNN3D(nn.Module):
    """3D CNN alternative model (returns raw logits)"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x).squeeze(-1) 
