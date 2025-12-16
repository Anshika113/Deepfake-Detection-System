import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import cv2
from collections import Counter

class DeepfakeDataset(Dataset):
    def __init__(self, metadata_path, sequence_length=16, mode='train',
                 img_size=224, frame_sampling='uniform'):
        self.metadata = pd.read_csv(metadata_path)
        self.sequence_length = sequence_length
        self.mode = mode
        self.img_size = img_size
        self.frame_sampling = frame_sampling
        if 'video_path' not in self.metadata.columns:
            raise ValueError("metadata.csv must contain 'video_path' column")
        if 'video_id' not in self.metadata.columns:
            self.metadata['video_id'] = self.metadata['video_path'].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
       
        self.video_groups = self.metadata.groupby('video_id')
        self.video_ids = list(self.video_groups.groups.keys())

        train_ids, test_ids = train_test_split(
            self.video_ids, test_size=0.2, random_state=42)
        train_ids, val_ids = train_test_split(
            train_ids, test_size=0.25, random_state=42)
        self.split_ids = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        self.filtered_metadata = self.metadata[
            self.metadata['video_id'].isin(self.split_ids[mode])
        ]

        self.train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.split_ids[self.mode])

    def _load_frame(self, frame_path):
        """Load a raw image frame from path."""
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame path not found: {frame_path}")
        
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"cv2 failed to read image: {frame_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            transformed = self.train_transform(image=img)
        else:
            transformed = self.val_transform(image=img)

        return transformed['image']

    def _get_frames_for_video(self, video_id):
        """Return paths + labels for one video."""
        video_frames = self.video_groups.get_group(video_id)

        frame_paths = video_frames['video_path'].tolist()
        labels = video_frames['label'].tolist()

        n = len(frame_paths)
        if n == 0:
            raise ValueError(f"No frames for video_id {video_id}")

        if n < self.sequence_length:
            repeat_times = (self.sequence_length // n) + 1
            frame_paths = (frame_paths * repeat_times)[:self.sequence_length]
            labels = (labels * repeat_times)[:self.sequence_length]
            return frame_paths, labels

        if self.frame_sampling == 'uniform':
            indices = np.linspace(0, n - 1, num=self.sequence_length, dtype=int).tolist()
            frame_paths = [frame_paths[i] for i in indices]
            labels = [labels[i] for i in indices]
        else:
            indices = sorted(random.sample(range(n), self.sequence_length))
            frame_paths = [frame_paths[i] for i in indices]
            labels = [labels[i] for i in indices]

        return frame_paths, labels

    def __getitem__(self, idx):
        video_id = self.split_ids[self.mode][idx]
        frame_paths, labels = self._get_frames_for_video(video_id)

        frames_list = [self._load_frame(fp) for fp in frame_paths]
        frames = torch.stack(frames_list)  

        if isinstance(labels[0], str):
            label_map = {'real': 0, 'fake': 1}
            numeric = [label_map.get(l.lower(), 0) for l in labels]
        else:
            numeric = [int(l) for l in labels]

        majority = Counter(numeric).most_common(1)[0][0]
        label_tensor = torch.tensor(majority, dtype=torch.float32)

        return {
            'frames': frames,
            'label': label_tensor,
            'video_id': video_id
        }

def create_dataloaders(
    metadata_path,
    batch_size=16,
    sequence_length=16,
    frame_size=224,
    frame_sampling='uniform',
    num_workers=4,
    pin_memory=False,
    root="data"
):
    img_size = frame_size

    train_set = DeepfakeDataset(
        metadata_path,
        sequence_length=sequence_length,
        mode='train',
        img_size=img_size,
        frame_sampling=frame_sampling
    )
    val_set = DeepfakeDataset(
        metadata_path,
        sequence_length=sequence_length,
        mode='val',
        img_size=img_size,
        frame_sampling=frame_sampling
    )
    test_set = DeepfakeDataset(
        metadata_path,
        sequence_length=sequence_length,
        mode='test',
        img_size=img_size,
        frame_sampling=frame_sampling
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
