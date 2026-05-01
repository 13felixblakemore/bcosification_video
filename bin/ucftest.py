import os

import torch
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Lambda, Normalize
from torch.utils.data import DataLoader
from pathlib import Path

UCF101_PATH = os.getenv("UCF101_PATH")  # root folder containing 'UCF-101'
ANNOTATION_PATH = "ucfTrainTestlist"  # folder with train/test split files
FRAMES_PER_CLIP = 8
STEP_BETWEEN_CLIPS = 8
BATCH_SIZE = 2

transform = Compose([

    Lambda(lambda x: x.permute(3, 0, 1, 2).float()),
    Lambda(lambda x: x / 255.0),
    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
])

train_md = torch.load("ucf101_train_metadata.pt")
val_md = torch.load("ucf101_eval_metadata.pt")

train_dataset = UCF101(
    root=UCF101_PATH,
    annotation_path=ANNOTATION_PATH,
    frames_per_clip=FRAMES_PER_CLIP,
    step_between_clips=STEP_BETWEEN_CLIPS,
    train=True,

    #transform=transform,
    _precomputed_metadata=train_md
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

for i, (video, audio, label) in enumerate(train_loader):
    print("Batch", i)
    print("Labels:", label)

    if i >= 10:
        break