import torch
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.dataframe.dataframe_wrapper import iterate
from torchvision.datasets import UCF101

from bcos import settings
from bcos.data.datamodules import VideoOnlyDataset
from bcos.data.presets import UCF101ClassificationPresetEval, UCF101ClassificationPresetTrain
from bcos.models.standard_models import I3DBcos

train_md = torch.load("ucf101_train_metadata.pt")

train_transform = UCF101ClassificationPresetTrain(
            crop_size=224,
            is_bcos=True,
)
eval_transform = UCF101ClassificationPresetEval(
            crop_size=224,
            is_bcos=True,
)


train_dataset = UCF101(
    root=settings.UCF101_PATH,
    annotation_path="ucfTrainTestlist",
    frames_per_clip=8,
    fold=2,
    transform=train_transform,
    step_between_clips=16,
    train=True,
    _precomputed_metadata=train_md,
)

train_dataset = VideoOnlyDataset(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)

# ====== INSPECT VIDEOS & LABELS ======
"""for i, (video, label) in enumerate(train_loader):
    # video: [B, C, T, H, W]
    print("Batch", i)
    print("Labels:", label)
    print("Video:", video.shape)

    if i >= 10:  # just first 10 batches
        break"""

videos, labels = next(iter(train_loader))
videos, labels = videos.cuda(), labels.cuda()

model = I3DBcos
epochs = 100
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
model.cuda()


for i in range(epochs):
    optimizer.zero_grad()
    outputs = model(videos)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"Step {i}: loss = {loss.item()}")