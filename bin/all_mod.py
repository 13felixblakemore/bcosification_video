import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
from pytorchvideo.data import make_clip_sampler
from torch import nn
import bcos.settings
from bcos.settings import UCF101_PATH

"""from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample
)"""

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)


class UCF101DataModule(pytorch_lightning.LightningDataModule):

  # Dataset configuration
  _TRAIN_PATH = "ucfTrainTestlist/trainlist01.txt"
  _TEST_PATH = "ucfTrainTestlist/testlist01.txt"
  print(UCF101_PATH)
  _CLIP_DURATION = 2  # Duration of sampled clip for each video
  _BATCH_SIZE = 1
  _NUM_WORKERS = 1  # Number of parallel processes fetching data

  def train_dataloader(self):
    """
    Create the Kinetics train partition from the list of video labels
    in {self._DATA_PATH}/train
    """
    """    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(8),
                        Lambda(lambda x: x / 255.0),
                        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(244),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )"""
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=self._TRAIN_PATH,
        clip_sampler=make_clip_sampler("random", self._CLIP_DURATION),
        video_path_prefix=UCF101_PATH,
        decode_audio=False,
        #transform=train_transform,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )

  def val_dataloader(self):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/val
    """
    """    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(8),
                        Lambda(lambda x: x / 255.0),
                        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ]
                ),
            ),
        ]
    )"""
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=self._TRAIN_PATH,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
        decode_audio=False,
        video_path_prefix=UCF101_PATH,
        #transform=val_transform,
    )
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )


from pytorchvideo.models.hub import i3d_r50


class MyI3D(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.base = i3d_r50(pretrained=True)

        # Replace block 2
        #self.base.blocks[2] = MyCustomBlock()

        in_features = self.base.blocks[-1].proj.in_features

        # Replace classifier
        self.base.blocks[-1].proj = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
  def __init__(self):
      super().__init__()
      self.model = MyI3D()

  def forward(self, x):
      return self.model(x)

  def training_step(self, batch, batch_idx):
      # The model expects a video tensor of shape (B, C, T, H, W), which is the
      # format provided by the dataset
      y_hat = self.model(batch["video"])

      # Compute cross entropy loss, loss.backwards will be called behind the scenes
      # by PyTorchLightning after being returned from this method.
      loss = F.cross_entropy(y_hat, batch["label"])

      # Log the train loss to Tensorboard
      self.log("train_loss", loss, prog_bar=True)

      return loss

  def validation_step(self, batch, batch_idx):
      y_hat = self.model(batch["video"])
      loss = F.cross_entropy(y_hat, batch["label"])
      self.log("val_loss", loss)
      return loss

  def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=1e-1)

def train():
    classification_module = VideoClassificationLightningModule()
    print("Done CM")
    data_module = UCF101DataModule()
    print("Done DM")
    num_gpus = 1
    trainer = pytorch_lightning.Trainer(max_epochs=1)
    print("Done trainer")
    trainer.fit(classification_module, data_module)

if __name__=="__main__":
    train()