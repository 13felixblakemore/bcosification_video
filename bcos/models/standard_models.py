from torch import nn
from torchvision.models import ResNet
from torchvision.models import DenseNet
from pytorchvideo.models.hub import i3d_r50
import torch
import torch.nn.functional as F

## START: ------------------- For standard models -------------------------------
class MyResNet(ResNet):
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class MyDenseNet(DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
## END: ------------------- For standard models -------------------------------

## ------------------- For Bcos models -------------------------------
class ResNetBcos(ResNet):
    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        # x = self.logit_layer(x) # Defined in BcosifyNetwork class

        return x

class DenseNetBcos(DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.classifier(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


class I3D(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = i3d_r50(pretrained=pretrained)
        self.blocks = self.model.blocks
        self.head = getattr(self.model, "head", None)

    def forward(self, x):
        # x: (B, C, T, H, W)
        out = self.model(x)
        return out


class I3DBcos(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = i3d_r50(pretrained=pretrained)
        self.blocks = self.model.blocks
        self.head = getattr(self.model, "head", None)

    def forward(self, x):
        # x: (B, C, T, H, W)
        if x.shape[1] != self.model.blocks[0].conv.weight.shape[1]:
            # permute channels from last dim to dim=1
            # assuming input shape is (B, T, H, W, C)
            x = x.permute(0, 4, 1, 2, 3)
        out = self.model(x)
        return out
