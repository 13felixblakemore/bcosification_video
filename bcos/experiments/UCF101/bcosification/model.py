import sys

from torch import nn
from torchvision.models.densenet import DenseNet121_Weights, _load_state_dict
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
    ResNet18_Weights,
    ResNet50_Weights,
)
from pytorchvideo.models.hub import i3d_r50

from bcos.models.standard_models import DenseNetBcos, ResNetBcos, I3DBcos
from bcosify3D import BcosifyNetwork

__all__ = ["get_model"]

def get_torch_model_modified(arch_name: str, model_config):
    if arch_name=='resnet18':
        tv_model = ResNetBcos(BasicBlock, [2, 2, 2, 2])
        weight_type = model_config["weights"]
        if weight_type:
            weights = ResNet18_Weights.verify(model_config["weights"])
            tv_model.load_state_dict(weights.get_state_dict(progress=False))
        return tv_model
    if arch_name=='resnet50':
        tv_model = ResNetBcos(Bottleneck, [3, 4, 6, 3])
        weight_type = model_config["weights"]
        if weight_type:
            weights = ResNet50_Weights.verify(model_config["weights"])
            tv_model.load_state_dict(weights.get_state_dict(progress=False))
        return tv_model
    if arch_name=='densenet121':
        tv_model = DenseNetBcos(32, (6, 12, 24, 16), 64)
        weight_type = model_config["weights"]
        if weight_type:
            weights = DenseNet121_Weights.verify(model_config["weights"])
            _load_state_dict(model=tv_model, weights=weights, progress=False)
        return tv_model
    if arch_name == "i3d":
        model = I3DBcos(pretrained=True)
        return model

def get_model(model_config) -> nn.Module:
    # extract args
    arch_name = model_config["name"]

    model = BcosifyNetwork(get_torch_model_modified(arch_name, model_config), model_config, add_channels=True, logit_layer=True) 

    # For standard changes
    standard_changes = model_config.get("standard_changes", None)

    replace = False
    if replace:
        # 1) Replace model.model.blocks.0.pool
        old_pool = model.model.blocks[0].pool
        model.model.blocks[0].pool = nn.AvgPool3d(
            kernel_size=old_pool.kernel_size,
            stride=old_pool.stride,
            padding=old_pool.padding,
        )

        # 2) Replace model.model.blocks.2
        old_pool = model.model.blocks[2]
        model.model.blocks[2] = nn.AvgPool3d(
            kernel_size=old_pool.kernel_size,
            stride=old_pool.stride,
            padding=old_pool.padding,
        )

        for name, module in model.named_modules():
            if isinstance(module, (nn.MaxPool2d, nn.MaxPool3d)):
                print("FOUND MAXPOOL:", name, module)

    # Making all the bias parameters None
    #print("keeping bias")
    print("Removing bias parameters (making None)")
    for mod in model.modules():
        print(mod)
        if hasattr(mod, "bias") and mod.bias is not None:
          mod.bias = None

    return model