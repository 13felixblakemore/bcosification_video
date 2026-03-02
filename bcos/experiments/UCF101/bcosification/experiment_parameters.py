import math  # noqa

import torch.nn
from torch import nn
from torch.nn import CrossEntropyLoss

from bcos.data.presets import (
    ImageNetClassificationPresetEval,
    ImageNetClassificationPresetTrain, UCF101ClassificationPresetTrain, UCF101ClassificationPresetEval,
)
from bcos.experiments.utils import (
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
)
from bcos.modules import norms
from bcos.modules.losses import (
    BinaryCrossEntropyLoss,
    UniformOffLabelsBCEWithLogitsLoss,
)
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 101
NUM_TRAIN_EXAMPLES: int = 50000 # not sure
NUM_EVAL_EXAMPLES: int = 3925 # not sure

DEFAULT_BATCH_SIZE = 8  # could be causing noisy gradients?
DEFAULT_NUM_EPOCHS = 30
DEFAULT_LR =1e-4
DEFAULT_CROP_SIZE = 224

DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered3d)  # bnu-linear
DEFAULT_OPTIMIZER = OptimizerFactory(name="Adam", lr=DEFAULT_LR, bcosify=True, b_opt = False)
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
)

DEFAULTS = dict(
    data=dict(
        train_transform=UCF101ClassificationPresetTrain(
            crop_size=DEFAULT_CROP_SIZE,
            is_bcos=True,
        ),
        test_transform=UCF101ClassificationPresetEval(
            crop_size=DEFAULT_CROP_SIZE,
            is_bcos=True,
        ),
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=4,
        num_classes=NUM_CLASSES,
    ),
    model=dict(
        is_bcos=True,
        args=dict(
            num_classes=NUM_CLASSES,
            norm_layer=DEFAULT_NORM_LAYER,
            logit_bias=-math.log(NUM_CLASSES - 1),
        ),
        bcos_args=dict(
            b=2,
            max_out=1,
        ),
    ),
    criterion=CrossEntropyLoss(),
    test_criterion=CrossEntropyLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
    use_agc=True, 
)

# helper
def update_default(new_config, **kwargs):
    # kwargs is a dict
    return update_config(DEFAULTS, new_config)

RESNET_DEPTHS = [18, 50]
resnets = {
    f"resnet_{depth}"+\
    f"{underscore+weight if weight=='V1' else ''}": update_default(
        dict(
            model=dict(
                name=f"resnet{depth}",
                last_layer_name = "fc", # For replacing the last fc layer with conv1x1
                weights=f"ResNet{depth}_Weights.DEFAULT" if weight == "V2" else f"IMAGENET1K_V1" if weight=='V1' and depth==50 else None,
                bcosify_args = dict(
                    fix_b = True, # Fixed b value (=2)
                    use_bias = False, # No bias
                    norm_layer = "BnUncV2", # Modified Batch Norm
                    manual_optim=False, # For manual optimization of b values
                    gap = True, # Global Average Pooling reorder works with conv1x1 for the last linear layer
                    act_layer = True, # ReLU activation layer
                ),
                standard_changes = {"maxpool": nn.AvgPool2d(kernel_size=3, stride=2, padding=1)},
            ),
        )
    )
    for depth in RESNET_DEPTHS
    for weight in ["V2", "V1"]
    for underscore in ["_"]
}

DENSENET_DEPTHS = [121]
densenets = {
    f"densenet_{depth}": update_default(
        dict(
            model=dict(
                name=f"densenet{depth}",
                last_layer_name = "classifier", # For replacing the last fc layer with conv1x1
                weights=f"DenseNet{depth}_Weights.DEFAULT",
                bcosify_args = dict(
                    fix_b = True, # Fixed b value (=2)
                    use_bias = False, # No bias
                    norm_layer = "BnUncV2", # Modified Batch Norm
                    manual_optim=False, # For manual optimization of b values
                    gap = True, # Global Average Pooling reorder works with conv1x1 for the last linear layer
                    act_layer = True, # ReLU activation layer
                ),
                standard_changes = {"features[3]": nn.AvgPool2d(kernel_size=3, stride=2, padding=1)},
            ),
        )
    )
    for depth in DENSENET_DEPTHS
}
# -------------------------------------------------------------------------
I3D_DEPTHS = ["50"]
i3ds = {
    f"i3d": update_default(
        dict(
            model=dict(
                name=f"i3d",
                last_layer_name = "classifier", # For replacing the last fc layer with conv1x1
                weights=f"I3D_Weights.DEFAULT",
                bcosify_args = dict(
                    fix_b = True, # Fixed b value (=2)
                    use_bias = False, # No bias
                    norm_layer = "BnUncV2", # Modified Batch Norm
                    manual_optim=False, # For manual optimization of b values
                    gap = True, # Global Average Pooling reorder works with conv1x1 for the last linear layer
                    act_layer = True, # ReLU activation layer
                ),
                standard_changes = {"features[3]": nn.AvgPool3d(kernel_size=3, stride=2, padding=1)},
            ),
        )
    )
    for depth in I3D_DEPTHS
}

#  sanity

CONFIGS = dict()
CONFIGS.update(resnets)
CONFIGS.update(densenets)
CONFIGS.update(i3ds)
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[5,420, 1337]))


if __name__ == "__main__":
    configs_cli(CONFIGS)
