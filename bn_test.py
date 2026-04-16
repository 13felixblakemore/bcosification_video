from types import SimpleNamespace
import pathlib

from bcos.experiments.utils import Experiment
from bcos.training.bcosify_trainer import BcosifyTrainer
from bcos.training.trainer import ClassificationLitModel
import pytorch_lightning as pl
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
from torch import nn

from bcos.data.presets import UCF101ClassificationPresetEval
from bcos.modules import BatchNormUncentered3d
from evaluate import load_model_and_config

bcos_args = SimpleNamespace(
    base_directory="./experiments",
    dataset="UCF101",
    base_network="bcosification",
    experiment_name="i3d",
    reload="last",
    weights=None,
    ema=False,
    batch_size=8,
    no_cuda=False,
)
standard_args = SimpleNamespace(
    base_directory="./experiments",
    dataset="UCF101",
    base_network="standard",
    experiment_name="i3d",
    reload="last",
    weights=None,
    ema=False,
    batch_size=8,
    no_cuda=False,
)
exp = Experiment("UCF101", "bcosification", "i3d")
config = exp.config.copy()

# get and set seed
seed = exp.config.get("seed", 42)
pl.seed_everything(seed, workers=True)

trainer = ClassificationLitModel  # For automatic optimization

bcosify_args = exp.config["model"].get("bcosify_args", None)
if bcosify_args is not None and bcosify_args.get("manual_optim", False):
    trainer = BcosifyTrainer
    print("B-Cosifying")

# init model
bcos_model = trainer(
    "UCF101",
    "bcosification",
    "i3d"
)
standard_model = trainer(
    "UCF101",
    "standard",
    "i3d"
)

_, bcos_config = load_model_and_config(bcos_args)
# Making all the bias parameters None
print("Removing bias parameters (making None)")
for mod in bcos_model.modules():
  if hasattr(mod, "bias") and mod.bias is not None:
      mod.bias = None

# compare original BN vs converted BN on same activation tensor
bcos_model.eval()
standard_model.eval()

bcos_transform = UCF101ClassificationPresetEval(
        crop_size=224,
        is_bcos=True,
    )
transform = UCF101ClassificationPresetEval(
        crop_size=224,
        is_bcos=False,
    )


bn = nn.BatchNorm3d(64)
unc = BatchNormUncentered3d.from_standard_module(bn, bcos_config)

x = torch.randn(8,64,8,32,32)

bn.eval()
unc.eval()

print((bn(x)-unc(x)).abs().mean())

bn.train()
unc.train()

print((bn(x)-unc(x)).abs().mean())