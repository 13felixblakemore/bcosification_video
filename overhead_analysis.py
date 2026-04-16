import os

import torch
from ptflops import get_model_complexity_info

from bcos.experiments.utils import CHECKPOINT_LAST_FILENAME, Experiment
from bcos.models.standard_models import I3DBcos
from bcosify3D import BcosifyNetwork

ckpt = CHECKPOINT_LAST_FILENAME
path = os.path.join("experiments/UCF101/bcosification/i3d", ckpt)
s_path = os.path.join("experiments/UCF101/standard/i3d", ckpt)
checkpoint = torch.load(path, map_location=torch.device('cuda'))
s_checkpoint = torch.load(s_path, map_location=torch.device('cuda'))

exp = Experiment("UCF101", "bcosification", "i3d")
config = exp.config.copy()

model = I3DBcos(True)
model_bcos = BcosifyNetwork(model, config["model"])

# If checkpoint contains only state_dict
for model, checkpoint in [(model, s_checkpoint), (model_bcos, checkpoint)]:
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    macs, params = get_model_complexity_info(
        model,
        (6, 8, 224, 224),  # C,T,H,W for your video model
        as_strings=True,
        print_per_layer_stat=False,
    )

    print("MACs:", macs)
    print("Params:", params)