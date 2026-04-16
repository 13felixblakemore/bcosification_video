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
    #print(model.state_dict().keys())
    #print(checkpoint["state_dict"].keys())
    #continue
    new_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        new_key = k.replace("model.model.model.", "model.model.")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)

    macs, params = get_model_complexity_info(
        model,
        (6, 8, 224, 224),  # C,T,H,W for your video model
        as_strings=True,
        print_per_layer_stat=False,
    )

    print("MACs:", macs)
    print("Params:", params)


#model = model_bcos
model.eval().cuda()
x = torch.randn(1, 6, 8, 224, 224).cuda()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with torch.no_grad():
        _ = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))