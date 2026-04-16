import torch
from ptflops import get_model_complexity_info

from bcos.experiments.utils import CHECKPOINT_LAST_FILENAME
from bcos.models.standard_models import I3DBcos

ckpt = CHECKPOINT_LAST_FILENAME
checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))

model = I3DBcos(True)

# If checkpoint contains only state_dict
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
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