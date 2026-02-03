from bcos.data.datamodules import UCF101DataModule
from bcos.experiments.utils import Experiment

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x

exp = Experiment("UCF101", "bcosification", "i3d")

config = exp.config  # gets the config
print(config)
datamodule = UCF101DataModule(config)
i=0


for video, target in tqdm(datamodule):
    print(target.numpy())
    i = i + 1
    if i == 20:
        break

