from bcos.data.datamodules import UCF101DataModule
from bcos.experiments.utils import Experiment
from tqdm.auto import tqdm

exp = Experiment("UCF101", "bcosification", "i3d")
config = exp.config

datamodule = UCF101DataModule(config["datamodule"])

# IMPORTANT: Lightning requires setup()
datamodule.setup("fit")

loader = datamodule.train_dataloader()

for i, (video, target) in enumerate(tqdm(loader)):
    print(target.cpu().numpy())
    if i == 20:
        break