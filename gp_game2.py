# parse args
import argparse
import pathlib

import torch
from torchvision.datasets import UCF101

from bcos import settings
from bcos.data.datamodules import UCF101DataModule
from bcos.data.presets import UCF101ClassificationPresetTrain, UCF101ClassificationPresetEval
from bcos.experiments.UCF101.bcosification.experiment_parameters import CONFIGS
from bcos.experiments.UCF101.bcosification.model import get_model
from evaluate import load_model_and_config


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Explain an image/vid", add_help=add_help
    )
    parser.add_argument(
        "--base_directory",
        default="./experiments",
        help="The base directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific Lightning .ckpt file to load"
    )
    return parser

def make_2x2_grid(videos):
    v0, v1, v2, v3 = videos
    top = torch.cat([v0, v1], dim=-1)
    bottom = torch.cat([v2, v3], dim=-1)
    return torch.cat([top, bottom], dim=-2)

def game(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)
    model.eval()
    print(model_config)
    dm = UCF101DataModule(model_config["data"])

    dm.setup("fit")

    loader = dm.train_dataloader()

    videos, labels = next(iter(loader))   # [B,C,T,H,W], [B]

    print(type(videos), videos.shape)
    print(type(labels), labels.shape)

    grid_video = make_2x2_grid([videos[0], videos[1], videos[2], videos[3]])
    grid_labels = labels[:4]

    print("grid_video shape:", grid_video.shape)
    print(grid_video.max(), grid_video.min())
    print("grid_labels:", grid_labels)
    with torch.no_grad():
        out = model(grid_video.unsqueeze(0).to(device))
    print(out.shape)
# sort clips by confidence
# choose top 500 clips and store

# compile 100 games
    # choose 4 random clips
    # compile each frame together into a quad frame
    # feed into model
    # compute explanations of the video with respect to one of the classes
    # evaluate how much of the top k% contribution is in the correct quadrant

# print out total score

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    game(args)