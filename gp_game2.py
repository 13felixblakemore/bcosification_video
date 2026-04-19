# parse args
import argparse
import pathlib

import torch

from bcos.data.datamodules import UCF101GridDataModule
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

config = {
    "batch_size": 2,
    "num_workers": 4,
    "frames_per_clip": 8,
    "step_between_clips": 32,
    "fold": 2,
    "train_transform": UCF101ClassificationPresetTrain,
    "test_transform": UCF101ClassificationPresetEval,
    "same_class_grid": False,
}

def game(args):
    # load model
    model, model_config = load_model_and_config(args)
    model.eval()

    dm = UCF101GridDataModule(config)
    print("dm")
    dm.setup("fit")
    print("dm fit")
    loader = dm.train_dataloader()
    print("loader")
    grid_video, labels, indices = next(iter(loader))
    print("batch")

    print(grid_video.shape)  # [B, C, T, 2H, 2W]
    print(labels.shape)  # [B, 4]
    print(indices.shape)  # [B, 4]
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