# parse args
import argparse
import pathlib

import torch

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

def game(args):
    # load model
    config = CONFIGS["i3d"]
    config = config["model"]
    model = get_model(config)

    cp = args.checkpoint
    original_posixpath = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        ckpt = torch.load(cp, map_location="cpu")
    finally:
        pathlib.PosixPath = original_posixpath
    print(ckpt.keys())
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
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