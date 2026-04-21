# parse args
import argparse
import os
import pathlib
import sys

import torch
import torch.nn.functional as F
from cv2.version import contrib
from matplotlib import pyplot as plt
from torch.version import cuda
from torchvision.datasets import UCF101
from torchvision.utils import save_image

from bcos import settings
from bcos.common import get_inx2label_ucf101, gradient_to_video
from bcos.data.datamodules import UCF101DataModule
from bcos.data.presets import UCF101ClassificationPresetTrain, UCF101ClassificationPresetEval
from bcos.experiments.UCF101.bcosification.experiment_parameters import CONFIGS
from bcos.experiments.UCF101.bcosification.model import get_model
from evaluate import load_model_and_config
from grid import sample_unique_class_grid, make_2x2_grid, collect_high_confidence_clips, sample_top_confidence_grid, sample_with_blank, sample_two_clips_two_blank


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")

        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Handle Lightning checkpoints
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 🔧 Fix key mismatches (VERY important for your setup)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k

            # Common prefix issues in your repo
            #new_key = new_key.replace("model.model.model.", "model.model.")
            #new_key = new_key.replace("model.model.", "model.")

            new_state_dict[new_key] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        print("Loaded checkpoint.")
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))

        # Optional debug
        if "epoch" in checkpoint:
            print("Checkpoint epoch:", checkpoint["epoch"])
    model.eval()
    print(model_config)
    dm = UCF101DataModule(model_config["data"])

    dm.setup("fit")

    loader = dm.train_dataloader()

    clips_by_class = collect_high_confidence_clips(
        model=model,
        loader=loader,
        device=device,
        confidence_threshold=0.9,  # try 0.5 if this is too strict
        max_per_class=20,
        max_batches=100,
    )

    print("Found high-confidence clips for", len(clips_by_class), "classes")

    total_scores = []
    total = 0
    for step in range(100):
        print(step)
        grid_video, grid_labels, confs, quads = sample_two_clips_two_blank(clips_by_class)

        scores, count = explain(model, args, grid_video, grid_labels, quads)
        total += count
        if scores:
            total_scores.append(scores)

    print("total clips: ", total)
    # --- aggregate ---
    metrics = ["energy_score"]

    avg_results = {m: 0.0 for m in metrics}
    count = 0


    for grid_scores in total_scores:  # each grid
        for s in grid_scores:  # each quadrant
            for m in metrics:
                avg_results[m] += s[m]
            count += 1

    for m in metrics:
        avg_results[m] /= count

    print("\n=== Average GP Results over 20 grids ===")
    for k, v in avg_results.items():
        print(f"{k}: {v:.4f}")


def explain(model, args, video_tensor, labels, true_quad=None):
    device = next(model.parameters()).device
    base_video = video_tensor.to(device).unsqueeze(0)   # [1, C, T, H, W]

    count = 0
    scores = []

    #print("labels:", labels)

    for quadrant, label in enumerate(labels):
        if label == -1:
            continue
        x = base_video.clone().detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        with torch.enable_grad(), model.explanation_mode():
            out = model(x)
            pred = out.topk(10, 1)

            logit = out[0, label]
            pred_class = out.argmax(dim=1).item()
            confidence = F.softmax(out, dim=1)[0, pred_class].item()

            if pred_class == label and confidence > 0.9:
                count += 1
                pass
            else:
                continue

            logit.backward(inputs=[x])

        if x.grad is None:
            raise RuntimeError("x.grad is None")

        grad = x.grad.detach().clone()
        linear_mapping = grad.squeeze(0)

        contribs = (x * linear_mapping).squeeze(0)

        contribs = contribs.sum(0)

        debug_quadrant_masses(contribs)

        gp_score = gp_scores_from_linear_map(contribs, quadrant)
        scores.append(gp_score)
    return scores, count

def gp_scores_from_linear_map(
    linear_map: torch.Tensor,
    target_quadrant: int,
    topk_percent: float = 0.1,
):
    """
    Compute GP game scores from a linear mapping.

    Args:
        linear_map: Tensor of shape [T, H, W] or [1, T, H, W]
                    (already reduced over channels if needed)
        target_quadrant: int in {0,1,2,3}
            0 = top-left
            1 = top-right
            2 = bottom-left
            3 = bottom-right
        topk_percent: fraction for top-k evaluation (e.g. 0.1 = top 10%)

    Returns:
        dict with:
            energy_score: float
    """

    # --- ensure shape [T, H, W]
    if linear_map.dim() == 4:
        linear_map = linear_map.squeeze(0)

    print(linear_map.shape)
    assert linear_map.dim() == 3, "Expected [T,H,W]"

    T, H, W = linear_map.shape

    print(linear_map.min(), linear_map.max())
    # --- use positive contributions only
    contrib = torch.relu(linear_map)

    total_mass = contrib.sum()
    if total_mass == 0:
        # avoid division by zero
        return {
            "energy_score": 0.0
        }

    # --- define quadrant masks
    h_mid = H // 2
    w_mid = W // 2

    masks = [
        (slice(None), slice(0, h_mid), slice(0, w_mid)),  # 0 TL
        (slice(None), slice(0, h_mid), slice(w_mid, W)),  # 1 TR
        (slice(None), slice(h_mid, H), slice(0, w_mid)),  # 2 BL
        (slice(None), slice(h_mid, H), slice(w_mid, W)),  # 3 BR
    ]

    # --- energy per quadrant
    quad_energy = []
    for m in masks:
        quad_energy.append(contrib[m].sum())

    quad_energy = torch.stack(quad_energy)

    # --- 1. Energy-based GP score
    energy_score = (quad_energy[target_quadrant] / total_mass).item()


    return {
        "energy_score": energy_score
    }

def debug_quadrant_masses(linear_map):
    contrib = torch.relu(linear_map)
    T, H, W = contrib.shape
    h_mid = H // 2
    w_mid = W // 2

    q0 = contrib[:, :h_mid, :w_mid].sum().item()
    q1 = contrib[:, :h_mid, w_mid:].sum().item()
    q2 = contrib[:, h_mid:, :w_mid].sum().item()
    q3 = contrib[:, h_mid:, w_mid:].sum().item()
    total = q0 + q1 + q2 + q3

    print("Quadrant masses:", [q0, q1, q2, q3])
    print("Normalised:", [q0/total, q1/total, q2/total, q3/total])

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    game(args)