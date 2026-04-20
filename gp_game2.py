# parse args
import argparse
import pathlib
import sys

import torch
from torch.version import cuda
from torchvision.datasets import UCF101
from torchvision.utils import save_image

from bcos import settings
from bcos.common import get_inx2label_ucf101
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

import random
from collections import defaultdict

def collect_high_confidence_clips(
    model,
    loader,
    device,
    confidence_threshold=0.7,
    max_per_class=10,
    max_batches=10,
):
    """
    Collect correctly classified, high-confidence clips.

    Returns:
        clips_by_class: dict[class_idx] -> list of (video_tensor_cpu, label, confidence)
    """
    model.eval()
    clips_by_class = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break

            videos = videos.to(device)
            labels = labels.to(device)

            out = model(videos)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            true_class_probs = probs[torch.arange(labels.size(0), device=device), labels]

            for i in range(labels.size(0)):
                label = int(labels[i].item())
                pred = int(preds[i].item())
                conf = float(true_class_probs[i].item())

                if pred == label and conf >= confidence_threshold:
                    if len(clips_by_class[label]) < max_per_class:
                        clips_by_class[label].append((
                            videos[i].detach().cpu(),
                            label,
                            conf
                        ))

    return clips_by_class


def sample_unique_class_grid(clips_by_class, seed=42):
    """
    Sample 4 clips from 4 different classes and return a 2x2 grid.
    """
    rng = random.Random(seed)

    valid_classes = [c for c, clips in clips_by_class.items() if len(clips) > 0]
    if len(valid_classes) < 4:
        raise RuntimeError(
            f"Only found {len(valid_classes)} classes with high-confidence clips. Need at least 4."
        )

    chosen_classes = rng.sample(valid_classes, 4)

    chosen_clips = []
    for c in chosen_classes:
        clip, label, conf = rng.choice(clips_by_class[c])
        chosen_clips.append((clip, label, conf))

    videos = [x[0] for x in chosen_clips]
    labels = torch.tensor([x[1] for x in chosen_clips], dtype=torch.long)
    confs = [x[2] for x in chosen_clips]

    grid_video = make_2x2_grid(videos)
    return grid_video, labels, confs


def sample_top_confidence_grid(clips_by_class):
    """
    Deterministic version: take the top-confidence clip from 4 classes with highest available confidence.
    """
    best_per_class = []
    for c, clips in clips_by_class.items():
        if len(clips) > 0:
            best_clip = max(clips, key=lambda x: x[2])
            best_per_class.append(best_clip)

    if len(best_per_class) < 4:
        raise RuntimeError(
            f"Only found {len(best_per_class)} classes with high-confidence clips. Need at least 4."
        )

    # pick the 4 strongest classes overall
    best_per_class = sorted(best_per_class, key=lambda x: x[2], reverse=True)[:4]

    videos = [x[0] for x in best_per_class]
    labels = torch.tensor([x[1] for x in best_per_class], dtype=torch.long)
    confs = [x[2] for x in best_per_class]

    grid_video = make_2x2_grid(videos)
    return grid_video, labels, confs

def make_2x2_grid(videos):
    v0, v1, v2, v3 = videos
    top = torch.cat([v0, v1], dim=-1)
    bottom = torch.cat([v2, v3], dim=-1)
    return torch.cat([top, bottom], dim=-2)

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
        confidence_threshold=0.8,  # try 0.5 if this is too strict
        max_per_class=10,
        max_batches=10,
    )

    print("Found high-confidence clips for", len(clips_by_class), "classes")

    # Random unique-class grid
    grid_video, grid_labels, confs = sample_unique_class_grid(clips_by_class, seed=42)

    # Or strongest 4 classes overall:
    # grid_video, grid_labels, confs = sample_top_confidence_grid(clips_by_class)

    print("Grid labels:", grid_labels.tolist())
    print("Grid confidences:", confs)
    print("Grid class names:", [get_inx2label_ucf101(int(l)) for l in grid_labels.tolist()])
    print("Grid video shape:", grid_video.shape)

    scores = explain(model, args, grid_video, grid_labels)
    print(scores)


def explain(model, args, video_tensor, labels):
    device = next(model.parameters()).device
    base_video = video_tensor.to(device).unsqueeze(0)   # [1, C, T, H, W]

    scores = []
    maps = []

    print("labels:", labels.tolist())

    for quadrant, label in enumerate(labels.tolist()):
        x = base_video.clone().detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        with torch.enable_grad(), model.explanation_mode():
            out = model(x)
            pred = out.topk(10, 1)
            print(pred)
            print(f"quadrant={quadrant}, label={label}, logit={out[0, label].item():.6f}")

            logit = out[0, label]
            logit.backward()

        if x.grad is None:
            raise RuntimeError("x.grad is None")

        grad = x.grad.detach().clone()

        # B-cos contribution map, not raw grad
        linear_mapping = (x.detach() * grad).sum(dim=1).squeeze(0)   # [T, H, W]

        gp_score = gp_scores_from_linear_map(linear_mapping, quadrant)
        scores.append(gp_score)
        maps.append(linear_mapping)

    for i in range(3):
        diff = (maps[i] - maps[i + 1]).abs().max().item()
        print(f"max abs diff map{i} vs map{i+1}: {diff}")

    return scores

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
            peak_correct: int (0 or 1)
            quadrant_pred: int
            quadrant_correct: int (0 or 1)
            topk_score: float
    """

    # --- ensure shape [T, H, W]
    if linear_map.dim() == 4:
        linear_map = linear_map.squeeze(0)

    assert linear_map.dim() == 3, "Expected [T,H,W]"

    T, H, W = linear_map.shape

    # --- use positive contributions only
    contrib = torch.relu(linear_map)

    total_mass = contrib.sum()
    if total_mass == 0:
        # avoid division by zero
        return {
            "energy_score": 0.0,
            "peak_correct": 0,
            "quadrant_pred": -1,
            "quadrant_correct": 0,
            "topk_score": 0.0,
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

    # --- 2. Peak-based GP (classic pointing game)
    flat_idx = contrib.view(-1).argmax()
    t_idx = flat_idx // (H * W)
    hw_idx = flat_idx % (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    if h_idx < h_mid and w_idx < w_mid:
        peak_quad = 0
    elif h_idx < h_mid and w_idx >= w_mid:
        peak_quad = 1
    elif h_idx >= h_mid and w_idx < w_mid:
        peak_quad = 2
    else:
        peak_quad = 3

    peak_correct = int(peak_quad == target_quadrant)

    # --- 3. Quadrant classification (which has most mass)
    quadrant_pred = int(torch.argmax(quad_energy))
    quadrant_correct = int(quadrant_pred == target_quadrant)

    # --- 4. Top-k mass score
    flat = contrib.view(-1)
    k = max(1, int(topk_percent * flat.numel()))

    topk_vals, topk_idx = torch.topk(flat, k)

    # convert indices to quadrant
    correct_count = 0
    for idx in topk_idx:
        idx = idx.item()
        t = idx // (H * W)
        hw = idx % (H * W)
        h = hw // W
        w = hw % W

        if h < h_mid and w < w_mid:
            q = 0
        elif h < h_mid and w >= w_mid:
            q = 1
        elif h >= h_mid and w < w_mid:
            q = 2
        else:
            q = 3

        if q == target_quadrant:
            correct_count += 1

    topk_score = correct_count / k

    return {
        "energy_score": energy_score,
        "peak_correct": peak_correct,
        "quadrant_pred": quadrant_pred,
        "quadrant_correct": quadrant_correct,
        "topk_score": topk_score,
    }

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    game(args)