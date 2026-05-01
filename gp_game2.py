# This is my contribution

# parse args
import argparse
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from cv2.version import contrib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.version import cuda
from torchvision.datasets import UCF101
from torchvision.utils import save_image

from bcos import settings
from bcos.common import get_inx2label_ucf101, gradient_to_video, linear_mapping_to_heatmap, smooth_heatmap_np
from bcos.data.datamodules import UCF101DataModule
from bcos.data.presets import UCF101ClassificationPresetTrain, UCF101ClassificationPresetEval
from bcos.experiments.UCF101.bcosification.experiment_parameters import CONFIGS
from bcos.experiments.UCF101.bcosification.model import get_model
from evaluate import load_model_and_config
from grid import sample_unique_class_grid, make_2x2_grid, collect_high_confidence_clips, sample_top_confidence_grid, sample_with_blank, sample_two_clips_two_blank


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Grid Pointing Game", add_help=add_help
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

        state_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            new_key = new_key.replace("model.model.model.", "model.model.")

            new_state_dict[new_key] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        print("Loaded checkpoint.")

        if "epoch" in checkpoint:
            print("Checkpoint epoch:", checkpoint["epoch"])

    model.eval()
    print(model_config)
    dm = UCF101DataModule(model_config["data"])
    dm.setup("test")

    loader = DataLoader(
        dm.eval_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    clips_by_class = collect_high_confidence_clips(
        model=model,
        loader=loader,
        device=device,
        confidence_threshold=0.9,
        max_per_class=5,
        max_batches=30,
    )

    print("Found high-confidence clips for", len(clips_by_class), "classes")

    total_scores = []
    total = 0

    num_samples = 20
    for step in range(num_samples):
        print(f"{step}/{num_samples}")
        grid_video, grid_labels, confs = sample_unique_class_grid(clips_by_class, step + 43)
        scores, count = explain(model, args, grid_video, grid_labels)

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

    print("count: ", count)
    print("avg results: ", avg_results)
    for m in metrics:
        avg_results[m] /= count

    print("\n=== Average GP Results over 20 grids ===")
    for k, v in avg_results.items():
        print(f"{k}: {v:.4f}")


def explain(model, args, video_tensor, labels, true_quad=None):
    device = next(model.parameters()).device
    base_video = video_tensor.to(device).unsqueeze(0)

    count = 0
    scores = []

    for quadrant, label in enumerate(labels):
        if label == -1:
            continue
        x = base_video.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        with torch.enable_grad(), model.explanation_mode():
            out = model(x)

            logit = out[0, label]
            pred_class = out.argmax(dim=1).item()
            confidence = F.softmax(out, dim=1)[0, label].item()

            if pred_class == label:
                count += 1
                pass
            else:
                continue

            logit.backward(inputs=[x])

        if x.grad is None:
            raise RuntimeError("x.grad is None")

        grad = x.grad.detach().clone()
        linear_mapping = grad.squeeze(0)

        gp_score = gp_scores_from_linear_map(linear_mapping, target_quadrant=quadrant, vid=x)
        scores.append(gp_score)
    return scores, count

def gp_scores_from_linear_map(
    linear_map: torch.Tensor,
    target_quadrant: int,
    topk_percent: float = 0.1,
    vid = None
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

    contribs = (vid * linear_map).squeeze(0)

    contribs = contribs.sum(0)
    debug_quadrant_masses(contribs)
    # --- ensure shape [T, H, W]
    if contribs.dim() == 4:
        linear_map = linear_map.squeeze(0)

    print(contribs.shape)
    assert contribs.dim() == 3, "Expected [T,H,W]"

    T, H, W = contribs.shape

    print(contribs.min(), contribs.max())
    # --- use positive contributions only
    contribs = torch.relu(contribs)

    total_mass = contribs.sum()
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
        quad_energy.append(contribs[m].sum())

    quad_energy = torch.stack(quad_energy)

    scores = quad_energy / total_mass
    print("scores: ", scores)
    # --- 1. Energy-based GP score
    energy_score = (quad_energy[target_quadrant] / total_mass).item()

    #if (scores > 0.1).all():
    #    plot_grid(linear_map, vid)

    if energy_score > 0.8:
        plot_grid(linear_map, vid)

    return {
        "energy_score": energy_score
    }


def plot_grid(linear_mapping, vid):
    # shape of vid and linmap is [C, T, H, W], summing over first dimension gives the contribution map per location per frame
    print(linear_mapping.shape)
    print(vid.shape)
    vid = vid.squeeze(0)

    contribs = (vid * linear_mapping).sum(0, keepdim=True)  # [1, T, H, W]

    print("Contribs ", contribs.shape)

    # Normalise each pixel vector (r, g, b, 1-r, 1-g, 1-b) s.t. max entry is 1, maintaining direction
    rgb_grad = linear_mapping / (
        linear_mapping.abs().max(0, keepdim=True).values + 1e-12
    )
    print("rgb grad ", rgb_grad.shape)

    # clip off values below 0 (i.e., set negatively weighted channels to 0 weighting)
    rgb_grad = rgb_grad.clamp(min=0)

    # normalise s.t. each pair (e.g., r and 1-r) sums to 1 and only use resulting rgb values
    #pair = rgb_grad[:3] + rgb_grad[3:]
    #rgb_grad = rgb_grad[:3] / (pair + 1e-12)  # [3, T, H, W]
    rgb_grad = rgb_grad[:3]
    rgb_grad = 1 - rgb_grad
    print("rgb grad ", rgb_grad.shape)
    # Set alpha value to the strength (L2 norm) of each location's gradient
    alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
    # Only show positive contributions
    contribs = contribs.squeeze(0)
    alpha = torch.where(contribs < 0, 1e-12, alpha)
    # [1, T, H, W] -> [T, 1, H, W]
    print("Alpha: ", alpha.shape)
    alpha_2d = alpha.permute(1, 0, 2, 3)
    alpha_2d = F.avg_pool2d(alpha_2d, kernel_size=15, stride=1, padding=(15 - 1) // 2)
    alpha = alpha_2d.permute(1, 0, 2, 3)  # back to [1, T, H, W]
    alpha = (alpha / torch.quantile(alpha, q=99.0 / 100)).clip(0, 1)

    rgb_grad = torch.concatenate([rgb_grad, alpha], dim=0)  # [4, T, H, W]
    T = rgb_grad.shape[1]
    print("RBG grad shape ", rgb_grad.shape)

    # Reshaping to [T, H, W, C]
    grad_video = [rgb_grad[:, t].permute(1, 2, 0).detach().cpu().numpy() for t in range(T)]
    print("GRADVID ", np.array(grad_video).shape)

    heatmap = linear_mapping_to_heatmap(vid, linear_mapping)
    heatmap = smooth_heatmap_np(heatmap)

    vid = vid[:3].permute(1, 2, 3, 0)
    vid = np.array(vid.cpu().detach())
    for t, frame in enumerate(vid):
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(os.path.join(args.base_directory, f"og_{t}.png"), bbox_inches='tight')
        plt.close()

    for t, frame_expl in enumerate(np.array(grad_video)):
        plt.imshow(frame_expl)
        plt.axis('off')
        plt.savefig(os.path.join(args.base_directory, f"explanation_{t:03d}.png"), bbox_inches='tight')
        plt.close()



    for t, frame in enumerate(heatmap):
        plt.imshow(vid[t])  # original frame
        plt.imshow(heatmap[t], cmap='jet', alpha=0.5)  # overlay
        plt.axis('off')
        plt.savefig(os.path.join(args.base_directory, f"heatmap_{t:03d}.png"), bbox_inches='tight')
        plt.close()

    sys.exit()


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