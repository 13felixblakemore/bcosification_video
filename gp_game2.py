# This is my contribution

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from bcos.common import linear_mapping_to_heatmap, smooth_heatmap_np
from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config
from grid import collect_high_confidence_clips, sample_unique_class_grid


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


def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.model.model.", "model.model.")
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print("Loaded checkpoint.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if "epoch" in checkpoint:
        print("Checkpoint epoch:", checkpoint["epoch"])

    return model


def get_loader(model_config, batch_size=8):
    dm = UCF101DataModule(model_config["data"])
    dm.setup("test")

    return DataLoader(
        dm.eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


def game(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)

    if args.checkpoint is not None:
        model = load_checkpoint(model, args.checkpoint, device)

    model.eval()
    loader = get_loader(model_config)

    clips_by_class = collect_high_confidence_clips(
        model=model,
        loader=loader,
        device=device,
        confidence_threshold=0.9,
        max_per_class=5,
        max_batches=30,
    )

    print("Found high-confidence clips for", len(clips_by_class), "classes")

    all_scores = []
    total_correct = 0
    num_samples = 20

    for step in range(num_samples):
        print(f"{step}/{num_samples}")

        grid_video, grid_labels, confs = sample_unique_class_grid(
            clips_by_class,
            step + 43
        )

        scores, count = explain(
            model=model,
            video_tensor=grid_video,
            labels=grid_labels,
            output_dir=args.base_directory,
        )

        total_correct += count
        all_scores.extend(scores)

    print("total clips:", total_correct)
    print("count:", len(all_scores))

    if len(all_scores) == 0:
        print("No valid scores found.")
        return

    avg_score = sum(all_scores) / len(all_scores)
    print(f"Score: {avg_score:.4f}")


def explain(model, video_tensor, labels, output_dir=None):
    device = next(model.parameters()).device
    base_video = video_tensor.to(device).unsqueeze(0)

    scores = []
    count = 0

    for quadrant, label in enumerate(labels):
        if label == -1:
            continue

        x = base_video.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        with torch.enable_grad(), model.explanation_mode():
            out = model(x)

            logit = out[0, label]
            pred_class = out.argmax(dim=1).item()

            if pred_class != label:
                continue

            count += 1
            logit.backward(inputs=[x])

        if x.grad is None:
            raise RuntimeError("x.grad is None")

        linear_mapping = x.grad.detach().squeeze(0)

        score = gp_score_from_linear_map(
            linear_map=linear_mapping,
            target_quadrant=quadrant,
            vid=x.detach(),
            output_dir=output_dir,
        )

        scores.append(score)

    return scores, count


def get_quadrant_slices(H, W):
    h_mid = H // 2
    w_mid = W // 2

    return [
        (slice(None), slice(0, h_mid), slice(0, w_mid)),
        (slice(None), slice(0, h_mid), slice(w_mid, W)),
        (slice(None), slice(h_mid, H), slice(0, w_mid)),
        (slice(None), slice(h_mid, H), slice(w_mid, W)),
    ]


def gp_score_from_linear_map(
    linear_map: torch.Tensor,
    target_quadrant: int,
    vid: torch.Tensor,
    output_dir=None,
    save_threshold=0.8,
):
    contribs = (vid * linear_map).squeeze(0)
    contribs = contribs.sum(0)
    contribs = torch.relu(contribs)

    assert contribs.dim() == 3, "Expected [T,H,W]"

    T, H, W = contribs.shape
    total_mass = contribs.sum()

    if total_mass.item() <= 1e-12:
        return 0.0

    masks = get_quadrant_slices(H, W)

    quad_energy = []
    for mask in masks:
        quad_energy.append(contribs[mask].sum())

    quad_energy = torch.stack(quad_energy)
    scores = quad_energy / total_mass

    print("scores:", scores)

    energy_score = scores[target_quadrant].item()

    if output_dir is not None and energy_score > save_threshold:
        plot_grid(linear_map, vid, output_dir)

    return energy_score


def save_plot(path):
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_grid(linear_mapping, vid, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    vid = vid.squeeze(0)

    contribs = (vid * linear_mapping).sum(0, keepdim=True)

    rgb_grad = linear_mapping / (
        linear_mapping.abs().max(0, keepdim=True).values + 1e-12
    )

    rgb_grad = rgb_grad.clamp(min=0)
    rgb_grad = rgb_grad[:3]
    rgb_grad = 1 - rgb_grad

    alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
    contribs = contribs.squeeze(0)
    alpha = torch.where(contribs < 0, 1e-12, alpha)

    alpha_2d = alpha.permute(1, 0, 2, 3)
    alpha_2d = F.avg_pool2d(alpha_2d, kernel_size=15, stride=1, padding=7)
    alpha = alpha_2d.permute(1, 0, 2, 3)
    alpha = (alpha / torch.quantile(alpha, q=0.99)).clip(0, 1)

    rgb_grad = torch.cat([rgb_grad, alpha], dim=0)
    T = rgb_grad.shape[1]

    grad_video = [
        rgb_grad[:, t].permute(1, 2, 0).detach().cpu().numpy()
        for t in range(T)
    ]

    heatmap = linear_mapping_to_heatmap(vid, linear_mapping)
    heatmap = smooth_heatmap_np(heatmap)

    vid = vid[:3].permute(1, 2, 3, 0)
    vid = np.array(vid.cpu().detach())

    for t, frame in enumerate(vid):
        plt.imshow(frame)
        save_plot(os.path.join(output_dir, f"og_{t:03d}.png"))

    for t, frame_expl in enumerate(np.array(grad_video)):
        plt.imshow(frame_expl)
        save_plot(os.path.join(output_dir, f"explanation_{t:03d}.png"))

    for t, frame in enumerate(heatmap):
        plt.imshow(vid[t])
        plt.imshow(frame, cmap="jet", alpha=0.5)
        save_plot(os.path.join(output_dir, f"heatmap_{t:03d}.png"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    game(args)