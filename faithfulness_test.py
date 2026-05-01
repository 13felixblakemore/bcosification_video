import argparse

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import load_model_and_config
from gp_game2 import load_checkpoint, get_loader


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Faithfulness test", add_help=add_help
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


def minmax_norm(logits):
    mins = logits.min(dim=1, keepdim=True).values
    maxs = logits.max(dim=1, keepdim=True).values
    return (logits - mins) / (maxs - mins + 1e-8)


def mask_topk_explanation(model, videos, target_classes, device, k_percentile=0.7):
    videos_for_grad = videos.detach().clone().to(device)
    videos_for_grad.requires_grad_(True)

    target_classes = target_classes.to(device).long()

    model.zero_grad()

    logits = model(videos_for_grad)
    target_logits = logits[
        torch.arange(videos_for_grad.size(0), device=device),
        target_classes
    ]
    target_logits.sum().backward()

    linear_mapping = videos_for_grad.grad

    contributions = (videos_for_grad * linear_mapping).sum(dim=1)
    explanation_scores = linear_mapping.norm(p=2, dim=1)

    explanation_scores = torch.where(
        contributions > 0,
        explanation_scores,
        torch.zeros_like(explanation_scores)
    )

    B, T, H, W = explanation_scores.shape

    explanation_scores_2d = explanation_scores.reshape(B * T, 1, H, W)
    explanation_scores_2d = F.avg_pool2d(
        explanation_scores_2d,
        kernel_size=15,
        stride=1,
        padding=7
    )
    explanation_scores = explanation_scores_2d.reshape(B, T, H, W)

    flat_scores = explanation_scores.view(B, -1)
    denom = torch.quantile(flat_scores, q=0.98, dim=1).view(B, 1, 1, 1)
    explanation_scores = (explanation_scores / (denom + 1e-8)).clamp(0, 1)

    flat_scores = explanation_scores.view(B, -1)

    k = int(k_percentile * flat_scores.size(1))
    k = max(k, 1)

    _, top_idx = torch.topk(flat_scores, k, dim=1)

    flat_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    flat_mask.scatter_(1, top_idx, True)

    important_mask = flat_mask.view(B, T, H, W)
    important_mask = important_mask.unsqueeze(1).expand_as(videos)

    if videos.size(1) == 6:
        mask_values = torch.zeros(videos.size(1), device=device)
        mask_values[:3] = 0
        mask_values[3:] = 1
    else:
        mask_values = torch.zeros(videos.size(1), device=device)

    mask_values = mask_values.view(1, videos.size(1), 1, 1, 1)

    masked_videos = torch.where(
        important_mask,
        mask_values,
        videos.detach().clone()
    )

    return masked_videos.detach()


def test_faithfulness(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)

    if args.checkpoint is not None:
        model = load_checkpoint(model, args.checkpoint, device)

    model.eval()
    loader = get_loader(model_config, batch_size=4)

    faithfulness_scores = []
    max_batches = 200

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        videos, labels = batch
        videos = videos.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(videos)
            norm_preds = minmax_norm(logits)
            preds = logits.argmax(dim=1)
            correct_mask = preds == labels

        if correct_mask.sum() == 0:
            continue

        videos = videos[correct_mask]
        labels = labels[correct_mask]
        preds = preds[correct_mask]
        norm_preds = norm_preds[correct_mask]

        target_classes = labels

        original_scores = norm_preds[
            torch.arange(videos.size(0), device=device),
            target_classes
        ]

        masked_videos = mask_topk_explanation(
            model,
            videos,
            target_classes,
            device,
            k_percentile=0.7
        )

        with torch.no_grad():
            masked_logits = model(masked_videos)
            masked_norm_preds = minmax_norm(masked_logits)

            masked_scores = masked_norm_preds[
                torch.arange(masked_videos.size(0), device=device),
                target_classes
            ]

            batch_scores = torch.abs(original_scores - masked_scores)
            faithfulness_scores.extend(batch_scores.detach().cpu().tolist())

    if len(faithfulness_scores) == 0:
        print("No correctly classified clips were evaluated.")
        return None, None

    mean_faithfulness = np.mean(faithfulness_scores)
    std_faithfulness = np.std(faithfulness_scores)

    print(f"Faithfulness: {mean_faithfulness:.4f} ± {std_faithfulness:.4f}")
    print(f"Evaluated clips: {len(faithfulness_scores)}")

    return mean_faithfulness, std_faithfulness


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    test_faithfulness(args)