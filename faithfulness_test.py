import argparse

import numpy as np
import torch.nn.functional as F
import torch
from evaluate import load_model_and_config
from gp_game2 import load_checkpoint, get_loader


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Frame pointing game", add_help=add_help
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

def test_faithfulness(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)

    if args.checkpoint is not None:
        model = load_checkpoint(model, args.checkpoint, device)

    model.eval()
    loader = get_loader(model_config, batch_size=4)

    faithfulness_scores = []
    max_batches = 100

    for batch_idx, batch in enumerate(loader):
        videos, labels = batch
        videos = videos.to(device)
        labels = labels.to(device)

        if batch_idx >= max_batches:
            break

        with torch.no_grad():
            logits = model(videos)
            probs = F.softmax(logits, dim=1)

            # only test on correct
            preds = probs.argmax(dim=1)
            correct_mask = preds == labels

        if correct_mask.sum() == 0:
            continue

        videos = videos[correct_mask]
        labels = labels[correct_mask]
        probs = probs[correct_mask]
        preds = preds[correct_mask]

        # Use the correctly predicted class as the explanation target
        target_classes = labels

        original_scores = probs[
            torch.arange(videos.size(0), device=device),
            target_classes
        ]

        masked_videos = mask_topk_contributions(model, videos, target_classes, device, k_percentile=0.2)
        with torch.no_grad():
            masked_logits = model(masked_videos)
            masked_probs = F.softmax(masked_logits, dim=1)

            masked_scores = masked_probs[
                torch.arange(masked_videos.size(0), device=device),
                target_classes
            ]

            batch_scores = torch.abs(original_scores - masked_scores)
            faithfulness_scores.extend(batch_scores.detach().cpu().tolist())

    mean_faithfulness = np.mean(faithfulness_scores)
    std_faithfulness = np.std(faithfulness_scores)

    print(f"Faithfulness: {mean_faithfulness:.4f} ± {std_faithfulness:.4f}")
    print(f"Evaluated clips: {len(faithfulness_scores)}")

    return mean_faithfulness, std_faithfulness

def mask_topk_contributions(model, videos, target_classes, device, k_percentile=0.1):
    videos_for_grad = videos.detach().clone()
    videos_for_grad.requires_grad_(True)

    model.zero_grad()

    logits = model(videos_for_grad)
    target_logits = logits[
        torch.arange(videos_for_grad.size(0), device=device),
        target_classes
    ]

    target_logits.sum().backward()

    contributions = videos_for_grad * videos_for_grad.grad
    contribution_maps = contributions.sum(dim=1)
    contribution_maps = contribution_maps.clamp(min=0)

    B = contribution_maps.size(0)
    flat_contribs = contribution_maps.view(B, -1)

    threshold = torch.quantile(flat_contribs, 1-k_percentile, dim=1)
    important_mask = contribution_maps >= threshold.view(B, 1, 1, 1)
    important_mask = important_mask.unsqueeze(1).expand_as(videos)

    masked_videos = videos.detach().clone()

    rgb_mask = important_mask[:, :3]
    inv_mask = important_mask[:, 3:]

    masked_videos[:, :3][rgb_mask] = 0
    masked_videos[:, 3:][inv_mask] = 1

    return masked_videos

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    test_faithfulness(args)