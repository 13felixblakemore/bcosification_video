import argparse
from collections import defaultdict

import torch

from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config
from grid import collect_high_confidence_clips, add_blank_frames, add_second_clip, add_blank_frames_full


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
            new_key = new_key.replace("model.model.model.", "model.model.")
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

    dm.setup("test")

    loader = dm.train_dataloader()
    # 1. Change to a dictionary of lists: {class_id: [score1, score2, ...]}
    class_scores = defaultdict(list)

    for batch_idx, (videos, labels) in enumerate(loader):
        print(f"Processing batch {batch_idx}")
        videos = add_blank_frames_full(videos)

        # explain now returns a dictionary of {label: score}
        batch_results = explain(model, args, videos, labels)

        # 2. Merge batch results into our main tracker
        for label, score in batch_results.items():
            class_scores[label].append(score)

    # 3. Calculate averages per class
    print("\n--- Per-Class Results ---")
    per_class_averages = {}
    for label, scores in class_scores.items():
        avg = sum(scores) / len(scores)
        per_class_averages[label] = avg
        print(f"Class {label}: {avg:.4f} (based on {len(scores)} samples)")

    # 4. Overall average (optional)
    all_scores = [s for scores in class_scores.values() for s in scores]
    print(f"\nGlobal Average: {sum(all_scores) / len(all_scores):.4f}")

    return per_class_averages

def explain_joint(model, args, clip, labels):
    device = next(model.parameters()).device
    base_video = clip.to(device).unsqueeze(0)   # [1, C, T, H, W]

    scores = []

    x = base_video.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)

    for i, label in enumerate(labels):
        with torch.enable_grad(), model.explanation_mode():
            out = model(x)

            logit = out[0, label]
            logit.backward(inputs=[x])

        if x.grad is None:
            raise RuntimeError("x.grad is None")

        grad = x.grad.detach().clone().squeeze(0)
        grad = grad[:3].clamp_min(0)
        grad = grad.sum(0)
        # then keep only top 10% of gradients

        T,H,W = grad.shape
        if i ==0:
            frames = [0,1,2,3]
        else:
            frames = [4,5,6,7]
        frame_contrib = 0
        total_contrib = 0
        for t in range(T):
            if t in frames:
                frame_contrib += grad[t].sum(dim=(0,1)).item()
            total_contrib += grad[t].sum(dim=(0,1)).item()
        scores.append(frame_contrib/total_contrib)
    return scores


def explain(model, args, batch, labels):
    device = next(model.parameters()).device
    scores = {}
    batch = batch.to(device)

    for idx, vid in enumerate(batch):
        # Add the batch dimension back for the model [1, C, T, H, W]
        x = vid.unsqueeze(0).clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        with torch.enable_grad(), model.explanation_mode():
            out = model(x)
            # Target the specific label for THIS video in the batch
            current_label = labels[idx].item()
            logit = out[0, current_label]
            logit.backward()

        if x.grad is None:
            continue  # Or raise error

        # Process gradients
        grad = x.grad.detach().clone().squeeze(0)  # [C, T, H, W]
        grad = grad[:3].clamp_min(0).sum(0)  # [T, H, W] (Positive RGB contrib)

        T, H, W = grad.shape
        frames = [3, 4]  # Your target "signal" frames

        frame_contrib = 0
        total_contrib = 0

        for t in range(T):
            step_sum = grad[t].sum().item()
            if t in frames:
                frame_contrib += step_sum
            total_contrib += step_sum

        # Store score keyed by the label
        # Note: If a batch has two of the same class, this overwrites.
        # Better to return a list of tuples or use the logic in game()
        scores[idx] = (current_label, frame_contrib / (total_contrib + 1e-8))

    # Return a list of (label, score) to handle duplicate classes in one batch
    return {label: score for idx, (label, score) in scores.items()}


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    game(args)