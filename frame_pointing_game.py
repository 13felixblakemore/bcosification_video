import argparse

import torch

from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config
from grid import collect_high_confidence_clips, add_blank_frames


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
        max_per_class=2,
        max_batches=20,
    )

    print("Found high-confidence clips for", len(clips_by_class), "classes")

    total_scores = []

    for step in range(20):
        print(step)
        clip, label = add_blank_frames(clips_by_class)
        # clip, labels = add_second_clip(clips_by_class)
        score = explain(model, args, clip, label)
        total_scores.append(score)

    print(total_scores)

def explain(model, args, clip, label):
    device = next(model.parameters()).device
    base_video = clip.to(device).unsqueeze(0)   # [1, C, T, H, W]

    scores = []

    x = base_video.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)

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
    frames = [3,4]
    frame_contrib = 0
    total_contrib = 0
    for t in range(T):
        if t in frames:
            frame_contrib += grad.sum(dims=(1,2)).item()
        total_contrib += grad.sum(dims=(1,2)).item()
    return frame_contrib / total_contrib


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    game(args)