# This is my contribution

import torch.nn.functional as F
import argparse
import torch
from torch.utils.data import DataLoader

from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config
from gp_game2 import load_checkpoint, get_loader
from grid import collect_high_confidence_clips, add_blank_frames, add_second_clip


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


def game(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)
    model = load_checkpoint(model, args.checkpoint, device)

    model.eval()

    loader = get_loader(model_config, 1)

    clips_by_class = collect_high_confidence_clips(
        model=model,
        loader=loader,
        device=device,
        confidence_threshold=0.9,
        max_per_class=30,
        max_batches=200,
    )

    print("Found high-confidence clips for", len(clips_by_class), "classes")

    total_scores = []

    num_samples = 500
    for step in range(num_samples):
        print(step)
        clip, labels = add_second_clip(clips_by_class, step)
        score = explain_joint(model, args, clip, labels)
        total_scores.append(score)

    flat_scores = [s for scores in total_scores for s in scores]

    print("Number of scores:", len(flat_scores))
    print("Number of attempted samples:", len(total_scores))

    if flat_scores:
        avgs = sum(flat_scores) / len(flat_scores)
        print("Average score:", avgs)
    else:
        print("No valid scores found.")

def explain_joint(model, args, clip, labels):
    device = next(model.parameters()).device
    base_video = clip.to(device).unsqueeze(0)

    count = 0
    scores = []

    x = base_video.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)

    for i, label in enumerate(labels):
        x.grad = None
        model.zero_grad(set_to_none=True)

        with torch.enable_grad(), model.explanation_mode():
            out = model(x)

            logit = out[0, label]
            pred_class = out.argmax(dim=1).item()
            confidence = F.softmax(out, dim=1)[0, label].item()

            if pred_class != label:
                continue

            logit.backward(inputs=[x])

        if x.grad is None:
            raise RuntimeError("x.grad is None")

        grad = x.grad.detach().clone().squeeze(0)


        contribs = grad.to(device) * clip.to(device).squeeze()

        contribs = contribs[:3].clamp_min(0)
        contribs = contribs.sum(0)

        T,H,W = contribs.shape
        if i == 0:
            frames = [0,1,2,3]
        else:
            frames = [4,5,6,7]

        # score = correct contribution / total
        frame_contrib = 0
        total_contrib = 0
        for t in range(T):
            if t in frames:
                frame_contrib += contribs[t].sum(dim=(0,1)).item()
            total_contrib += contribs[t].sum(dim=(0,1)).item()
        scores.append(frame_contrib/total_contrib)

    return scores



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    game(args)