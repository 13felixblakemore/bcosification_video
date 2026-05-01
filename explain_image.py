# This code is my contribution. Used to create explanations for 2D or 3D. Requires adaptation.

import argparse
import sys
import time

import cv2
import numpy as np
from fontTools.unicodedata import block

from bcos.common import get_inx2label_imagenette
from bcos.common import get_inx2label_ucf101 as idx2label
from pathlib import Path
from evaluate import evaluate, load_model_and_config
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os

from gp_game2 import load_checkpoint

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x  # noqa: E731
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from bcos.data.presets import ImageNetClassificationPresetEval, UCF101ClassificationPresetEval


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
        "--dataset",
        choices=["ImageNet", "CIFAR10", "ImageNette", "UCF101"],
        default="UCF101",
        help="The dataset.",
    )
    parser.add_argument(
        "--base_network", help="The model config or base network to use."
    )
    parser.add_argument("--experiment_name", help="The name of the experiment to run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--reload",
        default="last",
        help="What ckpt to load. ['last', 'best', 'epoch_<N>', 'best_any']"
    )
    group.add_argument(
        "--weights",
        metavar="PATH",
        type=Path,
        help="Specific weight state dict to load.",
    )

    parser.add_argument(
        "--ema",
        default=False,
        action="store_true",
        help="Load the EMA stored version if it exists. Not applicable for reload='best_any'.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific Lightning .ckpt file to load"
    )

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size to use. Default is 1"
    )
    parser.add_argument(
        "--no-cuda",
        default=False,
        action="store_true",
        help="Force into not using cuda.",
    )

    parser.add_argument(
        "--image_path", type=str, help="The image path."
    )

    return parser

def explain_image(args, image_path):
    global device
    if args.no_cuda:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    img = Image.open(image_path)

    transform = ImageNetClassificationPresetEval(
        crop_size=224,
        is_bcos=True,
    )

    img = transform(img)
    img = img[None]
    img = img.to(device)

    model, config = load_model_and_config(args)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(name, module)
            break

    module = model.model.conv1.weight
    print(module.shape)
    symmetry = (module[:, :3] + module[:, 3:]).abs().mean()
    print(symmetry)

    model.eval()

    expl_out = model.explain(img)

    plt.imshow(expl_out["explanation"])
    path_to_save = os.path.join(args.base_directory, "explanation.png")

    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()

def debug(model, video_tensor):
    model.eval()

    x = video_tensor.detach().clone().requires_grad_(True)

    print(len(model.model.blocks))

    block_0 = model.model.blocks[0]
    block_1 = model.model.blocks[1]
    block_2 = model.model.blocks[2]
    block_3 = model.model.blocks[3]
    block_4 = model.model.blocks[4]
    block_5 = model.model.blocks[5]
    block_6 = model.model.blocks[6]

    for name, module in model.named_children():
        print(name, "->", module.__class__.__name__)

    out = model.bcosifynormalize(x)
    f = out.sum()

    if x.grad is not None:
        x.grad.zero_()
    f.backward()

    grad = x.grad.detach().clone()
    recon = (x * grad).sum()

    print("Debugging")
    print("input shape: ", x.shape)
    print("conv output shape:", out.shape)
    print("scalar f = sum(conv(x)):", f.item())
    print("reconstructed:", recon.item())
    print("difference:", (f - recon).item())
    print("max abs grad:", grad.abs().max().item())

    import inspect
    print(inspect.getsource(model.bcosifynormalize.forward))

    return {
        "scalar": f.detach(),
        "reconstruction": recon.detach(),
        "difference": (f - recon).detach(),
        "grad": grad,
        "output": out.detach(),
    }

def explain_video(args, video_path=None, vid_tensor=None):
    global device
    if args.no_cuda:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = False

    if video_path:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        start = 16
        length = 8
        frames = frames[start:start+length]
        video_tensor = torch.tensor(np.stack(frames))
        transform = UCF101ClassificationPresetEval(
            crop_size=224,
            is_bcos=True,
        )

        video_tensor = transform(video_tensor)
        video_tensor = video_tensor.unsqueeze(0)
    else:
        video_tensor = vid_tensor

    video_tensor = video_tensor.to(device)

    if video_tensor.grad is not None:
        video_tensor.grad.zero_()

    model, config = load_model_and_config(args)
    model = load_checkpoint(model, args.checkpoint, device=device)

    model.eval()

    logits = model(video_tensor)
    pred_val, pred_idx = logits.max(dim=1)

    print("Predicted class:", pred_idx.item(), idx2label(pred_idx))
    print("Logit value:", pred_val.item())

    expl_out = model.explain_video(video_tensor)
    grad_video = expl_out["explanation"]

    contribs = expl_out["contribution_map"].squeeze(0)

    video_tensor = torch.tensor(np.stack(frames))
    video_tensor = transform(video_tensor)
    frames = video_tensor[:3].permute(1, 2, 3, 0).detach().cpu().numpy()

    for t, frame in enumerate(contribs):
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(os.path.join(args.base_directory, f"contrib{t:03d}.png"), bbox_inches='tight')
        plt.close()

    for t, frame_expl in enumerate(grad_video):
        plt.imshow(frame_expl)
        plt.axis('off')
        plt.savefig(os.path.join(args.base_directory, f"explanation_{t:03d}.png"), bbox_inches='tight')
        plt.close()


def plot_vid(grad_video, frames, frame_scores, save_path=None):
    frames = np.asarray(frames)
    frame_scores = np.asarray(frame_scores).squeeze()
    T = len(frame_scores)

    fig = plt.figure(figsize=(3 * T, 8))
    gs = fig.add_gridspec(3, T, height_ratios=[1,1,1])

    weighted_frames = []
    scores = np.array(frame_scores, dtype=float)

    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    norm_scores = norm_scores ** 0.5

    # weighted greyscale
    for t in range(len(frames)):
        frame = frames[t].astype(float)
        grey = frame.mean(axis=2, keepdims=True)
        grey = (grey - grey.min()) / (grey.max() - grey.min() + 1e-8)
        grey = np.repeat(grey, 3, axis=2)

        frame_out = grey * norm_scores[t]
        weighted_frames.append(frame_out.clip(0, 1))

    # temporal weighted greyscale
    for t in range(T):
        ax_img = fig.add_subplot(gs[0, t])
        ax_img.imshow(weighted_frames[t])
        ax_img.set_title(f"{t}\n{frame_scores[t]:.2f}", fontsize=10)
        ax_img.axis("off")

    # spatial explanation
    for t in range(T):
        ax_img = fig.add_subplot(gs[1, t])
        ax_img.imshow(grad_video[t])
        ax_img.axis("off")

    # original frames
    for t in range(T):
        ax_img = fig.add_subplot(gs[2, t])
        ax_img.imshow(frames[t])
        ax_img.axis("off")

    plt.tight_layout()

    if save_path is not None:
        print("Save path: ", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("show")
        plt.show()


def plot_frame_importance_with_frames(
    grad_video,
    frames,
    frame_scores,
    save_path=None,
    title="Frame importance over time",
):
    frames = np.asarray(frames)
    frame_scores = np.asarray(frame_scores).squeeze()
    T = len(frame_scores)

    fig = plt.figure(figsize=(3 * T, 10))
    gs = fig.add_gridspec(3, T, height_ratios=[2, 1, 1])

    # graph of frame importance
    ax_plot = fig.add_subplot(gs[0, :])
    x = np.arange(T)
    ax_plot.plot(x, frame_scores, marker="o")
    ax_plot.set_xticks(x)
    ax_plot.set_xlabel("Frame index")
    ax_plot.set_ylabel("Importance")
    ax_plot.set_title(title)
    ax_plot.grid(True, alpha=0.3)

    max_idx = int(np.argmax(frame_scores))
    ax_plot.axvline(max_idx, linestyle="--", alpha=0.7)
    ax_plot.scatter([max_idx], [frame_scores[max_idx]], s=80)

    for t in range(T):
        ax_img = fig.add_subplot(gs[1, t])
        ax_img.imshow(grad_video[t])
        ax_img.set_title(f"{t}\n{frame_scores[t]:.2f}", fontsize=10)
        ax_img.axis("off")

        if t == max_idx:
            for spine in ax_img.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
                spine.set_visible(True)

    # original frames
    for t in range(T):
        ax_img = fig.add_subplot(gs[2, t])
        ax_img.imshow(frames[t])
        ax_img.set_title(f"{t}\n{frame_scores[t]:.2f}", fontsize=10)
        ax_img.axis("off")

        if t == max_idx:
            for spine in ax_img.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.tight_layout()

    if save_path is not None:
        print("Save path: ", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    image = args.image_path
    explain_video(args, image)
