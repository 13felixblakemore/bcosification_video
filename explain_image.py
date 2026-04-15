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
    #print("Prediction:", idx2label[expl_out["prediction"]])

    plt.imshow(expl_out["explanation"])
    path_to_save = os.path.join(args.base_directory, "explanation.png")

    # Saving the plot
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()

def debug(model, video_tensor):
    """
    Test Euler reconstruction on the first B-Cos conv only.

    Checks whether:
        f(x) = sum(stem.conv(x))
    satisfies:
        f(x) == <x, grad_x f>

    Parameters
    ----------
    model : nn.Module
        Your loaded B-Cosified I3D model.
    video_tensor : torch.Tensor
        Input video tensor of shape [1, C, T, H, W]
    """
    model.eval()

    # Fresh leaf tensor with grad enabled
    x = video_tensor.detach().clone().requires_grad_(True)

    print(len(model.model.blocks))
    # Get first block and first conv
    block_0 = model.model.blocks[0]
    block_1 = model.model.blocks[1]
    block_2 = model.model.blocks[2]
    block_3 = model.model.blocks[3]
    block_4 = model.model.blocks[4]
    block_5 = model.model.blocks[5]
    block_6 = model.model.blocks[6]

    for name, module in model.named_children():
        print(name, "->", module.__class__.__name__)

    # Forward through only the first conv
    y = model.bcosifynormalize(x)
    f = y.sum()                  # scalar

    # Backward
    if x.grad is not None:
        x.grad.zero_()
    f.backward()

    grad = x.grad.detach().clone()
    recon = (x * grad).sum()

    print("=== Debug: stem conv only ===")
    print("input shape: ", x.shape)
    print("conv output shape:", y.shape)
    print("scalar f = sum(conv(x)):", f.item())
    print("reconstructed <x, grad>:", recon.item())
    print("difference f - recon:", (f - recon).item())
    print("max abs grad:", grad.abs().max().item())

    import inspect
    print(inspect.getsource(model.bcosifynormalize.forward))

    return {
        "scalar": f.detach(),
        "reconstruction": recon.detach(),
        "difference": (f - recon).detach(),
        "grad": grad,
        "output": y.detach(),
    }

def explain_video(args, video_path):
    global device
    if args.no_cuda:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

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
    length = 5
    frames = frames[start:start+length]

    transform = UCF101ClassificationPresetEval(
        crop_size=224,
        is_bcos=True,
    )
    # choose which frame to keep
    t_keep = [0,1,2,3,4]  # change this manually each run

    # replacement (use mean frame for stability)
    black = np.zeros_like(frames[0])

    new_frames = []
    for i in range(len(frames)):
        if i in t_keep:
            new_frames.append(frames[i])  # real frame
        else:
            new_frames.append(black)  # replace others

    frames = new_frames

    og = torch.tensor(np.stack(frames))  # [T,H,W,C]
    video_tensor = transform(og)
    video_tensor = video_tensor.to(device)
    video_tensor = video_tensor.unsqueeze(0)
    if video_tensor.grad is not None:
        video_tensor.grad.zero_()

    model, config = load_model_and_config(args)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(name, module)
            break

    #module = model.model.conv1.weight
    #print(module.shape)
    #symmetry = (module[:, :3] + module[:, 3:]).abs().mean()
    #print(symmetry)
    #sys.exit()
    model.eval()

    logits = model(video_tensor)
    norm_video_tensor = model.bcosifynormalize(video_tensor)
    pred_val, pred_idx = logits.max(dim=1)

    print("Predicted class:", pred_idx.item(), idx2label(pred_idx))
    print("Logit value:", pred_val.item())
    logits = logits[0]


    expl_out = model.explain_video(norm_video_tensor)
    print("Prediction:", idx2label(expl_out["prediction"]))
    print(pred_idx.item(), expl_out["prediction"])

    target_logit = logits[expl_out["prediction"]]

    reconstructed_logit = expl_out["reconstructed_logit"]

    linear_map = expl_out["dynamic_linear_weights"]
    lm_logit = (video_tensor * linear_map).sum(dim=(1,2,3,4))
    print(f"Actual vs reconstructed: {target_logit} ({lm_logit})")

    print("TESTING------------------------")
    print("pred_idx:", pred_idx.item())
    print("expl_pred:", expl_out["prediction"])
    print("video shape:", video_tensor.shape)
    print("lm shape:", linear_map.shape)

    lm_logit = (video_tensor * linear_map).sum(dim=(1, 2, 3, 4))
    print("target_logit:", target_logit.item())
    print("lm_logit:", lm_logit.item())
    print("reconstructed_logit:", reconstructed_logit.item())
    print("difference:", (target_logit - lm_logit).item())

    debug_out = debug(model, video_tensor)

    frame_scores = expl_out["frame_scores"]
    frame_path = os.path.join(args.base_directory, f"temporal_explanation.png")
    plot_frame_importance_with_frames(frames, frame_scores, frame_path)

    contribs = expl_out["contribution_map"].squeeze(0)

    for t, frame in enumerate(contribs):
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(os.path.join(args.base_directory, f"contrib{t:03d}.png"), bbox_inches='tight')
        plt.close()

    grad_video = expl_out["explanation"]
    for t, frame_expl in enumerate(grad_video):
        plt.imshow(frame_expl)
        plt.axis('off')
        plt.savefig(os.path.join(args.base_directory, f"explanation_{t:03d}.png"), bbox_inches='tight')
        plt.close()

def plot_frame_importance_with_frames(
    frames,
    frame_scores,
    save_path=None,
    title="Frame importance over time",
):
    """
    frames: array-like of shape [T, H, W, 3]
    frame_scores: array-like of shape [T]
    """
    frames = np.asarray(frames)
    frame_scores = np.asarray(frame_scores).squeeze()
    T = len(frame_scores)

    fig = plt.figure(figsize=(2 * T, 5))
    gs = fig.add_gridspec(2, T, height_ratios=[2, 1])

    # Top plot
    ax_plot = fig.add_subplot(gs[0, :])
    x = np.arange(T)
    ax_plot.plot(x, frame_scores, marker="o")
    ax_plot.set_xticks(x)
    ax_plot.set_xlabel("Frame index")
    ax_plot.set_ylabel("Importance")
    ax_plot.set_title(title)
    ax_plot.grid(True, alpha=0.3)

    # Highlight max frame
    max_idx = int(np.argmax(frame_scores))
    ax_plot.axvline(max_idx, linestyle="--", alpha=0.7)
    ax_plot.scatter([max_idx], [frame_scores[max_idx]], s=80)

    # Bottom row: frames
    for t in range(T):
        ax_img = fig.add_subplot(gs[1, t])
        ax_img.imshow(frames[t])
        ax_img.set_title(f"{t}\n{frame_scores[t]:.2f}", fontsize=10)
        ax_img.axis("off")

        # Highlight most important frame
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
        print("show")
        plt.show()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    image = args.image_path
    explain_video(args, image)
