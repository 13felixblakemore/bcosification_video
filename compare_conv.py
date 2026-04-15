import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bcos.experiments.utils import Experiment
# CHANGE THIS IMPORT TO MATCH YOUR PROJECT
# Example:
from bcos.modules.bcosconv3d import BcosConv3d
from bcos.modules.bcosifyconv3d import BcosifyConv3d


def load_video_rgb(
    video_path: str,
    num_frames: int = 16,
    resize_hw: tuple[int, int] = (224, 224),
    device: str = "cuda",
) -> torch.Tensor:
    """
    Load a video as a float tensor in [0,1], shape [1, 3, T, H, W].
    Frames are uniformly sampled over the whole video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")

    frames = np.stack(frames, axis=0)  # [N, H, W, 3]
    total_frames = frames.shape[0]

    if total_frames >= num_frames:
        idx = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = frames[idx]
    else:
        # Repeat last frame if too short
        pad_count = num_frames - total_frames
        pad = np.repeat(frames[-1: ], pad_count, axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    frames = torch.from_numpy(frames).float() / 255.0  # [T, H, W, 3]
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)   # [1, 3, T, H, W]

    # Resize spatial dims only
    b, c, t, h, w = frames.shape
    frames_2d = frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    frames_2d = F.interpolate(
        frames_2d,
        size=resize_hw,
        mode="bilinear",
        align_corners=False,
    )
    frames = frames_2d.reshape(b, t, c, resize_hw[0], resize_hw[1]).permute(0, 2, 1, 3, 4)

    return frames


def add_inverse_channels(x_rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, 3, T, H, W] -> [B, 6, T, H, W] as [R,G,B,1-R,1-G,1-B].
    """
    if x_rgb.ndim != 5 or x_rgb.shape[1] != 3:
        raise ValueError(f"Expected [B,3,T,H,W], got {tuple(x_rgb.shape)}")
    return torch.cat([x_rgb, 1.0 - x_rgb], dim=1)


def make_regular_conv(device: str = "cuda") -> nn.Conv3d:
    """
    Example stem-like Conv3d. Adjust params to match your I3D stem if needed.
    """
    conv = nn.Conv3d(
        in_channels=3,
        out_channels=64,
        kernel_size=(5, 7, 7),
        stride=(1, 2, 2),
        padding=(2, 3, 3),
        bias=False,
    ).to(device)
    return conv


def copy_regular_to_bcos_stem(conv3d: nn.Conv3d) -> BcosifyConv3d:
    """
    Create a BcosifyConv3d from a regular Conv3d.

    IMPORTANT:
    You may need to adapt this depending on your class signature.
    Common possibilities:
      - BcosifyConv3d(conv3d, b=1)
      - BcosifyConv3d(conv=conv3d, b=1)
      - BcosifyConv3d.from_conv(conv3d, b=1)
    """
    exp = Experiment("UCF101", "bcosification", "i3d")
    config = exp.config.copy()
    config["b"] = 1
    print(config)
    # Try your likely constructor here:
    bcos_conv = BcosifyConv3d.from_standard_module(conv3d, config["model"])

    # If your implementation does NOT automatically expand 3->6 channels,
    # uncomment this manual conversion block and adjust parameter access.
    #
    with torch.no_grad():
         w = conv3d.weight.detach().clone()  # [out, 3, kt, kh, kw]
         w6 = torch.cat([w, -w], dim=1) / 2  # [out, 6, kt, kh, kw]

         # Example if the wrapped inner conv lives at .linear:
         bcos_conv.linear.in_channels = 6
         bcos_conv.linear.weight = nn.Parameter(w6)

    return bcos_conv


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a - b).abs()
    denom = a.abs().mean().clamp_min(1e-12)

    print(f"\n{name}")
    print(f"  shape a: {tuple(a.shape)}")
    print(f"  shape b: {tuple(b.shape)}")
    print(f"  mean(|a-b|): {diff.mean().item():.8f}")
    print(f"  max(|a-b|):  {diff.max().item():.8f}")
    print(f"  rel mean diff: {(diff.mean() / denom).item():.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--media", type=str, default="video")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA requested but not available.")

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    if args.media == "video":
        x_rgb = load_video_rgb(
            str(path)
        )
        x_6 = add_inverse_channels(x_rgb)

    print("Input RGB shape:", tuple(x_rgb.shape))
    print("Input 6ch shape:", tuple(x_6.shape))
    print("RGB min/max:", x_rgb.min().item(), x_rgb.max().item())
    print("6ch min/max:", x_6.min().item(), x_6.max().item())

    conv = make_regular_conv(device=device).eval()
    bcos_conv = copy_regular_to_bcos_stem(conv).eval()

    with torch.no_grad():
        y_regular = conv(x_rgb)
        y_bcos = bcos_conv(x_6)

    compare_tensors("regular conv vs bcos conv", y_regular, y_bcos)

    # Optional: compare first few scalar values to inspect sign/scale behaviour
    flat_reg = y_regular.flatten()
    flat_bcos = y_bcos.flatten()

    print("\nFirst 10 values:")
    for i in range(min(10, flat_reg.numel(), flat_bcos.numel())):
        print(
            f"{i:02d} | regular={flat_reg[i].item(): .8f} "
            f"| bcos={flat_bcos[i].item(): .8f} "
            f"| diff={abs(flat_reg[i] - flat_bcos[i]).item(): .8f}"
        )


if __name__ == "__main__":
    main()