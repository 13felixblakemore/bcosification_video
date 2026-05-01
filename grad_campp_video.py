# this is taken from

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from evaluate import load_model_and_config
from pytorch_grad_cam.utils.image import show_cam_on_image
from gp_game import load_checkpoint

class GradCAMPlusPlus3D(GradCAMPlusPlus):
    def get_target_width_height(self, input_tensor):
        return (
            input_tensor.size(-1),  # W
            input_tensor.size(-2),  # H
            input_tensor.size(-3),  # T
        )

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", default="./experiments")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--target_class", type=int, default=None)
    parser.add_argument("--target_layer", type=str, default="model.blocks.4")
    parser.add_argument("--save_path", type=str, default="gradcampp_video.png")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=224)
    return parser

def load_standard_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = load_model_and_config(args)
    model = load_checkpoint(model, args.checkpoint, device)

    model.to(device)
    model.eval()
    return model, model_config


def get_module_by_name(model, name):
    modules = dict(model.named_modules())

    if name not in modules:
        print("Available modules:")
        for k in modules.keys():
            print(k)
        raise ValueError(f"Target layer '{name}' not found.")

    return modules[name]


def read_video(video_path, num_frames=8, crop_size=224):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    start_idx = 16
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.arange(start_idx, start_idx + num_frames)


    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, _ = frame.shape
        scale = 256 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h))

        y0 = (new_h - crop_size) // 2
        x0 = (new_w - crop_size) // 2
        frame = frame[y0:y0 + crop_size, x0:x0 + crop_size]

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames were read.")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames_np = np.stack(frames).astype(np.float32) / 255.0

    x = torch.from_numpy(frames_np).permute(3, 0, 1, 2)

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]

    x = (x - mean) / std
    x = x.unsqueeze(0)  # [1,C,T,H,W]

    return x, frames_np


def run_library_gradcampp(model, input_tensor, target_layer, target_class=None):
    if target_class is None:
        targets = None
    else:
        targets = [ClassifierOutputTarget(target_class)]

    with GradCAMPlusPlus3D(
        model=model,
        target_layers=[target_layer],
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
        )

        logits = cam.outputs

    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    return grayscale_cam[0], logits.detach(), target_class


def save_gradcam_figure(frames, grayscale_cam, save_path):
    T = frames.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(2.2 * T, 4.5))

    if T == 1:
        axes = axes[:, None]

    for t in range(T):
        axes[0, t].imshow(frames[t])
        axes[0, t].set_title(f"Frame {t}")
        axes[0, t].axis("off")

        cam_t = grayscale_cam[t]
        cam_t = cv2.resize(cam_t, (frames.shape[2], frames.shape[1]))

        overlay = show_cam_on_image(
            frames[t],
            cam_t,
            use_rgb=True,
            image_weight=0.55,
        )

        axes[1, t].imshow(overlay)
        axes[1, t].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Grad-CAM++")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = get_parser().parse_args()
    args.dataset = "UCF101"
    args.base_network = "standard"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = load_standard_model(args)

    x, frames = read_video(
        args.video_path,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
    )
    x = x.to(device)

    target_layer = get_module_by_name(model, args.target_layer)

    grayscale_cam, logits, used_target_class = run_library_gradcampp(
        model=model,
        input_tensor=x,
        target_layer=target_layer,
        target_class=args.target_class,
    )

    pred_class = int(logits.argmax(dim=1).item())

    print("Predicted class:", pred_class)
    print("Target class:", used_target_class)
    print("CAM shape:", grayscale_cam.shape)

    save_gradcam_figure(frames, grayscale_cam, args.save_path)


if __name__ == "__main__":
    main()