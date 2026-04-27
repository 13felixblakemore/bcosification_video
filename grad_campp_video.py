import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from evaluate import load_model_and_config
from pytorch_grad_cam.utils.image import show_cam_on_image


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


# -----------------------------
# DO NOT CHANGE MODEL LOADING
# -----------------------------
def load_standard_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = load_model_and_config(args)

    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.model.model.", "model.model.")
            new_state_dict[new_key] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()
    return model, model_config


def get_module_by_name(model, name):
    modules = dict(model.named_modules())

    if name not in modules:
        print("\nAvailable modules:")
        for k in modules.keys():
            print(k)
        raise ValueError(f"Target layer '{name}' not found.")

    return modules[name]


def read_video(video_path, num_frames=8, crop_size=224):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)

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

    frames_np = np.stack(frames).astype(np.float32) / 255.0  # [T,H,W,3]

    x = torch.from_numpy(frames_np).permute(3, 0, 1, 2)  # [C,T,H,W]

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]

    x = (x - mean) / std
    x = x.unsqueeze(0)  # [1,C,T,H,W]

    return x, frames_np


class GradCAMPlusPlus3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        if activations.ndim != 5:
            raise ValueError(
                f"Expected 5D target activations [B,C,T,H,W], got {activations.shape}"
            )

        # [B,C,T,H,W]
        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3

        eps = 1e-8

        # Grad-CAM++ alpha coefficients
        denominator = (
            2 * grads_power_2
            + (activations * grads_power_3).sum(dim=(2, 3, 4), keepdim=True)
            + eps
        )

        alpha = grads_power_2 / denominator

        positive_gradients = F.relu(gradients)

        # Channel weights: [B,C,1,1,1]
        weights = (alpha * positive_gradients).sum(dim=(2, 3, 4), keepdim=True)

        # Weighted combination over channels -> [B,T,H,W]
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        # Upsample from target-layer resolution to input video resolution
        # cam: [B,T,H,W] -> [B,1,T,H,W]
        cam = cam.unsqueeze(1)

        _, _, input_t, input_h, input_w = input_tensor.shape

        cam = F.interpolate(
            cam,
            size=(input_t, input_h, input_w),
            mode="trilinear",
            align_corners=False,
        )

        cam = cam.squeeze(1)  # [B,T,H,W]

        # Normalize per video
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + eps)

        return cam.detach().cpu().numpy()[0], logits.detach(), target_class


def save_gradcam_figure(frames, grayscale_cam, save_path):
    """
    frames: [T,H,W,3], float in [0,1]
    grayscale_cam: [T,H,W], float in [0,1]
    """

    T = frames.shape[0]

    if grayscale_cam.shape[0] != T:
        raise ValueError(f"CAM has {grayscale_cam.shape[0]} frames but video has {T}.")

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

    print(f"Saved Grad-CAM++ figure to {save_path}")


def main():
    args = get_parser().parse_args()

    args.dataset = "UCF101"
    args.base_network = "standard"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = load_standard_model(args)
    print(model_config)

    x, frames = read_video(
        args.video_path,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
    )
    x = x.to(device)

    target_layer = get_module_by_name(model, args.target_layer)

    cam_extractor = GradCAMPlusPlus3D(
        model=model,
        target_layer=target_layer,
    )

    try:
        grayscale_cam, logits, used_target_class = cam_extractor(
            input_tensor=x,
            target_class=args.target_class,
        )

        pred_class = int(logits.argmax(dim=1).item())

        print("Predicted class:", pred_class)
        print("Target class:", used_target_class)
        print("CAM shape:", grayscale_cam.shape)

        save_gradcam_figure(frames, grayscale_cam, args.save_path)

    finally:
        cam_extractor.remove_hooks()


if __name__ == "__main__":
    main()