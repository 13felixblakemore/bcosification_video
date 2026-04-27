import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config
from grid import (
    collect_high_confidence_clips,
    sample_unique_class_grid,
    add_second_clip,
)


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Run Grad-CAM++ frame-pointing and grid-pointing games for I3D video models.",
        add_help=add_help,
    )

    parser.add_argument("--base_directory", default="./experiments")
    parser.add_argument("--checkpoint", type=str, default=None)

    # Keep this configurable so you can use standard I3D by default, but override if needed.
    parser.add_argument("--dataset", type=str, default="UCF101")
    parser.add_argument("--base_network", type=str, default="standard")
    parser.add_argument("--experiment_name", type=str, default="i3d")
    parser.add_argument("--reload", type=str, default="last")
    parser.add_argument("--ema", action="store_true")

    # Grad-CAM++ settings
    parser.add_argument("--target_layer", type=str, default="model.blocks.4")

    # Data collection settings
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--max_per_class", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # Game settings
    parser.add_argument("--num_gp_games", type=int, default=500)
    parser.add_argument("--num_fp_games", type=int, default=100)
    parser.add_argument(
        "--game",
        choices=["gp", "fp", "both"],
        default="both",
        help="Which game to run: grid pointing, frame pointing, or both.",
    )
    parser.add_argument(
        "--require_correct_prediction",
        action="store_true",
        help="Only score examples where the model predicts the explained label.",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=43,
        help="Offset used when sampling synthetic games.",
    )

    return parser


# -----------------------------------------------------------------------------
# Model loading: intentionally follows your existing pattern
# -----------------------------------------------------------------------------

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
        print("Loaded checkpoint.")
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))

        if "epoch" in checkpoint:
            print("Checkpoint epoch:", checkpoint["epoch"])

    model.to(device)
    model.eval()
    return model, model_config


# -----------------------------------------------------------------------------
# Layer lookup
# -----------------------------------------------------------------------------

def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())

    if name not in modules:
        print("\nAvailable modules:")
        for k in modules.keys():
            print(k)
        raise ValueError(f"Target layer '{name}' not found.")

    return modules[name]


# -----------------------------------------------------------------------------
# 3D Grad-CAM++
# -----------------------------------------------------------------------------

class GradCAMPlusPlus3D:
    """
    Grad-CAM++ extended from 2D [B,C,H,W] activations to 3D video activations
    [B,C,T,H,W]. The output CAM is [B,T,H,W].
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, target_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_tensor: [1,C,T,H,W]
            target_class: class index to explain

        Returns:
            cam: [T,H,W], non-negative and normalised to [0,1] over the whole video
            logits: [1,num_classes]
        """
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            logits = self.model(input_tensor)
            score = logits[:, target_class].sum()
            score.backward(retain_graph=False)

        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        if activations.ndim != 5:
            raise ValueError(
                f"Expected target layer output [B,C,T,H,W], got {tuple(activations.shape)}. "
                "Choose a 3D convolutional/residual layer before pooling."
            )

        eps = 1e-8

        # Grad-CAM++ alpha coefficients, extended across spatio-temporal positions.
        grads_power_2 = gradients.pow(2)
        grads_power_3 = gradients.pow(3)

        denominator = (
            2.0 * grads_power_2
            + (activations * grads_power_3).sum(dim=(2, 3, 4), keepdim=True)
            + eps
        )
        alpha = grads_power_2 / denominator

        # Channel weights: [B,C,1,1,1]
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3, 4), keepdim=True)

        # Weighted channel combination: [B,T,H,W]
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        # Upsample from activation resolution to input resolution.
        _, _, input_t, input_h, input_w = input_tensor.shape
        cam = cam.unsqueeze(1)  # [B,1,T,H,W]
        cam = F.interpolate(
            cam,
            size=(input_t, input_h, input_w),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)  # [B,T,H,W]

        # Normalise per video. This preserves temporal comparability within a clip.
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + eps)

        return cam.detach()[0], logits.detach()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def energy_in_quadrant(cam: torch.Tensor, target_quadrant: int) -> float:
    """
    Args:
        cam: [T,H,W], non-negative
        target_quadrant: 0 TL, 1 TR, 2 BL, 3 BR
    """
    if cam.ndim != 3:
        raise ValueError(f"Expected cam [T,H,W], got {tuple(cam.shape)}")

    cam = cam.clamp_min(0)
    total = cam.sum().item()
    if total <= 0:
        return 0.0

    T, H, W = cam.shape
    h_mid = H // 2
    w_mid = W // 2

    masks = [
        (slice(None), slice(0, h_mid), slice(0, w_mid)),
        (slice(None), slice(0, h_mid), slice(w_mid, W)),
        (slice(None), slice(h_mid, H), slice(0, w_mid)),
        (slice(None), slice(h_mid, H), slice(w_mid, W)),
    ]

    return (cam[masks[target_quadrant]].sum().item() / total)


def energy_in_frame_range(cam: torch.Tensor, start: int, end: int) -> float:
    """
    Args:
        cam: [T,H,W], non-negative
        start, end: frame interval [start, end)
    """
    if cam.ndim != 3:
        raise ValueError(f"Expected cam [T,H,W], got {tuple(cam.shape)}")

    cam = cam.clamp_min(0)
    total = cam.sum().item()
    if total <= 0:
        return 0.0

    return (cam[start:end].sum().item() / total)


# -----------------------------------------------------------------------------
# Games
# -----------------------------------------------------------------------------

def run_grid_pointing_game(
    model: torch.nn.Module,
    cam_extractor: GradCAMPlusPlus3D,
    clips_by_class: Dict[int, List],
    num_games: int,
    seed_offset: int,
    require_correct_prediction: bool,
) -> Dict[str, float]:
    """
    Builds 2x2 grids of four different classes. For each class logit, measure the
    fraction of Grad-CAM++ energy that falls inside the correct quadrant.
    """
    device = next(model.parameters()).device

    scores: List[float] = []
    considered = 0
    skipped_wrong_pred = 0

    for step in range(num_games):
        if step % 25 == 0:
            print(f"GP game {step}/{num_games}")

        grid_video, grid_labels, confs = sample_unique_class_grid(
            clips_by_class,
            step + seed_offset,
        )

        x = grid_video.to(device).unsqueeze(0)  # [1,C,T,H,W]

        for quadrant, label in enumerate(grid_labels):
            if label == -1:
                continue

            considered += 1
            cam, logits = cam_extractor(x, int(label))
            pred_class = int(logits.argmax(dim=1).item())

            if require_correct_prediction and pred_class != int(label):
                skipped_wrong_pred += 1
                continue

            scores.append(energy_in_quadrant(cam, quadrant))

    avg = float(sum(scores) / len(scores)) if scores else 0.0
    return {
        "gp_energy_score": avg,
        "gp_scored": len(scores),
        "gp_considered": considered,
        "gp_skipped_wrong_pred": skipped_wrong_pred,
    }


def run_frame_pointing_game(
    model: torch.nn.Module,
    cam_extractor: GradCAMPlusPlus3D,
    clips_by_class: Dict[int, List],
    num_games: int,
    seed_offset: int,
    require_correct_prediction: bool,
) -> Dict[str, float]:
    """
    Builds a video by concatenating two clips from different classes. For each
    class logit, measure the fraction of Grad-CAM++ energy in that class's half.
    """
    device = next(model.parameters()).device

    scores: List[float] = []
    considered = 0
    skipped_wrong_pred = 0

    for step in range(num_games):
        if step % 25 == 0:
            print(f"FP game {step}/{num_games}")

        clip, labels = add_second_clip(clips_by_class, step + seed_offset)
        x = clip.to(device).unsqueeze(0)  # [1,C,T,H,W]

        T = x.shape[2]
        mid = T // 2

        for i, label in enumerate(labels):
            if label == -1:
                continue

            considered += 1
            cam, logits = cam_extractor(x, int(label))
            pred_class = int(logits.argmax(dim=1).item())

            if require_correct_prediction and pred_class != int(label):
                skipped_wrong_pred += 1
                continue

            if i == 0:
                start, end = 0, mid
            else:
                start, end = mid, T

            scores.append(energy_in_frame_range(cam, start, end))

    avg = float(sum(scores) / len(scores)) if scores else 0.0
    return {
        "fp_energy_score": avg,
        "fp_scored": len(scores),
        "fp_considered": considered,
        "fp_skipped_wrong_pred": skipped_wrong_pred,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def game(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = load_standard_model(args)
    print(model_config)

    dm = UCF101DataModule(model_config["data"])
    dm.setup("test")

    # Use eval dataset but shuffled, matching the intention in your GP script.
    loader = DataLoader(
        dm.eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    clips_by_class = collect_high_confidence_clips(
        model=model,
        loader=loader,
        device=device,
        confidence_threshold=args.confidence_threshold,
        max_per_class=args.max_per_class,
        max_batches=args.max_batches,
    )

    print("Found high-confidence clips for", len(clips_by_class), "classes")

    target_layer = get_module_by_name(model, args.target_layer)
    print("Using target layer:", args.target_layer)

    cam_extractor = GradCAMPlusPlus3D(model=model, target_layer=target_layer)

    try:
        results = {}

        if args.game in ["gp", "both"]:
            gp_results = run_grid_pointing_game(
                model=model,
                cam_extractor=cam_extractor,
                clips_by_class=clips_by_class,
                num_games=args.num_gp_games,
                seed_offset=args.seed_offset,
                require_correct_prediction=args.require_correct_prediction,
            )
            results.update(gp_results)

        if args.game in ["fp", "both"]:
            fp_results = run_frame_pointing_game(
                model=model,
                cam_extractor=cam_extractor,
                clips_by_class=clips_by_class,
                num_games=args.num_fp_games,
                seed_offset=args.seed_offset,
                require_correct_prediction=args.require_correct_prediction,
            )
            results.update(fp_results)

        print("\n=== Grad-CAM++ Pointing Game Results ===")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    finally:
        cam_extractor.remove_hooks()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    game(args)
