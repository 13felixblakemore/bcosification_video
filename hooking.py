import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch

from evaluate import load_model_and_config
from bcos.data.presets import UCF101ClassificationPresetEval
from explain_image import get_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_video_clip(
    video_path: str,
    start: int = 16,
    length: int = 10,
    crop_size: int = 224,
) -> torch.Tensor:
    """
    Load a video clip and return a tensor of shape [1, C, T, H, W].
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

    if len(frames) < start + length:
        raise ValueError(
            f"Video only has {len(frames)} frames, but need at least {start + length}."
        )

    frames = frames[start:start + length]

    transform = UCF101ClassificationPresetEval(
        crop_size=crop_size,
        is_bcos=True,
    )

    # [T, H, W, C]
    video_tensor = torch.tensor(np.stack(frames))
    video_tensor = transform(video_tensor)  # expected -> [C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, C, T, H, W]

    return video_tensor


def register_block_hooks(model: torch.nn.Module) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[Any]]:
    """
    Register forward hooks on model.blocks and save each block's input/output.
    """
    saved: Dict[str, Dict[str, torch.Tensor]] = {}
    handles: List[Any] = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            # Most modules take one tensor input
            x_in = inputs[0]
            saved[name] = {
                "input": x_in.detach().clone(),
                "output": output.detach().clone(),
            }
            print(f"{name}: {tuple(x_in.shape)} -> {tuple(output.shape)}")
        return hook

    if not hasattr(model, "blocks"):
        raise AttributeError("Model does not have attribute 'blocks'.")

    for i, block in enumerate(model.blocks):
        handle = block.register_forward_hook(make_hook(f"block_{i}"))
        handles.append(handle)

    return saved, handles


def register_named_hooks_by_type(
    model: torch.nn.Module,
    type_name: str,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[Any]]:
    """
    Optional helper: hook every module of a given class name, e.g. 'BatchNormUncentered3d'.
    """
    saved: Dict[str, Dict[str, torch.Tensor]] = {}
    handles: List[Any] = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            x_in = inputs[0]
            saved[name] = {
                "input": x_in.detach().clone(),
                "output": output.detach().clone(),
            }
            print(f"{name} ({type(module).__name__}): {tuple(x_in.shape)} -> {tuple(output.shape)}")
        return hook

    for name, module in model.named_modules():
        if type(module).__name__ == type_name:
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    return saved, handles


def remove_hooks(handles: List[Any]) -> None:
    for handle in handles:
        handle.remove()


def print_saved_summary(saved: Dict[str, Dict[str, torch.Tensor]]) -> None:
    print("\nSaved activations summary:")
    for name, tensors in saved.items():
        x_in = tensors["input"]
        x_out = tensors["output"]
        print(
            f"{name}: "
            f"in={tuple(x_in.shape)} "
            f"out={tuple(x_out.shape)} "
            f"in_mean={x_in.mean().item():.4f} "
            f"out_mean={x_out.mean().item():.4f}"
        )


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    global device
    if args.no_cuda:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    # Replace with however you normally load your model/config
    model, config = load_model_and_config(args)
    model = model.to(device)
    model.eval()

    video_tensor = load_video_clip(
        video_path=args.image_path
    )

    print("Input video tensor shape:", tuple(video_tensor.shape))

    # Hook top-level blocks
    saved_blocks, block_handles = register_block_hooks(model.model)

    # Optional: hook all BNUncentered layers too
    bn_handles = []
    saved_bn = {}
    if args.hook_bn:
        saved_bn, bn_handles = register_named_hooks_by_type(model, "BatchNormUncentered3d")

    with torch.no_grad():
        logits = model(video_tensor)

    print("\nLogits shape:", tuple(logits.shape))
    print("Pred class:", logits.argmax(dim=1).item())
    print("Pred logit:", logits.max(dim=1).values.item())

    remove_hooks(block_handles)
    remove_hooks(bn_handles)

    print_saved_summary(saved_blocks)

    if args.hook_bn:
        print("\nSaved BatchNormUncentered3d activations summary:")
        print_saved_summary(saved_bn)

    # Optional: save activations to disk for later debugging
    out_path = Path("debug_saved_activations.pt")
    torch.save(
        {
            "blocks": saved_blocks,
            "bn": saved_bn,
            "logits": logits.detach().cpu(),
        },
        out_path,
    )
    print(f"\nSaved activations to {out_path.resolve()}")


if __name__ == "__main__":
    main()