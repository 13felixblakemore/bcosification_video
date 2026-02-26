import argparse

import cv2
import numpy as np

from bcos.common import get_inx2label_imagenette as idx2label
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
    model.eval()

    expl_out = model.explain(img)
    #print("Prediction:", idx2label[expl_out["prediction"]])

    plt.imshow(expl_out["explanation"])
    path_to_save = os.path.join(args.base_directory, "explanation.png")

    # Saving the plot
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()

def explain_video(args, video_path):
    global device
    if args.no_cuda:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    transform = UCF101ClassificationPresetEval(
        crop_size=224,
        is_bcos=True,
    )

    # 3. Stack frames → [T, C, H, W] and batch dim → [1, T, C, H, W]
    video_tensor = torch.tensor(np.stack(frames)) # T, H, W, C
    print(video_tensor.shape)
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
    print(video_tensor.shape)
    video_tensor = video_tensor.unsqueeze(0)  # [1, T, C, H, W]
    print(video_tensor.shape)

    # If model expects [B, C, T, H, W]:
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)

    video_tensor = transform(video_tensor)
    video_tensor = video_tensor.to(device)

    model, config = load_model_and_config(args)
    model.eval()

    expl_out = model.explain_video(video_tensor)
    print("Prediction:", idx2label[expl_out["prediction"]])

    grad_video = expl_out["explanation"]  # list of [H,W,4] or array [T,H,W,4]
    for t, frame_expl in enumerate(grad_video):
        plt.imshow(frame_expl)
        plt.axis('off')
        plt.savefig(os.path.join(args.base_directory, f"explanation_{t:03d}.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    image = args.image_path
    explain_video(args, image)
