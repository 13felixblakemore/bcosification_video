import argparse
from bcos.common import BcosUtilMixin
from pathlib import Path
from evaluate import evaluate, load_model_and_config
from PIL import Image
import matplotlib.pyplot as plt
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x  # noqa: E731
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from bcos.data.presets import ImageNetClassificationPresetEval

def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Explain an image", add_help=add_help
    )
    parser.add_argument(
        "--base_directory",
        default="./experiments",
        help="The base directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=["ImageNet", "CIFAR10", "ImageNette"],
        default="ImageNette",
        help="The dataset.",
    )
    parser.add_argument(
        "--base_network", help="The model config or base network to use."
    )
    parser.add_argument("--experiment_name", help="The name of the experiment to run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--reload", help="What ckpt to load. ['last', 'best', 'epoch_<N>', 'best_any']"
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
    model, config = load_model_and_config(args)
    model.eval()

    img = Image.open(image_path)
    transform = ImageNetClassificationPresetEval(
        crop_size=224,
        is_bcos=True,
    )

    img = transform(img)
    img = img[None]

    expl_out = model.explain(img)
    print("Prediction:", idx2label[expl_out["prediction"]])

    plt.imshow(expl_out["explanation"])
    path_to_save = str(args.base_directory, "/explanation.png")

    # Saving the plot
    plt.savefig(path_to_save, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    image = args.image_path
    explain_image(args, image)
