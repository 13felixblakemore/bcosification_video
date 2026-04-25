import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config


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
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific Lightning .ckpt file to load"
    )
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)

    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")

        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Handle Lightning checkpoints
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 🔧 Fix key mismatches (VERY important for your setup)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k

            # Common prefix issues in your repo
            new_key = new_key.replace("model.model.model.", "model.model.")
            #new_key = new_key.replace("model.model.", "model.")

            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    print(model_config)
    dm = UCF101DataModule(model_config["data"])

    dm.setup("test")

    loader = DataLoader(
        dm.eval_dataset,
        batch_size=1,
        shuffle=True,  # ✅ force shuffle
        num_workers=4,  # match your config if needed
        pin_memory=True
    )

    batches = 100
    faithfulness_list = check_faithfulness(model, loader, args, batches)
    faithfulness_list = np.array(faithfulness_list)

    mean_error = faithfulness_list.mean()
    std_error = faithfulness_list.std()

    faithfulness = 1 - mean_error

    print(f"Faithfulness: {faithfulness:.4f}")
    print(f"Mean error: {mean_error:.4f} ± {std_error:.4f}")

def check_faithfulness(model, loader, args, batch_lim):
    device = next(model.parameters()).device

    faithfulness = []

    for batch_idx, (videos, labels) in enumerate(loader):
        print(batch_idx)
        videos = videos.to(device)
        model.zero_grad(set_to_none=True)
        if batch_idx == batch_lim:
            break
        with torch.enable_grad(), model.explanation_mode():
            x = videos.clone().detach().requires_grad_(True)
            out = model.model(x)
            pred_out = out.max(1)

            to_be_explained_logit = pred_out.values
            print("Explaining logits: ", to_be_explained_logit)
            to_be_explained_logit.backward(inputs=[x])

        grads = x.grad.detach().clone()
        print("grads: ", grads.shape)
        reconstructed_logits = (x * grads).sum(dim=(1, 2, 3, 4)).detach().clone()
        print("Reconstructed logits: ", reconstructed_logits)
        # compare error between reconstructed logit and actual logit
        error = (abs(reconstructed_logits) - abs(to_be_explained_logit)) / abs(reconstructed_logits)
        faithfulness.append(error.detach().cpu())
    return faithfulness


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False
    main(args)