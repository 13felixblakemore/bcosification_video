import argparse
import sys

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

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        print("Loaded checkpoint.")
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))

        print("\nMODEL block 5:")
        for k in model.state_dict().keys():
            if "blocks.5" in k:
                print(k)

        print("\nMODEL block 6:")
        for k in model.state_dict().keys():
            if "blocks.6" in k:
                print(k)

        print("\nCHECKPOINT block 5:")
        for k in new_state_dict.keys():
            if "blocks.5" in k:
                print(k)

        print("\nCHECKPOINT block 6:")
        for k in new_state_dict.keys():
            if "blocks.6" in k:
                print(k)

        filtered_state_dict = {
            k: v for k, v in new_state_dict.items()
            if k.startswith("model.model.")
        }

        model.load_state_dict(filtered_state_dict, strict=False)

        # Optional debug
        if "epoch" in checkpoint:
            print("Checkpoint epoch:", checkpoint["epoch"])
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

    batches = 10
    faithfulness = check_faithfulness(model, loader, args, batches)
    print("Faithfulness: ", faithfulness)

def check_faithfulness(model, loader, args, batch_lim):
    device = next(model.parameters()).device

    for batch_idx, (videos, labels) in enumerate(loader):
        videos = videos.to(device)
        model.zero_grad(set_to_none=True)
        if batch_idx >= batch_lim:
            with torch.enable_grad(), model.explanation_mode():
                x = videos.clone().detach().requires_grad_(True)
                out = model(x)
                pred_out = out.max(1)

                to_be_explained_logit = pred_out.values
                print("Explaining logits: ", to_be_explained_logit)
                to_be_explained_logit.backward(inputs=[x])

            grads = x.grad.detach().clone()
            print("grads: ", grads.shape)
            reconstructed_logits = (x * grads).sum(dim=(1, 2, 3, 4)).detach().clone()
            print("Reconstructed logits: ", reconstructed_logits)
            sys.exit()
            # compare error between reconstructed logit and actual logit
    faithfulness = 0
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