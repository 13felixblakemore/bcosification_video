# This file is my contribution

import argparse
from collections import defaultdict
import torch
from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config
from gp_game2 import load_checkpoint, get_loader


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Evaluate Per-Class Accuracy", add_help=add_help)
    parser.add_argument("--base_directory", default="./experiments", help="The base directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt file")
    return parser


def run_accuracy_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)

    if args.checkpoint is not None:
        model = load_checkpoint(model, args.checkpoint, device)

    model.eval()

    loader = get_loader(model_config, batch_size=4)

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    print("Starting")
    num_batches = 100

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(loader):
            if batch_idx > num_batches:
                break

            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            _, preds = torch.max(outputs, 1)

            for label, pred in zip(labels, preds):
                label_id = label.item()
                if label_id == pred.item():
                    class_correct[label_id] += 1
                class_total[label_id] += 1

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{num_batches}")

    print("PER-CLASS ACCURACY:")

    per_class_results = {}
    all_correct = 0
    all_total = 0

    for label_id in sorted(class_total.keys()):
        correct = class_correct[label_id]
        total = class_total[label_id]
        acc = correct / total
        per_class_results[label_id] = acc

        all_correct += correct
        all_total += total

        print(f"Class {label_id:3}: {acc:.4f} ({correct}/{total})")

    print(f"Global Accuracy: {all_correct / all_total:.4f}")

    return per_class_results


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False

    run_accuracy_eval(args)