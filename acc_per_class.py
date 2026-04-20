import argparse
from collections import defaultdict
import torch
from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Evaluate Per-Class Accuracy", add_help=add_help)
    parser.add_argument("--base_directory", default="./experiments", help="The base directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt file")
    return parser


def run_accuracy_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model and Config (Same as your script)
    model, model_config = load_model_and_config(args)

    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 🔧 Apply your specific key fixes
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.model.model.", "model.model.")
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=False)
        print("Loaded checkpoint.")

    model.to(device)
    model.eval()

    # 2. Setup Data
    dm = UCF101DataModule(model_config["data"])
    dm.setup("test")
    loader = dm.test_dataloader()

    # Trackers: {class_id: count}
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    print("Starting evaluation...")
    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(loader):
            videos, labels = videos.to(device), labels.to(device)

            # Forward pass
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)

            # Update per-class stats
            for label, pred in zip(labels, preds):
                label_id = label.item()
                if label_id == pred.item():
                    class_correct[label_id] += 1
                class_total[label_id] += 1

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{num_batches}")

    # 3. Report Results
    print("\n" + "=" * 30)
    print("PER-CLASS ACCURACY")
    print("=" * 30)

    per_class_results = {}
    all_correct = 0
    all_total = 0

    # Sort by class ID for clean output
    for label_id in sorted(class_total.keys()):
        correct = class_correct[label_id]
        total = class_total[label_id]
        acc = correct / total
        per_class_results[label_id] = acc

        all_correct += correct
        all_total += total

        print(f"Class {label_id:3}: {acc:.4f} ({correct}/{total})")

    print("=" * 30)
    print(f"Global Accuracy: {all_correct / all_total:.4f}")
    print("=" * 30)

    return per_class_results


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Matching your experiment setup
    args.dataset = "UCF101"
    args.base_network = "bcosification"
    args.experiment_name = "i3d"
    args.reload = "last"
    args.ema = False

    run_accuracy_eval(args)