from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os


def plot_multiple_scalars(log_dir, tags, smoothing=0.0):
    plt.figure(figsize=(8, 5))

    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        if smoothing > 0:
            smoothed = []
            last = values[0]
            for v in values:
                last = smoothing * last + (1 - smoothing) * v
                smoothed.append(last)
            values = smoothed

        plt.plot(steps, values, label=tag)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("TensorBoard Scalars")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot.png", bbox_inches="tight", dpi=300)
    plt.close()


def compare_scalar_between_runs(log_dirs,
                                labels,
                                tag="val_acc1",
                                smoothing=0.0,
                                save_path="comparison_plot.png"):

    plt.figure(figsize=(8, 5))

    for log_dir, label in zip(log_dirs, labels):

        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        events = ea.Scalars(tag)

        steps = [e.step for e in events]
        values = [e.value for e in events]

        # 🔑 Filter to max_epoch
        filtered = [(s, val) for s, val in zip(steps, values) if ep <= 76000]
        if not filtered:
            continue

        epochs, values = zip(*filtered)

        if smoothing > 0:
            smoothed = []
            last = values[0]
            for v in values:
                last = smoothing * last + (1 - smoothing) * v
                smoothed.append(last)
            values = smoothed

        plt.plot(epochs, values, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Top 1 Val Acc")
    plt.title(f"Comparing B")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def print_all_tensorboard_tags(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()

    print("\n=== TensorBoard Tags ===\n")
    for tag_type, tag_list in tags.items():
        print(f"{tag_type.upper()}:")
        for tag in tag_list:
            print(f"  - {tag}")
        print()

log_dir = "tb_logs/experiments/UCF101/standard/i3d/i3d/standard"
log_dir_2 = "tb_logs/experiments/UCF101/bcosification/i3d/i3d/version_7"

logs = ["21","19","20","18"]
log_dirs = ["tb_logs/experiments/UCF101/bcosification/i3d/i3d/version_" + version for version in logs]
tags = ["val_acc1"]
labels = ["b=1", "b=1.5", "b=2", "b=3"]
compare_scalar_between_runs(log_dirs, labels)