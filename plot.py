from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

# My contribution

# Plot multiple tensorboard scalars from a single run
def plot_multiple_scalars(log_dir, tags):
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

        plt.plot(steps, values, label=tag)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Scalars")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scalars.png", bbox_inches="tight", dpi=300)
    plt.close()


# Plot a scalar over different runs
def compare_scalar_between_runs(log_dirs,
                                labels,
                                tag="val_acc1",
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

        # filter to max step
        filtered = [(s, val) for s, val in zip(steps, values) if s <= 76000]
        if not filtered:
            continue

        steps, values = zip(*filtered)

        plt.plot(steps, values, label=label)

    plt.xlabel("Step")
    plt.ylabel("Top 1 Val Acc")
    plt.title("Comparing B")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def print_all_tensorboard_tags(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()

    print("TensorBoard Tags:\n")
    for tag_type, tag_list in tags.items():
        print(f"{tag_type.upper()}:")
        for tag in tag_list:
            print(tag)
        print()

# usage
log_dir = "tb_logs/experiments/UCF101/standard/i3d/i3d/standard"
log_dir_2 = "tb_logs/experiments/UCF101/bcosification/i3d/i3d/version_7"

logs = ["21","19","20","18"]
log_dirs = ["tb_logs/experiments/UCF101/bcosification/i3d/i3d/version_" + version for version in logs]
tags = ["val_acc1"]
labels = ["b=1", "b=1.5", "b=2", "b=3"]
compare_scalar_between_runs(log_dirs, labels)