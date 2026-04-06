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

"""
SCALARS:
  - hp_metric
  - lr-AdamW
  - train_loss
  - train_acc1_step
  - train_acc5_step
  - epoch
  - val_loss
  - val_acc1
  - val_acc5
  - train_acc1_epoch
  - train_acc5_epoch
"""


log_dir = "tb_logs/experiments/UCF101/standard/i3d/i3d/standard"
tags = ["train_acc1_step"]

plot_multiple_scalars(log_dir, tags)