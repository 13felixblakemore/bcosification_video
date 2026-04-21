import argparse
from collections import defaultdict
import torch
import torch.nn.functional as F

from bcos.data.datamodules import UCF101DataModule
from evaluate import load_model_and_config
from grid import collect_high_confidence_clips, add_second_clip


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Joint Temporal Pointing Game", add_help=add_help)
    parser.add_argument("--base_directory", default="./experiments")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser


def game(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model_and_config(args)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {k.replace("model.model.model.", "model.model."): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded.")

    model.eval()
    dm = UCF101DataModule(model_config["data"])
    dm.setup("test")
    loader = dm.test_dataloader()

    # 1. Collect clips the model actually knows (important for 3D B-CoS)
    print("Collecting high-confidence clips...")
    clips_by_class = collect_high_confidence_clips(
        model=model,
        loader=loader,
        device=device,
        confidence_threshold=0.7,
        max_per_class=5,
        max_batches=200,
    )

    # 2. Tracking: {class_id: [localization_ratios]}
    class_localization_scores = defaultdict(list)

    # 3. Run the Joint Game
    num_steps = 100
    for step in range(num_steps):
        # add_second_clip returns a video and a list/tuple of [label_first_half, label_second_half]
        clip, labels = add_second_clip(clips_by_class, step)

        # Calculate scores for both actions in the same video
        joint_results = explain_joint(model, clip, labels)

        for label, ratio in joint_results:
            class_localization_scores[label].append(ratio)

        if step % 10 == 0:
            print(f"Step {step}/{num_steps} processed.")

    # 4. Final Per-Class Analysis
    print("\n" + "=" * 40)
    print(f"{'Class ID':<10} | {'Mean Loc. Ratio':<15} | {'Samples'}")
    print("-" * 40)

    all_ratios = []
    for cls_id in sorted(class_localization_scores.keys()):
        ratios = class_localization_scores[cls_id]
        mean_ratio = sum(ratios) / len(ratios)
        all_ratios.extend(ratios)
        print(f"{cls_id:<10} | {mean_ratio:<15.4f} | {len(ratios)}")

    if all_ratios:
        print("=" * 40)
        print(f"Global Joint Localization Average: {sum(all_ratios) / len(all_ratios):.4f}")


def explain_joint(model, clip, labels):
    """
    Calculates how much the model's explanation for 'label'
    falls into the correct temporal window of a concatenated video.
    """
    device = next(model.parameters()).device
    x = clip.to(device).unsqueeze(0).clone().detach().requires_grad_(True)

    results = []

    for i, label in enumerate(labels):
        model.zero_grad(set_to_none=True)

        with torch.enable_grad(), model.explanation_mode():
            out = model(x)
            logit = out[0, label]
            logit.backward(retain_graph=False)  # Retain to explain the second label next

        grad = x.grad.detach().clone().squeeze(0)
        rgb_grad = grad / (
                grad.abs().max(0, keepdim=True).values + 1e-12
        )

        contribs = (x * grad).sum(0, keepdim=True)

        # clip off values below 0 (i.e., set negatively weighted channels to 0 weighting)
        rgb_grad = rgb_grad.clamp(min=0)
        pair = rgb_grad[:3] + rgb_grad[3:]
        rgb_grad = rgb_grad[:3] / (pair + 1e-12)  # [3, T, H, W]

        # Set alpha value to the strength (L2 norm) of each location's gradient
        alpha = grad.norm(p=2, dim=0, keepdim=True)
        # Only show positive contributions
        alpha = torch.where(contribs < 0, 1e-12, alpha)
        # [1, T, H, W] -> [T, 1, H, W]
        alpha = alpha.squeeze(0)
        print("Alpha: ", alpha.shape)
        alpha_2d = alpha.permute(1, 0, 2, 3)
        alpha_2d = F.avg_pool2d(alpha_2d, kernel_size=5, stride=1, padding=(5 - 1) // 2)
        alpha = alpha_2d.permute(1, 0, 2, 3)  # back to [1, T, H, W]
        alpha = (alpha / torch.quantile(alpha, q=98.0 / 100)).clip(0, 1)

        rgb_grad = torch.concatenate([rgb_grad, alpha], dim=0)
        grad = rgb_grad
        #flat = grad.reshape(-1)
        #k = max(1, int(0.001 * flat.numel()))
        #topk_vals, _ = torch.topk(flat, k)
        #threshold = topk_vals[-1]

        #grad = grad.clone()
        #grad[grad < threshold] = 0

        #print((grad > 0).float().mean())

        T = grad.shape[0]
        # Logic: First label should be in first half, second in second half
        midpoint = T // 2
        correct_frames = range(0, midpoint) if i == 0 else range(midpoint, T)

        frame_contrib = 0
        total_contrib = 0

        for t in range(T):
            t_sum = grad[t].sum().item()
            if t in correct_frames:
                frame_contrib += t_sum
            total_contrib += t_sum

        ratio = frame_contrib / (total_contrib + 1e-8)
        results.append((label, ratio))

        # Reset grad for the next label in the same video
        x.grad.zero_()

    return results


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # Explicitly setting these for your dissertation environment
    args.dataset, args.base_network, args.experiment_name = "UCF101", "bcosification", "i3d"
    args.reload, args.ema = "last", False
    game(args)