import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# ---- adapt these imports to your project ----
from evaluate import load_model_and_config
from bcos.data.presets import UCF101ClassificationPresetEval
from bcos.common import get_inx2label_ucf101 as idx2label


# =========================================================
# Configuration
# =========================================================

@dataclass
class GameConfig:
    num_games: int = 500
    frames_per_clip: int = 8
    crop_size: int = 224
    confidence_threshold: float = 0.70
    positive_only: bool = True
    heatmap_smooth: int = 9
    heatmap_percentile: float = 95.0
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Utility functions
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_video_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def sample_frame_from_video(video_path: str, frame_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
    frames = load_video_frames(video_path)
    if len(frames) == 0:
        raise ValueError(f"No frames found in {video_path}")
    if frame_idx is None:
        frame_idx = len(frames) // 2
    frame_idx = max(0, min(frame_idx, len(frames) - 1))
    return frames[frame_idx], frame_idx


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(frame)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.array(pil)


def make_2x2_grid(frames_4: List[np.ndarray]) -> np.ndarray:
    """
    frames_4: list of 4 RGB frames, same H,W,3
    order: [top-left, top-right, bottom-left, bottom-right]
    """
    tl, tr, bl, br = frames_4
    top = np.concatenate([tl, tr], axis=1)
    bottom = np.concatenate([bl, br], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    return grid


def repeat_grid_as_video(grid_frame: np.ndarray, T: int) -> np.ndarray:
    """
    Returns [T, H, W, C]
    """
    return np.repeat(grid_frame[None, ...], T, axis=0)


def linear_mapping_to_heatmap(
    video: torch.Tensor,
    linear_mapping: torch.Tensor,
    smooth: int = 9,
    percentile: float = 95.0,
    positive_only: bool = True,
) -> np.ndarray:
    """
    video: [C, T, H, W]
    linear_mapping: [C, T, H, W]
    returns: [T, H, W] in [0,1]
    """
    contribs = (video * linear_mapping).sum(0)  # [T,H,W]

    heatmap = contribs.clamp(min=0) if positive_only else contribs.abs()

    if smooth > 1:
        heatmap = heatmap.unsqueeze(1)  # [T,1,H,W]
        heatmap = F.avg_pool2d(
            heatmap,
            kernel_size=smooth,
            stride=1,
            padding=(smooth - 1) // 2,
        ).squeeze(1)

    flat = heatmap.flatten(1)  # [T, H*W]
    q = torch.quantile(flat, percentile / 100.0, dim=1, keepdim=True)
    heatmap = heatmap / (q.unsqueeze(-1) + 1e-12)
    heatmap = heatmap.clamp(0, 1)

    return heatmap.detach().cpu().numpy()


def aggregate_temporal_heatmap(heatmap_t: np.ndarray) -> np.ndarray:
    """
    heatmap_t: [T,H,W]
    returns: [H,W]
    """
    return heatmap_t.sum(axis=0)


def cell_slices(H: int, W: int) -> Dict[int, Tuple[slice, slice]]:
    h2, w2 = H // 2, W // 2
    return {
        0: (slice(0, h2), slice(0, w2)),   # top-left
        1: (slice(0, h2), slice(w2, W)),   # top-right
        2: (slice(h2, H), slice(0, w2)),   # bottom-left
        3: (slice(h2, H), slice(w2, W)),   # bottom-right
    }


def pointing_game_hit(heatmap: np.ndarray, target_cell: int) -> int:
    """
    heatmap: [H,W]
    """
    H, W = heatmap.shape
    cells = cell_slices(H, W)

    best_cell = None
    best_value = -np.inf

    for c, (ys, xs) in cells.items():
        val = heatmap[ys, xs].max()
        if val > best_value:
            best_value = val
            best_cell = c

    return int(best_cell == target_cell)


def attribution_mass_in_cell(heatmap: np.ndarray, target_cell: int) -> float:
    H, W = heatmap.shape
    cells = cell_slices(H, W)
    ys, xs = cells[target_cell]
    total = float(heatmap.sum())
    if total <= 1e-12:
        return 0.0
    return float(heatmap[ys, xs].sum() / total)


# =========================================================
# Candidate mining
# =========================================================

@dataclass
class Candidate:
    video_path: str
    label: int
    confidence: float
    pred: int
    frame_idx: int


def get_prediction_and_confidence(
    model,
    transform,
    video_path: str,
    frames_per_clip: int,
    device: torch.device,
) -> Optional[Tuple[int, float]]:
    frames = load_video_frames(video_path)
    if len(frames) == 0:
        return None

    # simple middle clip sampling
    mid = len(frames) // 2
    start = max(0, mid - frames_per_clip // 2)
    clip = frames[start:start + frames_per_clip]

    if len(clip) < frames_per_clip:
        clip = clip + [clip[-1]] * (frames_per_clip - len(clip))

    video_np = np.stack(clip)  # [T,H,W,C]
    video_tensor = torch.tensor(video_np)
    video_tensor = transform(video_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    return int(pred.item()), float(conf.item())


def mine_high_confidence_candidates(
    model,
    transform,
    samples: List[Tuple[str, int]],
    frames_per_clip: int,
    device: torch.device,
    confidence_threshold: float,
) -> List[Candidate]:
    candidates = []

    for video_path, true_label in tqdm(samples, desc="Mining candidates"):
        result = get_prediction_and_confidence(
            model=model,
            transform=transform,
            video_path=video_path,
            frames_per_clip=frames_per_clip,
            device=device,
        )
        if result is None:
            continue

        pred, conf = result
        if pred == true_label and conf >= confidence_threshold:
            frames = load_video_frames(video_path)
            frame_idx = len(frames) // 2
            candidates.append(
                Candidate(
                    video_path=video_path,
                    label=true_label,
                    confidence=conf,
                    pred=pred,
                    frame_idx=frame_idx,
                )
            )

    return candidates


# =========================================================
# Game construction
# =========================================================

def build_game(
    target: Candidate,
    distractors: List[Candidate],
    grid_size: int = 224,
) -> Tuple[np.ndarray, int]:
    """
    Returns:
      grid_frame: [2*grid_size, 2*grid_size, 3]
      target_cell: int in {0,1,2,3}
    """
    target_frame, _ = sample_frame_from_video(target.video_path, target.frame_idx)
    target_frame = resize_frame(target_frame, grid_size)

    distractor_frames = []
    for d in distractors:
        fr, _ = sample_frame_from_video(d.video_path, d.frame_idx)
        distractor_frames.append(resize_frame(fr, grid_size))

    cells = [target_frame] + distractor_frames
    perm = list(range(4))
    random.shuffle(perm)

    arranged = [None] * 4
    target_cell = None
    for src_idx, dst_idx in enumerate(perm):
        arranged[dst_idx] = cells[src_idx]
        if src_idx == 0:
            target_cell = dst_idx

    grid_frame = make_2x2_grid(arranged)
    return grid_frame, target_cell


def choose_distractors(
    candidates: List[Candidate],
    target_label: int,
    k: int = 3,
) -> List[Candidate]:
    pool = [c for c in candidates if c.label != target_label]
    # try to use distinct distractor classes
    random.shuffle(pool)

    chosen = []
    used_labels = set()
    for c in pool:
        if c.label in used_labels:
            continue
        chosen.append(c)
        used_labels.add(c.label)
        if len(chosen) == k:
            break

    if len(chosen) < k:
        raise RuntimeError("Not enough distractor candidates with distinct labels.")
    return chosen


# =========================================================
# Explanation wrapper
# =========================================================

def explain_target_class_heatmap(
    model,
    transform,
    grid_video_np: np.ndarray,
    target_label: int,
    cfg: GameConfig,
) -> np.ndarray:
    """
    grid_video_np: [T,H,W,C]
    returns aggregated spatial heatmap [H,W]
    """

    video_tensor = torch.tensor(grid_video_np)
    video_tensor = transform(video_tensor).to(cfg.device)  # [C,T,H,W]
    video_tensor = video_tensor.unsqueeze(0)               # [1,C,T,H,W]

    # This assumes your model exposes a suitable explanation API.
    # If your implementation differs, adapt this part only.
    expl_out = model.explain_video(video_tensor, class_idx=target_label)

    # Option A: model already returns a heatmap [T,H,W]
    if "heatmap" in expl_out:
        heatmap_t = expl_out["heatmap"]
        if isinstance(heatmap_t, torch.Tensor):
            heatmap_t = heatmap_t.detach().cpu().numpy()
        return aggregate_temporal_heatmap(heatmap_t)

    # Option B: compute from contribution / linear mapping manually
    if "linear_mapping" in expl_out:
        linear_mapping = expl_out["linear_mapping"]  # [C,T,H,W]
        if isinstance(linear_mapping, np.ndarray):
            linear_mapping = torch.from_numpy(linear_mapping).to(video_tensor.device)

        video_for_expl = video_tensor.squeeze(0)
        heatmap_t = linear_mapping_to_heatmap(
            video=video_for_expl,
            linear_mapping=linear_mapping,
            smooth=cfg.heatmap_smooth,
            percentile=cfg.heatmap_percentile,
            positive_only=cfg.positive_only,
        )
        return aggregate_temporal_heatmap(heatmap_t)

    raise RuntimeError("Explanation output did not contain 'heatmap' or 'linear_mapping'.")


# =========================================================
# Main evaluation
# =========================================================

def run_spatial_grid_pointing_game(
    model,
    transform,
    candidates: List[Candidate],
    cfg: GameConfig,
) -> Dict[str, Any]:
    """
    Runs 500 spatial games.
    """
    results = []

    # sort by confidence so we truly use high-confidence samples
    candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)

    if len(candidates) < cfg.num_games:
        raise RuntimeError(f"Need at least {cfg.num_games} candidates, got {len(candidates)}")

    targets = candidates[:cfg.num_games]

    for i, target in enumerate(tqdm(targets, desc="Spatial grid pointing game")):
        distractors = choose_distractors(candidates, target_label=target.label, k=3)

        grid_frame, target_cell = build_game(
            target=target,
            distractors=distractors,
            grid_size=cfg.crop_size,
        )

        grid_video = repeat_grid_as_video(grid_frame, cfg.frames_per_clip)  # [T,H,W,C]

        heatmap = explain_target_class_heatmap(
            model=model,
            transform=transform,
            grid_video_np=grid_video,
            target_label=target.label,
            cfg=cfg,
        )

        hit = pointing_game_hit(heatmap, target_cell)
        mass = attribution_mass_in_cell(heatmap, target_cell)

        results.append({
            "game_idx": i,
            "target_video": target.video_path,
            "target_label": target.label,
            "target_confidence": target.confidence,
            "target_cell": target_cell,
            "hit": hit,
            "mass_in_target_cell": mass,
        })

    hit_rate = float(np.mean([r["hit"] for r in results]))
    mean_mass = float(np.mean([r["mass_in_target_cell"] for r in results]))

    return {
        "num_games": len(results),
        "pointing_game_accuracy": hit_rate,
        "mean_attribution_mass_in_target_cell": mean_mass,
        "results": results,
    }


# =========================================================
# Example entry point
# =========================================================

def load_ucf_samples_from_split(split_txt: str) -> List[Tuple[str, int]]:
    """
    Placeholder:
    return list of (video_path, true_label_idx)
    Adapt this to your UCF101 split format.
    """
    samples = []
    with open(split_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Example expected format: /path/to/video.avi,label
            video_path, label = line.split(",")
            samples.append((video_path, int(label)))
    return samples


def main():
    cfg = GameConfig()
    set_seed(cfg.random_seed)
    device = torch.device(cfg.device)

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    class DummyArgs:
        no_cuda = False
        dataset = "UCF101"
        base_network = "i3d"
        experiment_name = "bcosification"
        reload = "best"
        weights = None

    args = DummyArgs()
    model, _ = load_model_and_config(args)
    model = model.to(device)
    model.eval()

    transform = UCF101ClassificationPresetEval(
        crop_size=cfg.crop_size,
        is_bcos=True,
    )

    # -------------------------------------------------
    # Load sample list
    # -------------------------------------------------
    # Replace this with your actual split file
    split_txt = "ucfTrainTestlist/testlist02.txt"
    samples = load_ucf_samples_from_split(split_txt)

    # -------------------------------------------------
    # Mine high-confidence, correctly classified candidates
    # -------------------------------------------------
    candidates = mine_high_confidence_candidates(
        model=model,
        transform=transform,
        samples=samples,
        frames_per_clip=cfg.frames_per_clip,
        device=device,
        confidence_threshold=cfg.confidence_threshold,
    )

    print(f"Found {len(candidates)} high-confidence correctly classified candidates")

    # -------------------------------------------------
    # Run spatial grid pointing game
    # -------------------------------------------------
    summary = run_spatial_grid_pointing_game(
        model=model,
        transform=transform,
        candidates=candidates,
        cfg=cfg,
    )

    print("\n=== Spatial Grid Pointing Game Results ===")
    print(f"Number of games: {summary['num_games']}")
    print(f"Pointing-game accuracy: {100 * summary['pointing_game_accuracy']:.2f}%")
    print(f"Mean attribution mass in target cell: {100 * summary['mean_attribution_mass_in_target_cell']:.2f}%")

    # Optional save
    os.makedirs("grid_pointing_results", exist_ok=True)
    out_path = os.path.join("grid_pointing_results", "spatial_gp_summary.pt")
    torch.save(summary, out_path)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()