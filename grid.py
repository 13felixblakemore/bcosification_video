import random
from collections import defaultdict

import torch

def collect_high_confidence_clips(
    model,
    loader,
    device,
    confidence_threshold=0.7,
    max_per_class=10,
    max_batches=10,
):
    """
    Collect correctly classified, high-confidence clips.

    Returns:
        clips_by_class: dict[class_idx] -> list of (video_tensor_cpu, label, confidence)
    """
    model.eval()
    clips_by_class = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break

            videos = videos.to(device)
            labels = labels.to(device)

            out = model(videos)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            true_class_probs = probs[torch.arange(labels.size(0), device=device), labels]

            for i in range(labels.size(0)):
                label = int(labels[i].item())
                pred = int(preds[i].item())
                conf = float(true_class_probs[i].item())

                if pred == label and conf >= confidence_threshold:
                    if len(clips_by_class[label]) < max_per_class:
                        clips_by_class[label].append((
                            videos[i].detach().cpu(),
                            label,
                            conf
                        ))

    return clips_by_class


def sample_unique_class_grid(clips_by_class, seed=42):
    """
    Sample 4 clips from 4 different classes and return a 2x2 grid.
    """
    rng = random.Random(seed)

    valid_classes = [c for c, clips in clips_by_class.items() if len(clips) > 0]
    if len(valid_classes) < 4:
        raise RuntimeError(
            f"Only found {len(valid_classes)} classes with high-confidence clips. Need at least 4."
        )

    chosen_classes = rng.sample(valid_classes, 4)

    chosen_clips = []
    for c in chosen_classes:
        clip, label, conf = rng.choice(clips_by_class[c])
        chosen_clips.append((clip, label, conf))

    videos = [x[0] for x in chosen_clips]
    labels = torch.tensor([x[1] for x in chosen_clips], dtype=torch.long)
    confs = [x[2] for x in chosen_clips]

    grid_video = make_2x2_grid(videos)
    return grid_video, labels, confs


def sample_top_confidence_grid(clips_by_class):
    """
    Deterministic version: take the top-confidence clip from 4 classes with highest available confidence.
    """
    best_per_class = []
    for c, clips in clips_by_class.items():
        if len(clips) > 0:
            best_clip = max(clips, key=lambda x: x[2])
            best_per_class.append(best_clip)

    if len(best_per_class) < 4:
        raise RuntimeError(
            f"Only found {len(best_per_class)} classes with high-confidence clips. Need at least 4."
        )

    # pick the 4 strongest classes overall
    best_per_class = sorted(best_per_class, key=lambda x: x[2], reverse=True)[:4]

    videos = [x[0] for x in best_per_class]
    labels = torch.tensor([x[1] for x in best_per_class], dtype=torch.long)
    confs = [x[2] for x in best_per_class]

    grid_video = make_2x2_grid(videos)
    return grid_video, labels, confs

def make_2x2_grid(videos):
    v0, v1, v2, v3 = videos
    top = torch.cat([v0, v1], dim=-1)
    bottom = torch.cat([v2, v3], dim=-1)
    return torch.cat([top, bottom], dim=-2)

def sample_with_blank(clips_by_class, quadrant=None, device="cuda"):
    """
    Create a 2x2 grid where only one quadrant contains a real clip,
    and the others are blank (zeros).

    Args:
        clips_by_class: dict[class_id] -> list of (video_tensor, confidence)
        quadrant: int in {0,1,2,3} or None (random)
        device: torch device

    Returns:
        grid_video: [C, T, 2H, 2W]
        grid_labels: list of 4 labels (only one real, others = -1)
        confidence: float (confidence of selected clip)
        quadrant: int (where the real clip was placed)
    """

    import random
    import torch

    # --- pick random class and clip
    cls = random.choice(list(clips_by_class.keys()))
    video, _, conf = random.choice(clips_by_class[cls])

    video = video.to(device)   # [C, T, H, W]
    C, T, H, W = video.shape

    # --- create blank clip
    blank = torch.zeros_like(video)

    # --- choose quadrant
    if quadrant is None:
        quadrant = random.randint(0, 3)

    # --- assign clips
    videos = [blank.clone() for _ in range(4)]
    labels = torch.tensor([-1, -1, -1, -1], dtype=torch.long)

    videos[quadrant] = video
    labels[quadrant] = cls

    # --- build grid
    def make_2x2_grid(videos):
        v0, v1, v2, v3 = videos
        top = torch.cat([v0, v1], dim=-1)     # concat width
        bottom = torch.cat([v2, v3], dim=-1)
        return torch.cat([top, bottom], dim=-2)  # concat height

    grid_video = make_2x2_grid(videos)

    return grid_video, labels, conf, quadrant

def sample_two_clips_two_blank(
    clips_by_class,
    device="cpu",
    ensure_different_classes=True,
    seed=None,
):
    """
    Create a 2x2 grid with:
        - 2 real clips
        - 2 blank quadrants

    Returns:
        grid_video: [C, T, 2H, 2W]
        grid_labels: list of 4 labels (-1 for blank)
        confs: list of 4 confidences (0 for blank)
        real_quadrants: list of quadrants containing real clips
    """

    import random
    import torch

    rng = random.Random(seed)

    # --- pick classes ---
    classes = list(clips_by_class.keys())

    if ensure_different_classes:
        if len(classes) < 2:
            raise RuntimeError("Need at least 2 classes")
        c1, c2 = rng.sample(classes, 2)
    else:
        c1 = rng.choice(classes)
        c2 = rng.choice(classes)

    # --- pick clips ---
    v1, _, conf1 = rng.choice(clips_by_class[c1])
    v2, _, conf2 = rng.choice(clips_by_class[c2])

    v1 = v1.to(device)
    v2 = v2.to(device)

    C, T, H, W = v1.shape

    # --- blank ---
    blank = torch.zeros_like(v1)

    # --- choose 2 quadrants for real clips ---
    quadrants = [0, 1, 2, 3]
    real_quadrants = rng.sample(quadrants, 2)

    videos = [blank.clone() for _ in range(4)]
    labels = [-1, -1, -1, -1]
    confs = [0.0, 0.0, 0.0, 0.0]

    # assign clips
    videos[real_quadrants[0]] = v1
    labels[real_quadrants[0]] = c1
    confs[real_quadrants[0]] = conf1

    videos[real_quadrants[1]] = v2
    labels[real_quadrants[1]] = c2
    confs[real_quadrants[1]] = conf2

    # --- grid builder ---
    def make_2x2_grid(videos):
        v0, v1, v2, v3 = videos
        top = torch.cat([v0, v1], dim=-1)
        bottom = torch.cat([v2, v3], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    grid_video = make_2x2_grid(videos)

    return grid_video, labels, confs, real_quadrants

def add_blank_frames(clips_by_class, seed=42):
    rng = random.Random(seed)

    # --- pick classes ---
    classes = list(clips_by_class.keys())

    class_1 = rng.choice(classes)
    clip, _, conf = rng.choice(clips_by_class[class_1]) # C,T,H,W

    C,T,H,W = clip.shape

    blank = torch.zeros_like(clip)

    blank_ind = [0,1,2,5,6,7]

    for t in range(T):
        if t in blank_ind:
            clip[:,t] = blank.clone()[:,t]

    return clip, class_1

def add_blank_frames_full(batch):
    B,C,T,H,W = batch.shape
    blank_ind = [0, 1, 2, 5, 6, 7]
    blank = torch.zeros_like(batch)
    for t in range(T):
        if t in blank_ind:
            batch[:,:,t] = blank.clone()[:,:,t]
    return batch

def add_second_clip(clips_by_class, seed=42):
    rng = random.Random(seed)

    # --- pick classes ---
    classes = list(clips_by_class.keys())
    c1, c2 = rng.sample(classes, 2)

    clip1, _, _ = rng.choice(clips_by_class[c1])  # (C, T, H, W)
    clip2, _, _ = rng.choice(clips_by_class[c2])  # (C, T, H, W)

    C, T, H, W = clip1.shape

    # --- ensure we have enough frames ---
    assert T >= 8, f"Need at least 8 frames, got {T}"

    # --- stride-2 sampling: 0,2,4,6 ---
    idx = torch.arange(0, 8, 2)  # [0,2,4,6]

    first_half = clip1[:, idx]   # (C, 4, H, W)
    second_half = clip2[:, idx]  # (C, 4, H, W)

    # --- concatenate to get 8 frames total ---
    combined = torch.cat([first_half, second_half], dim=1)  # (C, 8, H, W)

    return combined, [c1, c2]

def add_second_clip_full(batch, labels, seed=42):
    rng = random.Random(seed)
    B, C, T, H, W = batch.shape
    # --- pick classes ---

    idx1 = rng.randint(0, B-1)
    idx2 = rng.randint(0, B-1)

    v1 = batch[idx1]
    v2 = batch[idx2]

    l1 = labels[0]
    l2 = labels[1]

    for t in range(T//2):
        v2[:, t] = v1[:, t].clone()

    return v2, (l1, l2)



