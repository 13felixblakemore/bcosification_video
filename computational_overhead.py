import argparse
import time
import json
import numpy as np
import torch

from evaluate import load_model_and_config

# -----------------------------
# Optional imports
# -----------------------------
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from torch.profiler import profile, ProfilerActivity


# -----------------------------
# Utils
# -----------------------------

def set_determinism(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model Loading
# -----------------------------

def load_model(args, device):
    model, config = load_model_and_config(args)

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        state_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.model.model.", "model.model.")
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()

    return model, config


# -----------------------------
# Inputs
# -----------------------------

def make_inputs_std(batch_size, device):
    return torch.randn((batch_size, 3, 16, 224, 224)).to(device)


def make_inputs_bcos(batch_size, device):
    return torch.randn((batch_size, 6, 16, 224, 224)).to(device)


# -----------------------------
# Benchmarks
# -----------------------------

def benchmark_latency(model, inputs, n_warmup=20, n_runs=50):
    times = []

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.model(inputs)
        synchronize()

        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.model(inputs)
            synchronize()
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)

    return {
        "mean": float(times.mean()),
        "std": float(times.std()),
        "p95": float(np.percentile(times, 95)),
    }


def benchmark_throughput(model, inputs, duration=5.0):
    count = 0
    start = time.perf_counter()

    with torch.no_grad():
        while time.perf_counter() - start < duration:
            _ = model.model(inputs)
            count += inputs.shape[0]

    synchronize()
    elapsed = time.perf_counter() - start

    return count / elapsed


def benchmark_memory(model, inputs):
    if not torch.cuda.is_available():
        return -1

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model.model(inputs)

    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def benchmark_flops(model, inputs):
    if not FVCORE_AVAILABLE:
        return -1

    try:
        flops = FlopCountAnalysis(model.model, inputs)
        return float(flops.total())
    except Exception as e:
        print("FLOPs failed:", e)
        return -1


def benchmark_profile(model, inputs, device, steps=10):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False
    ) as prof:

        with torch.no_grad():
            for _ in range(steps):
                _ = model.model(inputs)

    key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"

    table = prof.key_averages().table(
        sort_by=key,
        row_limit=10
    )

    return table


# -----------------------------
# Main benchmark loop
# -----------------------------

def run_benchmark(model_std, model_bcos, batch_sizes, device):

    results = {}

    for b in batch_sizes:
        print(f"\nBatch size: {b}")

        inputs_std = make_inputs_std(b, device)
        inputs_bcos = make_inputs_bcos(b, device)

        # --- Standard ---
        lat_std = benchmark_latency(model_std, inputs_std)
        thr_std = benchmark_throughput(model_std, inputs_std)
        mem_std = benchmark_memory(model_std, inputs_std)
        flops_std = benchmark_flops(model_std, inputs_std)

        # --- B-cos ---
        lat_bcos = benchmark_latency(model_bcos, inputs_bcos)
        thr_bcos = benchmark_throughput(model_bcos, inputs_bcos)
        mem_bcos = benchmark_memory(model_bcos, inputs_bcos)
        flops_bcos = benchmark_flops(model_bcos, inputs_bcos)

        overhead = ((lat_bcos["mean"] - lat_std["mean"]) / lat_std["mean"]) * 100

        print("\n--- Results ---")
        print(f"Latency std:  {lat_std['mean']:.6f}s")
        print(f"Latency bcos: {lat_bcos['mean']:.6f}s")
        print(f"Overhead:     {overhead:.2f}%")

        print(f"Throughput std:  {thr_std:.2f}")
        print(f"Throughput bcos: {thr_bcos:.2f}")

        print(f"Memory std:  {mem_std:.2f} MB")
        print(f"Memory bcos: {mem_bcos:.2f} MB")

        print(f"FLOPs std:  {flops_std:.2e}")
        print(f"FLOPs bcos: {flops_bcos:.2e}")

        # --- Profiling only once ---
        if b == 1:
            print("\n--- PROFILING STANDARD MODEL ---")
            profile_std = benchmark_profile(model_std, inputs_std, device)
            print(profile_std)

            print("\n--- PROFILING BCOS MODEL ---")
            profile_bcos = benchmark_profile(model_bcos, inputs_bcos, device)
            print(profile_bcos)
        else:
            profile_std = None
            profile_bcos = None

        results[b] = {
            "latency_std": lat_std,
            "latency_bcos": lat_bcos,
            "throughput_std": thr_std,
            "throughput_bcos": thr_bcos,
            "memory_std_MB": mem_std,
            "memory_bcos_MB": mem_bcos,
            "flops_std": flops_std,
            "flops_bcos": flops_bcos,
            "overhead_percent": overhead,
            "profile_std": profile_std,
            "profile_bcos": profile_bcos
        }

    return results


# -----------------------------
# CLI
# -----------------------------

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_directory", default="./experiments")

    parser.add_argument("--checkpoint_standard", default=None)
    parser.add_argument("--checkpoint_bcos", default=None)

    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8, 32])

    parser.add_argument("--output", default="benchmark_results.json")

    return parser


# -----------------------------
# Main
# -----------------------------

def main():
    args = get_parser().parse_args()

    set_determinism()
    device = get_device()

    # -------- STANDARD --------
    args_std = argparse.Namespace(**vars(args))
    args_std.experiment_name = "i3d"
    args_std.base_network = "standard"
    args_std.checkpoint = args.checkpoint_standard
    args_std.dataset = "UCF101"
    args_std.reload = "last"
    args_std.ema = False

    model_std, _ = load_model(args_std, device)

    # -------- BCOS --------
    args_bcos = argparse.Namespace(**vars(args))
    args_bcos.experiment_name = "i3d"
    args_bcos.base_network = "bcosification"
    args_bcos.checkpoint = args.checkpoint_bcos
    args_bcos.dataset = "UCF101"
    args_bcos.reload = "last"
    args_bcos.ema = False

    model_bcos, _ = load_model(args_bcos, device)

    # -------- Run --------
    results = run_benchmark(
        model_std,
        model_bcos,
        args.batch_sizes,
        device
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()