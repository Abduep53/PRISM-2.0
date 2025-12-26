import time
import numpy as np
import torch

@torch.no_grad()
def benchmark_inference(model, device: torch.device, batch_size: int, channels: int, clip_len: int, num_joints: int, warmup: int, iters: int):
    model.eval()
    x = torch.rand((batch_size, channels, clip_len, num_joints), device=device)

    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    mean_ms = float(times.mean())
    p95_ms = float(np.percentile(times, 95))
    fps = float((1000.0 / mean_ms) * batch_size)
    return {"mean_latency_ms": mean_ms, "p95_latency_ms": p95_ms, "throughput_fps": fps}

def model_size_mb(model) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return float(total / (1024 * 1024))
