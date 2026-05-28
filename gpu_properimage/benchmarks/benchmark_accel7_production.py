#!/usr/bin/env python
"""Production-style benchmark for acceleration plan 7.

This benchmark measures the FFT-heavy subtraction core with:
- CPU path: SciPy FFT.
- GPU pipeline path: Torch CUDA FFT, including host-to-device copies.
- GPU persistent path: Torch CUDA FFT with image tensors reused on device.

The code intentionally avoids importing properimage so the benchmark can run
even when optional image-cleaning dependencies are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import scipy.fft
import scipy.optimize
import torch


BASE_COLUMNS = [
    "Pixels",
    "K",
    "P",
    "Bkg",
    "D",
    "Auto",
    "Pipeline mean ms",
    "Persistent mean ms",
    "CPU mean ms",
    "Speedup",
    "Persistent speedup",
    "Kernel ms",
    "H2D copy ms",
    "Init ms",
    "Image alloc ms",
    "Result copy ms",
    "Solve ms",
    "Cleanup ms",
    "GPU RAM peak delta MB",
    "GPU VRAM peak delta MB",
    "CPU RAM peak delta MB",
    "Symmetry max abs",
    "Max abs M error",
    "Max abs b error",
    "Finite",
]


@dataclass(frozen=True)
class BenchConfig:
    pixels: int
    k: int
    p: int
    bkg: int
    density: int
    auto: bool
    repeats: int


class RssSampler:
    def __init__(self, interval: float = 0.001):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        gc.collect()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    @property
    def delta_mb(self) -> float:
        return max(0, self.peak - self.baseline) / 1024**2


def now() -> float:
    return time.perf_counter()


def ms(start: float, stop: float) -> float:
    return (stop - start) * 1000.0


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)
    psf /= np.sum(psf)
    return psf


def synthesize(config: BenchConfig, seed: int):
    rng = np.random.default_rng(seed)
    shape = (config.pixels, config.pixels)
    ref = rng.normal(1000.0, 25.0, shape).astype(np.float32)
    new = rng.normal(1000.0, 25.0, shape).astype(np.float32)

    n_sources = max(8, config.pixels * config.pixels // config.density)
    xs = rng.integers(config.k, config.pixels - config.k, size=n_sources)
    ys = rng.integers(config.k, config.pixels - config.k, size=n_sources)
    fluxes = rng.uniform(3000.0, 45000.0, size=n_sources).astype(np.float32)
    psf_ref = gaussian_psf(config.k, 2.1)
    psf_new = gaussian_psf(config.k, 2.8)
    half = config.k // 2
    for x, y, flux in zip(xs, ys, fluxes):
        slx = slice(x - half, x + half + 1)
        sly = slice(y - half, y + half + 1)
        ref[slx, sly] += flux * psf_ref
        new[slx, sly] += 0.82 * flux * psf_new

    gradient = np.linspace(-1.0, 1.0, config.pixels, dtype=np.float32)
    bkg = config.bkg * 3.0 * gradient[None, :]
    gamma = (12.0 + bkg).astype(np.float32)
    new = new + gamma
    mask = np.zeros(shape, dtype=bool)

    return ref, new, psf_ref, psf_new, gamma, mask


def fft_shift_np(array, shift):
    phase = np.zeros(array.shape, dtype=np.float32)
    for axis, amount in enumerate(shift):
        axis_shape = [1] * array.ndim
        axis_shape[axis] = array.shape[axis]
        freq = scipy.fft.fftfreq(array.shape[axis]).reshape(axis_shape)
        phase += freq.astype(np.float32) * amount
    return array * np.exp((-2j * np.pi * phase).astype(np.complex64))


def fft_shift_torch(array, shift):
    phase = torch.zeros(array.shape, device=array.device, dtype=torch.float32)
    for axis, amount in enumerate(shift):
        axis_shape = [1] * array.ndim
        axis_shape[axis] = array.shape[axis]
        freq = torch.fft.fftfreq(
            array.shape[axis], device=array.device, dtype=torch.float32
        ).reshape(axis_shape)
        phase = phase + freq * float(amount)
    return array * torch.exp((-2j * math.pi * phase).to(torch.complex64))


def cpu_run(ref, new, psf_ref, psf_new, gamma):
    start = now()
    shape = ref.shape
    psf_ref_hat = scipy.fft.fftn(psf_ref, s=shape, norm="ortho")
    psf_new_hat = scipy.fft.fftn(psf_new, s=shape, norm="ortho")
    ref_hat = scipy.fft.fftn(ref, norm="ortho")
    new_hat = scipy.fft.fftn(new, norm="ortho")
    dhr = fft_shift_np(psf_new_hat * ref_hat, (-0.35, 0.15))
    dhn = fft_shift_np(psf_ref_hat * new_hat, (-0.15, 0.35))
    norm_a = np.abs(psf_ref_hat) ** 2
    norm_b = np.abs(psf_new_hat) ** 2

    solve_start = now()

    def cost(vec):
        b = float(vec[0])
        norm = np.sqrt(norm_a + norm_b * b**2 + 1e-6)
        residual = (
            scipy.fft.ifftn(dhn / norm, norm="ortho")
            - b * scipy.fft.ifftn(dhr / norm, norm="ortho")
            - gamma / np.sqrt(1.0 + b**2)
        )
        return np.array([np.mean(np.abs(residual.real), dtype=np.float64)])

    result = scipy.optimize.least_squares(
        cost,
        np.array([0.82], dtype=np.float64),
        bounds=([0.1], [2.0]),
        max_nfev=8,
        jac="2-point",
    )
    solve_ms = ms(solve_start, now())
    b = float(result.x[0])
    norm = np.sqrt(norm_a + norm_b * b**2 + 1e-6)
    d_hat = (dhn - b * dhr) / norm
    image = scipy.fft.ifftn(d_hat, norm="ortho").real.astype(np.float32)
    total_ms = ms(start, now())
    return image, b, {"CPU mean ms": total_ms, "Solve ms": solve_ms}


def prepare_gpu(ref, new, psf_ref, psf_new, gamma):
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = now()
    ref_t = torch.from_numpy(ref)
    new_t = torch.from_numpy(new)
    pr_t = torch.from_numpy(psf_ref)
    pn_t = torch.from_numpy(psf_new)
    gamma_t = torch.from_numpy(gamma)
    t1 = now()
    ref_g = ref_t.to(device, non_blocking=False)
    new_g = new_t.to(device, non_blocking=False)
    pr_g = pr_t.to(device, non_blocking=False)
    pn_g = pn_t.to(device, non_blocking=False)
    gamma_g = gamma_t.to(device, non_blocking=False)
    torch.cuda.synchronize()
    t2 = now()
    return (ref_g, new_g, pr_g, pn_g, gamma_g), ms(t0, t1), ms(t1, t2)


def gpu_core(state, copy_result=True):
    ref_g, new_g, psf_ref_g, psf_new_g, gamma_g = state
    shape = ref_g.shape
    kernel_start = now()
    psf_ref_hat = torch.fft.fftn(psf_ref_g, s=shape, norm="ortho")
    psf_new_hat = torch.fft.fftn(psf_new_g, s=shape, norm="ortho")
    ref_hat = torch.fft.fftn(ref_g, norm="ortho")
    new_hat = torch.fft.fftn(new_g, norm="ortho")
    dhr = fft_shift_torch(psf_new_hat * ref_hat, (-0.35, 0.15))
    dhn = fft_shift_torch(psf_ref_hat * new_hat, (-0.15, 0.35))
    norm_a = torch.abs(psf_ref_hat) ** 2
    norm_b = torch.abs(psf_new_hat) ** 2
    torch.cuda.synchronize()
    kernel_ms = ms(kernel_start, now())

    solve_start = now()

    def cost(vec):
        b = float(vec[0])
        norm = torch.sqrt(norm_a + norm_b * b**2 + 1e-6)
        residual = (
            torch.fft.ifftn(dhn / norm, norm="ortho")
            - b * torch.fft.ifftn(dhr / norm, norm="ortho")
            - gamma_g / math.sqrt(1.0 + b**2)
        )
        value = torch.mean(torch.abs(residual.real))
        torch.cuda.synchronize()
        return np.array([float(value.cpu().numpy())])

    result = scipy.optimize.least_squares(
        cost,
        np.array([0.82], dtype=np.float64),
        bounds=([0.1], [2.0]),
        max_nfev=8,
        jac="2-point",
    )
    b = float(result.x[0])
    torch.cuda.synchronize()
    solve_ms = ms(solve_start, now())

    result_start = now()
    norm = torch.sqrt(norm_a + norm_b * b**2 + 1e-6)
    d_hat = (dhn - b * dhr) / norm
    image_g = torch.fft.ifftn(d_hat, norm="ortho").real
    if copy_result:
        image = image_g.cpu().numpy()
    else:
        image = None
    torch.cuda.synchronize()
    result_ms = ms(result_start, now())
    return image, b, {
        "Kernel ms": kernel_ms,
        "Solve ms": solve_ms,
        "Result copy ms": result_ms,
    }


def gpu_pipeline_run(ref, new, psf_ref, psf_new, gamma):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    init_start = now()
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    init_ms = ms(init_start, now())
    state, alloc_ms, h2d_ms = prepare_gpu(ref, new, psf_ref, psf_new, gamma)
    start = now()
    image, b, timings = gpu_core(state, copy_result=True)
    core_total = ms(start, now())
    cleanup_start = now()
    del state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cleanup_ms = ms(cleanup_start, now())
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    total = init_ms + alloc_ms + h2d_ms + core_total + cleanup_ms
    timings.update(
        {
            "Pipeline mean ms": total,
            "Init ms": init_ms,
            "Image alloc ms": alloc_ms,
            "H2D copy ms": h2d_ms,
            "Cleanup ms": cleanup_ms,
            "GPU VRAM peak delta MB": peak_vram,
        }
    )
    return image, b, timings


def validation_matrix_cpu(ref, new, gamma):
    ref64 = ref.astype(np.float64)
    new64 = new.astype(np.float64)
    gamma64 = gamma.astype(np.float64)
    matrix = np.array(
        [
            [np.mean(ref64 * ref64), np.mean(ref64 * new64)],
            [np.mean(new64 * ref64), np.mean(new64 * new64)],
        ],
        dtype=np.float64,
    )
    vector = np.array(
        [np.mean(ref64 * gamma64), np.mean(new64 * gamma64)],
        dtype=np.float64,
    )
    return matrix, vector


def validation_matrix_gpu(ref, new, gamma):
    state, _, _ = prepare_gpu(ref, new, ref[:1, :1], new[:1, :1], gamma)
    ref_g, new_g, _, _, gamma_g = state
    matrix = torch.stack(
        [
            torch.stack([torch.mean(ref_g * ref_g), torch.mean(ref_g * new_g)]),
            torch.stack([torch.mean(new_g * ref_g), torch.mean(new_g * new_g)]),
        ]
    )
    vector = torch.stack(
        [torch.mean(ref_g * gamma_g), torch.mean(new_g * gamma_g)]
    )
    matrix_cpu = matrix.cpu().numpy().astype(np.float64)
    vector_cpu = vector.cpu().numpy().astype(np.float64)
    del state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return matrix_cpu, vector_cpu


def mean_dict(dicts):
    keys = sorted({key for item in dicts for key in item})
    return {
        key: statistics.fmean(float(item.get(key, 0.0)) for item in dicts)
        for key in keys
    }


def round_float(value):
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    return f"{float(value):.4f}"


def run_config(config: BenchConfig, seed: int):
    ref, new, psf_ref, psf_new, gamma, _ = synthesize(config, seed)

    warm_image, _, _ = gpu_pipeline_run(ref, new, psf_ref, psf_new, gamma)
    del warm_image
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    cpu_results = []
    with RssSampler() as cpu_mem:
        for _ in range(config.repeats):
            cpu_image, cpu_b, cpu_timing = cpu_run(
                ref, new, psf_ref, psf_new, gamma
            )
            cpu_results.append(cpu_timing)
    cpu_mean = mean_dict(cpu_results)

    gpu_results = []
    with RssSampler() as gpu_mem:
        for _ in range(config.repeats):
            gpu_image, gpu_b, gpu_timing = gpu_pipeline_run(
                ref, new, psf_ref, psf_new, gamma
            )
            gpu_results.append(gpu_timing)
    gpu_mean = mean_dict(gpu_results)

    state, persistent_alloc_ms, persistent_h2d_ms = prepare_gpu(
        ref, new, psf_ref, psf_new, gamma
    )
    persistent_results = []
    for _ in range(config.repeats):
        persistent_start = now()
        persistent_image, persistent_b, timing = gpu_core(
            state, copy_result=True
        )
        timing["Persistent mean ms"] = ms(persistent_start, now())
        persistent_results.append(timing)
    persistent_mean = mean_dict(persistent_results)
    del state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    cpu_matrix, cpu_vector = validation_matrix_cpu(ref, new, gamma)
    gpu_matrix, gpu_vector = validation_matrix_gpu(ref, new, gamma)
    symmetry = np.max(np.abs(gpu_matrix - gpu_matrix.T))
    max_m_error = np.max(np.abs(cpu_matrix - gpu_matrix))
    max_b_error = np.max(np.abs(cpu_vector - gpu_vector))
    finite = bool(
        np.isfinite(cpu_image).all()
        and np.isfinite(gpu_image).all()
        and np.isfinite(persistent_image).all()
        and np.isfinite([cpu_b, gpu_b, persistent_b]).all()
        and np.isfinite(gpu_matrix).all()
        and np.isfinite(gpu_vector).all()
    )

    pipeline_ms = gpu_mean["Pipeline mean ms"]
    persistent_ms = persistent_mean["Persistent mean ms"]
    cpu_ms = cpu_mean["CPU mean ms"]

    row = {
        "Pixels": config.pixels,
        "K": config.k,
        "P": config.p,
        "Bkg": config.bkg,
        "D": config.density,
        "Auto": str(config.auto),
        "Pipeline mean ms": pipeline_ms,
        "Persistent mean ms": persistent_ms,
        "CPU mean ms": cpu_ms,
        "Speedup": cpu_ms / pipeline_ms,
        "Persistent speedup": cpu_ms / persistent_ms,
        "Kernel ms": persistent_mean["Kernel ms"],
        "H2D copy ms": gpu_mean["H2D copy ms"],
        "Init ms": gpu_mean["Init ms"],
        "Image alloc ms": gpu_mean["Image alloc ms"],
        "Result copy ms": persistent_mean["Result copy ms"],
        "Solve ms": persistent_mean["Solve ms"],
        "Cleanup ms": gpu_mean["Cleanup ms"],
        "GPU RAM peak delta MB": gpu_mem.delta_mb,
        "GPU VRAM peak delta MB": gpu_mean["GPU VRAM peak delta MB"],
        "CPU RAM peak delta MB": cpu_mem.delta_mb,
        "Symmetry max abs": symmetry,
        "Max abs M error": max_m_error,
        "Max abs b error": max_b_error,
        "Finite": str(finite),
    }
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pixels",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096],
        help="Image widths/heights to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gpu_properimage/docs/accel7_production_benchmark.csv"),
    )
    return parser.parse_args()


def warmup_cuda():
    tensor = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    _ = torch.fft.fftn(tensor, norm="ortho")
    torch.cuda.synchronize()
    del tensor
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available.")

    warmup_cuda()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configs = [
        BenchConfig(
            pixels=pixels,
            k=21,
            p=2,
            bkg=1,
            density=max(4096, pixels * pixels // 256),
            auto=True,
            repeats=args.repeats if pixels < 4096 else max(2, args.repeats - 1),
        )
        for pixels in args.pixels
    ]

    rows = []
    for offset, config in enumerate(configs):
        print(f"running {config.pixels}x{config.pixels} repeats={config.repeats}")
        rows.append(run_config(config, args.seed + offset))

    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=BASE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round_float(row[key]) for key in BASE_COLUMNS})

    print(args.output)
    for row in rows:
        print(
            f"{row['Pixels']} px: CPU {row['CPU mean ms']:.2f} ms, "
            f"pipeline {row['Pipeline mean ms']:.2f} ms, "
            f"persistent {row['Persistent mean ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
