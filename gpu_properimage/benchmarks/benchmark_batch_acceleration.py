#!/usr/bin/env python
"""Benchmark batch CPU/GPU subtraction scheduling on real FITS folders."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from properimage.acceleration import (
    AccelerationConfig,
    clear_acceleration_cache,
    resolve_cuda_devices,
    subtract_batch,
)
from properimage.cli import _pair_from_dirs


REPO_ROOT = Path(__file__).resolve().parents[2]

FIELDS = [
    "Mode",
    "Batch size",
    "Devices",
    "CPU workers",
    "GPU workers",
    "Prefetch",
    "Throughput pairs/s",
    "Mean ms",
    "P50 ms",
    "P95 ms",
    "P99 ms",
    "Elapsed ms",
    "Cache hit rate",
    "Failures",
]


def percentile(values, pct):
    if not values:
        return ""
    values = sorted(values)
    index = min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1))))
    return values[index]


def run_mode(mode, pairs, args):
    if mode == "cpu":
        devices = "cpu"
        use_gpu = False
    elif mode == "single-gpu":
        visible = resolve_cuda_devices(AccelerationConfig(devices="auto"))
        if not visible:
            return None
        devices = (visible[0],)
        use_gpu = True
    else:
        devices = "auto"
        use_gpu = True

    clear_acceleration_cache()
    config = AccelerationConfig(
        devices=devices,
        cpu_workers=args.cpu_workers,
        io_workers=args.io_workers,
        gpu_workers=args.gpu_workers,
        prefetch=args.prefetch,
        persistent=not args.no_persistent,
    )
    start = time.perf_counter()
    summary = subtract_batch(
        pairs,
        acceleration=config,
        use_gpu=use_gpu,
        beta=not args.no_beta,
        shift=not args.no_shift,
        fitted_psf=not args.no_fitted_psf,
        smooth_psf=args.smooth_psf,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    timings = [task.elapsed_ms for task in summary.tasks]
    cache_hit_rate = (
        summary.cache_hits / len(summary.tasks) if summary.tasks else 0.0
    )
    return {
        "Mode": mode,
        "Batch size": len(summary.tasks),
        "Devices": "cpu" if not summary.devices else ",".join(map(str, summary.devices)),
        "CPU workers": summary.cpu_workers,
        "GPU workers": summary.gpu_workers,
        "Prefetch": args.prefetch,
        "Throughput pairs/s": summary.throughput_pairs_per_s,
        "Mean ms": statistics.mean(timings) if timings else "",
        "P50 ms": percentile(timings, 50),
        "P95 ms": percentile(timings, 95),
        "P99 ms": percentile(timings, 99),
        "Elapsed ms": elapsed_ms,
        "Cache hit rate": cache_hit_rate,
        "Failures": len(summary.failed),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", default=str(REPO_ROOT / "data" / "ref"))
    parser.add_argument("--new-dir", default=str(REPO_ROOT / "data" / "new"))
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "gpu_properimage"
            / "docs"
            / "batch_acceleration_benchmark.csv"
        ),
    )
    parser.add_argument("--modes", nargs="+", default=["cpu", "single-gpu", "multi-gpu"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cpu-workers", default="auto")
    parser.add_argument("--io-workers", default="auto")
    parser.add_argument("--gpu-workers", default="auto")
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--no-persistent", action="store_true")
    parser.add_argument("--smooth-psf", action="store_true")
    parser.add_argument("--no-fitted-psf", action="store_true")
    parser.add_argument("--no-beta", action="store_true")
    parser.add_argument("--no-shift", action="store_true")
    args = parser.parse_args(argv)

    pairs = _pair_from_dirs(args.ref_dir, args.new_dir) * args.repeats
    rows = []
    for mode in args.modes:
        row = run_mode(mode, pairs, args)
        if row is not None:
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
