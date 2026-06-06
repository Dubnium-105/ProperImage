#!/usr/bin/env python
"""Sweep single-GPU batch subtraction settings for maximum throughput."""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from properimage.acceleration import (  # noqa: E402
    AccelerationConfig,
    clear_acceleration_cache,
    get_cuda_device_report,
    subtract_batch,
)
from properimage.cli import _pair_from_dirs  # noqa: E402


FIELDS = [
    "Rank",
    "Device",
    "Device name",
    "Pairs",
    "Repeat pairs",
    "CPU workers",
    "IO workers",
    "GPU workers",
    "Prefetch",
    "Elapsed ms",
    "Throughput pairs/s",
    "Task mean ms",
    "Task p50 ms",
    "Task p95 ms",
    "Failures",
    "Finite",
]


def percentile(values, pct):
    if not values:
        return ""
    values = sorted(values)
    index = min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1))))
    return values[index]


def cleanup(device_id):
    clear_acceleration_cache()
    gc.collect()
    try:
        import cupy as cp

        with cp.cuda.Device(device_id):
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def run_config(args, pairs, device_id, device_name, gpu_workers, prefetch):
    cleanup(device_id)
    config = AccelerationConfig(
        devices=(device_id,),
        cpu_workers=args.cpu_workers,
        io_workers=args.io_workers,
        gpu_workers=gpu_workers,
        prefetch=prefetch,
        persistent=not args.no_persistent,
    )
    started = time.perf_counter()
    summary = subtract_batch(
        pairs,
        acceleration=config,
        use_gpu=True,
        beta=not args.no_beta,
        shift=not args.no_shift,
        fitted_psf=not args.no_fitted_psf,
        smooth_psf=args.smooth_psf,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    timings = [task.elapsed_ms for task in summary.tasks]
    finite = all(task.finite for task in summary.tasks if task.ok)
    cleanup(device_id)
    return {
        "Rank": "",
        "Device": device_id,
        "Device name": device_name,
        "Pairs": len(summary.tasks),
        "Repeat pairs": args.repeat_pairs,
        "CPU workers": summary.cpu_workers,
        "IO workers": args.io_workers,
        "GPU workers": gpu_workers,
        "Prefetch": prefetch,
        "Elapsed ms": elapsed_ms,
        "Throughput pairs/s": summary.throughput_pairs_per_s,
        "Task mean ms": statistics.mean(timings) if timings else "",
        "Task p50 ms": percentile(timings, 50),
        "Task p95 ms": percentile(timings, 95),
        "Failures": len(summary.failed),
        "Finite": finite,
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
            / "single_gpu_throughput_sweep.csv"
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeat-pairs", type=int, default=2)
    parser.add_argument("--cpu-workers", default="auto")
    parser.add_argument("--io-workers", default="auto")
    parser.add_argument("--gpu-workers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--prefetch", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--no-persistent", action="store_true")
    parser.add_argument("--smooth-psf", action="store_true")
    parser.add_argument("--no-fitted-psf", action="store_true")
    parser.add_argument("--no-beta", action="store_true")
    parser.add_argument("--no-shift", action="store_true")
    args = parser.parse_args(argv)

    report = get_cuda_device_report(AccelerationConfig(devices=(args.device,)))
    if not report:
        raise RuntimeError(f"CUDA device {args.device} is not available")
    device_name = report[0]["name"]
    base_pairs = _pair_from_dirs(args.ref_dir, args.new_dir)
    pairs = base_pairs * args.repeat_pairs

    if args.warmup:
        run_config(args, base_pairs[:1], args.device, device_name, 1, 1)

    rows = []
    for gpu_workers in args.gpu_workers:
        for prefetch in args.prefetch:
            row = run_config(
                args, pairs, args.device, device_name, gpu_workers, prefetch
            )
            rows.append(row)
            print(
                "gpu_workers={gpu_workers} prefetch={prefetch} "
                "throughput={throughput:.4f} pairs/s failures={failures}".format(
                    gpu_workers=gpu_workers,
                    prefetch=prefetch,
                    throughput=row["Throughput pairs/s"],
                    failures=row["Failures"],
                )
            )

    rows.sort(key=lambda row: float(row["Throughput pairs/s"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["Rank"] = rank

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"best throughput={rows[0]['Throughput pairs/s']:.4f} pairs/s")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
