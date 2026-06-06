#!/usr/bin/env python
"""Single-GPU memory and throughput stress test for real FITS subtraction."""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from properimage.acceleration import (
    AccelerationConfig,
    clear_acceleration_cache,
    get_cuda_device_report,
    subtract_batch,
)
from properimage.cli import _pair_from_dirs


FIELDS = [
    "Round",
    "Pairs",
    "Device",
    "Device name",
    "CPU workers",
    "GPU workers",
    "Prefetch",
    "Elapsed ms",
    "Throughput pairs/s",
    "Task mean ms",
    "Task p50 ms",
    "Task p95 ms",
    "Failures",
    "Finite",
    "RSS before MB",
    "RSS after MB",
    "RSS cleanup MB",
    "RSS delta MB",
    "RSS cleanup delta MB",
    "VRAM used before MB",
    "VRAM used after MB",
    "VRAM used cleanup MB",
    "VRAM delta MB",
    "VRAM cleanup delta MB",
    "CuPy pool used before MB",
    "CuPy pool used after MB",
    "CuPy pool used cleanup MB",
    "CuPy pool total before MB",
    "CuPy pool total after MB",
    "CuPy pool total cleanup MB",
]


def mb(value):
    return float(value) / 1024**2


def percentile(values, pct):
    if not values:
        return ""
    values = sorted(values)
    index = min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1))))
    return values[index]


def snapshot(cp, process, device_id):
    with cp.cuda.Device(device_id):
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        cp.cuda.Stream.null.synchronize()
        return {
            "rss_mb": mb(process.memory_info().rss),
            "vram_used_mb": mb(total_mem - free_mem),
            "pool_used_mb": mb(pool.used_bytes()),
            "pool_total_mb": mb(pool.total_bytes()),
        }


def cleanup(cp, device_id):
    clear_acceleration_cache()
    gc.collect()
    with cp.cuda.Device(device_id):
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()


def run_round(round_index, pairs, args, cp, process, device_id, device_name):
    cleanup(cp, device_id)
    before = snapshot(cp, process, device_id)
    config = AccelerationConfig(
        devices=(device_id,),
        cpu_workers=args.cpu_workers,
        io_workers=args.io_workers,
        gpu_workers=args.gpu_workers,
        prefetch=args.prefetch,
        persistent=not args.no_persistent,
    )
    summary = subtract_batch(
        pairs,
        acceleration=config,
        use_gpu=True,
        beta=not args.no_beta,
        shift=not args.no_shift,
        fitted_psf=not args.no_fitted_psf,
        smooth_psf=args.smooth_psf,
    )
    after = snapshot(cp, process, device_id)
    cleanup(cp, device_id)
    cleanup_snap = snapshot(cp, process, device_id)
    timings = [task.elapsed_ms for task in summary.tasks]
    finite = all(task.finite for task in summary.tasks if task.ok)
    return {
        "Round": round_index,
        "Pairs": len(summary.tasks),
        "Device": device_id,
        "Device name": device_name,
        "CPU workers": summary.cpu_workers,
        "GPU workers": summary.gpu_workers,
        "Prefetch": args.prefetch,
        "Elapsed ms": summary.elapsed_ms,
        "Throughput pairs/s": summary.throughput_pairs_per_s,
        "Task mean ms": statistics.mean(timings) if timings else "",
        "Task p50 ms": percentile(timings, 50),
        "Task p95 ms": percentile(timings, 95),
        "Failures": len(summary.failed),
        "Finite": finite,
        "RSS before MB": before["rss_mb"],
        "RSS after MB": after["rss_mb"],
        "RSS cleanup MB": cleanup_snap["rss_mb"],
        "RSS delta MB": after["rss_mb"] - before["rss_mb"],
        "RSS cleanup delta MB": cleanup_snap["rss_mb"] - before["rss_mb"],
        "VRAM used before MB": before["vram_used_mb"],
        "VRAM used after MB": after["vram_used_mb"],
        "VRAM used cleanup MB": cleanup_snap["vram_used_mb"],
        "VRAM delta MB": after["vram_used_mb"] - before["vram_used_mb"],
        "VRAM cleanup delta MB": (
            cleanup_snap["vram_used_mb"] - before["vram_used_mb"]
        ),
        "CuPy pool used before MB": before["pool_used_mb"],
        "CuPy pool used after MB": after["pool_used_mb"],
        "CuPy pool used cleanup MB": cleanup_snap["pool_used_mb"],
        "CuPy pool total before MB": before["pool_total_mb"],
        "CuPy pool total after MB": after["pool_total_mb"],
        "CuPy pool total cleanup MB": cleanup_snap["pool_total_mb"],
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
            / "single_gpu_memory_stress.csv"
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat-pairs", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--cpu-workers", default="auto")
    parser.add_argument("--io-workers", default="auto")
    parser.add_argument("--gpu-workers", default=1)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--no-persistent", action="store_true")
    parser.add_argument("--smooth-psf", action="store_true")
    parser.add_argument("--no-fitted-psf", action="store_true")
    parser.add_argument("--no-beta", action="store_true")
    parser.add_argument("--no-shift", action="store_true")
    args = parser.parse_args(argv)

    import cupy as cp

    report = get_cuda_device_report(AccelerationConfig(devices=(args.device,)))
    if not report:
        raise RuntimeError(f"CUDA device {args.device} is not available")
    device_name = report[0]["name"]
    pairs = _pair_from_dirs(args.ref_dir, args.new_dir) * args.repeat_pairs
    process = psutil.Process()
    rows = []
    for round_index in range(1, args.rounds + 1):
        row = run_round(
            round_index, pairs, args, cp, process, args.device, device_name
        )
        rows.append(row)
        print(
            "round {round}: throughput={throughput:.3f} pairs/s "
            "rss_cleanup_delta={rss:.2f}MB "
            "vram_cleanup_delta={vram:.2f}MB failures={failures}".format(
                round=round_index,
                throughput=row["Throughput pairs/s"],
                rss=row["RSS cleanup delta MB"],
                vram=row["VRAM cleanup delta MB"],
                failures=row["Failures"],
            )
        )
        if args.cooldown > 0:
            time.sleep(args.cooldown)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
