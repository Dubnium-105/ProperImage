#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Command line tools for ProperImage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .acceleration import (
    AccelerationConfig,
    get_cuda_device_report,
    resolve_cuda_devices,
    subtract_batch,
)


def _parse_devices(value):
    if value in ("auto", "cpu", "off"):
        return value
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _fits_files(path):
    path = Path(path)
    patterns = ("*.fits", "*.fit", "*.fts", "*.FITS", "*.FIT", "*.FTS")
    files = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted(set(files))


def _pair_from_dirs(ref_dir, new_dir):
    refs = _fits_files(ref_dir)
    news = _fits_files(new_dir)
    ref_by_name = {path.name: path for path in refs}
    new_by_name = {path.name: path for path in news}
    names = sorted(set(ref_by_name) & set(new_by_name))
    if names:
        return [(ref_by_name[name], new_by_name[name]) for name in names]
    if len(refs) != len(news):
        raise ValueError(
            "ref-dir and new-dir must contain matching FITS names or the same "
            "number of FITS files."
        )
    return list(zip(refs, news))


def build_subtract_batch_parser():
    parser = argparse.ArgumentParser(
        prog="properimage-subtract-batch",
        description="Run ProperImage subtraction over paired FITS files.",
    )
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--cpu-workers", default="auto")
    parser.add_argument("--io-workers", default="auto")
    parser.add_argument("--gpu-workers", default="auto")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument(
        "--max-gpu-memory-fraction", type=float, default=0.85
    )
    parser.add_argument(
        "--no-persistent", action="store_true", help="Disable ref cache reuse."
    )
    parser.add_argument(
        "--use-gpu",
        default="auto",
        choices=("auto", "true", "false", "cpu", "gpu"),
    )
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--smooth-psf", action="store_true")
    parser.add_argument("--no-fitted-psf", action="store_true")
    parser.add_argument("--no-beta", action="store_true")
    parser.add_argument("--no-shift", action="store_true")
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--inf-loss", type=float, default=0.25)
    parser.add_argument("--stress-repeats", type=int, default=1)
    parser.add_argument("--require-multi-gpu", action="store_true")
    return parser


def _normalize_use_gpu(value):
    if value in ("true", "gpu"):
        return True
    if value in ("false", "cpu"):
        return False
    return "auto"


def subtract_batch_main(argv=None):
    parser = build_subtract_batch_parser()
    args = parser.parse_args(argv)
    devices = _parse_devices(args.devices)
    config = AccelerationConfig(
        devices=devices,
        cpu_workers=args.cpu_workers,
        io_workers=args.io_workers,
        gpu_workers=args.gpu_workers,
        prefetch=args.prefetch,
        persistent=not args.no_persistent,
        max_gpu_memory_fraction=args.max_gpu_memory_fraction,
    )
    device_ids = resolve_cuda_devices(config, strict=False)
    if args.require_multi_gpu and len(device_ids) < 2:
        print(
            "properimage-subtract-batch: fewer than two CUDA devices are "
            "visible",
            file=sys.stderr,
        )
        for item in get_cuda_device_report(config):
            print(
                "device {device_id}: {name} free={free_mb:.1f}MB "
                "total={total_mb:.1f}MB".format(**item),
                file=sys.stderr,
            )
        return 2

    pairs = _pair_from_dirs(args.ref_dir, args.new_dir)
    if args.stress_repeats > 1:
        pairs = pairs * args.stress_repeats
    manifest = args.manifest or str(Path(args.output_dir) / "manifest.csv")
    summary = subtract_batch(
        pairs,
        output_dir=args.output_dir,
        manifest=manifest,
        acceleration=config,
        use_gpu=_normalize_use_gpu(args.use_gpu),
        align=args.align,
        smooth_psf=args.smooth_psf,
        fitted_psf=not args.no_fitted_psf,
        beta=not args.no_beta,
        shift=not args.no_shift,
        iterative=args.iterative,
        inf_loss=args.inf_loss,
    )
    print(
        "completed {done}/{total} pairs in {elapsed:.2f} s; "
        "throughput={throughput:.3f} pairs/s; manifest={manifest}".format(
            done=sum(1 for task in summary.tasks if task.ok),
            total=len(summary.tasks),
            elapsed=summary.elapsed_ms / 1000.0,
            throughput=summary.throughput_pairs_per_s,
            manifest=manifest,
        )
    )
    if summary.failed:
        for task in summary.failed[:5]:
            print(
                f"failed index={task.index} ref={task.ref} "
                f"new={task.new}: {task.error}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(subtract_batch_main())
