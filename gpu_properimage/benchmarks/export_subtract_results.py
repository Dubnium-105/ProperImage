#!/usr/bin/env python
"""Export CPU and GPU subtraction D images for ref/new FITS pairs."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REF_DIR = REPO_ROOT / "data" / "ref"
DEFAULT_NEW_DIR = REPO_ROOT / "data" / "new"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "res"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.csv"


CSV_COLUMNS = [
    "mode",
    "backend",
    "ref",
    "new",
    "shape",
    "elapsed_ms",
    "finite",
    "mean",
    "std",
    "output",
]

MODE_PARAMS = {
    "fixed_beta_no_shift": {
        "align": False,
        "iterative": False,
        "beta": False,
        "shift": False,
        "fitted_psf": True,
        "smooth_psf": False,
    },
    "default_beta_shift": {
        "align": False,
        "iterative": False,
        "beta": True,
        "shift": True,
        "fitted_psf": True,
        "smooth_psf": False,
    },
}


def add_torch_cuda_dll_path():
    try:
        import torch
    except ImportError:
        return

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.exists():
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{torch_lib}{os.pathsep}{current_path}"
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(torch_lib))


def fits_files(folder):
    suffixes = {".fit", ".fits", ".fts"}
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def discover_pairs(ref_dir, new_dir):
    ref_files = fits_files(ref_dir)
    new_files = fits_files(new_dir)
    new_by_name = {path.name: path for path in new_files}
    pairs = [
        (ref_file, new_by_name[ref_file.name])
        for ref_file in ref_files
        if ref_file.name in new_by_name
    ]
    if len(pairs) == len(ref_files) == len(new_files):
        return pairs
    if len(ref_files) != len(new_files):
        raise ValueError(
            f"Cannot pair {len(ref_files)} ref files with "
            f"{len(new_files)} new files"
        )
    return list(zip(ref_files, new_files))


def safe_stem(path):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return stem or "image"


def warmup_cupy(cp):
    arr = cp.ones((32, 32), dtype=cp.float32)
    _ = cp.fft.fftn(arr)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()


def write_fits(output, image, ref, new, backend, mode, elapsed_ms):
    data = np.asarray(np.real(image), dtype=np.float32)
    header = fits.Header()
    header["BACKEND"] = backend
    header["MODE"] = mode
    header["REF"] = ref.name
    header["NEW"] = new.name
    header["ELAPSMS"] = float(elapsed_ms)
    fits.writeto(output, data, header=header, overwrite=True)
    return data


def run_one(subtract, cp, ref, new, backend, use_gpu, mode, out_dir):
    params = MODE_PARAMS[mode]
    gc.collect()
    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    d_img, _, _, _ = subtract(str(ref), str(new), use_gpu=use_gpu, **params)
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    output = out_dir / backend / f"{safe_stem(ref)}__D.fits"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = write_fits(output, d_img, ref, new, backend, mode, elapsed_ms)
    return {
        "mode": mode,
        "backend": backend,
        "ref": str(ref),
        "new": str(new),
        "shape": "x".join(str(v) for v in data.shape),
        "elapsed_ms": elapsed_ms,
        "finite": bool(np.isfinite(data).all()),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "output": str(output),
    }


def round_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--new-dir", type=Path, default=DEFAULT_NEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PARAMS),
        default="default_beta_shift",
        help="Subtraction mode to export.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    add_torch_cuda_dll_path()

    import cupy as cp

    warmup_cupy(cp)

    from properimage import subtract

    rows = []
    for ref, new in discover_pairs(args.ref_dir, args.new_dir):
        for backend, use_gpu in [("cpu", False), ("gpu", True)]:
            print(f"exporting {backend}: {ref.name}")
            rows.append(
                run_one(
                    subtract,
                    cp,
                    ref,
                    new,
                    backend,
                    use_gpu,
                    args.mode,
                    args.output_dir,
                )
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round_value(row[key]) for key in CSV_COLUMNS})

    print(args.manifest)


if __name__ == "__main__":
    main()
