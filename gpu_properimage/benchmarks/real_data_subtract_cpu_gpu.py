#!/usr/bin/env python
"""Run real ProperImage CPU/GPU subtraction on data/*.fit images."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REF = REPO_ROOT / "data" / "aligned_eso085-030-004.fit"
DEFAULT_NEW = REPO_ROOT / "data" / "aligned_eso085-030-005.fit"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "gpu_properimage"
    / "docs"
    / "real_data_subtract_cpu_gpu.csv"
)


CSV_COLUMNS = [
    "Mode",
    "Ref",
    "New",
    "Shape",
    "CPU ms",
    "GPU ms",
    "Speedup",
    "Mask true",
    "CPU D finite",
    "GPU D finite",
    "CPU P finite",
    "GPU P finite",
    "CPU S finite",
    "GPU S finite",
    "CPU D mean",
    "GPU D mean",
    "CPU D std",
    "GPU D std",
    "CPU S mean",
    "GPU S mean",
    "CPU S std",
    "GPU S std",
    "D max abs error",
    "D mean abs error",
    "P max abs error",
    "S max abs error",
    "S mean abs error",
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
    """Let CuPy find NVRTC DLLs when a CUDA Torch wheel provides them."""
    try:
        import torch
    except ImportError:
        return

    path = Path(torch.__file__).resolve().parent / "lib"
    if path.exists():
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{path}{os.pathsep}{current_path}"
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))


def warmup_cupy(cp):
    arr = cp.ones((32, 32), dtype=cp.float32)
    _ = cp.fft.fftn(arr)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()


def run_subtract(subtract, cp, ref, new, mode_name, params, use_gpu):
    gc.collect()
    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    d_img, p_img, s_img, mask = subtract(
        str(ref),
        str(new),
        use_gpu=use_gpu,
        **params,
    )
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "mode": mode_name,
        "elapsed_ms": elapsed_ms,
        "D": d_img,
        "P": p_img,
        "S": s_img,
        "mask": mask,
    }


def stats(prefix, result):
    d_real = np.real(result["D"])
    return {
        f"{prefix} D finite": bool(np.isfinite(result["D"]).all()),
        f"{prefix} P finite": bool(np.isfinite(result["P"]).all()),
        f"{prefix} S finite": bool(np.isfinite(result["S"]).all()),
        f"{prefix} D mean": float(np.mean(d_real)),
        f"{prefix} D std": float(np.std(d_real)),
        f"{prefix} S mean": float(np.mean(result["S"])),
        f"{prefix} S std": float(np.std(result["S"])),
    }


def compare(cpu, gpu):
    d_cpu = np.real(cpu["D"])
    d_gpu = np.real(gpu["D"])
    return {
        "D max abs error": float(np.max(np.abs(d_cpu - d_gpu))),
        "D mean abs error": float(np.mean(np.abs(d_cpu - d_gpu))),
        "P max abs error": float(np.max(np.abs(cpu["P"] - gpu["P"]))),
        "S max abs error": float(np.max(np.abs(cpu["S"] - gpu["S"]))),
        "S mean abs error": float(np.mean(np.abs(cpu["S"] - gpu["S"]))),
    }


def row_for_mode(subtract, cp, ref, new, mode_name, params):
    cpu = run_subtract(subtract, cp, ref, new, mode_name, params, False)
    gpu = run_subtract(subtract, cp, ref, new, mode_name, params, True)
    row = {
        "Mode": mode_name,
        "Ref": ref.name,
        "New": new.name,
        "Shape": "x".join(str(v) for v in cpu["D"].shape),
        "CPU ms": cpu["elapsed_ms"],
        "GPU ms": gpu["elapsed_ms"],
        "Speedup": cpu["elapsed_ms"] / gpu["elapsed_ms"],
        "Mask true": int(np.sum(cpu["mask"])),
    }
    row.update(stats("CPU", cpu))
    row.update(stats("GPU", gpu))
    row.update(compare(cpu, gpu))
    return row


def round_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref",
        type=Path,
        default=DEFAULT_REF,
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=DEFAULT_NEW,
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Directory of reference FITS files. Overrides --ref.",
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=None,
        help="Directory of new FITS files. Overrides --new.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_PARAMS),
        default=["fixed_beta_no_shift", "default_beta_shift"],
        help="Subtraction modes to run.",
    )
    return parser.parse_args()


def fits_files(folder):
    suffixes = {".fit", ".fits", ".fts"}
    return sorted(
        path for path in folder.iterdir() if path.is_file()
        and path.suffix.lower() in suffixes
    )


def discover_pairs(args):
    if args.ref_dir is None and args.new_dir is None:
        return [(args.ref, args.new)]
    if args.ref_dir is None or args.new_dir is None:
        raise ValueError("--ref-dir and --new-dir must be provided together")

    ref_files = fits_files(args.ref_dir)
    new_files = fits_files(args.new_dir)
    new_by_name = {path.name: path for path in new_files}
    pairs = []
    for ref_file in ref_files:
        if ref_file.name in new_by_name:
            pairs.append((ref_file, new_by_name[ref_file.name]))

    if len(pairs) == len(ref_files) == len(new_files):
        return pairs

    if len(ref_files) != len(new_files):
        raise ValueError(
            f"Cannot pair {len(ref_files)} ref files with "
            f"{len(new_files)} new files"
        )
    return list(zip(ref_files, new_files))


def main():
    args = parse_args()
    add_torch_cuda_dll_path()

    import cupy as cp

    warmup_cupy(cp)

    from properimage import subtract

    rows = []
    for ref, new in discover_pairs(args):
        for name in args.modes:
            params = MODE_PARAMS[name]
            print(f"running {name}: {ref.name} -> {new.name}")
            rows.append(row_for_mode(subtract, cp, ref, new, name, params))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round_value(row[key]) for key in CSV_COLUMNS})

    print(args.output)
    for row in rows:
        print(
            f"{row['Mode']}: CPU {row['CPU ms']:.2f} ms, "
            f"GPU {row['GPU ms']:.2f} ms, speedup {row['Speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
