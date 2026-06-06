#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Acceleration helpers for ProperImage subtraction workloads."""

from __future__ import annotations

import csv
import logging
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_DEFAULT_CONFIG_LOCK = threading.Lock()


@dataclass(frozen=True)
class AccelerationConfig:
    """Runtime resource controls for GPU/CPU subtraction workloads."""

    devices: Any = "auto"
    cpu_workers: Any = "auto"
    io_workers: Any = "auto"
    gpu_workers: Any = "auto"
    prefetch: int = 2
    persistent: bool = True
    max_gpu_memory_fraction: float = 0.85


@dataclass(frozen=True)
class SubtractTaskResult:
    """Result metadata for one batch subtraction task."""

    index: int
    ref: str
    new: str
    ok: bool
    device_id: Optional[int]
    elapsed_ms: float
    output_d: Optional[str] = None
    output_p: Optional[str] = None
    output_s_corr: Optional[str] = None
    output_mask: Optional[str] = None
    finite: Optional[bool] = None
    max_abs_d: Optional[float] = None
    mean_abs_d: Optional[float] = None
    error: Optional[str] = None
    cache_hit: bool = False


@dataclass(frozen=True)
class SubtractBatchResult:
    """Summary and per-task metadata from :func:`subtract_batch`."""

    tasks: tuple[SubtractTaskResult, ...]
    elapsed_ms: float
    devices: tuple[int, ...]
    cpu_workers: int
    gpu_workers: int
    cache_hits: int = 0

    @property
    def ok(self):
        return all(task.ok for task in self.tasks)

    @property
    def failed(self):
        return tuple(task for task in self.tasks if not task.ok)

    @property
    def throughput_pairs_per_s(self):
        if self.elapsed_ms <= 0:
            return math.nan
        return len(self.tasks) / (self.elapsed_ms / 1000.0)


@dataclass(frozen=True)
class TuningTrial:
    """One measured candidate from acceleration auto tuning."""

    config: AccelerationConfig
    throughput_pairs_per_s: float
    elapsed_ms: float
    failures: int
    finite: bool


@dataclass(frozen=True)
class TuningResult:
    """Selected acceleration configuration and candidate measurements."""

    config: AccelerationConfig
    trials: tuple[TuningTrial, ...] = ()
    strategy: str = "heuristic"

    @property
    def best_trial(self):
        if not self.trials:
            return None
        return max(self.trials, key=lambda trial: trial.throughput_pairs_per_s)


_DEFAULT_CONFIG = AccelerationConfig()
_REF_CACHE: dict[tuple[Any, ...], Any] = {}
_REF_CACHE_LOCK = threading.Lock()


def configure_acceleration(**kwargs):
    """Set the process-wide default acceleration configuration."""
    global _DEFAULT_CONFIG
    with _DEFAULT_CONFIG_LOCK:
        _DEFAULT_CONFIG = replace(_DEFAULT_CONFIG, **kwargs)
        return _DEFAULT_CONFIG


def get_acceleration_config():
    """Return the process-wide default acceleration configuration."""
    with _DEFAULT_CONFIG_LOCK:
        return _DEFAULT_CONFIG


def clear_acceleration_cache():
    """Clear CPU-side persistent preparation caches."""
    with _REF_CACHE_LOCK:
        _REF_CACHE.clear()


def resolve_cuda_devices(config=None, strict=False):
    """Return visible CUDA device ids for a configuration."""
    config = config or get_acceleration_config()
    devices = config.devices
    if devices in ("cpu", "off", None, False):
        return ()
    if isinstance(devices, int):
        return (devices,)
    if isinstance(devices, str) and devices != "auto":
        return tuple(
            int(part.strip())
            for part in devices.split(",")
            if part.strip() != ""
        )
    if isinstance(devices, Sequence) and not isinstance(devices, str):
        return tuple(int(device) for device in devices)

    try:
        import cupy as cp

        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        if strict:
            raise RuntimeError("No usable CUDA devices were found.") from exc
        logger.info("CUDA device discovery failed: %s", exc)
        return ()
    return tuple(range(count))


def resolve_worker_count(value, default, minimum=1):
    """Normalize worker settings such as ``auto`` and integer strings."""
    if value in (None, "auto"):
        return max(minimum, int(default))
    count = int(value)
    if count < minimum:
        raise ValueError("worker counts must be positive")
    return count


def default_cpu_workers(config=None):
    """Choose a conservative CPU worker count for GPU pipelines."""
    config = config or get_acceleration_config()
    cores = os.cpu_count() or 1
    visible_gpus = len(resolve_cuda_devices(config, strict=False))
    return max(1, cores - visible_gpus)


def _candidate_values(values, upper):
    candidates = []
    for value in values:
        value = max(1, min(int(value), upper))
        if value not in candidates:
            candidates.append(value)
    return candidates


def heuristic_acceleration_config(base=None, use_gpu="auto"):
    """Return a hardware-aware config without running benchmark trials."""
    base = base or get_acceleration_config()
    devices = resolve_cuda_devices(base, strict=False)
    if use_gpu in (False, "off", "cpu", None) or not devices:
        workers = resolve_worker_count(
            base.cpu_workers,
            max(1, os.cpu_count() or 1),
        )
        return replace(
            base,
            devices="cpu",
            cpu_workers=workers,
            io_workers=resolve_worker_count(base.io_workers, workers),
            gpu_workers=1,
            prefetch=max(1, int(base.prefetch)),
        )

    cores = os.cpu_count() or 1
    gpu_count = len(devices)
    cpu_workers = resolve_worker_count(
        base.cpu_workers, max(1, cores - gpu_count)
    )
    per_gpu_workers = resolve_worker_count(
        base.gpu_workers, min(8, max(1, cpu_workers // gpu_count))
    )
    prefetch = max(int(base.prefetch), per_gpu_workers * gpu_count)
    return replace(
        base,
        devices=devices,
        cpu_workers=cpu_workers,
        io_workers=resolve_worker_count(base.io_workers, cpu_workers),
        gpu_workers=per_gpu_workers,
        prefetch=prefetch,
    )


def tune_acceleration(
    pairs,
    *,
    acceleration=None,
    use_gpu="auto",
    max_trials=8,
    sample_size=6,
    **subtract_kwargs,
):
    """Measure candidate worker settings and return the fastest config."""
    base = acceleration or get_acceleration_config()
    heuristic = heuristic_acceleration_config(base, use_gpu=use_gpu)
    pairs = tuple(pairs)
    if not pairs:
        return TuningResult(config=heuristic, strategy="empty")

    sample = pairs[: max(1, min(int(sample_size), len(pairs)))]
    if heuristic.devices in ("cpu", "off", None, False):
        cpu_candidates = _candidate_values(
            [1, heuristic.cpu_workers, (os.cpu_count() or 1)],
            os.cpu_count() or 1,
        )
        configs = [
            replace(
                heuristic,
                cpu_workers=workers,
                io_workers=workers,
                gpu_workers=1,
                prefetch=max(1, min(len(sample), workers)),
            )
            for workers in cpu_candidates
        ]
    else:
        cores = os.cpu_count() or 1
        gpu_candidates = _candidate_values(
            [1, 2, 4, 6, 8, heuristic.gpu_workers],
            max(1, cores),
        )
        prefetch_candidates = _candidate_values(
            [1, 2, 3, 4, 6, 8, heuristic.prefetch],
            max(1, len(sample)),
        )
        configs = []
        for gpu_workers in gpu_candidates:
            for prefetch in prefetch_candidates:
                configs.append(
                    replace(
                        heuristic,
                        gpu_workers=gpu_workers,
                        prefetch=prefetch,
                    )
                )
                if len(configs) >= max_trials:
                    break
            if len(configs) >= max_trials:
                break

    trials = []
    best = None
    for config in configs[: max(1, int(max_trials))]:
        clear_acceleration_cache()
        result = subtract_batch(
            sample,
            acceleration=config,
            use_gpu=use_gpu,
            **subtract_kwargs,
        )
        finite = all(task.finite for task in result.tasks if task.ok)
        trial = TuningTrial(
            config=config,
            throughput_pairs_per_s=result.throughput_pairs_per_s,
            elapsed_ms=result.elapsed_ms,
            failures=len(result.failed),
            finite=finite,
        )
        trials.append(trial)
        if trial.failures == 0 and trial.finite:
            if best is None:
                best = trial
            elif trial.throughput_pairs_per_s > best.throughput_pairs_per_s:
                best = trial

    clear_acceleration_cache()
    if best is None:
        return TuningResult(
            config=heuristic, trials=tuple(trials), strategy="heuristic"
        )
    return TuningResult(
        config=best.config, trials=tuple(trials), strategy="measured"
    )


@contextmanager
def cpu_thread_context(workers):
    """Temporarily constrain common native CPU thread pools."""
    workers = resolve_worker_count(workers, workers)
    previous = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    value = str(workers)
    try:
        for name in _THREAD_ENV_VARS:
            os.environ[name] = value
        try:
            import pyfftw

            old_pyfftw_threads = pyfftw.config.NUM_THREADS
            pyfftw.config.NUM_THREADS = workers
        except Exception:
            old_pyfftw_threads = None
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
        if old_pyfftw_threads is not None:
            try:
                import pyfftw

                pyfftw.config.NUM_THREADS = old_pyfftw_threads
            except Exception:
                pass


@contextmanager
def cuda_device_context(device_id, use_stream=True):
    """Bind CuPy work to one CUDA device and optional stream."""
    if device_id is None:
        yield
        return
    import cupy as cp

    with cp.cuda.Device(device_id):
        stream = cp.cuda.Stream(non_blocking=True) if use_stream else None
        if stream is None:
            yield
        else:
            with stream:
                yield
            stream.synchronize()


def get_cuda_device_report(config=None):
    """Inspect CUDA devices for benchmark and stress-test manifests."""
    devices = resolve_cuda_devices(config, strict=False)
    report = []
    if not devices:
        return report
    try:
        import cupy as cp
    except Exception:
        return report
    for device_id in devices:
        with cp.cuda.Device(device_id):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            name = props.get("name", b"")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            report.append(
                {
                    "device_id": device_id,
                    "name": name,
                    "free_mb": free_mem / 1024**2,
                    "total_mb": total_mem / 1024**2,
                }
            )
    return report


def _pair_label(value):
    if isinstance(value, (str, os.PathLike)):
        return str(value)
    filename = getattr(value, "filename", None)
    if filename:
        return str(filename)
    return repr(value)


def _ref_cache_key(ref, smooth_psf):
    if isinstance(ref, (str, os.PathLike)):
        path = Path(ref)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = (None, None)
        return ("path", str(path.resolve()), stamp, bool(smooth_psf))
    return ("object", id(ref), bool(smooth_psf))


def _prepare_pair(index, ref, new, smooth_psf, fitted_psf, persistent):
    from .single_image import SingleImage, SingleImageGaussPSF

    cls = SingleImageGaussPSF if fitted_psf else SingleImage
    cache_hit = False
    if isinstance(ref, (str, os.PathLike)) or isinstance(
        new, (str, os.PathLike)
    ):
        prepared_ref = str(ref) if isinstance(ref, os.PathLike) else ref
        prepared_new = str(new) if isinstance(new, os.PathLike) else new
        return index, ref, new, prepared_ref, prepared_new, cache_hit

    if isinstance(ref, cls):
        prepared_ref = ref
    elif persistent:
        key = _ref_cache_key(ref, smooth_psf)
        with _REF_CACHE_LOCK:
            prepared_ref = _REF_CACHE.get(key)
        if prepared_ref is None:
            prepared_ref = cls(ref, smooth_psf=smooth_psf)
            with _REF_CACHE_LOCK:
                _REF_CACHE[key] = prepared_ref
        else:
            cache_hit = True
    else:
        prepared_ref = cls(ref, smooth_psf=smooth_psf)

    if isinstance(new, cls):
        prepared_new = new
    else:
        prepared_new = cls(new, smooth_psf=smooth_psf)
    return index, ref, new, prepared_ref, prepared_new, cache_hit


def _sanitize_name(value, fallback):
    if isinstance(value, (str, os.PathLike)):
        stem = Path(value).stem
        if stem:
            return stem
    return fallback


def _write_outputs(output_dir, index, ref, new, result):
    from astropy.io import fits

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_name = _sanitize_name(ref, f"ref{index:04d}")
    new_name = _sanitize_name(new, f"new{index:04d}")
    prefix = output_dir / f"{index:04d}_{ref_name}__{new_name}"
    d_img, p_img, s_img, mask = result
    paths = {
        "output_d": str(prefix.with_suffix(".D.fits")),
        "output_p": str(prefix.with_suffix(".P.fits")),
        "output_s_corr": str(prefix.with_suffix(".S_corr.fits")),
        "output_mask": str(prefix.with_suffix(".mask.fits")),
    }
    fits.writeto(paths["output_d"], np.asarray(d_img.real), overwrite=True)
    fits.writeto(paths["output_p"], np.asarray(p_img), overwrite=True)
    fits.writeto(paths["output_s_corr"], np.asarray(s_img), overwrite=True)
    fits.writeto(
        paths["output_mask"], np.asarray(mask, dtype=np.uint8), overwrite=True
    )
    return paths


def _run_prepared_subtract(
    prepared,
    device_id,
    config,
    output_dir,
    return_results,
    subtract_kwargs,
):
    from .operations import subtract

    index, ref, new, prepared_ref, prepared_new, cache_hit = prepared
    use_gpu = subtract_kwargs.pop("use_gpu", "auto")
    if device_id is not None and use_gpu == "auto":
        use_gpu = True
    if device_id is None:
        use_gpu = False if use_gpu == "auto" else use_gpu
    local_config = replace(config, devices=(() if device_id is None else (device_id,)))
    start = time.perf_counter()
    try:
        with cuda_device_context(device_id):
            result = subtract(
                prepared_ref,
                prepared_new,
                use_gpu=use_gpu,
                acceleration=local_config,
                **subtract_kwargs,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        d_img, _, s_img, _ = result
        finite = bool(np.all(np.isfinite(np.asarray(d_img).real))) and bool(
            np.all(np.isfinite(np.asarray(s_img)))
        )
        paths = {}
        if output_dir is not None:
            paths = _write_outputs(output_dir, index, ref, new, result)
        if not return_results:
            result = None
        max_abs = float(np.max(np.abs(np.asarray(d_img).real)))
        mean_abs = float(np.mean(np.abs(np.asarray(d_img).real)))
        return SubtractTaskResult(
            index=index,
            ref=_pair_label(ref),
            new=_pair_label(new),
            ok=True,
            device_id=device_id,
            elapsed_ms=elapsed_ms,
            finite=finite,
            max_abs_d=max_abs,
            mean_abs_d=mean_abs,
            cache_hit=cache_hit,
            **paths,
        ), result
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (
            SubtractTaskResult(
                index=index,
                ref=_pair_label(ref),
                new=_pair_label(new),
                ok=False,
                device_id=device_id,
                elapsed_ms=elapsed_ms,
                error=str(exc),
                cache_hit=cache_hit,
            ),
            None,
        )


def _choose_device(devices, counts):
    if not devices:
        return None
    return min(devices, key=lambda device_id: counts.get(device_id, 0))


def subtract_batch(
    pairs,
    *,
    output_dir=None,
    return_results=False,
    acceleration=None,
    manifest=None,
    **subtract_kwargs,
):
    """Run many subtractions with bounded CPU/GPU parallelism."""
    config = acceleration or get_acceleration_config()
    pairs = tuple(pairs)
    use_gpu = subtract_kwargs.get("use_gpu", "auto")
    if use_gpu in (False, "off", "cpu", None):
        devices = ()
    else:
        devices = resolve_cuda_devices(config, strict=(use_gpu is True))
    cpu_workers = resolve_worker_count(
        config.cpu_workers, default_cpu_workers(config)
    )
    io_workers = resolve_worker_count(config.io_workers, cpu_workers)
    default_gpu_workers = 1 if devices else cpu_workers
    gpu_workers = resolve_worker_count(config.gpu_workers, default_gpu_workers)
    prefetch = max(1, int(config.prefetch))
    smooth_psf = bool(subtract_kwargs.get("smooth_psf", False))
    fitted_psf = bool(subtract_kwargs.get("fitted_psf", True))
    start = time.perf_counter()
    task_results = []
    returned_results = {}
    device_counts = {device_id: 0 for device_id in devices}
    compute_futures = set()
    prepare_futures = set()
    prepare_meta = {}
    next_index = 0
    compute_executor = ThreadPoolExecutor(
        max_workers=max(1, len(devices) * gpu_workers if devices else cpu_workers)
    )

    try:
        with cpu_thread_context(1 if devices else cpu_workers):
            with ThreadPoolExecutor(max_workers=io_workers) as prepare_executor:
                while next_index < len(pairs) and len(prepare_futures) < prefetch:
                    ref, new = pairs[next_index]
                    future = prepare_executor.submit(
                        _prepare_pair,
                        next_index,
                        ref,
                        new,
                        smooth_psf,
                        fitted_psf,
                        config.persistent,
                    )
                    prepare_futures.add(future)
                    prepare_meta[future] = (next_index, ref, new)
                    next_index += 1

                while prepare_futures or compute_futures:
                    done_prepare, prepare_futures = wait(
                        prepare_futures,
                        timeout=0.01,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done_prepare:
                        meta = prepare_meta.pop(future, None)
                        try:
                            prepared = future.result()
                        except Exception as exc:
                            index, ref, new = meta or (-1, "", "")
                            task_results.append(
                                SubtractTaskResult(
                                    index=index,
                                    ref=_pair_label(ref),
                                    new=_pair_label(new),
                                    ok=False,
                                    device_id=None,
                                    elapsed_ms=0.0,
                                    error=str(exc),
                                )
                            )
                            continue
                        device_id = _choose_device(devices, device_counts)
                        if device_id is not None:
                            device_counts[device_id] += 1
                        kwargs_copy = dict(subtract_kwargs)
                        compute_futures.add(
                            compute_executor.submit(
                                _run_prepared_subtract,
                                prepared,
                                device_id,
                                config,
                                output_dir,
                                return_results,
                                kwargs_copy,
                            )
                        )

                    while next_index < len(pairs) and len(prepare_futures) < prefetch:
                        ref, new = pairs[next_index]
                        future = prepare_executor.submit(
                            _prepare_pair,
                            next_index,
                            ref,
                            new,
                            smooth_psf,
                            fitted_psf,
                            config.persistent,
                        )
                        prepare_futures.add(future)
                        prepare_meta[future] = (next_index, ref, new)
                        next_index += 1

                    done_compute, compute_futures = wait(
                        compute_futures,
                        timeout=0.01,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done_compute:
                        task_result, result = future.result()
                        task_results.append(task_result)
                        if return_results:
                            returned_results[task_result.index] = result
    finally:
        compute_executor.shutdown(wait=True)

    task_results.sort(key=lambda item: item.index)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    summary = SubtractBatchResult(
        tasks=tuple(task_results),
        elapsed_ms=elapsed_ms,
        devices=tuple(devices),
        cpu_workers=cpu_workers,
        gpu_workers=gpu_workers,
        cache_hits=sum(1 for task in task_results if task.cache_hit),
    )
    if manifest is not None:
        write_manifest(manifest, summary)
    if return_results:
        return summary, tuple(
            returned_results.get(index) for index in range(len(pairs))
        )
    return summary


def write_manifest(path, batch_result):
    """Write batch task metadata as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "ref",
        "new",
        "ok",
        "device_id",
        "elapsed_ms",
        "output_d",
        "output_p",
        "output_s_corr",
        "output_mask",
        "finite",
        "max_abs_d",
        "mean_abs_d",
        "error",
        "cache_hit",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for task in batch_result.tasks:
            writer.writerow({field: getattr(task, field) for field in fields})
