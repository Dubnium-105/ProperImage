# ProperImage GPU Acceleration Status

## Current Status (2026-06-06)

### Acceleration Tracks

| Plan | Path | Method | Repository | Status |
|------|------|--------|------------|--------|
| Plan 7 | FFT domain | CuPy/cuFFT plus batch scheduling | This repository | Implemented and validated on real FITS data |
| Plan 8 | Spatial domain | CUDA fused kernel | Dubnium-105/ois | Implemented on the `gpu-ois-v0.2` branch |

### Plan 7 Scope

**Target**: accelerate `properimage.operations.subtract()`.

**Main bottlenecks**:
- Repeated IFFT calls inside `scipy.optimize` callbacks.
- Final variance-correction FFT/IFFT work.
- PSF rendering and background estimation.

**Implemented**:
- Added `subtract(..., use_gpu="auto")` as the default path, preferring
  CuPy/cuFFT when available and falling back to CPU otherwise.
- `use_gpu=True` strictly requires GPU, while `use_gpu=False` forces CPU.
- Added a CuPy Fourier-domain shift implementation for the GPU path.
- Kept residual images and masks on GPU inside optimizer callbacks, returning
  only scalar costs to SciPy.
- Normalized scalar masks to full-size boolean masks so real FITS inputs work
  in the optimizer cost slicing path.
- Added benchmark and export scripts under `gpu_properimage/benchmarks/`.
- Recorded synthetic and real-data CPU/GPU comparisons under
  `gpu_properimage/docs/`.
- Added `AccelerationConfig`, `subtract_batch`, and
  `properimage-subtract-batch` for CPU/GPU worker control, bounded prefetch,
  task-level multi-GPU dispatch, manifest output, and multi-GPU stress testing.

**Current limitations**:
- `SingleImage` construction, PSF rendering, and SEP background estimation are
  still CPU-side, but batch mode can overlap that work with GPU subtraction
  tasks.
- `scipy.optimize` remains CPU-side, although repeated FFT/IFFT work can now
  run on GPU.
- The installed CuPy package must match the host CUDA runtime.
- Multi-GPU support is task-level parallelism; one individual image pair is not
  split across devices.
- Benchmark results in this folder are reference measurements from one
  development environment. Re-run the benchmark scripts on target systems
  before choosing production worker and prefetch settings.

### Documentation

- `PLAN.md`: plan 7 target, bottlenecks, and strategy.
- `ACCEL7_PRODUCTION_BENCHMARK.md`: synthetic benchmark results.
- `batch_acceleration_benchmark.csv`: optional real-folder batch scheduler
  benchmark output.
- `SINGLE_GPU_MEMORY_STRESS.md`: single-GPU repeated-run memory/VRAM stress
  test summary.
- `SINGLE_GPU_THROUGHPUT_SWEEP.md`: single-GPU throughput sweep using the
  real FITS test workflow.
- `REAL_DATA_CPU_GPU_SUBTRACT.md`: original two-file real-data benchmark.
- `REAL_FOLDER_CPU_GPU_SUBTRACT.md`: paired `data/ref` and `data/new`
  real-data benchmark.
- `README.md`: documentation index and ignored artifact policy.
