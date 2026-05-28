# ProperImage GPU Acceleration Status

## Current Status (2026-05-28)

### Acceleration Tracks

| Plan | Path | Method | Repository | Status |
|------|------|--------|------------|--------|
| Plan 7 | FFT domain | CuPy/cuFFT | This repository | Implemented and validated on real FITS data |
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

**Current limitations**:
- `SingleImage` construction, PSF rendering, and SEP background estimation are
  still CPU-side.
- `scipy.optimize` remains CPU-side, although repeated FFT/IFFT work can now
  run on GPU.
- The installed CuPy package must match the host CUDA runtime.

### Documentation

- `PLAN.md`: plan 7 target, bottlenecks, and strategy.
- `ACCEL7_PRODUCTION_BENCHMARK.md`: synthetic benchmark results.
- `REAL_DATA_CPU_GPU_SUBTRACT.md`: original two-file real-data benchmark.
- `REAL_FOLDER_CPU_GPU_SUBTRACT.md`: paired `data/ref` and `data/new`
  real-data benchmark.
- `README.md`: documentation index and local artifact policy.
