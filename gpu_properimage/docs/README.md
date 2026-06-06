# ProperImage GPU Acceleration Documentation

This folder documents the CuPy/cuFFT implementation and batch scheduling tools
for GPU-accelerated `properimage.operations.subtract`.

## User Usage

The public API defaults to `use_gpu="auto"`: it tries the CuPy/cuFFT backend
first and falls back to CPU when GPU support is unavailable.

```python
from properimage.operations import subtract

D, P, Scorr, mask = subtract(
    ref=ref_path,
    new=new_path,
    fitted_psf=True,
    beta=True,
    shift=True,
)
```

Install the GPU extra matching the target CUDA runtime before relying on the
automatic GPU path. For CUDA 12 installations:

```console
python -m pip install properimage[gpu-cu12]
```

For CUDA 11 installations, use `properimage[gpu-cu11]`. Set `use_gpu=True` to
require GPU and raise an error if CuPy/CUDA cannot be used. Set
`use_gpu=False` to force the CPU backend.

On Windows, some CuPy installations need NVRTC DLLs from another CUDA package.
If a CUDA-enabled Torch wheel is installed, ProperImage will automatically
expose Torch's CUDA DLL directory to the current process before checking the
CuPy backend.

## What Runs On GPU

The current implementation moves the FFT-heavy subtraction path onto CuPy:

- PSF and image FFTs.
- Fourier-domain shifts.
- Repeated IFFT residual calculations inside `scipy.optimize` callbacks.
- Final decorrelated subtraction, PSF, and corrected score FFT/IFFT work.

The following stages are still CPU-side:

- FITS loading and `SingleImage` construction.
- PSF modeling/rendering.
- SEP background estimation.
- The SciPy optimizer driver, which receives scalar costs from the GPU path.

This means GPU speedups are most visible for `beta=True` and/or `shift=True`,
where the optimizer repeatedly evaluates FFT/IFFT-heavy residuals.

## Batch Processing

Batch subtraction can keep CPU preprocessing and CUDA work active across many
paired FITS files:

```console
properimage-subtract-batch \
  --ref-dir data/ref \
  --new-dir data/new \
  --output-dir res \
  --devices auto \
  --prefetch 4
```

The command writes a manifest CSV with per-pair device id, elapsed time, finite
status, output paths, and failure details. `--devices cpu` or `--use-gpu false`
forces the CPU path.

For multi-GPU validation, require at least two CUDA devices and repeat the same
workload as a stress test:

```console
properimage-subtract-batch \
  --ref-dir data/ref \
  --new-dir data/new \
  --output-dir res \
  --devices auto \
  --prefetch 4 \
  --stress-repeats 20 \
  --require-multi-gpu
```

## Benchmark Commands

The benchmark CSV files in this directory record one development environment's
measurements. Treat the values as reference data for regression checks, not as
portable hardware claims. Re-run the scripts on target systems before choosing
production defaults.

Single-GPU memory and VRAM stress test:

```console
python gpu_properimage/benchmarks/stress_single_gpu_memory.py \
  --ref-dir data/ref \
  --new-dir data/new \
  --rounds 5 \
  --device 0 \
  --prefetch 2 \
  --gpu-workers 1 \
  --no-beta \
  --no-shift \
  --output gpu_properimage/docs/single_gpu_memory_stress.csv
```

Single-GPU throughput sweep:

```console
python gpu_properimage/benchmarks/benchmark_single_gpu_throughput.py \
  --ref-dir data/ref \
  --new-dir data/new \
  --device 0 \
  --repeat-pairs 2 \
  --gpu-workers 1 2 \
  --prefetch 1 2 3 4 \
  --warmup \
  --no-beta \
  --no-shift \
  --output gpu_properimage/docs/single_gpu_throughput_sweep.csv
```

Extended single-GPU worker sweep:

```console
python gpu_properimage/benchmarks/benchmark_single_gpu_throughput.py \
  --ref-dir data/ref \
  --new-dir data/new \
  --device 0 \
  --repeat-pairs 2 \
  --gpu-workers 3 4 6 8 \
  --prefetch 2 3 4 6 8 \
  --warmup \
  --no-beta \
  --no-shift \
  --output gpu_properimage/docs/single_gpu_throughput_sweep_more_workers.csv
```

Synthetic production-style benchmark:

```console
python gpu_properimage/benchmarks/benchmark_accel7_production.py \
  --pixels 1024 2048 4096 \
  --repeats 3 \
  --output gpu_properimage/docs/accel7_production_benchmark.csv
```

Batch scheduling benchmark on real folders:

```console
python gpu_properimage/benchmarks/benchmark_batch_acceleration.py \
  --ref-dir data/ref \
  --new-dir data/new \
  --modes cpu single-gpu multi-gpu \
  --repeats 5 \
  --output gpu_properimage/docs/batch_acceleration_benchmark.csv
```

CPU/GPU comparison for a single real pair:

```console
python gpu_properimage/benchmarks/real_data_subtract_cpu_gpu.py \
  --ref data/aligned_eso085-030-004.fit \
  --new data/aligned_eso085-030-005.fit \
  --modes fixed_beta_no_shift default_beta_shift \
  --output gpu_properimage/docs/real_data_subtract_cpu_gpu.csv
```

CPU/GPU comparison for paired folders:

```console
python gpu_properimage/benchmarks/real_data_subtract_cpu_gpu.py \
  --ref-dir data/ref \
  --new-dir data/new \
  --modes fixed_beta_no_shift default_beta_shift \
  --output gpu_properimage/docs/real_folder_cpu_gpu_subtract.csv
```

Export CPU/GPU subtraction FITS outputs:

```console
python gpu_properimage/benchmarks/export_subtract_results.py \
  --ref-dir data/ref \
  --new-dir data/new \
  --mode default_beta_shift \
  --output-dir res \
  --manifest res/manifest.csv
```

`data/ref` and `data/new` are paired by identical FITS filenames. If filenames
do not match one-to-one, the benchmark script falls back to sorted order when
the two folders contain the same number of files.

## Documents

- `STATUS.md`: current implementation status, limitations, and cross-repo notes.
- `PLAN.md`: original plan 7 target, bottlenecks, strategy, and implementation
  status.
- `ACCEL7_PRODUCTION_BENCHMARK.md`: synthetic production-style CPU/GPU
  benchmark with pipeline and persistent timings.
- `REAL_DATA_CPU_GPU_SUBTRACT.md`: CPU/GPU subtraction test for the original
  two FITS files in `data/`.
- `REAL_FOLDER_CPU_GPU_SUBTRACT.md`: CPU/GPU subtraction test for three paired
  FITS files from `data/ref` and `data/new`.
- `SINGLE_GPU_MEMORY_STRESS.md`: repeated single-GPU memory/VRAM stress test.
- `SINGLE_GPU_THROUGHPUT_SWEEP.md`: single-GPU throughput sweep for the
  bundled real FITS test workflow.

## CSV Outputs

- `accel7_production_benchmark.csv`
- `real_data_subtract_cpu_gpu.csv`
- `real_folder_cpu_gpu_subtract.csv`
- `single_gpu_memory_stress.csv`
- `single_gpu_throughput_sweep.csv`
- `single_gpu_throughput_sweep_more_workers.csv`

## Ignored Artifacts

Large FITS input/output artifacts are intentionally ignored:

- `data/ref/`
- `data/new/`
- `res/`

Use `gpu_properimage/benchmarks/export_subtract_results.py` to regenerate
CPU/GPU subtraction FITS outputs into `res/`.
