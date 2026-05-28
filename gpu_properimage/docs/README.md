# ProperImage GPU Acceleration Docs

This folder records acceleration plan 7, the CuPy/cuFFT implementation for
`properimage.operations.subtract`.

## User-Facing GPU Usage

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

Install the GPU extra matching the local CUDA runtime before relying on the
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

## Benchmark And Export Commands

Synthetic production-style benchmark:

```console
python gpu_properimage/benchmarks/benchmark_accel7_production.py \
  --pixels 1024 2048 4096 \
  --repeats 3 \
  --output gpu_properimage/docs/accel7_production_benchmark.csv
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
  FITS files from local `data/ref` and `data/new`.

## CSV Outputs

- `accel7_production_benchmark.csv`
- `real_data_subtract_cpu_gpu.csv`
- `real_folder_cpu_gpu_subtract.csv`

## Local Artifacts

The large local FITS input/output artifacts are intentionally ignored:

- `data/ref/`
- `data/new/`
- `res/`

Use `gpu_properimage/benchmarks/export_subtract_results.py` to regenerate
CPU/GPU subtraction FITS outputs into `res/`.
