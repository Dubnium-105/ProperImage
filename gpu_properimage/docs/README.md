# ProperImage GPU Acceleration Docs

This folder records acceleration plan 7, the CuPy/cuFFT implementation for
`properimage.operations.subtract`.

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
