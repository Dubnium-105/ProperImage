# Single-GPU Memory Stress Test

## Purpose

This report checks whether the single-GPU batch subtraction path releases CPU
memory, GPU memory, and CuPy memory-pool allocations across repeated real-data
runs in one Python process.

The benchmark is intended as a regression check for resource cleanup. It is
not a universal performance claim; throughput depends on image dimensions,
subtraction options, CUDA hardware, drivers, and the available CPU cores.

## Workload

| Setting | Value |
|---|---:|
| Input data | Paired FITS files from `data/ref` and `data/new` |
| CUDA devices | 1 |
| Rounds | 5 |
| Pairs per round | 3 |
| GPU workers | 1 |
| Prefetch | 2 |
| Subtraction mode | `--no-beta --no-shift` |

Each round clears the ProperImage acceleration cache, runs garbage collection,
synchronizes CUDA, and releases CuPy default and pinned memory pools before
recording post-cleanup memory.

```console
python gpu_properimage/benchmarks/stress_single_gpu_memory.py \
  --ref-dir data/ref \
  --new-dir data/new \
  --rounds 5 \
  --repeat-pairs 1 \
  --device 0 \
  --prefetch 2 \
  --gpu-workers 1 \
  --no-beta \
  --no-shift \
  --output gpu_properimage/docs/single_gpu_memory_stress.csv
```

## Results

All 15 subtraction tasks completed successfully and produced finite outputs.

| Metric | Result |
|---|---:|
| Throughput min | 0.110 pairs/s |
| Throughput max | 0.115 pairs/s |
| Throughput mean | 0.114 pairs/s |
| Mean round elapsed | 26.44 s |
| Failures | 0 |
| CuPy pool cleanup total | 0.0 MB every round |

Resource behavior:

- Round 1 showed expected warm-up growth after CUDA, CuPy, and FFT
  initialization.
- Rounds 2-5 showed stable post-cleanup GPU memory: cleanup delta was
  0.00 MB every round.
- Rounds 2-5 CPU RSS cleanup deltas were 3.80 MB, 1.96 MB, 0.41 MB, and
  0.73 MB, with no monotonic growth trend.

## Conclusion

No repeated-run GPU memory leak was observed after warm-up. CuPy memory-pool
allocations returned to zero after cleanup, and CPU RSS remained stable within a
small residual range.

Detailed per-round data is in `single_gpu_memory_stress.csv`.
