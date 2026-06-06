# Single-GPU Throughput Sweep

## Purpose

This report estimates the best observed single-GPU batch subtraction throughput
for the real FITS test set under `data/ref` and `data/new`.

The sweep does not write FITS outputs, so the measurement focuses on
subtraction and scheduling throughput rather than output-disk throughput.
Results are workload-specific and should be remeasured on each production
environment.

## Workload

| Setting | Value |
|---|---:|
| Input data | Paired FITS files from `data/ref` and `data/new` |
| CUDA devices | 1 |
| Base pairs | 3 |
| Repeat factor | 2 |
| Tasks per configuration | 6 |
| Subtraction mode | `--no-beta --no-shift` |
| Warm-up | 1 subtraction task before measurement |

Initial sweep:

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

Extended worker sweep:

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

## Best Observed Configurations

| Rank | GPU workers | Prefetch | Throughput pairs/s | Elapsed ms | Failures | Finite |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 4 | 0.1667 | 35996.66 | 0 | true |
| 2 | 8 | 6 | 0.1660 | 36154.02 | 0 | true |
| 3 | 8 | 8 | 0.1652 | 36314.12 | 0 | true |
| 4 | 6 | 6 | 0.1650 | 36352.90 | 0 | true |
| 5 | 6 | 8 | 0.1648 | 36418.29 | 0 | true |

The best observed throughput was **0.1667 pairs/s**, or about
**6.00 seconds per pair** at the batch level for this workload.

## Interpretation

The best one-worker configuration reached 0.1198 pairs/s. The best two-worker
configuration reached 0.1532 pairs/s. The extended sweep reached
0.1667 pairs/s with eight GPU workers.

For this data set, `gpu_workers=8` and `prefetch=4` produced the highest
measured throughput. The 6-8 worker range is close to a plateau, so this value
should be treated as a measured tuning result for this workload rather than a
global default.

Detailed ranked results are in `single_gpu_throughput_sweep.csv` and
`single_gpu_throughput_sweep_more_workers.csv`.
