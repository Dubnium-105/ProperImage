# Real Data CPU/GPU Subtraction

Date: 2026-05-28

## Input

- Reference: `data/aligned_eso085-030-004.fit`
- New image: `data/aligned_eso085-030-005.fit`
- Shape: `682x1024`

## Dependencies

Install the project runtime dependencies plus a CuPy wheel matching the local
CUDA runtime. On CUDA 12 systems, for example:

```powershell
python -m pip install `
  astroscrappy cupy-cuda12x tinynpydb pyfftw
```

## Command

```powershell
python `
  gpu_properimage\benchmarks\real_data_subtract_cpu_gpu.py `
  --output gpu_properimage\docs\real_data_subtract_cpu_gpu.csv
```

## Results

| Mode | CPU ms | GPU ms | Speedup | Mask true | CPU D finite | GPU D finite | CPU S finite | GPU S finite | D max abs error | D mean abs error | P max abs error | S max abs error | S mean abs error |
|:---|---:|---:|---:|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|
| fixed_beta_no_shift | 3474.85 | 3091.90 | 1.12386 | 560 | True | True | True | True | 0.0244356 | 0.0235708 | 4.17964e-06 | 0.00296631 | 9.40252e-06 |
| default_beta_shift | 12748.2 | 3301.06 | 3.86186 | 560 | True | True | True | True | 0.256075 | 0.023801 | 4.20306e-06 | 0.00291526 | 9.15708e-06 |

Raw CSV: `gpu_properimage/docs/real_data_subtract_cpu_gpu.csv`
