# Real Folder CPU/GPU Subtraction

Date: 2026-05-28

## Input

- Reference folder: `data/ref`
- New image folder: `data/new`
- Pairing: by identical FITS filename
- Pairs tested: 3
- Modes:
  - `fixed_beta_no_shift`: `beta=False`, `shift=False`
  - `default_beta_shift`: `beta=True`, `shift=True`

## Command

```powershell
python `
  gpu_properimage\benchmarks\real_data_subtract_cpu_gpu.py `
  --ref-dir data\ref `
  --new-dir data\new `
  --output gpu_properimage\docs\real_folder_cpu_gpu_subtract.csv
```

## Results

| Mode | Ref/New | Shape | CPU ms | GPU ms | Speedup | Finite | D mean abs error | S mean abs error |
|:---|:---|:---|---:|---:|---:|:---|---:|---:|
| fixed_beta_no_shift | `20260203T131943__SAC NGC  784__aligned_crop.fts` | 1322x1992 | 11016.5 | 9119.79 | 1.20797 | True | 0.0270286 | 6.53253e-06 |
| default_beta_shift | `20260203T131943__SAC NGC  784__aligned_crop.fts` | 1322x1992 | 376236 | 20971.2 | 17.9406 | True | 0.0240184 | 8.70983e-06 |
| fixed_beta_no_shift | `20260203T132612__NGC 803__aligned_crop.fts` | 1325x1993 | 9349.16 | 7924.25 | 1.17982 | True | 1.6385e-06 | 8.31916e-11 |
| default_beta_shift | `20260203T132612__NGC 803__aligned_crop.fts` | 1325x1993 | 170536 | 12393.7 | 13.7598 | True | 9.16995e-05 | 5.03353e-09 |
| fixed_beta_no_shift | `20260203T132707__IC 196__aligned_crop.fts` | 1325x1994 | 8976.73 | 7156.28 | 1.25439 | True | 0.0282789 | 7.47011e-11 |
| default_beta_shift | `20260203T132707__IC 196__aligned_crop.fts` | 1325x1994 | 94508.2 | 9649.81 | 9.79378 | True | 0.0213841 | 3.84148e-08 |

Raw CSV: `gpu_properimage/docs/real_folder_cpu_gpu_subtract.csv`

## Notes

- All CPU/GPU `D`, `P`, and `S` outputs were finite.
- `mix_mask` was scalar for these FITS files, so `properimage.operations.subtract`
  now normalizes scalar masks to a full-size boolean mask before cost slicing.
