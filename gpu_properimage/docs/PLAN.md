# ProperImage GPU Acceleration (方案7: CuPy/cuFFT)

## Target
FFT-based subtraction in properimage/operations.py subtract()

## Bottleneck
- scipy.optimize callback loop with repeated IFFT: 50-90%
- Final variance correction FFT/IFFT: 20-40%
- PSF rendering + background: 5-20%

## Strategy
Replace numpy.fft with CuPy/cuFFT, keep data on GPU through entire pipeline.

Key challenges:
- scipy.optimize runs on CPU; cost() must minimize GPU↔CPU roundtrips
- sep.Background (C) not GPU-compatible — needs alternative
- fourier_shift uses numpy FFT internally — blind spot
- SingleImage internal state needs GPU refactor

## Expected speedup
8-20x for 4096² images on RTX 3060+

## Implementation status
- Added `subtract(..., use_gpu=True)` as an explicit opt-in GPU path.
- The subtraction FFT/IFFT chain now uses CuPy/cuFFT when enabled.
- Added a CuPy implementation of `scipy.ndimage.fourier_shift` for frequency
  domain arrays, keeping shift operations on GPU.
- Optimization callbacks keep full image residuals and masks on GPU, returning
  only scalar costs to SciPy.
- CPU behavior remains the default and still uses the existing NumPy/pyFFTW
  backend.

## Remaining CPU sections
- `SingleImage` construction, PSF modeling/rendering, and SEP background
  estimation remain CPU-side.
- `scipy.optimize` remains CPU-side, but its repeated FFT/IFFT work can now
  execute on GPU.
