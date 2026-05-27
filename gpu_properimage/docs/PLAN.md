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
