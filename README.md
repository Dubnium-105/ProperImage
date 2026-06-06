# Proper image treatments

[![QuatroPe](https://img.shields.io/badge/QuatroPe-Applications-1c5896)](https://quatrope.github.io/)
[![Build Status](https://github.com/quatrope/ProperImage/actions/workflows/testing.yml/badge.svg?branch=master)](https://github.com/quatrope/ProperImage/actions/workflows/python-properimage-ci.yml)
[![Documentation Status](https://readthedocs.org/projects/properimage/badge/?version=latest)](http://properimage.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/quatrope/ProperImage/branch/master/graph/badge.svg)](https://codecov.io/gh/quatrope/ProperImage)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://badge.fury.io/py/properimage)
[![License](https://img.shields.io/pypi/l/properimage?color=blue)](https://tldrlegal.com/license/bsd-3-clause-license-(revised))
[![ascl:1904.025](https://img.shields.io/badge/ascl-1904.025-blue.svg?colorB=262255)](http://ascl.net/1904.025)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This code is inspired on [Zackay & Ofek 2017](http://arxiv.org/abs/1512.06872)  papers *How to coadd images?* (see References below).

* It can perform a PSF estimation using [Karhunen-Löeve expansion](https://en.wikipedia.org/wiki/Karhunen–Loève_theorem), which is based on [Lauer 2002](https://doi.org/10.1117/12.461035) work.

* It can perform the statistical proper coadd of several images.

* It can also perform a proper-subtraction of images.

* Images need to be aligned and registered, or at least [astroalign](https://github.com/toros-astro/astroalign) must be installed.

* Contains a nice plot module for PSF visualization (_needs matplotlib_)

## Installation

To install from PyPI

```console
$ pip install properimage
```

To install GPU support, choose the extra matching your CUDA runtime:

```console
$ pip install properimage[gpu-cu12]
$ pip install properimage[gpu-cu11]
```

## Quick usage

### PSF estimation

```python
>>> from properimage import singleimage as si
>>> with si.SingleImage(frame, smooth_psf=False) as sim:
...     a_fields, psf_basis = sim.get_variable_psf(inf_loss=0.15)
```

### Proper-subtraction of images

To create a proper-subtraction of images:

```python
>>> from properimage.operations import subtract
>>> D, P, Scorr, mask = subtract(ref=ref_path, new=new_path, smooth_psf=False, fitted_psf=True,
...                             align=False, iterative=False, beta=False, shift=False)
```

Where `D`, `P`, `Scorr` refer to the images defined by the same name in [Zackay & Ofek](https://iopscience.iop.org/article/10.3847/0004-637X/830/1/27/meta) paper.

### GPU-accelerated subtraction

`subtract()` can use a CuPy/cuFFT backend for the FFT-heavy subtraction path.
By default, `use_gpu="auto"` tries GPU first and falls back to CPU when a
compatible CuPy/CUDA runtime is not available. Install the GPU extra matching
your CUDA runtime to make the automatic GPU path available. For CUDA 12:

```console
$ pip install properimage[gpu-cu12]
```

```python
>>> from properimage.operations import subtract
>>> D, P, Scorr, mask = subtract(
...     ref=ref_path,
...     new=new_path,
...     fitted_psf=True,
...     beta=True,
...     shift=True,
... )
```

Use `use_gpu=True` to require GPU and raise an error if CuPy/CUDA is not
available. Use `use_gpu=False` to force the CPU backend.

For new users, GPU installation does not change the public subtraction API:
existing code such as `subtract(ref, new)` continues to work. The only runtime
behavior change is that, when a compatible GPU backend is installed,
`use_gpu="auto"` may choose CuPy/cuFFT instead of the CPU FFT backend. Results
should be numerically close but may not be bit-for-bit identical because the FFT
backend changes. Integrated systems that require the previous CPU-only behavior
can keep it by passing `use_gpu=False`.

Current GPU coverage is focused on FFT/IFFT, Fourier-domain shifts, and the
repeated optimizer residual calculations. Image loading, PSF modeling,
background estimation, and the SciPy optimizer itself still run on CPU, so the
largest gains appear when `beta` and/or `shift` optimization performs repeated
frequency-domain work.

For folder-scale production runs, use the batch API or CLI to keep CPU
preprocessing and one or more CUDA devices busy:

```python
>>> from properimage import AccelerationConfig, subtract_batch, tune_acceleration
>>> pairs = [(ref_path, new_path), ...]
>>> tuning = tune_acceleration(
...     pairs,
...     acceleration=AccelerationConfig(devices="auto"),
...     use_gpu="auto",
... )
>>> result = subtract_batch(
...     pairs,
...     output_dir="res",
...     acceleration=tuning.config,
...     use_gpu="auto",
... )
>>> result.throughput_pairs_per_s
```

```console
$ properimage-subtract-batch --ref-dir data/ref --new-dir data/new \
    --output-dir res --devices auto --auto-tune
```

`devices="auto"` uses all CUDA devices visible to CuPy, including virtualized
devices exposed by the target CUDA runtime. Use `devices="cpu"` or
`--use-gpu false` to force CPU execution.

The CLI pairs FITS files by identical filename in `--ref-dir` and `--new-dir`.
If names do not match, it falls back to sorted order when both folders contain
the same number of FITS files. Each output pair writes `D`, `P`, `S_corr`, and
`mask` FITS files under `--output-dir`, plus a `manifest.csv` with per-pair
status, device id, timing, finite checks, and output paths.

Common CLI options:

```console
$ properimage-subtract-batch --ref-dir data/ref --new-dir data/new \
    --output-dir res --devices auto --auto-tune

$ properimage-subtract-batch --ref-dir data/ref --new-dir data/new \
    --output-dir res --devices cpu --use-gpu false

$ properimage-subtract-batch --ref-dir data/ref --new-dir data/new \
    --output-dir res --devices 0 --gpu-workers 6 --prefetch 4
```

Use `--auto-tune` for first runs on a new machine or workload. For repeated
production runs, use the measured `--gpu-workers` and `--prefetch` values
explicitly to avoid spending time on tuning each run.

Additional benchmark scripts and result notes live in
[`gpu_properimage/docs`](gpu_properimage/docs/README.md).

For the full documentation refer to [readthedocs](https://properimage.readthedocs.io).

## Rerefences

> Zackay, B., & Ofek, E. O. (2017). How to Coadd Images. I. Optimal Source Detection and Photometry of Point Sources Using Ensembles of Images. The Astrophysical Journal, 836(2), 187. [Arxiv version](http://arxiv.org/abs/1512.06872)
>
> Zackay, B., & Ofek, E. O. (2017). How to Coadd Images. II. A Coaddition Image that is Optimal for Any Purpose in the Background-dominated Noise Limit. The Astrophysical Journal, 836(2), 188. [Arxiv version](http://arxiv.org/abs/1512.06879)
>
>Zackay, B., Ofek, E. O., & Gal-Yam, A. (2016). Proper Image Subtrraction-Optimal Transient Detection, Photometry, and Hypothesis Testing. [The Astrophysical Journal, 830(1), 27.](https://iopscience.iop.org/article/10.3847/0004-637X/830/1/27/meta)
>
>Lauer, T. (2002, December). Deconvolution with a spatially-variant PSF. In [Astronomical Data Analysis II (Vol. 4847, pp. 167-174). International Society for Optics and Photonics.](https://doi.org/10.1117/12.461035)
