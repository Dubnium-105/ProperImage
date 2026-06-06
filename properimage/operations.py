#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  operations.py
#
#  Copyright 2020 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#

"""operations module of ProperImage package for astronomical image analysis.

This module contains algorithm implementations for coadding and subtracting
astronomical images.
"""

import logging
import os
from pathlib import Path
import time
import warnings

import astroalign as aa

from astropy.stats import sigma_clipped_stats

from joblib import Parallel, delayed

import numpy as np

from scipy import optimize
from scipy.ndimage import center_of_mass
from scipy.ndimage import fourier_shift

import sep

from . import utils as u
from .acceleration import get_acceleration_config, resolve_cuda_devices
from .single_image import SingleImage as si

try:
    import cupy as cp
except Exception:  # pragma: no cover - depends on optional GPU extra
    cp = None

try:
    import pyfftw

    _fftwn = pyfftw.interfaces.numpy_fft.fftn  # noqa
    _ifftwn = pyfftw.interfaces.numpy_fft.ifftn  # noqa
except ImportError:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2

logger = logging.getLogger(__name__)

aa.PIXEL_TOL = 0.5
eps = np.finfo(np.float64).eps
OPT_BORDER = 100  # Border size for optimization cost calculation
_GPU_BACKEND_ERROR = {}
_GPU_BACKEND_READY = {}


def _cupy_fourier_shift(array, shift):
    """CuPy equivalent of scipy.ndimage.fourier_shift for complex FFT data."""
    phase = cp.zeros(array.shape, dtype=cp.float64)
    for axis, amount in enumerate(shift):
        axis_shape = [1] * array.ndim
        axis_shape[axis] = array.shape[axis]
        frequencies = cp.fft.fftfreq(array.shape[axis]).reshape(axis_shape)
        phase = phase + frequencies * amount
    return array * cp.exp(-2j * cp.pi * phase)


def _resolve_fft_backend(use_gpu, acceleration=None):
    """Return FFT helpers for the requested backend."""
    if use_gpu in (False, "off", "cpu", None):
        return np, _fftwn, _ifftwn, fourier_shift, np.asarray, False

    if use_gpu in (True, "on", "gpu"):
        strict_gpu = True
    elif use_gpu == "auto":
        strict_gpu = False
    else:
        raise ValueError(
            "use_gpu must be one of True, False, or 'auto'."
        )

    config = acceleration or get_acceleration_config()
    devices = resolve_cuda_devices(config, strict=strict_gpu)
    device_id = devices[0] if devices else None

    if not _cupy_backend_ready(device_id):
        message = (
            "CuPy/cuFFT backend is unavailable. Install a CUDA-matched "
            "CuPy extra such as properimage[gpu-cu12] or "
            "properimage[gpu-cu11]."
        )
        if _GPU_BACKEND_ERROR.get(device_id) is not None:
            message = (
                f"{message} Original error: "
                f"{_GPU_BACKEND_ERROR[device_id]}"
            )
        if strict_gpu:
            raise RuntimeError(message)
        logger.info("%s Falling back to CPU FFT backend.", message)
        return np, _fftwn, _ifftwn, fourier_shift, np.asarray, False

    if device_id is not None:
        cp.cuda.Device(device_id).use()

    return (
        cp,
        cp.fft.fftn,
        cp.fft.ifftn,
        _cupy_fourier_shift,
        cp.asnumpy,
        True,
    )


def _cupy_backend_ready(device_id=None):
    """Check whether CuPy can run a small FFT on the active CUDA device."""
    global _GPU_BACKEND_ERROR, _GPU_BACKEND_READY

    if device_id in _GPU_BACKEND_READY:
        return _GPU_BACKEND_READY[device_id]

    if cp is None:
        _GPU_BACKEND_ERROR[device_id] = "CuPy is not installed."
        _GPU_BACKEND_READY[device_id] = False
        return False

    _add_torch_cuda_dll_directory()

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA devices were reported by CuPy.")
        context = cp.cuda.Device(device_id) if device_id is not None else None
        if context is None:
            probe = cp.ones((4, 4), dtype=cp.float32)
            cp.fft.fftn(probe)
            cp.cuda.Stream.null.synchronize()
        else:
            with context:
                probe = cp.ones((4, 4), dtype=cp.float32)
                cp.fft.fftn(probe)
                cp.cuda.Stream.null.synchronize()
        _GPU_BACKEND_READY[device_id] = True
        _GPU_BACKEND_ERROR[device_id] = None
    except Exception as exc:
        _GPU_BACKEND_READY[device_id] = False
        _GPU_BACKEND_ERROR[device_id] = exc
    return _GPU_BACKEND_READY[device_id]


def _add_torch_cuda_dll_directory():
    """Expose CUDA DLLs bundled by optional Torch wheels when present."""
    try:
        import torch
    except ImportError:
        return

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib.exists():
        return

    current_path = os.environ.get("PATH", "")
    torch_lib_text = str(torch_lib)
    path_parts = current_path.split(os.pathsep) if current_path else []
    if torch_lib_text not in path_parts:
        os.environ["PATH"] = f"{torch_lib}{os.pathsep}{current_path}"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(torch_lib_text)


def subtract(
    ref,
    new,
    align=False,
    inf_loss=0.25,
    smooth_psf=False,
    beta=True,
    shift=True,
    iterative=False,
    fitted_psf=True,
    use_gpu="auto",
    acceleration=None,
):
    """
    Subtract a pair of SingleImage instances.

    Parameters:
    -----------
    align : bool
        Whether to align the images before subtracting, default to False
    inf_loss : float
        Value of information loss in PSF estimation, lower limit is 0,
        upper is 1. Only valid if fitted_psf=False. Default is 0.25
    smooth_psf : bool
        Whether to smooth the PSF, using a noise reduction technique.
        Default to False.
    beta : bool
        Specify if using the relative flux scale estimation.
        Default to True.
    shift : bool
        Whether to include a shift parameter in the iterative
        methodology, in order to correct for misalignments.
        Default to True.
    iterative : bool
        Specify if an iterative estimation of the subtraction relative
        flux scale must be used. Default to False.
    fitted_psf : bool
        Whether to use a Gaussian fitted PSF. Overrides the use of
        auto-psf determination. Default to True.
    use_gpu : bool or "auto"
        Whether to use CuPy/cuFFT for the FFT-heavy subtraction path.
        Default is "auto", which uses GPU when a compatible CuPy/CUDA
        backend is available and otherwise falls back to CPU. Set True to
        require GPU or False to force CPU.
    acceleration : AccelerationConfig, optional
        Runtime device and worker configuration. Most users can leave this
        unset and rely on the process-wide defaults.

    Returns:
    --------
    D : np.ndarray(n, m) of float
        Subtracion image, Zackay's decorrelated D.
    P : np.ndarray(n, m) of float
        Subtracion image PSF. This is a full PSF image, with a size equal to D
    S_corr : np.ndarray of float
        Subtracion image S, Zackay's cross-correlated D x P
    mix_mask : np.ndarray of bool
        Mask of bad pixels for subtracion image, with True marking bad pixels
    """
    if fitted_psf:
        from .single_image import SingleImageGaussPSF as SI

        logger.info("Using single psf, gaussian modeled")
    else:
        from .single_image import SingleImage as SI

    if not isinstance(ref, SI):
        try:
            ref = SI(ref, smooth_psf=smooth_psf)
        except Exception:
            try:
                ref = SI(ref.data, smooth_psf=smooth_psf)
            except Exception:
                raise

    if not isinstance(new, SI):
        try:
            new = SI(new, smooth_psf=smooth_psf)
        except Exception:
            try:
                new = SI(new.data, smooth_psf=smooth_psf)
            except Exception:
                raise

    if align:
        registrd, registrd_mask = aa.register(new.data, ref.data)
        new._clean()
        #  should it be new = type(new)(  ?
        new = SI(
            registrd[: ref.data.shape[0], : ref.data.shape[1]],
            mask=registrd_mask[: ref.data.shape[0], : ref.data.shape[1]],
            borders=False,
            smooth_psf=smooth_psf,
        )
        # new.data = registered
        # new.data.mask = registered.mask

    # make sure that the alignment has delivered arrays of same size
    if new.data.data.shape != ref.data.data.shape:
        raise ValueError("N and R arrays are of different size")

    xp, fftn, ifftn, fshift, to_cpu, using_gpu = _resolve_fft_backend(
        use_gpu, acceleration=acceleration
    )

    t0 = time.time()
    mix_mask = np.ma.mask_or(new.data.mask, ref.data.mask)
    if np.isscalar(mix_mask) or getattr(mix_mask, "shape", ()) == ():
        mix_mask = np.zeros(new.data.shape, dtype=bool)

    zps, meanmags = u.transparency([ref, new])
    ref.zp = zps[0]
    new.zp = zps[1]
    n_zp = new.zp
    r_zp = ref.zp

    a_ref, psf_ref = ref.get_variable_psf(inf_loss)
    a_new, psf_new = new.get_variable_psf(inf_loss)

    if fitted_psf:
        #  I already know that a_ref and a_new are None, both of them
        #  And each psf is a list, first element a render,
        #  second element a model

        p_r = psf_ref[1]
        p_n = psf_new[1]

        p_r.x_mean = ref.data.data.shape[0] / 2.0
        p_r.y_mean = ref.data.data.shape[1] / 2.0
        p_n.x_mean = new.data.data.shape[0] / 2.0
        p_n.y_mean = new.data.data.shape[1] / 2.0
        p_r.bounding_box = None
        p_n.bounding_box = None

        p_n = p_n.render(np.zeros(new.data.data.shape))
        p_r = p_r.render(np.zeros(ref.data.data.shape))

        dx_ref, dy_ref = center_of_mass(p_r)  # [0])
        dx_new, dy_new = center_of_mass(p_n)  # [0])
    else:
        p_r = psf_ref[0]
        p_n = psf_new[0]

        dx_ref, dy_ref = center_of_mass(p_r)  # [0])
        dx_new, dy_new = center_of_mass(p_n)  # [0])
        if dx_new < 0.0 or dy_new < 0.0:
            raise ValueError("Impossible to acquire center of PSF in stamp")

    psf_ref_hat = fftn(xp.asarray(p_r), s=ref.data.shape, norm="ortho")
    psf_new_hat = fftn(xp.asarray(p_n), s=new.data.shape, norm="ortho")

    psf_ref_hat[psf_ref_hat.real == 0] = eps
    psf_new_hat[psf_new_hat.real == 0] = eps

    psf_ref_hat_conj = psf_ref_hat.conj()
    psf_new_hat_conj = psf_new_hat.conj()

    if using_gpu:
        ref_interped_hat = fftn(xp.asarray(ref.interped), norm="ortho")
        new_interped_hat = fftn(xp.asarray(new.interped), norm="ortho")
    else:
        ref_interped_hat = ref.interped_hat
        new_interped_hat = new.interped_hat

    D_hat_r = fshift(psf_new_hat * ref_interped_hat, (-dx_new, -dy_new))
    D_hat_n = fshift(psf_ref_hat * new_interped_hat, (-dx_ref, -dy_ref))

    norm_b = ref.var**2 * psf_new_hat * psf_new_hat_conj
    norm_a = new.var**2 * psf_ref_hat * psf_ref_hat_conj

    new_back = sep.Background(new.interped).back()
    ref_back = sep.Background(ref.interped).back()
    gamma = xp.asarray(new_back - ref_back)
    backend_mask = xp.asarray(mix_mask) if using_gpu else mix_mask
    b = n_zp / r_zp
    norm = xp.sqrt(norm_a + norm_b * b**2)

    def to_scalar(value):
        return float(to_cpu(value)) if using_gpu else value

    def optimizer_scalar(value):
        return float(np.asarray(value).ravel()[0])

    def masked_abs_sum(array, border=0, mean_normalized=False):
        real_array = array.real
        mask = backend_mask
        if border:
            real_array = real_array[border:-border, border:-border]
            mask = mask[border:-border, border:-border]
        real_array = xp.where(mask, 0, real_array)
        scale = real_array.shape[0] * real_array.shape[1]
        if mean_normalized:
            real_array = real_array / scale
        value = xp.sum(xp.abs(real_array))
        return to_scalar(value)

    if beta:
        if shift:  # beta==True & shift==True

            def cost(vec):
                b, dx, dy = [optimizer_scalar(item) for item in vec]
                gammap = gamma / xp.sqrt(new.var**2 + b**2 * ref.var**2)
                norm = xp.sqrt(norm_a + norm_b * b**2)
                dhn = D_hat_n / norm
                dhr = D_hat_r / norm
                b_n = (
                    ifftn(dhn, norm="ortho")
                    - ifftn(fshift(dhr, (dx, dy)), norm="ortho") * b
                    - xp.roll(gammap, (int(round(dx)), int(round(dy))))
                )
                return masked_abs_sum(
                    b_n, border=OPT_BORDER, mean_normalized=True
                )

            ti = time.time()
            vec0 = [b, 0.0, 0.0]
            bounds = ([0.1, -0.9, -0.9], [10.0, 0.9, 0.9])
            solv_beta = optimize.least_squares(
                cost,
                vec0,
                xtol=1e-5,
                jac="3-point",
                method="trf",
                bounds=bounds,
            )
            tf = time.time()

            if solv_beta.success:
                logger.info("Found that beta = {}".format(solv_beta.x))
                logger.info("Took only {} awesome seconds".format(tf - ti))
                logger.info(
                    "The solution was with cost {}".format(solv_beta.cost)
                )
                b, dx, dy = solv_beta.x
            else:
                logger.info("Least squares could not find our beta  :(")
                logger.info("Beta is overriden to be the zp ratio again")
                b = n_zp / r_zp
                dx = 0.0
                dy = 0.0
        elif iterative:  # beta==True & shift==False & iterative==True

            def F(b):
                b = optimizer_scalar(b)
                gammap = gamma / xp.sqrt(new.var**2 + b**2 * ref.var**2)
                norm = xp.sqrt(norm_a + norm_b * b**2)
                b_n = (
                    ifftn(D_hat_n / norm, norm="ortho")
                    - gammap
                    - b * ifftn(D_hat_r / norm, norm="ortho")
                )
                # robust_stats = lambda b: sigma_clipped_stats(
                #    b_n(b).real[100:-100, 100:-100])
                return masked_abs_sum(b_n)

            ti = time.time()
            solv_beta = optimize.minimize_scalar(
                F,
                method="bounded",
                bounds=[0.1, 10.0],
                options={"maxiter": 1000},
            )

            tf = time.time()
            if solv_beta.success:
                logger.info("Found that beta = {}".format(solv_beta.x))
                logger.info("Took only {} awesome seconds".format(tf - ti))
                b = solv_beta.x
            else:
                logger.info("Least squares could not find our beta  :(")
                logger.info("Beta is overriden to be the zp ratio again")
                b = n_zp / r_zp
            dx = dy = 0.0
        else:  # beta==True & shift==False & iterative==False

            def F(b):
                b = optimizer_scalar(b)
                gammap = gamma / xp.sqrt(new.var**2 + b**2 * ref.var**2)
                norm = xp.sqrt(norm_a + norm_b * b**2)
                b_n = (
                    ifftn(D_hat_n / norm, norm="ortho")
                    - gammap
                    - b * ifftn(D_hat_r / norm, norm="ortho")
                )
                return masked_abs_sum(b_n)

            ti = time.time()
            solv_beta = optimize.least_squares(
                F, b, ftol=1e-8, bounds=[0.1, 10.0], jac="2-point"
            )

            tf = time.time()
            if solv_beta.success:
                logger.info("Found that beta = {}".format(solv_beta.x))
                logger.info("Took only {} awesome seconds".format(tf - ti))
                logger.info(
                    "The solution was with cost {}".format(solv_beta.cost)
                )
                b = float(solv_beta.x[0])
            else:
                logger.info("Least squares could not find our beta  :(")
                logger.info("Beta is overriden to be the zp ratio again")
                b = n_zp / r_zp
            dx = dy = 0.0
    else:
        if shift:  # beta==False & shift==True
            gammap = gamma / xp.sqrt(new.var**2 + b**2 * ref.var**2)
            norm = xp.sqrt(norm_a + norm_b * b**2)
            dhn = D_hat_n / norm
            dhr = D_hat_r / norm

            def cost(vec):
                dx, dy = [optimizer_scalar(item) for item in vec]
                b_n = (
                    ifftn(dhn, norm="ortho")
                    - ifftn(fshift(dhr, (dx, dy)), norm="ortho") * b
                    - xp.roll(gammap, (int(round(dx)), int(round(dy))))
                )
                return masked_abs_sum(
                    b_n, border=OPT_BORDER, mean_normalized=True
                )

            ti = time.time()
            vec0 = [0.0, 0.0]
            bounds = ([-0.9, -0.9], [0.9, 0.9])
            solv_beta = optimize.least_squares(
                cost,
                vec0,
                xtol=1e-5,
                jac="3-point",
                method="trf",
                bounds=bounds,
            )
            tf = time.time()

            if solv_beta.success:
                logger.info("Found that shift = {}".format(solv_beta.x))
                logger.info("Took only {} awesome seconds".format(tf - ti))
                logger.info(
                    "The solution was with cost {}".format(solv_beta.cost)
                )
                dx, dy = solv_beta.x
            else:
                logger.info("Least squares could not find our shift  :(")
                dx = 0.0
                dy = 0.0
        else:  # beta==False & shift==False
            b = new.zp / ref.zp
            dx = 0.0
            dy = 0.0

    norm = norm_a + norm_b * b**2

    if dx == 0.0 and dy == 0.0:
        D_hat = (D_hat_n - b * D_hat_r) / xp.sqrt(norm)
    else:
        D_hat = (D_hat_n - fshift(b * D_hat_r, (dx, dy))) / xp.sqrt(
            norm
        )

    D = ifftn(D_hat, norm="ortho")
    has_nan = to_scalar(xp.any(xp.isnan(D.real)))
    if bool(has_nan):
        logger.warning("NaN values detected in D.real array after inverse FFT")

    d_zp = b / xp.sqrt(ref.var**2 * b**2 + new.var**2)
    P_hat = (psf_ref_hat * psf_new_hat * b) / (xp.sqrt(norm) * d_zp)

    P = ifftn(P_hat, norm="ortho").real
    P = to_cpu(P) if using_gpu else P
    dx_p, dy_p = center_of_mass(P)

    dx_pk, dy_pk = [val[0] for val in np.where(P == np.max(P))]
    if (np.abs(dx_p - dx_pk) > 30) or (np.abs(dy_p - dy_pk) > 30):
        logger.info("Resetting PSF center of mass to peak")
        dx_p = dx_pk
        dy_p = dy_pk

    S_hat = fshift(d_zp * D_hat * P_hat.conj(), (dx_p, dy_p))

    kr = ifftn(
        new.zp * psf_ref_hat_conj * b * psf_new_hat * psf_new_hat_conj / norm,
        norm="ortho",
    )

    kn = ifftn(
        new.zp * psf_new_hat_conj * psf_ref_hat * psf_ref_hat_conj / norm,
        norm="ortho",
    )

    V_en = ifftn(
        fftn(xp.asarray(new.data.filled(0) + 1.0), norm="ortho")
        * fftn(kn**2, s=new.data.shape),
        norm="ortho",
    )

    V_er = ifftn(
        fftn(xp.asarray(ref.data.filled(0) + 1.0), norm="ortho")
        * fftn(kr**2, s=ref.data.shape),
        norm="ortho",
    )

    S_corr = ifftn(S_hat, norm="ortho") / xp.sqrt(V_en + V_er)
    D = to_cpu(D) if using_gpu else D
    S_corr = to_cpu(S_corr) if using_gpu else S_corr
    logger.info("S_corr sigma_clipped_stats ")
    logger.info(
        "mean = {}, median = {}, std = {}\n".format(
            *sigma_clipped_stats(S_corr.real.flatten(), sigma=4.0)
        )
    )
    logger.info(
        "Subtraction performed in {} seconds\n\n".format(time.time() - t0)
    )

    return D, P, S_corr.real, mix_mask


def diff(*args, **kwargs):
    """Subtract images.

    Wrapper of `subtract`.
    """
    warnings.warn(
        "This is being deprecated in favour of `subtract`", DeprecationWarning
    )
    return subtract(*args, **kwargs)


def process_chunk_joblib(chunk, global_shape):
    S_hat = np.zeros(global_shape).astype(np.complex128)
    psf_hat_sum = np.zeros(global_shape).astype(np.complex128)
    mix_mask = chunk[0].data.mask

    for an_img in chunk:

        np.add(an_img.s_hat_comp, S_hat, out=S_hat, casting="same_kind")
        np.add(
            ((an_img.zp / an_img.var) ** 2) * an_img.psf_hat_sqnorm(),
            psf_hat_sum,
            out=psf_hat_sum,
        )
        mix_mask = np.ma.mask_or(mix_mask, an_img.data.mask)
    return S_hat.real, S_hat.imag, psf_hat_sum.real, psf_hat_sum.imag, mix_mask


def coadd(si_list, align=True, inf_loss=0.2, n_procs=2):
    """Coadd a list of SingleImage instances using R estimator with joblib.

    Parameters:
    -----------
    align : bool
        Whether to align the images before subtracting, default to False
    inf_loss : float
        Value of information loss in PSF estimation, lower limit is 0,
        upper is 1. Only valid if fitted_psf=False. Default is 0.25
    n_procs : int
        Number of parallel workers to use. If value is one then no parallelism
        is being used. Default 2.

    Returns:
    --------
    R : np.ndarray(n, m) of float
        Coadd image, Zackay's decorrelated R.
    P : np.ndarray(n, m) of float
        Coadd image PSF. This is a full PSF image, with a size equal to R
    mix_mask : np.ndarray of bool
        Mask of bad pixels for subtracion image, with True marking bad pixels
    """
    logger.info(f"Starting coadd operation with {len(si_list)} images")
    logger.info(
        f"Alignment: {align}, inf_loss: {inf_loss}, n_procs: {n_procs}"
    )

    for i_img, animg in enumerate(si_list):
        if not isinstance(animg, si):
            si_list[i_img] = si(animg)

    if align:
        logger.info("Aligning images for coadd")
        img_list = u._align_for_coadd(si_list)
        logger.info("Updating sources after alignment")
        for an_img in img_list:
            an_img.update_sources()
    else:
        logger.info("Using images without alignment")
        img_list = si_list

    shapex = np.min([an_img.data.shape[0] for an_img in img_list])
    shapey = np.min([an_img.data.shape[1] for an_img in img_list])
    global_shape = (shapex, shapey)
    logger.info(f"Global shape determined: {global_shape}")

    logger.info("Computing transparency and zero points")
    zps, meanmags = u.transparency(img_list)
    for j, an_img in enumerate(img_list):
        an_img.zp = zps[j]
        an_img._setup_kl_a_fields(inf_loss)
    logger.info(f"Zero points: {zps}")

    psf_shapes = [an_img.stamp_shape[0] for an_img in img_list]
    psf_shape = np.max(psf_shapes)
    psf_shape = (psf_shape, psf_shape)
    logger.info(f"PSF shape: {psf_shape}")

    n_jobs = n_procs
    if n_jobs > 1:
        logger.info(f"Using parallel processing with {n_jobs} workers")
        chunks = list(u.chunk_it(img_list, n_jobs))
        ti = time.time()
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_chunk_joblib)(chunk, global_shape)
            for chunk in chunks
        )
        tf = time.time()
        logger.info(f"Parallel processing completed in {tf - ti:.2f} seconds")

        logger.info("Combining results from parallel workers")
        S_hat = np.zeros(global_shape, dtype=np.complex128)
        P_hat = np.zeros(global_shape, dtype=np.complex128)
        mix_mask = np.zeros(global_shape, dtype=bool)
        for (
            S_hat_real,
            S_hat_imag,
            psf_hat_sum_real,
            psf_hat_sum_imag,
            mask,
        ) in results:
            np.add(S_hat_real, S_hat.real, out=S_hat.real)
            np.add(S_hat_imag, S_hat.imag, out=S_hat.imag)
            np.add(psf_hat_sum_real, P_hat.real, out=P_hat.real)
            np.add(psf_hat_sum_imag, P_hat.imag, out=P_hat.imag)
            mix_mask = np.ma.mask_or(mix_mask, mask)

        logger.info("Computing final coadd image")
        P_r_hat = np.sqrt(P_hat)
        P_r = _ifftwn(fourier_shift(P_r_hat, psf_shape))
        P_r = P_r / np.sum(P_r)
        R = _ifftwn(S_hat / np.sqrt(P_hat))
    else:
        logger.info("Using sequential processing (single thread)")
        S_hat = np.zeros(global_shape, dtype=np.complex128)
        P_hat = np.zeros(global_shape, dtype=np.complex128)
        mix_mask = img_list[0].data.mask

        ti = time.time()
        for an_img in img_list:
            np.add(an_img.s_hat_comp, S_hat, out=S_hat)
            np.add(
                ((an_img.zp / an_img.var) ** 2) * an_img.psf_hat_sqnorm(),
                P_hat,
                out=P_hat,
            )
            mix_mask = np.ma.mask_or(mix_mask, an_img.data.mask)
        tf = time.time()
        logger.info(
            f"Sequential processing completed in {tf - ti:.2f} seconds"
        )

        logger.info("Computing final coadd image")
        P_r_hat = np.sqrt(P_hat)
        P_r = _ifftwn(fourier_shift(P_r_hat, psf_shape))
        P_r = P_r / np.sum(P_r)
        R = _ifftwn(S_hat / P_r_hat)

    logger.info("Coadd operation completed successfully")
    return R, P_r, mix_mask


def stack_R(*args, **kwargs):
    """Coadd images.

    Wrapper of `coadd`.
    """
    warnings.warn(
        "This is being deprecated in favour of `coadd`", DeprecationWarning
    )
    return coadd(*args, **kwargs)
