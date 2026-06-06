#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2020 Toros astro team
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#

"""Properimage is a package for astronomical image processing.

It implements in particular algorithms for coaddition and subtraction
of CCD images. The methodology follows a hypothesis test scheme.
"""

__version__ = "0.7.3"

from .acceleration import (
    AccelerationConfig,
    clear_acceleration_cache,
    configure_acceleration,
    get_acceleration_config,
    subtract_batch,
)
from .operations import coadd, subtract
from .single_image import SingleImage


__all__ = [
    "subtract",
    "subtract_batch",
    "coadd",
    "SingleImage",
    "AccelerationConfig",
    "configure_acceleration",
    "get_acceleration_config",
    "clear_acceleration_cache",
]
