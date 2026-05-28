#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2020 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#

"""Minimal setup.py for backwards compatibility.

All package metadata, dependencies, and optional GPU extras are configured in
pyproject.toml. Install GPU support with extras such as:

    pip install properimage[gpu-cu12]
    pip install properimage[gpu-cu11]
"""

from setuptools import setup

setup()
