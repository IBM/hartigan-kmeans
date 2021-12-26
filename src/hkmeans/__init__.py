#
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""
...
"""

# init file

# import cython created shared object files
import hkmeans.c_package  # cython with cpp version

# import core functionality
from .hkmeans_main import *


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
