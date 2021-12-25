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
from hkmeans import *

from . import _version
__version__ = _version.get_versions()['version']
