# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

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
