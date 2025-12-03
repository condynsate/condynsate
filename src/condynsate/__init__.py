# -*- coding: utf-8 -*-
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

# Submodules always needs to be imported to ensure registration
from condynsate.project import Project # NOQA
from condynsate.simulator import Simulator # NOQA
from condynsate.visualizer import Visualizer # NOQA
from condynsate.animator import Animator # NOQA
from condynsate.keyboard import Keyboard # NOQA

__all__ = ["Project",
           "Simulator",
           "Visualizer",
           "Animator",
           "Keyboard",]


__version__ = '1.0.4'


import os
_root = os.path.split(__file__)[0]
_dirpath = os.path.join(_root, "__assets__")
_vals = [os.path.join(_dirpath, f) for f in os.listdir(_dirpath)]
_pairs = []
_accepted = ('.urdf', '.png', '.obj', '.stl', '.dae')
for _v in _vals:
    if _v.lower().endswith(_accepted):
        _pairs.append((os.path.basename(_v.lower()), _v))
__assets__ = dict(_pairs)
del(_root)
del(_dirpath)
del(_vals)
del(_pairs)
del(_accepted)
del(_v)
