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


__version__ = '1.0.0'


import os
_root = os.path.split(__file__)[0]
_dirpath = os.path.join(_root, "__assets__")
vals = [os.path.join(_dirpath, f) for f in os.listdir(_dirpath)]
keys = []
accepted = ('.urdf', '.png', '.obj', '.stl', '.dae')
for v in vals:
    if v.lower().endswith(accepted):
        keys.append(os.path.basename(v.lower()))
__assets__ = dict(zip(keys, vals))
del(_root)
del(_dirpath)
del(vals)
del(v)
del(keys)
