(C) Copyright, 2025 G. Schaer.

This work is licensed under a [GNU General Public License 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) and an MIT License.

SPDX-License-Identifier: GPL-3.0-only AND MIT

# Preamble
We present a Python-based, open-source educational tool called condynsate (**con**trol and **dyn**amics simulator) designed to ease the creation of computation-based dynamics and control demonstrations, assignments, and projects. Guided by a simulation and role-play pedagogy, projects built with condynsate mimic the experience of simulation-based games to enhance learning and retention.

condynsate was built at the University of Illinois Urbana-Champaign under the generous support of the Grainger College of Engineering Strategic Instructional Innovations Program (SIIP): [Computational Tools for Dynamics and Control grant](https://ae3.grainger.illinois.edu/programs/siip-grants/64459).

All materials, including the package and example usage, have been made publicly available at [https://github.com/condynsate/condynsate](https://github.com/condynsate/condynsate) and are licensed under the GPL-3.0-only and MIT licenses. To install, type
```bash
python3 -m pip install condynsate
```
in a Python-enabled terminal.

# The condynsate Package
Built on [PyBullet](https://pybullet.org/wordpress/) and [MeshCat](https://github.com/meshcat-dev/meshcat-python/), condynsate implements real-time simulation of [.stl](https://en.wikipedia.org/wiki/STL_(file_format)/) and [.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file/) defined rigid bodies and [.urdf](http://wiki.ros.org/urd/) defined articulated bodies with a browser-based 3D viewer to visualize simulations, a built-in animator to plot arbitrary simulation states, and a keyboard module which allows detection of key press events. These choices ensure condynsate has a broad scope of applicability by guaranteeing that any dynamic system that can be described by a .urdf file, a file format created by [Open Robotics](https://www.openrobotics.org/) for the [Robot Operating System](https://wiki.ros.org/) software, is supported.

# Detailed Installation Instruction
## Windows
A C++ compiler for C++ 2003 is needed. On Windows, we recommend using the Desktop development with C++ workload for [Microsoft C++ Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

Additionally, we strongly recommend installing condynsate in a virtual environment:

```powershell
C:\Users\username> python -m venv .venv
C:\Users\username> .venv\Scripts\activate.bat
```

When done installing and using condynsate, deactivate the virtual environment with:

```console
(.venv) user@device:~$ deactivate
```

### PyPi (Recommended)
[python>=3.8](https://www.python.org/), [pip](https://pip.pypa.io/en/stable/), and [git](https://git-scm.com/) are required.

To install condynsate:

```powershell
(.venv) C:\Users\username> pip install condynsate
```

### Source
[python>=3.8](https://www.python.org/), [pip](https://pip.pypa.io/en/stable/), and [git](https://git-scm.com/) are required.
To clone the repository:

```powershell
(.venv) C:\Users\username> git clone https://github.com/condynsate/condynsate.git
(.venv) C:\Users\username> cd condynsate
(.venv) C:\Users\username> git submodule update --init --recursive
```

To install condynsate:

```powershell
(.venv) C:\Users\username\condynsate> pip install -e .
```

## Linux
We strongly recommend installing condynsate in a virtual environment:

```console
user@device:~$ python3 -m venv .venv
user@device:~$ source .venv/bin/activate
```

On Debian/Ubuntu systems you may need to first install the python3-venv package. For Python 3.10 this can be installed with:

```console
user@device:~$ sudo apt update
user@device:~$ sudo apt install python3.10-venv
```

When done installing and using condynsate, deactivate the virtual environment with:

```console
(.venv) user@device:~$ deactivate
```

Additionally, on Debian/Ubuntu systems, to build condynsate you may need to first install the Python and Linux development headers. These can be installed with

```console
(.venv) user@device:~$ sudo apt update
(.venv) user@device:~$ sudo apt install build-essential python3-dev linux-headers-$(uname -r)
```

Finally, the package that provides keyboard interactivity uses [X](https://en.wikipedia.org/wiki/X_Window_System). This means that for keyboard interactivity to work

1. an X server must be running, and
2. the environment variable $DISPLAY must be set.

If these are not true, then keyboard interactivity will not work. All other features will work, though. For example, to use keyboard iteractivity on Ubuntu 22.04, you must first add

```console
WaylandEnable=false
```

to /etc/gdm3/custom.conf and then either reboot your system or run the command

```console
user@device:~$ systemctl restart gdm3
```

### PyPi (Recommended)
[python>=3.8](https://www.python.org/), [pip](https://pip.pypa.io/en/stable/), and [git](https://git-scm.com/) are required.

To install condynsate:

```console
(.venv) user@device:~$ pip install condynsate
```

### Source
[python>=3.8](https://www.python.org/), [pip](https://pip.pypa.io/en/stable/), and [git](https://git-scm.com/) are required.

To clone the repository:

```console
(.venv) user@device:~$ git clone https://github.com/condynsate/condynsate.git
(.venv) user@device:~$ cd condynsate
(.venv) user@device:~$ git submodule update --init --recursive

```

To install condynsate:

```console
(.venv) user@device:~/condynsate$ pip install -e .
```

On Debian/Ubuntu systems, you may need to first install the Python and Linux development headers. These can be installed with:

```console
(.venv) user@device:~/condynsate$ sudo apt update
(.venv) user@device:~/condynsate$ sudo apt install build-essential python3-dev linux-headers-$(uname -r)
```

# Documentation
condynsate documentation is found at [https://condynsate.github.io/condynsate/](https://condynsate.github.io/condynsate/).
