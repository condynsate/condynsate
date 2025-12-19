(C) Copyright, 2025 G. Schaer.

This work is licensed under a [GNU General Public License 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) and an MIT License.

SPDX-License-Identifier: GPL-3.0-only AND MIT

# Preamble
We present a Python-based, open-source software package called ``condynsate`` (**con**trol and **dyn**amics simulator) designed to ease the creation of computation-based dynamics and control demonstrations, assignments, and projects. Guided by a simulation and role-play pedagogy, projects built with ``condynsate`` mimics the experience of simulation-based games to enhance learning and retention. 

Users can import rigid and articulated bodies into a simulation environment, visualize the simulation environment in 3D, tweak the appearance of the visualization, create and display live-updating plots, read the state, apply arbitrary forces, and tweak the physics and appearance of all joints and links in each body, run full physics simulations, interact with the environment through keypresses, and record the visual outputs of ``condynsate``, all in real time.

All materials, including the package and example usage, have been made publicly available at [https://github.com/condynsate/condynsate](https://github.com/condynsate/condynsate) and are licensed under the GPL-3.0-only and MIT licenses. To install, type

```bash
python3 -m pip install condynsate
```

in a Python-enabled terminal.

# The condynsate Package
With a physics engine provided by ``PyBullet`` and 3D visualization provided by ``MeshCat``, ``condynsate`` implements real-time simulation of .stl and .obj defined rigid bodies and .urdf defined articulated bodies. It allows users to interact with simulation results through a browser-based 3D viewer to visualize simulations, a built-in animator to plot arbitrary simulation states, and a keyboard module that allows detection of key press events. These features equip ``condynsate`` with a broad scope of applicability by guaranteeing that any dynamic system that can be described by a .urdf file, a file format created by Open Robotics for the Robot Operating System software, is supported.

Documentation, tutorials, and examples were generated with the intent of educating instructors on the usage of ``condynsate`` to develop teaching demonstrations. The five included tutorials walk through the development of ``condynsate``-based projects, including the simulation environment, design and implementation of mechanical systems, application of internal and external forces and torques to the mechanical systems, real-time visualization and animation, keyboard interactivity, and methods of data collection.

 In addition, at least one example of usage is included for each of the five major modules: the ``Keyboard`` module, which provides keyboard interactivity; the ``Visualizer`` module, which provides 3D visualization; the ``Animator`` module, which provides GUI-based plotting; the ``Simulator`` module, which provides the physics simulation environment; and the ``Project`` module, which provides automatic interfacing with all other modules for easy project development. The ``Project`` module examples best demonstrate how educators may use ``condynsate`` to construct assignments and demonstrations. As such, the provided examples are common mechanical systems used in dynamics and control education: an inverted pendulum autonomously balanced on a 4-wheeled cart, a three-axis gyroscope with keyboard interactivity, and a double pendulum with real-time phase space plotting.

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

# Final Remarks
To date, projects built with ``condynsate`` have been successfully deployed in an undergraduate aerospace controls course during the Spring 2024 and Fall 2024 semesters at the University of Illinois Urbana-Champaign.  Post-semester student survey results showed that 69% of students felt better prepared to solve control problems using computational tools, and 86% stated they felt their Python skills improved.  We conclude that the deployment and continued development of computation-based curricula provide an enhanced approach for dynamics and control education. Computationally literate students are better equipped to exploit the full versatility of a computer to tackle complex problems, and we propose that ``condynsate`` can set students on the path of utilizing computational resources as a matter-of-course tool both at university and in their careers.
