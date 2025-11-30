(C) Copyright, 2025 G. Schaer.

This work is licensed under a [GNU General Public License 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) and an MIT License.

SPDX-License-Identifier: GPL-3.0-only AND MIT

# Preamble
Mechanical systems and controls are fundamental topics in the fields of mechanical, aerospace, and robotics engineering; however, conventional educational approaches rely heavily upon either classroom-constrained theory---directly antithetical to the dynamics these fields study---or laboratory demonstrations which are limited by cost and complexity. In response to these limitations, we developed a Python-based dynamic system simulation and visualization tool called condynsate (**con**trol and **dyn**amics simul**at**or) with the explicit philosophy of not limiting project complexity while simultaneously promoting ease of use.

Driven by the identification of computational literacy as a professional skill set highly valued in academia and industry, we focused on the later stage of undergraduate education, where less "recipe-style" and more creative implementations of tools and solutions are presented to students. We designed condynsate to be a flexible tool, built upon foundational programming experience from earlier-stage classes, that frees students to creatively approach real-world engineering problems based on their own initiative and drive for exploration. Our goal was to provide a viable approach to both increase the sophistication and ease of development of lecture demonstrations, assignments, and projects related to fundamental engineering concepts that govern mechanical systems (e.g., pendulum, gyroscope) and control systems (e.g., differential wheeled robot, autonomous quadrotor). 

Students who are computationally literate, not only at the level of running programs but also who "code to learn", are better equipped to exploit the full versatility of a computer to tackle complex problems. By designing condynsate to:

1. provide an environment in which students can interact with dynamic systems in adherence with the simulation and role-play pedagogy,
2. act as an introduction to programming in Python, and, most significantly, 
3. facilitate the design and implementation of controls and dynamics design projects without requiring laboratory equipment,

we deliver a computational tool that can be used to set students on the path of confidently utilizing computational resources as matter-of-course tools both at university and in their careers.

# The condynsate Package
condynsate is a Python-based, open-source educational tool built by G. Schaer at the University of Illinois at Urbana-Champaign under the Grainger College of Engineering 2023-2025 Strategic Instructional Innovations Program: [Computational Tools for Dynamics and Control grant](https://ae3.grainger.illinois.edu/programs/siip-grants/64459).

Built on [PyBullet](https://pybullet.org/wordpress/), [MeshCat](https://github.com/meshcat-dev/meshcat-python/), and [Tk](https://www.tcl-lang.org/), it implements nonlinear simulation of [stl](https://en.wikipedia.org/wiki/STL_(file_format)/) and [obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file/) defined rigid bodies and\or [urdf](http://wiki.ros.org/urd/) articulated bodies. A browser-based 3D viewer visualizes the simulation, and a built-in animator allows for plotting of arbitrary states in a Tk-based GUI, all in real-time. By simultaneously enabling keyboard interactivity, condynsate projects are designed to feel more like video games, familiar and fun, rather than conventional lab demos, all while providing similar educational benefits.

All materials, including the package and example usage, have been made publicly available at [https://github.com/condynsate/condynsate](https://github.com/condynsate/condynsate) and are licensed under the GPL-3.0-only and MIT licenses. 

# Installation
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
