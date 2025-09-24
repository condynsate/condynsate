Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

# condynsate

**condynsate** (**con**trol and **dyn**amics simul**at**or) is a python-based, open-source educational tool built by [G. Schaer](http://bretl.csl.illinois.edu/people) at the University of Illinois at Urbana-Champaign under the Grainger College of Engineering 2023-2025 Strategic Instructional Innovations Program: [Computational Tools for Dynamics and Control grant](https://ae3.grainger.illinois.edu/programs/siip-grants/64459). It is designed to aid the education of control and dynamics to aerospace, mechanical, and robotics engineering students by

1. providing a simulation environment in which students can see and interact with controlled and uncontrolled dynamic systems in familiar and [beneficial ways](https://doi.org/10.3390/educsci13070747), 
2. serving as a platform for introductory Python programming, and
3. equipping instructors with a streamlined method of implementing custom in-class demonstrations and lab demos without the need for physical equipment.

Built on [PyBullet](https://pybullet.org/wordpress/), [MeshCat](https://github.com/meshcat-dev/meshcat-python/), and [OpenCV](https://opencv.org/), it implements nonlinear simulation of [stl](https://en.wikipedia.org/wiki/STL_(file_format)/) or [obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file/) rigid bodies and\or [urdf](http://wiki.ros.org/urd/) articulated bodies. A browser-based 3D viewer visualizes the simulation and the evolution of individual states are plotted, all in real-time. By simultaneously enabling keyboard interactivity, condynsate projects are designed to feel more like video games, familiar and fun, rather than conventional lab demos all while providing similar educational benefits. 





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
[python>=3.8](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/) are required. 

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
```

To install condynsate:

```powershell
(.venv) C:\Users\username\condynsate> pip install -e .
```





## MacOS
### PyPi (Recommended)
Coming soon!

### Source
Coming soon!





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
[python>=3.8](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/) are required.

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





