============
Installation
============
Windows
-------
A C++ compiler for C++ 2003 is needed. We recommend using the Desktop development with C++ workload for `Microsoft C++ Build Tools 2022`_. 

Additionally, we strongly recommend installing condynsate in a virtual environment:

.. code-block:: console

   C:\Users\username> python -m venv .venv
   C:\Users\username> .venv\Scripts\activate.bat

When done installing and using condynsate, deactivate the virtual environment with:

.. code-block:: console

   (.venv) C:\Users\username> deactivate


**From PyPi (Recommended)**
***************************
`python>=3.8`_ and `pip`_ are required.

To install condynsate:

.. code-block:: console

   (.venv) C:\Users\username> pip install condynsate


From Source
***********
`python>=3.8`_, `pip`_, and `git`_ are required. 

To clone the repository: 

.. code-block:: console

   (.venv) C:\Users\username> git clone https://github.com/condynsate/condynsate.git
   (.venv) C:\Users\username> cd condynsate
   (.venv) C:\Users\username> git submodule update --init --recursive

To install condynsate:

.. code-block:: console   

   (.venv) C:\Users\username\condynsate> pip install -e .





Linux
-----
We strongly recommend installing condynsate in a virtual environment:

.. code-block:: console

   user@device:~$ python3 -m venv .venv
   user@device:~$ source .venv/bin/activate

On Debian/Ubuntu systems you may need to first install the python3-venv package. For Python 3.10 this can be installed with:

.. code-block:: console

   user@device:~$ sudo apt update
   user@device:~$ sudo apt install python3.10-venv

When done installing and using condynsate, deactivate the virtual environment with:

.. code-block:: console

   (.venv) user@device:~$ deactivate

Additionally, On Debian/Ubuntu systems, to build condynsate you may need to first install the Python and Linux development headers. These can be installed with:

.. code-block:: console

   user@device:~$ sudo apt update
   user@device:~$ sudo apt install build-essential python3-dev linux-headers-$(uname -r)

Finally, the package that provides keyboard interactivity uses `X`_. This means that for keyboard interactivity to work

1. an X server must be running, and

2. the environment variable $DISPLAY must be set.

If these are not true, then keyboard interactivity will not work. All other features will work, though. For example, to use keyboard iteractivity on Ubuntu 22.04, you must first add 

.. code-block:: console

   WaylandEnable=false

to /etc/gdm3/custom.conf and then either reboot your system or run the command

.. code-block:: console

   user@device:~$ systemctl restart gdm3


**From PyPi (Recommended)**
***************************
`python>=3.8`_ and `pip`_ are required. 

To install condynsate:

.. code-block:: console

   (.venv) user@device:~$ pip install condynsate


From Source
***********
`python>=3.8`_, `pip`_, and `git`_ are required.

To clone the repository: 

.. code-block:: console

   (.venv) user@device:~$ git clone https://github.com/condynsate/condynsate.git
   (.venv) user@device:~$ cd condynsate
   (.venv) user@device:~$ git submodule update --init --recursive

To install condynsate:

.. code-block:: console

   (.venv) user@device:~/condynsate$ pip install -e .




.. _Microsoft C++ Build Tools 2022: https://visualstudio.microsoft.com/visual-cpp-build-tools/

.. _python>=3.8: https://www.python.org/

.. _git: https://git-scm.com/

.. _pip: https://pip.pypa.io/en/stable/

.. _X: https://en.wikipedia.org/wiki/X_Window_System