==========
condynsate
==========

We present a Python-based, open-source software package called ``condynsate`` (\ **con**\ trol and **dyn**\ amics simul\ **at**\ or) designed to ease the creation of computation-based dynamics and control demonstrations, assignments, and projects. Guided by a simulation and role-play pedagogy, projects built with ``condynsate`` mimic the experience of simulation-based games to enhance learning and retention. 

With a physics engine provided by `PyBullet`_ and 3D visualization provided by `MeshCat`_, ``condynsate`` implements real-time simulation of `stl`_ and `obj`_ defined rigid bodies and `urdf`_ defined articulated bodies. It allows users to interact with simulation results through a browser-based 3D viewer to visualize simulations, a built-in animator to plot arbitrary simulation states, and a keyboard module that allows detection of key press events. These features equip ``condynsate`` with a broad scope of applicability by guaranteeing that any dynamic system that can be described by a urdf file, a file format created by `Open Robotics`_ for the `Robot Operating System`_ software, is supported.

Documentation, tutorials, and examples were generated with the intent of educating instructors on the usage of ``condynsate`` to develop teaching demonstrations. The five included tutorials walk through the development of ``condynsate``-based projects, including the simulation environment, design and implementation of mechanical systems, application of internal and external forces and torques to the mechanical systems, real-time visualization and animation, keyboard interactivity, and methods of data collection. 

In addition, at least one example of usage is included for each of the five major modules: the ``Keyboard`` module, which provides keyboard interactivity; the ``Visualizer`` module, which provides 3D visualization; the ``Animator`` module, which provides GUI-based plotting; the ``Simulator`` module, which provides the physics simulation environment; and the ``Project`` module, which provides automatic interfacing with all other modules for easy project development. 

``condynsate`` was built by `G. Schaer`_ and funded by the `Computational Tools for Dynamics and Control grant`_ through the University of Illinois Urbana-Champaign Grainger College of Engineering Academy for Excellence in Engineering Education (AE3) Strategic Instructional Innovations Program (SIIP).

All materials, including the package and example usage, have been made publicly available at `https://github.com/condynsate/condynsate`_ and are licensed under the GPL-3.0-only and MIT licenses. To install, type

.. code-block:: bash

   python3 -m pip install condynsate

in a Python-enabled terminal.

.. _PyBullet: https://pybullet.org/wordpress/
.. _Meshcat: https://github.com/meshcat-dev/meshcat-python/
.. _stl: https://en.wikipedia.org/wiki/STL_(file_format)/
.. _obj: https://en.wikipedia.org/wiki/Wavefront_.obj_file/
.. _urdf: http://wiki.ros.org/urd/
.. _Open Robotics: https://www.openrobotics.org/
.. _Robot Operating System: https://www.ros.org/
.. _G. Schaer: https://www.linkedin.com/in/grayson-schaer/
.. _Computational Tools for Dynamics and Control grant: https://ae3.grainger.illinois.edu/programs/siip-grants/64459
.. _https://github.com/condynsate/condynsate: https://github.com/condynsate/condynsate


Contents
--------

.. toctree::

   installation

   package

   examples

   tutorials