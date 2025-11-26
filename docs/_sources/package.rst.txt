######################
The condynsate Package
######################

**********************
The project Subpackage
**********************

The project Module
==================
.. automodule:: condynsate.project.project
   :no-members:

The Project Class
-----------------
.. autoclass:: condynsate.Project
   :no-members:
.. automethod:: condynsate.Project.load_urdf
.. automethod:: condynsate.Project.step
.. automethod:: condynsate.Project.reset
.. automethod:: condynsate.Project.refresh_visualizer
.. automethod:: condynsate.Project.refresh_animator
.. automethod:: condynsate.Project.await_keypress
.. automethod:: condynsate.Project.await_anykeys
.. automethod:: condynsate.Project.terminate

************************
The simulator Subpackage
************************

The simulator Module
====================
.. automodule:: condynsate.simulator.simulator
   :no-members:

The Simulator Class
-------------------
.. autoclass:: condynsate.Simulator
   :no-members:
.. automethod:: condynsate.Simulator.set_gravity
.. automethod:: condynsate.Simulator.load_urdf
.. automethod:: condynsate.Simulator.step
.. automethod:: condynsate.Simulator.reset
.. automethod:: condynsate.Simulator.terminate

The dataclasses Module
======================
.. automodule:: condynsate.simulator.dataclasses
   :members:

The objects Module
==================
.. automodule:: condynsate.simulator.objects
   :no-members:

The Body Class
--------------
.. autoclass:: condynsate.simulator.objects.Body
   :no-members:
.. automethod:: condynsate.simulator.objects.Body.set_initial_state
.. automethod:: condynsate.simulator.objects.Body.set_state
.. automethod:: condynsate.simulator.objects.Body.apply_force
.. automethod:: condynsate.simulator.objects.Body.apply_torque
.. automethod:: condynsate.simulator.objects.Body.reset
.. automethod:: condynsate.simulator.objects.Body.clear_visual_buffer

The Joint Class
---------------
.. autoclass:: condynsate.simulator.objects.Joint
   :no-members:
.. automethod:: condynsate.simulator.objects.Joint.set_dynamics
.. automethod:: condynsate.simulator.objects.Joint.set_initial_state
.. automethod:: condynsate.simulator.objects.Joint.set_state
.. automethod:: condynsate.simulator.objects.Joint.apply_torque
.. automethod:: condynsate.simulator.objects.Joint.reset

The Link Class
--------------
.. autoclass:: condynsate.simulator.objects.Link
   :no-members:
.. automethod:: condynsate.simulator.objects.Link.set_color
.. automethod:: condynsate.simulator.objects.Link.set_texture
.. automethod:: condynsate.simulator.objects.Link.set_dynamics
.. automethod:: condynsate.simulator.objects.Link.apply_force

*************************
The visualizer Subpackage
*************************

The visualizer Module
=====================
.. automodule:: condynsate.visualizer.visualizer
   :no-members:

The Visualizer Class
--------------------
.. autoclass:: condynsate.Visualizer
   :no-members:
.. automethod:: condynsate.Visualizer.set_grid
.. automethod:: condynsate.Visualizer.set_axes
.. automethod:: condynsate.Visualizer.set_background
.. automethod:: condynsate.Visualizer.set_spotlight
.. automethod:: condynsate.Visualizer.set_ptlight_1
.. automethod:: condynsate.Visualizer.set_ptlight_2
.. automethod:: condynsate.Visualizer.set_amblight
.. automethod:: condynsate.Visualizer.set_dirnlight
.. automethod:: condynsate.Visualizer.set_cam_position
.. automethod:: condynsate.Visualizer.set_cam_target
.. automethod:: condynsate.Visualizer.set_cam_zoom
.. automethod:: condynsate.Visualizer.set_cam_frustum
.. automethod:: condynsate.Visualizer.add_object
.. automethod:: condynsate.Visualizer.set_transform
.. automethod:: condynsate.Visualizer.set_material
.. automethod:: condynsate.Visualizer.reset
.. automethod:: condynsate.Visualizer.terminate

***********************
The animator Subpackage
***********************

The animator Module
===================
.. automodule:: condynsate.animator.animator
   :no-members:

The Animator Class
------------------
.. autoclass:: condynsate.Animator
   :no-members:
.. automethod:: condynsate.Animator.add_lineplot
.. automethod:: condynsate.Animator.add_barchart
.. automethod:: condynsate.Animator.start
.. automethod:: condynsate.Animator.refresh
.. automethod:: condynsate.Animator.barchart_set_value
.. automethod:: condynsate.Animator.lineplot_append_point
.. automethod:: condynsate.Animator.lineplot_set_data
.. automethod:: condynsate.Animator.reset
.. automethod:: condynsate.Animator.terminate

***********************
The keyboard Subpackage
***********************

The keyboard Module
===================
.. automodule:: condynsate.keyboard.keyboard
   :no-members:

The Keyboard Class
------------------
.. autoclass:: condynsate.Keyboard
   :no-members:
.. automethod:: condynsate.Keyboard.get_pressed
.. automethod:: condynsate.Keyboard.is_pressed
.. automethod:: condynsate.Keyboard.await_press
.. automethod:: condynsate.Keyboard.terminate