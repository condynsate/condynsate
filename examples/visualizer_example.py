# -*- coding: utf-8 -*-
"""
This module gives example usage of the Visualizer class. In this, we load a
cube into the visualizer and update its position and color while tracking it
with the camera.
"""
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

import time
import condynsate
from condynsate import __assets__ as assets
import numpy as np

if __name__ == "__main__":
    # Create an instance of the visualizer running at 60 fps
    vis = condynsate.Visualizer(frame_rate=60.0, record=False)

    # Turn off the axes and the ground plane
    vis.set_axes(False)
    vis.set_grid(False)

    # Set the initial position of the camera
    vis.set_cam_position((0.0, -3.0, 3.0))

    # Add a plane as the ground from an obj file and apply a texture
    vis.add_object('Ground', assets['plane.obj'],
                   tex_path=assets['concrete.png'],
                   scale=(10., 10., 1.))

    # Add a cube from an stl file.
    vis.add_object('Cube', assets['cube.stl'],
                   color=(0.121, 0.403, 0.749), # Set initial color
                   scale=(0.5, 0.5, 0.5), # Scale cube to 0.5x0.5x0.5
                   position=(0., 0., 0.25) # Set initial position on ground
                   )

    N = 1750
    P0 = np.array([0., 0., 0.25])
    P1 = np.array([-0.5, 0.5, 4])
    for i in range(N):
        t = i / (N-1)

        # Get the position and orientation to set the cube to
        p = (1-t)*P0 + t*P1
        roll = np.sin(5*t)
        pitch = np.sin(7*t)
        yaw = np.sin(11*t)

        # Transform the cube to the desired position and orientation
        vis.set_transform('Cube',
                          position=p,
                          roll=roll,
                          pitch=pitch,
                          yaw=yaw,
                          scale=(0.5, 0.5, 0.5), # You must reset the scale
                          # lest it return to default (1,1,1)
                          )

        # Select a new color to apply to the cube and apply it
        color = (0.121+np.sin(5*t),
                 0.403+np.cos(7*t),
                 0.749+np.cos(11*t))
        color = tuple(float(max(0,min(c,1))) for c in color)
        vis.set_material('Cube', color=color)

        # Set the camera target to the new position of the cube
        vis.set_cam_target(p)

        # Move the camera's position in a dramatic way
        vis.set_cam_position((3*np.sin(4*t), -3*np.cos(4*t), 3.+4.*t))

        # Run the updates at about triple the frame rate
        time.sleep(0.0056)

    # When done, terminate ensure all children threads exit gracefully
    vis.terminate()
