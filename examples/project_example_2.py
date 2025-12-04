# -*- coding: utf-8 -*-
"""
This module gives an example use case for the Project class. In this example,
we create a project that uses the visualizer and the keyboard facilitate
interaction with a 3 axis gyroscope.
"""
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from condynsate import Project
from condynsate import __assets__ as assets

def apply_torques(project, gryoscope):
    """
    Applies a set of torques to each of the gyroscope's gimbals according
    to keyboard inputs.

    Parameters
    ----------
    project : condynsate.Project
        The project class in which the simulation is occuring.
    gryoscope : condynsate.simulator.objects.Body
        The gyroscope object.

    Returns
    -------
    None.

    """
    # Determine what torques to apply based on key presses
    tau0 = 0.0 # Torque applied to outermost ring
    tau0 -= 0.5 * float(project.keyboard.is_pressed('q')) # Convert bool  float
    tau0 += 0.5 * float(project.keyboard.is_pressed('e'))

    tau1 = 0.0 # Torque applied to the middle ring
    tau1 -= 0.5 * float(project.keyboard.is_pressed('w'))
    tau1 += 0.5 * float(project.keyboard.is_pressed('s'))

    tau2 = 0.0 # Torque applied to the inner ring
    tau2 -= 0.5 * float(project.keyboard.is_pressed('a'))
    tau2 += 0.5 * float(project.keyboard.is_pressed('d'))

    # Apply the torques
    gryoscope.joints['base_to_ring3'].apply_torque(
        tau0, # Set the torque value
        draw_arrow=True, # Draw an arrow to visualize the applied torque
        arrow_scale=1.5, # Set the scaling of the arrow
        arrow_offset=0.5,) # Move the arrow away from the center of the

    gryoscope.joints['ring3_to_ring2'].apply_torque(tau1,
                                                    draw_arrow=True,
                                                    arrow_scale=1.5,
                                                    arrow_offset=0.8,)
    gryoscope.joints['ring2_to_ring1'].apply_torque(tau2,
                                                    draw_arrow=True,
                                                    arrow_scale=1.5,
                                                    arrow_offset=0.5,)

def set_color(gryoscope, max_omega=10.0):
    """
    Colors the inner wheel of the gyroscope according to its
    angular velocity.

    Parameters
    ----------
    gryoscope : condynsate.simulator.objects.Body
        The gyroscope object.
    max_omega : float, optional
        The maximum allowed angular speed of the core. This is used to
        scale the coloring

    Returns
    -------
    None.

    """
    # Get the angular velocity of the wheel in its body coordinates
    omega = gryoscope.links['core'].state.omega_in_body

    # In the case of the wheel, the +z body axis is the rotational axis,
    # so isolate that
    omega = omega[2]

    # Get a color based on the rotation rate
    r = min(max(omega / max_omega + 1., 0.), 1.)
    g = 1. - abs(omega) / max_omega
    b = min(max(1. - omega / max_omega, 0.), 1.)

    # Set the color of the core link
    gryoscope.links['core'].set_color((r, g, b))

def set_omega(project, gryoscope):
    """
    Sets the anular velocity of the inner wheel of the gyroscope
    according to keyboard inputs

    Parameters
    ----------
    project : condynsate.Project
        The project class in which the simulation is occuring.
    gryoscope : condynsate.simulator.objects.Body
        The gyroscope object.

    Returns
    -------
    None.

    """
    # The amount by which to iterate the wheel speed
    iter_val = 0.0
    iter_val -= 0.1 * float(project.keyboard.is_pressed('f'))
    iter_val += 0.1 * float(project.keyboard.is_pressed('r'))

    # Read the current wheel joint speed, iterate it, and set the new value
    old_omega = gryoscope.joints['ring1_to_core'].state.omega
    new_omega = old_omega + iter_val
    gryoscope.joints['ring1_to_core'].set_state(omega = new_omega)

if __name__ == "__main__":
    # Make an instance of project
    proj = Project(keyboard = True,
               visualizer = True,
               animator = False)

    # Turn off the axes and grid visualization.
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)

    # Load a plane with a carpet texture for the ground
    ground = proj.load_urdf(assets['plane_medium.urdf'], fixed=True)
    ground.links['plane'].set_texture(assets['carpet.png'])

    # Load and orient a 2 gimbal gyroscope.
    gyro = proj.load_urdf(assets['gyro.urdf'])
    gyro.set_initial_state(fixed=True)

    # Set the camera's position and focus on gyro
    proj.visualizer.set_cam_position((2.0, -4.0, 3.75))
    proj.visualizer.set_cam_target(gyro.center_of_mass)

    # Set joint damping for gimbals
    gyro.joints['base_to_ring3'].set_dynamics(damping=0.025)
    gyro.joints['ring3_to_ring2'].set_dynamics(damping=0.025)
    gyro.joints['ring2_to_ring1'].set_dynamics(damping=0.025)

    # Set no friction and max speed for core joint
    gyro.joints['ring1_to_core'].set_dynamics(damping=0.0, max_omega=10.0)

    # Set the initial speed of the core to 5 rad/sec
    gyro.joints['ring1_to_core'].set_initial_state(omega=5.0)

    # Reset the project to its initial state. This is required to
    # reset the simulation, reset the visualizer, and reset/start the
    # animator.
    proj.reset() # This returns 0 on success

    # Run a 10 second simulation loop
    while proj.simtime <= 10.:
        apply_torques(proj, gyro)
        set_omega(proj, gyro)
        set_color(gyro)
        proj.step(real_time=True, stable_step=False)

    # This is required to save any videos that are recorded and gracefully exit
    # all the children threads.
    proj.terminate()
