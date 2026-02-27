# -*- coding: utf-8 -*-
"""
This module implements the backend for the pendulum cart project. In this
project, we apply torques to the wheels of a cart to balance an inverted
pendulum atop the cart.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from time import sleep
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

def _make(visualization):
    # Create the project
    proj = Project(keyboard=True, visualizer=visualization, animator=False)

    # Turn off the axes and grid visualization.
    if visualization:
        proj.visualizer.set_axes(True)
        proj.visualizer.set_grid(True)
        proj.visualizer.set_background(top=(0.075, 0.075, 0.11),
                                       bottom=(0.05, 0.05, 0.05))

    # Turn off gravity
    proj.simulator.set_gravity((0., 0., 0.))

    # Load the cmg
    cmg = proj.load_urdf(assets['dual_axis_cmg.urdf'], fixed=False)

    # Set joint friction to small value and eliminate link air resistance
    cmg.joints['chassis_to_outer'].set_dynamics(damping=0.00025)
    cmg.joints['outer_to_inner'].set_dynamics(damping=0.00025)
    cmg.joints['inner_to_core'].set_dynamics(damping=0.0)
    for link in cmg.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    # Set the speed of the core to ~600 rpm
    cmg.joints['inner_to_core'].set_initial_state(omega=60.0)

    # Return the project and wheel objects
    proj.visualizer.set_cam_target(cmg.center_of_mass)
    return proj, cmg

def _stall(proj):
    try:
        # Await the user pressing enter (in available)
        proj.keyboard.await_press('enter')

    # If no keyboard exists, ignore this call
    except AttributeError:
        sleep(1.333)

def _state(cmg):
    return None

def _sim_loop(proj, cmg, get_torque, time, real_time):
    # Reset the project to its initial state.
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':np.array([]),}

    # Run a simulation loop
    while proj.simtime <= time:
        # Get the state of the system
        state = _state(cmg)

        # Get the controller torque
        torque = get_torque(proj)

        # Apply the controller
        # cmg.joints['base_to_chassis'].apply_torque(torque[0],
        #                                             draw_arrow=True,
        #                                             arrow_scale=300,
        #                                             arrow_offset=0.8,)
        cmg.joints['chassis_to_outer'].apply_torque(torque[1],
                                                    draw_arrow=True,
                                                    arrow_scale=600,
                                                    arrow_offset=0.8,)
        cmg.joints['outer_to_inner'].apply_torque(torque[2],
                                                  draw_arrow=True,
                                                  arrow_scale=600,
                                                  arrow_offset=0.5,)

        # # Ensure target speed of core
        # cmg.joints['inner_to_core'].set_initial_state(omega=10.0)

        # Update the data
        data['time'] = np.append(data['time'], proj.simtime)

        # Take a simulation step
        proj.step(real_time=real_time, stable_step=False)

    # Return the collected data
    return data

def run(controller, time=60.0, real_time=True):
    """
    Makes and runs a condynsate-based simulation of a dual axis CMG.
    The goal of the simulation is to apply torques to the gimbals such
    that the chassis remains oriented. These torques are provided by
    controller. At every time step, calls controller to get the torque applied
    to the gimbals based on the state of the system.

    Parameters
    ----------
    controller : function
        The controller function.
    real_time : boolean, optional
        A boolean flag that indicates if the simulation is run in real time
        with visualization (True) or as fast as possible with no visualization
        (False). Regardless of choice, simulation data is still gathered.

    Returns
    -------
    data : dictionary of array-likes with length n
        The data collected during the simulation.

    """
    # Build the project, run the simulation loop, terminate the project
    proj, cmg = _make(real_time)
    if real_time:
        _stall(proj)
    data = _sim_loop(proj, cmg, controller, time, real_time)
    proj.terminate()
    return data

def controller(proj):
    tau0 = 0.0 # Torque applied to the outer chassis
    tau0 -= 0.001 * float(proj.keyboard.is_pressed('q'))
    tau0 += 0.001 * float(proj.keyboard.is_pressed('e'))

    tau1 = 0.0 # Torque applied to the outer gimbal
    tau1 -= 0.001 * float(proj.keyboard.is_pressed('s'))
    tau1 += 0.001 * float(proj.keyboard.is_pressed('w'))

    tau2 = 0.0 # Torque applied to the inner gimbal
    tau2 -= 0.001 * float(proj.keyboard.is_pressed('a'))
    tau2 += 0.001 * float(proj.keyboard.is_pressed('d'))

    return (tau0, tau1, tau2)

if __name__ == "__main__":
    run(controller)
