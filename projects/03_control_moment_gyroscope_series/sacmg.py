# -*- coding: utf-8 -*-
"""
This module implements the backend for the single axis CMG project. In this
project, we use a single axis CMG to balance an inverted pendulum.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from time import sleep
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

def _make(initial_angle, initial_gimbal_angle, visualization):
    # Create the project
    proj = Project(keyboard=False, visualizer=visualization, animator=False)

    # Load a ground plane
    ground = proj.load_urdf(assets['plane_medium.urdf'], fixed=True)
    ground.links['plane'].set_texture(assets['tile_floor.png'])

    # Load the cmg
    cmg = proj.load_urdf(assets['sacmg.urdf'], fixed=True)
    cmg.set_initial_state(position=(0,0,1.2)) # Place cmg on the ground

    # Set joint friction to small value and eliminate link air resistance
    # Add joint limits to the A so that it collides with the base
    cmg.joints['A_to_B'].set_dynamics(damping=0.00005)
    cmg.joints['B_to_flywheel'].set_dynamics(damping=0.0)
    cmg.joints['base_to_A'].set_dynamics(damping=0.0, limits=(-1.4049, 1.4049))
    for link in cmg.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    # Set the speed of the core to ~50 rps
    cmg.joints['B_to_flywheel'].set_initial_state(omega=314)

    # Set the initial angle of the cmg pendulum
    cmg.joints['base_to_A'].set_initial_state(angle=initial_angle)
    cmg.joints['A_to_B'].set_initial_state(angle=initial_gimbal_angle)

    # Set some visualizer options
    if visualization:
        proj.visualizer.set_axes(False)
        proj.visualizer.set_grid(False)
        proj.visualizer.set_cam_position((-0.75, -4, 3.25))
        proj.visualizer.set_cam_target(cmg.center_of_mass)
        proj.refresh_visualizer()
    return proj, cmg

def _stall(proj):
    try:
        # Await the user pressing enter (in available)
        proj.keyboard.await_press('enter')

    # If no keyboard exists, ignore this call
    except AttributeError:
        sleep(1.333)

def _state(cmg):
    a = cmg.joints['base_to_A'].state
    b = cmg.joints['A_to_B'].state
    state = {'omega_alpha':a.omega,
             'omega_beta':b.omega,
             'alpha':a.angle,
             'beta':b.angle,}
    return state

def _sim_loop(proj, cmg, get_torque, time, real_time):
    # Reset the project to its initial state.
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':np.array([]),
            'omega_alpha':np.array([]),
            'omega_beta':np.array([]),
            'alpha':np.array([]),
            'beta':np.array([]),
            'tau_beta':np.array([]),}

    # Run a simulation loop
    while proj.simtime <= time:
        # Get the state of the system
        state = _state(cmg)

        # Get the controller torque
        torque = get_torque(state)
        torque = np.clip(torque, -0.0025, 0.0025)

        # Apply the controller
        cmg.joints['A_to_B'].apply_torque(torque,
                                          draw_arrow=True,
                                          arrow_scale=600,
                                          arrow_offset=0.8,)

        # Update the data
        data['time'] = np.append(data['time'], proj.simtime)
        data['omega_alpha']=np.append(data['omega_alpha'],state['omega_alpha'])
        data['omega_beta'] = np.append(data['omega_beta'], state['omega_beta'])
        data['alpha'] = np.append(data['alpha'], state['alpha'])
        data['beta'] = np.append(data['beta'], state['beta'])
        data['tau_beta'] = np.append(data['tau_beta'], torque)

        # Take a simulation step
        proj.step(real_time=real_time, stable_step=False)

    # Return the collected data
    return data

def run(initial_angle, controller,
        initial_gimbal_angle=0.0, time=20.0, real_time=True):
    """
    Makes and runs a condynsate-based simulation of a dual axis CMG.
    The goal of the simulation is to apply torques to the gimbals such
    that the chassis remains oriented. These torques are provided by
    controller. At every time step, calls controller to get the torque applied
    to the gimbals based on the state of the system.

    Parameters
    ----------
    initial_angle : float,
        The initial angle of the pendulum ring in radians.
    controller : function
        The controller function.
    initial_gimbal_angle : float, optional
        The initial angle of the gimbal joint in radians. The default is 0.
    time : float, optional
        The amount of time to run the simulation in seconds. The default is
        20.
    real_time : boolean, optional
        A boolean flag that indicates if the simulation is run in real time
        with visualization (True) or as fast as possible with no visualization
        (False). Regardless of choice, simulation data is still gathered.
        The default is True

    Returns
    -------
    data : dictionary of array-likes with length n
        The data collected during the simulation.

    """
    # Build the project, run the simulation loop, terminate the project
    proj, cmg = _make(initial_angle, initial_gimbal_angle, real_time)
    if real_time:
        _stall(proj)
    data = _sim_loop(proj, cmg, controller, time, real_time)
    proj.terminate()
    return data
