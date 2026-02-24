# -*- coding: utf-8 -*-
"""
This module implements the backend for the wheel on axle project. In this
project, we apply torque to a wheel on an freely rotating axle to
enforce a target angle.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from time import sleep
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

def _make(target):
    # Make an instance of project
    proj = Project(keyboard = False, visualizer = True, animator = False)

    # Turn off the axes and grid visualization.
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)

    # Make the lighting pretty
    proj.visualizer.set_amblight(intensity=0.3)

    # Load a plane with a tile texture for the ground
    ground = proj.load_urdf(assets['plane_medium.urdf'], fixed=True)
    ground.links['plane'].set_texture(assets['tile_floor.png'])

    # Load a wheel on an axle
    wheel = proj.load_urdf(assets['wheel.urdf'], fixed=True)

    # Set the wheel's target angle as its initial state
    wheel.joints['axle_to_target'].set_initial_state(angle=target)

    # After the initial state is set,
    # refresh the visualizer to reflect the change
    proj.refresh_visualizer()

    # Set the camera's position and focus on wheel
    proj.visualizer.set_cam_position((0.5, -0.75, 1.0))
    proj.visualizer.set_cam_target(wheel.center_of_mass)

    # Remove axle friction to 0
    wheel.joints['axle_to_wheel'].set_dynamics(damping=0.0)

    # Remove all air resistance on every link
    for link in wheel.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    # Return the project and wheel objects
    return proj, wheel

def _stall(proj):
    try:
        # Await the user pressing enter (in available)
        proj.keyboard.await_press('enter')

    # If no keyboard exists, ignore this call
    except AttributeError:
        sleep(1.333)

def _wheel_state(wheel):
    # Read the angle and rate of the wheel on the axle
    # Read the angle of the target arrow on the axle
    w_st = wheel.joints['axle_to_wheel'].state
    t_st = wheel.joints['axle_to_target'].state
    return (w_st.angle, w_st.omega, t_st.angle)

def _update_target(proj, wheel):
    # Find an interation value such that the target rotates 60 deg/s
    iter_val = 0.3333*np.pi*proj.simulator.dt

    try:
        # Use keypresses to determine if target angle will be updated
        itr = 0.0
        if proj.keyboard.is_pressed('q'):
            itr -= iter_val
        if proj.keyboard.is_pressed('e'):
            itr += iter_val

        # Update the target angle
        curr = wheel.joints['axle_to_target'].state.angle
        wheel.joints['axle_to_target'].set_state(angle=curr+itr)

    # If no keyboard exists, ignore this call
    except AttributeError:
        pass

def _sim_loop(proj, wheel, get_torque, disturbance, time):
    # Reset the project to its initial state. This is required to
    # reset the simulation, reset the visualizer, and reset/start the
    # animator
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':np.array([]),
            'angle':np.array([]),
            'angle_integral':np.array([]),
            'angular_rate':np.array([]),
            'target':np.array([]),
            'target_integral':np.array([]),
            'torque':np.array([]),
            'disturbance':np.array([])}

    # Run a simulation loop
    ang_int = 0.0
    tag_int = 0.0
    while proj.simtime <= time:
        # Update the target angle via keypresses (if available)
        _update_target(proj, wheel)

        # Get and apply the controller torque
        state = _wheel_state(wheel)
        ang_int += state[0]*proj.simulator.dt
        tag_int += state[2]*proj.simulator.dt
        state = {'angle':state[0],
                 'angle_integral':ang_int,
                 'angular_rate':state[1],
                 'target':state[2],
                 'target_integral':tag_int}
        torque = get_torque(state)
        wheel.joints['axle_to_wheel'].apply_torque(torque,
                                                   draw_arrow=True,
                                                   arrow_scale=1.0,
                                                   arrow_offset=0.075)

        # Apply the disturbance torque
        wheel.joints['axle_to_wheel'].apply_torque(disturbance)

        # Update the data
        data['time'] = np.append(data['time'], proj.simtime)
        data['angle'] = np.append(data['angle'], state['angle'])
        data['angle_integral'] = np.append(data['angle_integral'], ang_int)
        data['angular_rate'] = np.append(data['angular_rate'],
                                         state['angular_rate'])
        data['target'] = np.append(data['target'], state['target'])
        data['target_integral'] = np.append(data['target_integral'], tag_int)
        data['torque'] = np.append(data['torque'], torque)
        data['disturbance'] = np.append(data['disturbance'], disturbance)

        # Take a simulation step
        proj.step(real_time=True, stable_step=False)

    # Return the collected data
    return data

def run(target, controller, disturbance=0.0, time=15.0):
    """
    Makes and runs a condynsate-based simulation of a wheel on an axle.
    The goal of the simulation is to apply torques to the wheel such that
    it points in a desired direction. These torques are provided by controller.
    At every time step, calls controller to get the torque applied to the wheel
    based on the state of the wheel.

    Parameters
    ----------
    target : float
        The target angle of the wheel in radians.
    controller : function
        The controller function. Takes as argument a dictionary called state
        with the keys
            angle : float
                The current angle the wheel is facing in radians
            angle_integral : float
                The integral of the wheel's angle from the start of the
                simulation to now
            angular_rate : float
                The current angular rate of the wheel in radians / second
            target : float
                The current target angle in radians
            target_integral : float
                The integral of the target angle from the start of the
                simulation to now
        Returns a float which is the torque applied to the wheel in Nm.
    disturbance : float, optional
        The disturbance torque to apply to the wheel in Nm. The default is 0.0.
    time : float, optional
        The duration of the simulation. The default is 15.0.

    Returns
    -------
    data : dictionary of array-likes with length n
        The data collected during the simulation. Has the keys:
            time : list of n floats
                The time, in seconds, at which each data point is collected
            angle : list of n floats
                The angle of the wheel, in radians, at each of the n data
                collection points.
            angle_integral : list of n floats
                The integral of the wheel's angle, in radian-seconds, from the
                start of the simulation to each of the n data collection points
                in radian-seconds
            angular_rate: list of n floats
                The angular rate of the wheel, in radians per second, at each
                of the n data collection points.
            target : list of n floats
                The target angle, in radians, at each of the n data collection
                points.
            target_integral : list of n floats
                The integral of the target angle, in radian-seconds, from the
                start of the simulation to each of the n data collection points
            torque : list of n floats
                The torque applied to the wheel, in Newton-meters, at each of
                the n data collection points.

    """
    # Build the project, run the simulation loop, terminate the project
    proj, wheel = _make(target)
    _stall(proj)
    data = _sim_loop(proj, wheel, controller, disturbance, time)
    proj.terminate()
    return data
