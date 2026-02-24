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

def _make(initial_angle, visualization):
    # Create the project
    proj = Project(keyboard=False, visualizer=visualization, animator=False)

    # Turn off the axes and grid visualization.
    if visualization:
        proj.visualizer.set_axes(False)
        proj.visualizer.set_grid(False)

    # Load and orient the ground and walls
    ground = proj.load_urdf(assets['plane_medium.urdf'], fixed=True)
    ground.links['plane'].set_texture(assets['tile_floor.png'])
    left = proj.load_urdf(assets['half_plane_medium.urdf'], fixed=True)
    left.links['plane'].set_texture(assets['white_wall.png'])
    left.set_initial_state(roll=1.5708, yaw=1.5708, position=(-5,0,2.5))
    right = proj.load_urdf(assets['half_plane_medium.urdf'], fixed=True)
    right.links['plane'].set_texture(assets['white_wall.png'])
    right.set_initial_state(roll=1.5708, yaw=-1.5708, position=(5,0,2.5))
    back = proj.load_urdf(assets['half_plane_medium.urdf'], fixed=True)
    back.links['plane'].set_texture(assets['white_wall.png'])
    back.set_initial_state(roll=1.5708, position=(0,5,2.5))

    # Load and orient a cart carrying an inverted pendulum. Set initial state
    cart = proj.load_urdf(assets['cart.urdf'])
    cart.set_initial_state(position=(0,0,0.12501))
    cart.joints['chassis_to_arm'].set_initial_state(angle=initial_angle)

    # After the initial state is set, refresh the visualizer to show the change
    proj.refresh_visualizer()

    # Focus the camera on the cart
    if visualization:
        proj.visualizer.set_cam_target(cart.center_of_mass)

    # Remove all joint friction and link air resistance
    for joint in cart.joints.values():
        joint.set_dynamics(damping=0.0)
    for link in cart.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    # Return the project and wheel objects
    return proj, cart

def _stall(proj):
    try:
        # Await the user pressing enter (in available)
        proj.keyboard.await_press('enter')

    # If no keyboard exists, ignore this call
    except AttributeError:
        sleep(1.333)

def _state(cart):
    # Read the angle and rate of the pendulum
    # Read the mean angle and mean rate of the wheels
    p_st = cart.joints['chassis_to_arm'].state
    c_st = cart.links['chassis'].state
    x = c_st.position[0]
    x_dot = c_st.velocity[0]
    w_ang = -x / c_st.position[2]
    w_omg = -x_dot / c_st.position[2]
    return (p_st.angle, p_st.omega, w_ang, w_omg)

def _sim_loop(proj, cart, get_torque, time, real_time):
    # Reset the project to its initial state.
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':np.array([]),
            'angle':np.array([]),
            'angle_integral':np.array([]),
            'angular_rate':np.array([]),
            'wheel':np.array([]),
            'wheel_integral':np.array([]),
            'wheel_rate':np.array([]),
            'torque':np.array([]),}

    # Run a simulation loop
    ang_int = 0.0
    whl_int = 0.0
    while proj.simtime <= time:
        # Get the state of the system
        state = _state(cart)
        ang_int += state[0]*proj.simulator.dt
        whl_int += state[2]*proj.simulator.dt
        state = {'angle':state[0],
                 'angle_integral':ang_int,
                 'angular_rate':state[1],
                 'wheel':state[2],
                 'wheel_integral':whl_int,
                 'wheel_rate':state[3],}

        # Get the controller torque
        torque = get_torque(state)
        torque = np.clip(torque, -7.5, 7.5)
        wheels = ('chassis_to_wheel_1','chassis_to_wheel_2',
                  'chassis_to_wheel_3','chassis_to_wheel_4')

        # Apply the controller torque
        for wheel in wheels:
            # This will offset a drawn torque arrow out of the center of
            # the wheels so we can see them. It is required to be
            # different between the front wheels (1 and 2) and the rear
            # wheels (3 and 4) because they are oriented 180 degrees apart
            offset = ('3' in wheel or '4' in wheel)*0.1-0.05
            cart.joints[wheel].apply_torque(torque,
                                            draw_arrow=True,
                                            arrow_scale=0.067,
                                            arrow_offset=offset)


        # Update the data
        data['time'] = np.append(data['time'], proj.simtime)
        data['angle'] = np.append(data['angle'], state['angle'])
        data['angle_integral'] = np.append(data['angle_integral'], ang_int)
        data['angular_rate'] = np.append(data['angular_rate'],
                                         state['angular_rate'])
        data['wheel'] = np.append(data['wheel'], state['wheel'])
        data['wheel_integral'] = np.append(data['wheel_integral'], whl_int)
        data['wheel_rate'] = np.append(data['wheel_rate'], state['wheel_rate'])
        data['torque'] = np.append(data['torque'], torque)

        # Take a simulation step
        proj.step(real_time=real_time, stable_step=False)

    # Return the collected data
    return data

def run(initial_angle, controller, time=30.0, real_time=True):
    """
    Makes and runs a condynsate-based simulation of an inverted pendulum on
    a cart. The goal of the simulation is to apply torques to the wheels such
    that the pendulum remains upright. These torques are provided by
    controller. At every time step, calls controller to get the torque applied
    to the wheels based on the state of the system.

    Parameters
    ----------
    initial_angle : float
        The initial angle of the pendulum in radians. 0 is vertical.
    controller : function
        The controller function. Takes as argument a dictionary called state
        with the keys
            angle : float
                The current angle the pendulum in radians
            angle_integral : float
                The integral of the pendulum angle from the start of the
                simulation to now in radian-seconds
            angular_rate : float
                The current angular rate of the pendulum in radians / second
            wheel : float
                The current mean angle of the wheels in radians
            wheel_integral : float
                The integral of the mean wheel angle from the start of the
                simulation to now in radian-seconds
            wheel_rate : float
                The current mean angular rate of the wheels in radians / second
        Returns a float which is the torque applied to each wheel in Nm.
    time : float, optional
        The duration of the simulation. The default is 30.0.
    real_time : boolean, optional
        A boolean flag that indicates if the simulation is run in real time
        with visualization (True) or as fast as possible with no visualization
        (False). Regardless of choice, simulation data is still gathered.

    Returns
    -------
    data : dictionary of array-likes with length n
        The data collected during the simulation. Has the keys:
            time : list of n floats
                The time, in seconds, at which each data point is collected
            angle : list of n floats
                The angle of the pendulum, in radians, at each of the n data
                collection points.
            angle_integral : list of n floats
                The integral of the pendulum angle, in radian-seconds, from the
                start of the simulation to each of the n data collection points
            angular_rate: list of n floats
                The angular rate of the pendulum, in radians per second, at
                each of the n data collection points.
            wheel : list of n floats
                The mean angle of the wheels, in radians, at each of the n data
                collection points.
            wheel_integral : list of n floats
                The integral of the mean wheel angle, in radian-seconds, from
                the start of the simulation to each of the n data collection
                points
            wheel_rate: list of n floats
                The mean angular rate of the wheels, in radians per second,
                at each of the n data collection points.
            torque : list of n floats
                The torque applied to the wheels, in Newton-meters, at each of
                the n data collection points.

    """
    # Build the project, run the simulation loop, terminate the project
    proj, cart = _make(initial_angle, real_time)
    if real_time:
        _stall(proj)
    data = _sim_loop(proj, cart, controller, time, real_time)
    proj.terminate()
    return data
