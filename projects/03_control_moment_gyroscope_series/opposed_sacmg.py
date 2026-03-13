# -*- coding: utf-8 -*-
"""
This module implements the backend for the double axis CMG project. In this
project, we use a double axis CMG to steer a spacecraft.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from time import sleep
from time import time as now
from collections import deque
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

def _make(initial_gamma, visualization):
    # Create the project
    proj = Project(visualizer = visualization,
                   simulator_dt = 1.0/300.0,
                   simulator_gravity = (0.,0.,0.))

    # Load the cmg
    cmg = proj.load_urdf(assets['parallel_single_axis_cmg.urdf'], fixed=False)

    # Set joint friction to small value and eliminate link air resistance
    cmg.joints['S_to_A'].set_dynamics(damping=0.006) # Req for sim stability
    cmg.joints['S_to_B'].set_dynamics(damping=0.006) # Req for sim stability
    cmg.joints['A_to_Fa'].set_dynamics(damping=0.0)
    cmg.joints['B_to_Fb'].set_dynamics(damping=0.0)
    for link in cmg.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    # Set the speed of the cores to ~7.5 rps
    cmg.joints['A_to_Fa'].set_initial_state(omega=47.1)
    cmg.joints['B_to_Fb'].set_initial_state(omega=47.1)

    # Set the initial angle of the cmg pendulum
    cmg.joints['S_to_A'].set_initial_state(angle=initial_gamma)
    cmg.joints['S_to_B'].set_initial_state(angle=-initial_gamma)

    # Set some visualizer options
    if visualization:
        # Load a starfield and planet
        proj.visualizer.add_object('white_starfield',
                                   assets['128-starfield_5-r_cen-orig.obj'],
                                   color=(0.918, 0.929, 1.0),
                                   emissive_color=(0.918, 0.929, 1.0),
                                   scale=(175, 175, 175))
        proj.visualizer.add_object('red_starfield',
                                   assets['64-starfield_5-r_cen-orig.obj'],
                                   scale=(175, 175, 175),
                                   color=(1.0, 0.706, 0.424),
                                   emissive_color=(1.0, 0.706, 0.424),
                                   roll = 161.,
                                   pitch = 74.,
                                   yaw = 14.)
        proj.visualizer.add_object('blue_starfield',
                                   assets['32-starfield_5-r_cen-orig.obj'],
                                   scale=(175, 175, 175),
                                   color=(0.616, 0.741, 1.0),
                                   emissive_color=(0.616, 0.741, 1.0),
                                   roll = 145.,
                                   pitch = 71.,
                                   yaw = 102.)
        proj.visualizer.add_object('planet',
                                   assets['sphere_1_center_origin.stl'],
                                   scale=(5000, 5000, 5000),
                                   color=(0.938, 0.884, 0.766),
                                   emissive_color=(0.375, 0.354, 0.306),
                                   position=(0.0, 0.0, -2600))

        # Make the grid and axes invisible
        proj.visualizer.set_axes(False)
        proj.visualizer.set_grid(False)

        # Set the camera position and target
        proj.visualizer.set_cam_position((6, -6, 6))
        proj.visualizer.set_cam_target(cmg.center_of_mass)

        # Set the scene lighting
        proj.visualizer.set_ptlight_1(position=(3, -3, -6), intensity=0.65)
        proj.visualizer.set_ptlight_2(position=(6, 1, 6), intensity=0.5)
        proj.visualizer.set_spotlight(on=True, shadow=True, position=(6,0,0.5),
                                      intensity=0.7, distance=7.6)
        proj.visualizer.set_dirnlight(on=True, shadow=True, intensity=0.25)
        proj.visualizer.set_amblight(intensity=0.4)

        # Set the background color
        proj.visualizer.set_background((0.0, 0.0, 0.0),
                                       (0.2, 0.2, 0.4),)

        # Refresh the visualizer to reflect the changes we made
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
    s = cmg.state
    a = cmg.joints['S_to_A'].state
    state = {'omega_theta':-s.omega_in_body[1],
             'omega_gamma':a.omega,
             'theta':-s.ypr[1],
             'gamma':a.angle,}
    return state

def _get_des(program_number, sim_time):
    des = 0.0

    # Step programs
    if program_number==1:
        des = 10.0*np.pi/180.0 if sim_time > 2.0 else 0.0
    elif program_number==2:
        des = -15.0*np.pi/180.0 if sim_time > 2.0 else 0.0
    elif program_number==3:
        des = 30.0*np.pi/180.0 if sim_time > 2.0 else 0.0

    # Sequential step programs
    elif program_number==4:
        des = (min(sim_time,60.0)//5.0) * 5.0*np.pi/180.0
    elif program_number==5:
        des = -(min(sim_time,60.0)//5.0) * 10.0*np.pi/180.0
    elif program_number==6:
        des = (min(sim_time,60.0)//5.0) * 15.0*np.pi/180.0

    # Linear programs
    elif program_number==7:
        des = -sim_time*1.0*np.pi/180.0
    elif program_number==8:
        des = sim_time*2.0*np.pi/180.0
    elif program_number==9:
        des = -sim_time*4.0*np.pi/180.0

    # Sinusoidal programs
    elif program_number==10:
        des = 10.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))
    elif program_number==11:
        des = -15.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))
    elif program_number==12:
        des = 30.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))

    # Unrecognized program
    des = max(min(des, 0.4*np.pi), -0.4*np.pi)
    return des

def _sim_loop(proj, cmg, program, get_torque, time, real_time):
    # Reset the project to its initial state.
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':deque(),
            'omega_theta':deque(),
            'omega_gamma':deque(),
            'theta':deque(),
            'gamma':deque(),
            'tau_gamma':deque(),
            'theta_des':deque(),}

    # Run a simulation loop
    start = now()
    while proj.simtime <= time:
        # Get the state of the system
        state = _state(cmg)

        # Get the desired alpha
        theta_des = _get_des(program, proj.simtime)

        # Get the controller torque
        torque = get_torque(state, theta_des)
        torque = max(min(torque, 0.02), -0.02)

        # Apply the controller
        cmg.joints['S_to_A'].apply_torque(torque,
                                          draw_arrow=True,
                                          arrow_scale=150,
                                          arrow_offset=0.7,)
        cmg.joints['S_to_B'].apply_torque(-torque,
                                          draw_arrow=True,
                                          arrow_scale=150,
                                          arrow_offset=0.7,)

        # Update the data
        if int(round(1000*proj.simtime)) % 10 == 0.0:
            data['time'].append(proj.simtime)
            data['omega_theta'].append(state['omega_theta'])
            data['omega_gamma'].append(state['omega_gamma'])
            data['theta'].append(state['theta'])
            data['gamma'].append(state['gamma'])
            data['tau_gamma'].append(torque)
            data['theta_des'].append(theta_des)

        # Take a simulation step
        proj.step(real_time=real_time)

    # Get the last time step
    state = _state(cmg)
    theta_des = _get_des(program, proj.simtime)
    data['time'].append(proj.simtime)
    data['omega_theta'].append(state['omega_theta'])
    data['omega_gamma'].append(state['omega_gamma'])
    data['theta'].append(state['theta'])
    data['gamma'].append(state['gamma'])
    data['tau_gamma'].append(0.0)
    data['theta_des'].append(theta_des)

    # Return the collected data
    duration = now() - start
    print(f"Simulation took {duration:.2f} seconds.")
    for key, value in data.items():
        data[key] = np.array(value)
    return data

def run(initial_gamma, program, controller, time=30., real_time=True):
    """
    Makes and runs a condynsate-based simulation of a dual axis CMG.
    The goal of the simulation is to apply torques to the gimbals such
    that the chassis remains oriented. These torques are provided by
    controller. At every time step, calls controller to get the torque applied
    to the gimbals based on the state of the system.

    Parameters
    ----------
    psi_program : int
        An integer that selects which of the desired yaw programs is run.
        Each program gives a sequence of desired yaws as a function
        of time that will be passed to the controller.
    theta_program : int
        An integer that selects which of the desired pitch programs is run.
        Each program gives a sequence of desired pitches as a function
        of time that will be passed to the controller.
    controller : function
        The controller function.
    initial_beta : float, optional
        The initial angle of the outer gimbal in radians. The default is 0.
    initial_gamma : float, optional
        The initial angle of the inner gimbal in radians. The default is 0.
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
    proj, cmg = _make(initial_gamma, real_time)
    if real_time:
        _stall(proj)
    data = _sim_loop(proj, cmg, program, controller, time, real_time)
    proj.terminate()
    return data
