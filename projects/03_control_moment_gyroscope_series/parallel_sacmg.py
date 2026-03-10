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
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

def _make(initial_alpha, initial_beta, visualization):
    # Create the project
    proj = Project(keyboard=False, visualizer=visualization, animator=False)
    proj.simulator.set_gravity((0., 0., 0.))

    # Load the cmg
    cmg = proj.load_urdf(assets['parallel_single_axis_cmg.urdf'], fixed=False)

    # Set joint friction to small value and eliminate link air resistance
    cmg.joints['S_to_A'].set_dynamics(damping=0.001)
    cmg.joints['S_to_B'].set_dynamics(damping=0.001)
    cmg.joints['A_to_Fa'].set_dynamics(damping=0.0)
    cmg.joints['B_to_Fb'].set_dynamics(damping=0.0)
    for link in cmg.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    # Set the speed of the cores to ~10 rps
    cmg.joints['A_to_Fa'].set_initial_state(omega=78.5)
    cmg.joints['B_to_Fb'].set_initial_state(omega=78.5)

    # Set the initial angle of the cmg pendulum
    cmg.joints['S_to_A'].set_initial_state(angle=initial_alpha)
    cmg.joints['S_to_B'].set_initial_state(angle=initial_beta)

    # Set some visualizer options
    if visualization:
        proj.visualizer.set_axes(False)
        proj.visualizer.set_grid(True)
        proj.visualizer.set_cam_position((6, -6, 6))
        proj.visualizer.set_cam_target(cmg.center_of_mass)
        proj.visualizer.set_ptlight_1(position=(3, -3, -6), intensity=0.6)
        proj.visualizer.set_ptlight_2(position=(6, 1, 6), intensity=0.5)
        proj.visualizer.set_spotlight(on=True, shadow=True, position=(6,0,0.5),
                                      intensity=0.8, distance=7.6)
        proj.visualizer.set_amblight(intensity=0.3)
        proj.visualizer.set_background((0.0, 0.0, 0.0),
                                       (0.294, 0.380, 0.650),)
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
    b = cmg.joints['S_to_B'].state
    state = {'omega_theta':s.omega_in_body[1],
             'omega_phi':s.omega_in_body[0],
             'omega_alpha':a.omega,
             'omega_beta':b.omega,
             'theta':s.ypr[1],
             'phi':s.ypr[2],
             'alpha':a.angle,
             'beta':b.angle,}
    return state

def _get_des(program_number, sim_time):
    des = 0.0

    # Step programs
    if program_number==1:
        des = 5.0*np.pi/180.0 if sim_time > 2.0 else 0.0
    elif program_number==2:
        des = -15.0*np.pi/180.0 if sim_time > 2.0 else 0.0
    elif program_number==3:
        des = 30.0*np.pi/180.0 if sim_time > 2.0 else 0.0

    # Sequential step programs
    elif program_number==4:
        des = np.clip((sim_time//2.0) * 1.0*np.pi/180.0)
    elif program_number==5:
        des = -(sim_time//2.0) * 5.0*np.pi/180.0
    elif program_number==6:
        des = (sim_time//2.0) * 10.0*np.pi/180.0

    # Linear programs
    elif program_number==7:
        des = -sim_time*np.pi/180.0
    elif program_number==8:
        des = sim_time*4.5*np.pi/180.0
    elif program_number==9:
        des = -sim_time*9.0*np.pi/180.0

    # Sinusoidal programs
    elif program_number==10:
        des = 15.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))
    elif program_number==11:
        des = -30.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))
    elif program_number==12:
        des = 30.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*5.0))

    # Unrecognized program
    des = np.clip(des, -0.99*np.pi, 0.99*np.pi)
    return des

def _sim_loop(proj, cmg, program, get_torque, time, real_time):
    # Reset the project to its initial state.
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':np.array([]),
            'omega_theta':np.array([]),
            'omega_phi':np.array([]),
            'omega_alpha':np.array([]),
            'omega_beta':np.array([]),
            'theta':np.array([]),
            'phi':np.array([]),
            'alpha':np.array([]),
            'beta':np.array([]),
            'tau_alpha':np.array([]),
            'tau_beta':np.array([]),
            'theta_des':np.array([]),
            'phi_des':np.array([]),}

    # Run a simulation loop
    while proj.simtime <= time:
        # Set the yaw to 0 (fixed condition)
        cmg.set_state(yaw=0.0)

        # Get the state of the system
        state = _state(cmg)

        # Get the desired alpha
        theta_des = _get_des(program[0], proj.simtime)
        phi_des = _get_des(program[1], proj.simtime)

        # Get the controller torque
        torque = get_torque(state, theta_des, phi_des)
        torque = np.clip(torque, -0.005, 0.005)

        # Apply the controller
        cmg.joints['S_to_A'].apply_torque(torque[0],
                                          draw_arrow=True,
                                          arrow_scale=600,
                                          arrow_offset=0.7,)
        cmg.joints['S_to_B'].apply_torque(torque[1],
                                          draw_arrow=True,
                                          arrow_scale=600,
                                          arrow_offset=0.7,)

        # Update the data
        data['time'] = np.append(data['time'], proj.simtime)
        data['omega_theta']=np.append(data['omega_theta'],state['omega_theta'])
        data['omega_phi'] = np.append(data['omega_phi'], state['omega_phi'])
        data['omega_alpha']=np.append(data['omega_alpha'],state['omega_alpha'])
        data['omega_beta'] = np.append(data['omega_beta'], state['omega_beta'])
        data['theta'] = np.append(data['theta'], state['theta'])
        data['phi'] = np.append(data['phi'], state['phi'])
        data['alpha'] = np.append(data['alpha'], state['alpha'])
        data['beta'] = np.append(data['beta'], state['beta'])
        data['tau_alpha'] = np.append(data['tau_alpha'], torque[0])
        data['tau_beta'] = np.append(data['tau_beta'], torque[1])
        data['theta_des'] = np.append(data['theta_des'], theta_des)
        data['phi_des'] = np.append(data['phi_des'], phi_des)

        # Take a simulation step
        proj.step(real_time=real_time, stable_step=False)

    # Return the collected data
    return data

def run(program, controller,
        initial_alpha=-np.pi/4, initial_beta=np.pi/4,
        time=20.0, real_time=True):
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
    proj, cmg = _make(initial_alpha, initial_beta, real_time)
    if real_time:
        _stall(proj)
    data = _sim_loop(proj, cmg, program, controller, time, real_time)
    proj.terminate()
    return data

def controller(state, psi_des, theta_des):
    return -0.001, 0.001

if __name__ == "__main__":
    run((0, 0), controller, time=1.0)
    # data = run(1, 0, controller, time=10.0)

    # # Import plotting tool
    # import matplotlib.pyplot as plt
    # %matplotlib inline

    # # Plot the yaw and desired yaw as functions of time
    # plt.plot(data['time'], data['psi']*180/3.14, label='Yaw [deg]', lw=2.0)
    # plt.plot(data['time'], data['psi_des']*180/3.14, label='Desired Yaw [deg]', lw=2.0)
    # plt.legend()
    # plt.xlabel('Time [seconds]')
    # plt.axhline(c='k', lw=0.5)
    # plt.show()

    # # Plot the pitch and desired pitch as functions of time
    # plt.plot(data['time'], data['theta']*180/3.14, label='Pitch [deg]', lw=2.0)
    # plt.plot(data['time'], data['theta_des']*180/3.14, label='Desired Pitch [deg]', lw=2.0)
    # plt.legend()
    # plt.xlabel('Time [seconds]')
    # plt.axhline(c='k', lw=0.5)
    # plt.show()
