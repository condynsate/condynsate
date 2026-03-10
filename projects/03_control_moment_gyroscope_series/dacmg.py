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

def _make(initial_beta, initial_gamma, visualization):
    # Create the project
    proj = Project(keyboard=False, visualizer=visualization, animator=False)
    proj.simulator.set_gravity((0., 0., 0.))

    # Load the cmg
    cmg = proj.load_urdf(assets['double_axis_cmg.urdf'], fixed=False)

    # Set joint friction to small value and eliminate link air resistance
    # Add joint limits to the A so that it collides with the base
    cmg.joints['A_to_B'].set_dynamics(damping=0.00001)
    cmg.joints['B_to_C'].set_dynamics(damping=0.001)
    cmg.joints['C_to_flywheel'].set_dynamics(damping=0.0)
    for link in cmg.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    # Set the speed of the core to ~25 rps
    cmg.joints['C_to_flywheel'].set_initial_state(omega=157)

    # Set the initial angle of the cmg pendulum
    cmg.joints['A_to_B'].set_initial_state(angle=initial_beta)
    cmg.joints['B_to_C'].set_initial_state(angle=initial_gamma)

    # Set some visualizer options
    if visualization:
        proj.visualizer.set_axes(True)
        proj.visualizer.set_grid(True)
        proj.visualizer.set_cam_position((1.5, -2, 5))
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
    a = cmg.state
    b = cmg.joints['A_to_B'].state
    c = cmg.joints['B_to_C'].state
    state = {'omega_psi':a.omega_in_body[2],
             'omega_theta':a.omega_in_body[0],
             'omega_beta':b.omega,
             'omega_gamma':c.omega,
             'psi':a.ypr[0],
             'theta':a.ypr[2],
             'beta':b.angle,
             'gamma':c.angle,}
    return state

def _get_des(program_number, sim_time):
    # Step programs
    if program_number==1:
        return 10.0*np.pi/180.0 if sim_time > 2.0 else 0.0
    if program_number==2:
        return -45.0*np.pi/180.0 if sim_time > 2.0 else 0.0
    if program_number==3:
        return 90.0*np.pi/180.0 if sim_time > 2.0 else 0.0

    # Sequential step programs
    if program_number==4:
        return (sim_time//2.0) * 5.0*np.pi/180.0
    if program_number==5:
        return -(sim_time//2.0) * 10.0*np.pi/180.0
    if program_number==6:
        return (sim_time//2.0) * 20.0*np.pi/180.0

    # Linear programs
    if program_number==7:
        return -sim_time*4.5*np.pi/180.0
    if program_number==8:
        return sim_time*9.0*np.pi/180.0
    if program_number==9:
        return -sim_time*13.5*np.pi/180.0

    # Sinusoidal programs
    if program_number==10:
        return 45.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))
    if program_number==11:
        return -90.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))
    if program_number==12:
        return 90.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*5.0))

    # Unrecognized program
    return 0

def _sim_loop(proj, cmg, psi_prog, theta_prog, get_torque, time, real_time):
    # Reset the project to its initial state.
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':np.array([]),
            'omega_psi':np.array([]),
            'omega_theta':np.array([]),
            'omega_beta':np.array([]),
            'omega_gamma':np.array([]),
            'psi':np.array([]),
            'theta':np.array([]),
            'beta':np.array([]),
            'gamma':np.array([]),
            'tau_beta':np.array([]),
            'tau_gamma':np.array([]),
            'psi_des':np.array([]),
            'theta_des':np.array([]),}

    # Run a simulation loop
    while proj.simtime <= time:
        # Get the state of the system
        state = _state(cmg)

        # Get the desired alpha
        psi_des = _get_des(psi_prog, proj.simtime)
        theta_des = _get_des(theta_prog, proj.simtime)

        # Get the controller torque
        torque = get_torque(state, psi_des, theta_des)
        torque = np.clip(torque, -0.005, 0.005)

        # Apply the controller
        cmg.joints['A_to_B'].apply_torque(torque[0],
                                          draw_arrow=True,
                                          arrow_scale=300,
                                          arrow_offset=0.8,)
        cmg.joints['B_to_C'].apply_torque(torque[1],
                                          draw_arrow=True,
                                          arrow_scale=300,
                                          arrow_offset=0.8,)

        # Update the data
        data['time'] = np.append(data['time'], proj.simtime)
        data['omega_psi'] = np.append(data['omega_psi'], state['omega_psi'])
        data['omega_theta']=np.append(data['omega_theta'],state['omega_theta'])
        data['omega_beta'] = np.append(data['omega_beta'], state['omega_beta'])
        data['omega_gamma']=np.append(data['omega_gamma'],state['omega_gamma'])
        data['psi'] = np.append(data['psi'], state['psi'])
        data['theta'] = np.append(data['theta'], state['theta'])
        data['beta'] = np.append(data['beta'], state['beta'])
        data['gamma'] = np.append(data['gamma'], state['gamma'])
        data['tau_beta'] = np.append(data['tau_beta'], torque[0])
        data['tau_gamma'] = np.append(data['tau_gamma'], torque[1])
        data['psi_des'] = np.append(data['psi_des'], psi_des)
        data['theta_des'] = np.append(data['theta_des'], theta_des)

        # Take a simulation step
        proj.step(real_time=real_time, stable_step=False)

    # Return the collected data
    return data

def run(psi_program, theta_program, controller,
        initial_beta=0.0, initial_gamma=0.0, time=20.0, real_time=True):
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
    proj, cmg = _make(initial_beta, initial_gamma, real_time)
    if real_time:
        _stall(proj)
    data = _sim_loop(proj, cmg, psi_program, theta_program,
                     controller, time, real_time)
    proj.terminate()
    return data

def controller(state, psi_des, theta_des):
    # Define the controller
    K = np.array([[ 2.8e-11, -3.2e-06, 0.0025, -1.1e-07, -2.7e-18, -7.7e-10, 0.0016, 8.7e-11],
                  [ -0.0039, -0.00016, -1.6e-17, 0.0071, -0.029, -9.5e-08, -7.9e-18, -2e-10]])

    m_e = np.array([0, 0, 0, 0, 0, 0, 0, 0]) # The equilibrium nonlinear state vector
    n_e = np.array([0, 0])                   # The equilibrium nonlinear input vector

    # Define a desired alpha angle
    x_des = np.array([0, 0, 0, 0, psi_des, theta_des, 0, 0]) - m_e

    # Build the nonlinear state vector
    m = np.array([state['omega_psi'],
                  state['omega_theta'],
                  state['omega_beta'],
                  state['omega_gamma'],
                  state['psi'],
                  state['theta'],
                  state['beta'],
                  state['gamma']])

    # Build the linear state vector
    x = m - m_e

    # Build the reference tracking linear state vector
    z = x - x_des

    # Apply the feedback control law with our selected gain matrix to get the linear input vector
    u = -K@z

    # Convert the linear input vector into the nonlinear input vector
    n = u + n_e

    # Return the nonlinear torques as scalars
    tau_beta = n[0]
    tau_gamma = n[1]
    return tau_beta, tau_gamma

if __name__ == "__main__":
    data = run(1, 0, controller, time=10.0)

    # Import plotting tool
    import matplotlib.pyplot as plt
    %matplotlib inline

    # Plot the yaw and desired yaw as functions of time
    plt.plot(data['time'], data['psi']*180/3.14, label='Yaw [deg]', lw=2.0)
    plt.plot(data['time'], data['psi_des']*180/3.14, label='Desired Yaw [deg]', lw=2.0)
    plt.legend()
    plt.xlabel('Time [seconds]')
    plt.axhline(c='k', lw=0.5)
    plt.show()

    # Plot the pitch and desired pitch as functions of time
    plt.plot(data['time'], data['theta']*180/3.14, label='Pitch [deg]', lw=2.0)
    plt.plot(data['time'], data['theta_des']*180/3.14, label='Desired Pitch [deg]', lw=2.0)
    plt.legend()
    plt.xlabel('Time [seconds]')
    plt.axhline(c='k', lw=0.5)
    plt.show()
