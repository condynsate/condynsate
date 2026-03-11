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

def _make(initial_alpha, visualization):
    # Create the project
    proj = Project(keyboard=False, visualizer=visualization, animator=False)
    proj.simulator.set_gravity((0., 0., 0.))

    # Load the cmg
    cmg = proj.load_urdf(assets['parallel_single_axis_cmg.urdf'], fixed=False)

    # Set joint friction to small value and eliminate link air resistance
    cmg.joints['S_to_A'].set_dynamics(damping=0.017) # Req for sim stability
    cmg.joints['S_to_B'].set_dynamics(damping=0.017) # Req for sim stability
    cmg.joints['A_to_Fa'].set_dynamics(damping=0.0)
    cmg.joints['B_to_Fb'].set_dynamics(damping=0.0)
    for link in cmg.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    # Set the speed of the cores to ~10 rps
    cmg.joints['A_to_Fa'].set_initial_state(omega=62.8)
    cmg.joints['B_to_Fb'].set_initial_state(omega=62.8)

    # Set the initial angle of the cmg pendulum
    cmg.joints['S_to_A'].set_initial_state(angle=initial_alpha)
    cmg.joints['S_to_B'].set_initial_state(angle=-initial_alpha)

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
    state = {'omega_theta':-s.omega_in_body[1],
             'omega_gamma':a.omega,
             'theta':-s.ypr[1],
             'gamma':a.angle,}
    return state

def _get_des(program_number, sim_time):
    des = 0.0

    # Step programs
    if program_number==1:
        des = 5.0*np.pi/180.0 if sim_time > 2.0 else 0.0
    elif program_number==2:
        des = -10.0*np.pi/180.0 if sim_time > 2.0 else 0.0
    elif program_number==3:
        des = 15.0*np.pi/180.0 if sim_time > 2.0 else 0.0

    # Sequential step programs
    elif program_number==4:
        des = np.clip((sim_time//5.0) * 1.0*np.pi/180.0)
    elif program_number==5:
        des = -(sim_time//5.0) * 2.5*np.pi/180.0
    elif program_number==6:
        des = (sim_time//2.5) * 3.5*np.pi/180.0

    # Linear programs
    elif program_number==7:
        des = -sim_time*0.25*np.pi/180.0
    elif program_number==8:
        des = sim_time*0.75*np.pi/180.0
    elif program_number==9:
        des = -sim_time*2*np.pi/180.0

    # Sinusoidal programs
    elif program_number==10:
        des = 4.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))
    elif program_number==11:
        des = -8.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))
    elif program_number==12:
        des = 16.0*np.pi/180.0*np.sin(np.pi*sim_time/(2.0*10.0))

    # Unrecognized program
    des = np.clip(des, -0.99*np.pi, 0.99*np.pi)
    return des

def _sim_loop(proj, cmg, program, get_torque, time, real_time):
    # Reset the project to its initial state.
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':np.array([]),
            'omega_theta':np.array([]),
            'omega_gamma':np.array([]),
            'theta':np.array([]),
            'gamma':np.array([]),
            'tau_gamma':np.array([]),
            'theta_des':np.array([]),}

    # Run a simulation loop
    while proj.simtime <= time:
        # Get the state of the system
        state = _state(cmg)

        # Get the desired alpha
        theta_des = _get_des(program, proj.simtime)

        # Get the controller torque
        torque = get_torque(state, theta_des)
        torque = np.clip(torque, -0.004, 0.004)

        # Apply the controller
        cmg.joints['S_to_A'].apply_torque(torque,
                                          draw_arrow=True,
                                          arrow_scale=750,
                                          arrow_offset=0.7,)
        cmg.joints['S_to_B'].apply_torque(-torque,
                                          draw_arrow=True,
                                          arrow_scale=750,
                                          arrow_offset=0.7,)

        # Update the data
        data['time'] = np.append(data['time'], proj.simtime)
        data['omega_theta']=np.append(data['omega_theta'],state['omega_theta'])
        data['omega_gamma']=np.append(data['omega_gamma'],state['omega_gamma'])
        data['theta'] = np.append(data['theta'], state['theta'])
        data['gamma'] = np.append(data['gamma'], state['gamma'])
        data['tau_gamma'] = np.append(data['tau_gamma'], torque)
        data['theta_des'] = np.append(data['theta_des'], theta_des)

        # Take a simulation step
        proj.step(real_time=real_time, stable_step=False)

    # Return the collected data
    return data

def run(program, controller, initial_alpha=np.pi/4, time=45, real_time=True):
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
    proj, cmg = _make(initial_alpha, real_time)
    if real_time:
        _stall(proj)
    data = _sim_loop(proj, cmg, program, controller, time, real_time)
    proj.terminate()
    return data

# def controller(state, theta_des):
#     K = np.array([[-6.25727963e-05,  5.03773953e-03,  1.91818797e-02]])
#     T = np.array([[ 0.9957061 ,  0.        ,  0.        , -0.09257087],
#                   [ 0.        ,  0.0304825 , -0.9995353 ,  0.        ],
#                   [ 0.        ,  0.9995353 ,  0.0304825 ,  0.        ],
#                   [-0.09257087,  0.        ,  0.        , -0.9957061 ]])

#     m_e = np.array([0, 0, 0, 0.5*np.pi]) # The equilibrium nonlinear state vector
#     n_e = np.array([0]) # The equilibrium nonlinear input vector

#     # Define a desired alpha angle
#     x_des = np.array([0, theta_des, 0, 0]) - m_e

#     # Build the nonlinear state vector
#     m = np.array([state['omega_theta'],
#                   state['theta'],
#                   state['omega_gamma'],
#                   state['gamma'],])

#     # Build the linear state vector
#     x = m - m_e

#     # Build the reference tracking linear state vector
#     z = x - x_des

#     # Transform into controllable coordinates
#     z_tilda = T.T @ z
#     z_c = z_tilda[0:3]

#     # Apply the feedback control law with our selected gain matrix to get the linear input vector
#     u = -K@z_c

#     # Convert the linear input vector into the nonlinear input vector
#     n = u + n_e

#     # Return the nonlinear torques as scalars
#     tau_gamma = n[0]
#     return tau_gamma

# if __name__ == "__main__":
#     # Run a simulation
#     data = run(1, controller, time=45.0, real_time=False)

#     # Import plotting tool
#     import matplotlib.pyplot as plt
#     %matplotlib inline

#     # Plot the pitch, desired pitch, and input torque as functions of time
#     plt.plot(data['time'], data['theta']*180/3.14, label='Pitch [deg]', lw=2.0, c='r')
#     plt.plot(data['time'], data['theta_des']*180/3.14, label='Desired Pitch [deg]', lw=2.0, c='r', ls='--')
#     plt.plot(data['time'], 1000*data['tau_gamma'], label='Torque [mN-m]', lw=2.0, c='b')
#     plt.legend()
#     plt.xlabel('Time [seconds]')
#     plt.axhline(c='k', lw=0.5)
#     plt.show()
