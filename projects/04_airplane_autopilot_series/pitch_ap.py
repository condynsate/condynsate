# -*- coding: utf-8 -*-
"""
This module implements the backend for the airplane project. In this
project, we use develop an altitude autopilot.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from time import sleep
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

def _load_vis_env(proj, h0):
    # Load the ground and sun
    proj.visualizer.add_object('ground',
                               assets['sphere_1_center_origin.stl'],
                               scale=(20000, 20000, 20000),
                               tex_path=assets['farmland.png'],
                               tex_wrap=[1000, 1000],
                               tex_repeat=[13, 13],
                               emissive_color=(0.175, 0.175, 0.175),
                               position=(0.0, 0.0, -10000 - h0),)
    proj.visualizer.add_object('sun',
                               assets['sphere_1_center_origin.stl'],
                               scale=(150, 150, 150),
                               color=(1.0, 0.706, 0.424),
                               emissive_color=(1.0, 0.706, 0.424),
                               position=(707.1, -707.1, 100))

    # Make the grid and axes invisible
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)

    # Set the scene lighting
    proj.visualizer.set_ptlight_1(on=True, position=(0, 0, -12),
                                  intensity=0.75, shadow=True, distance=50)
    proj.visualizer.set_spotlight(on=True, position=(12.6, -12.6, 1.8),
                                  angle=0.35, intensity=1.0, distance=50,
                                  shadow=True)
    proj.visualizer.set_amblight(on=True, intensity=0.5)
    proj.visualizer.set_ptlight_2(on=False)
    proj.visualizer.set_dirnlight(on=False)
    proj.visualizer.set_dirnlight(on=False)

    # Look at the plane
    proj.visualizer.set_cam_position((0, 1, 0.125))
    proj.visualizer.set_cam_target((0, 0, 0))

    # Refresh the visualizer to reflect the changes we made
    proj.refresh_visualizer()

def _set_init_conds(plane, theta0, omega_theta0):
    # Rotate the prop at 2500rpm
    plane.joints['fuselage_to_nosecone'].set_initial_state(omega=261.8)

    # Set all control surfaces to 0
    plane.joints['fuselage_to_flaps'].set_initial_state(angle=0)
    plane.joints['fuselage_to_r_aileron'].set_initial_state(angle=0)
    plane.joints['fuselage_to_l_aileron'].set_initial_state(angle=0)
    plane.joints['fuselage_to_elevator'].set_initial_state(angle=0)
    plane.joints['fuselage_to_rudder'].set_initial_state(angle=0)

    # Apply initial state
    plane.set_initial_state(pitch = -theta0,
                            omega = (0.0, -omega_theta0, 0.0))

def _make(**kwargs):
    # Extract useful kwargs
    visualization = kwargs.get('real_time', True)
    h0 = kwargs.get('h', 100.0)
    omega_theta0 = kwargs.get('omega_theta', 0.0)
    theta0 = kwargs.get('theta', 0.060307)
    seed = kwargs.get('seed', 2357136050)

    # Initializer random
    rng = np.random.default_rng(seed=seed)

    # Create the project
    proj = Project(keyboard=False, visualizer=visualization, animator=False)
    proj.simulator.set_gravity((0.0, 0.0, 0.0))

    # Load the visual environment
    if visualization:
        _load_vis_env(proj, h0)

    # Load the plane, set its initial condition, and look at it
    plane = proj.load_urdf(assets['cessna172.urdf'], fixed=False)
    _set_init_conds(plane, theta0, omega_theta0)

    # Remove all friction
    for joint in plane.joints.values():
        joint.set_dynamics(damping=0.0)
    for link in plane.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    return proj, plane, rng

def _stall(proj):
    try:
        # Await the user pressing enter (in available)
        proj.refresh_visualizer()
        proj.keyboard.await_press('enter')
        proj.refresh_visualizer()

    # If no keyboard exists, ignore this call
    except AttributeError:
        proj.refresh_visualizer()
        sleep(5.0)
        proj.refresh_visualizer()

def _rho(h):
    return 1.225*np.exp(-h/10363.)

def _T(h):
    if h < 10000:
        return 288.2 - 0.00649*h
    return 223.3

def _mu(h):
    return 1.458e-6*_T(h)**(1.5) / (_T(h) + 110.4)

def _g(h):
    return (5.9722e24*6.67430e-11)/(6371e3+h)**2

def _cL(alpha, a_s, a_0, a_l0):
    # No stall
    if abs(alpha) <= a_s:
        return a_0*(alpha-a_l0)

    # Positive stalling region
    elif alpha > 0 and alpha < a_s+0.0873:
        a = -32.83*(a_0*(a_s+0.3491)-a_l0*a_0)
        b = 65.66*(a_0*(a_s**2+0.3491*a_s+0.0152)-a_l0*a_0*a_s)
        c = -32.83*(a_0*a_s**2*(a_s+0.3491)-a_l0*a_0*(a_s**2-0.0305))
        return a*alpha**2 + b*alpha + c

    # Negative stalling region
    elif alpha < 0 and alpha > -a_s-0.0873:
        a = 32.83*(a_0*(a_s+0.3491)+a_l0*a_0)
        b = 65.66*(a_0*(a_s**2+0.3491*a_s+0.0152)+a_l0*a_0*a_s)
        c = 32.83*(a_0*a_s**2*(a_s+0.3491)+a_l0*a_0*(a_s**2-0.0305))
        return a*alpha**2 + b*alpha + c

    # Complete stall
    return 0.0

def _wing_forces(state):
    # Wing param
    a_0 = 5.011
    s = 16.2
    c = 1.49
    ar = 7.52

    # Extract state info
    h = state['h']
    u_inf = state['u_inf']
    alpha = state['alpha']

    # Get the reynold's number
    re = _rho(h)*c*u_inf/_mu(h)

    # Calculate stall angle and 0 lift angle
    a_s = 0.0407*np.log(re) - 0.266
    a_l0 = -0.0436 * (np.arctan(re/11880. - 10.52) / np.pi + 0.5)

    # Get the lift and drag
    cL = _cL(alpha, a_s, a_0, a_l0)
    cDi = (a_0*(alpha-a_l0))**2 / (np.pi*ar)
    cDf = 0.074*re**(-0.2)
    L = 0.5 * _rho(h) * s * cL * u_inf**2
    D = 0.5 * _rho(h) * s * (cDi + cDf) * u_inf**2
    return L, D

def _tail_forces(state, delta):
    # Horizontal stab param
    a_0_t = 4.817
    s_t = 2.00
    ar_t = 6.32

    # Ele param
    a_0_e = 5.230
    s_e = 1.35
    ar_e = 9.37

    # Combined parameters
    a_s_te = 0.279
    a_l0_te = 0.0393
    c_te = 0.942

    # Extract state info
    h = state['h']
    u_inf = state['u_inf']
    alpha = state['alpha']

    # Get the reynold's number
    re = _rho(h)*c_te*u_inf/_mu(h)

    # Calculate stall angle and 0 lift angle
    a_s_te = 0.0407*np.log(re) - 0.266
    a_l0_te = 0.0436 * (np.arctan(re/11880. - 10.52) / np.pi + 0.5)

    # Get forces in world coords
    cL_t = _cL(alpha, a_s_te, a_0_t, a_l0_te)
    cDi_t = (a_0_t*(alpha-a_l0_te))**2 / (np.pi*ar_t)
    cL_e = _cL(alpha-delta, a_s_te, a_0_e, a_l0_te)
    cDi_e = (a_0_e*(alpha-delta-a_l0_te))**2 / (np.pi*ar_e)
    cDf_te = 0.074*re**(-0.2)
    L = 0.5 * _rho(h) * (s_t*cL_t + s_e*cL_e) * u_inf**2
    D = 0.5 * _rho(h) * (s_t*cDi_t + s_e*cDi_e + (s_t+s_e)*cDf_te) * u_inf**2
    return L, D

def  _body_forces(state):
    # Body parameters
    s = 5.59
    cDf = 0.095

    # Extract state info
    h = state['h']
    u_inf = state['u_inf']

    # Get the lift and drag
    L = 0.0
    D = 0.5 * _rho(h) * s * cDf * u_inf**2
    return L, D

def _prop_rps(P):
    P_hp = P / 745.7
    return 45 / (1 + np.exp(-0.038827 * (P_hp - 69.95)))

def _eta(u_inf, P):
    J = u_inf / (_prop_rps(P) * 1.905)
    if J>=0 and J<0.87:
        return -1.097*J*J + 1.908*J
    elif J>=0.87 and J<=1.05:
        return -25.62*J*J + 44.57*J - 18.65
    else:
        return 0.0

def _prop_forces(state, P):
    u_inf = state['u_inf']
    return P*_eta(u_inf, P) / u_inf

def _LD_to_plane(L, D, alpha, theta):
    Lx = -L*np.sin(theta-alpha)
    Lz = -L*np.cos(theta-alpha)
    Dx = -D*np.cos(theta-alpha)
    Dz = D*np.sin(theta-alpha)
    return np.array((Lx, 0.0, Lz)) + np.array((Dx, 0.0, Dz))

def _net_aero_force_torque(state, delta, P):
    # Parameters
    rw = 0.111
    rt = 4.560

    # Extract state info
    alpha = state['alpha']
    theta = state['theta']

    # Get the forces
    Lw, Dw = _wing_forces(state)
    Lt, Dt = _tail_forces(state, delta)
    Lb, Db = _body_forces(state)
    T = _prop_forces(state, P)

    # Get the net torque
    tau_y = -np.cos(alpha)*(Lw*rw + Lt*rt) - np.sin(alpha)*(Dw*rw + Dt*rt)

    # Convert to plane coords
    Fw = _LD_to_plane(Lw, Dw, alpha, theta)
    Ft = _LD_to_plane(Lt, Dt, alpha, theta)
    Fb = _LD_to_plane(Lb, Db, alpha, theta)
    Fp = np.array((np.cos(theta)*T, 0.0, -np.sin(theta)*T))
    F_net = Fw + Ft + Fb + Fp
    tau_net = np.array([0.0, tau_y, 0.0])
    return F_net, tau_net

def _plane_to_world(vec):
    return np.array((vec[0], -vec[1], -vec[2]))

def _state(plane, p_world, u_world, pitch):
    state = {'h' : p_world[2],
             'u_inf' : np.linalg.norm(u_world),
             'alpha' : pitch - np.atan2(u_world[2], u_world[0]),
             'omega_theta' : -plane.state.omega_in_body[1],
             'theta' : pitch,}
    return state

def _mass(plane):
    mass = 0.0
    for link in plane.links.values():
        mass += link.mass
    return mass

def _turbulence(turb_mag, time, p, s, o):
    tl=[b*(0.25*(np.sin(2/a*(time-c))+np.sin(np.pi/a*(time-c)))+0.5)
        for a,b,c in zip(p,s,o)]
    ts=[.25*b*(.25*(np.sin(2/(.1*a)*(time-c))+np.sin(np.pi/(.1*a)*(time-c)))+.5)
        for a,b,c in zip(p,s,o)]
    t = (np.array(tl) + np.array(ts)) * turb_mag
    return t

def _h_des(program_number, sim_time, h0):
    h_des = h0

    # Step programs
    if program_number==1:
        h_des = h0 - 25.0 if sim_time > 2.0 else h0
    elif program_number==2:
        h_des = h0 + 75.0 if sim_time > 2.0 else h0
    elif program_number==3:
        h_des = h0 + 300. if sim_time > 2.0 else h0

    # Sequential step programs
    elif program_number==4:
        h_des = (min(sim_time,60.0)//5.0) *-30.48/12.0 + h0
    elif program_number==5:
        h_des = (min(sim_time,60.0)//5.0) * 152.4/12.0 + h0
    elif program_number==6:
        h_des = (min(sim_time,60.0)//5.0) * 304.8/12.0 + h0

    # Linear programs
    elif program_number==7:
        h_des = min(sim_time,60.0)*-0.508 + h0
    elif program_number==8:
        h_des = min(sim_time,60.0)*2.5400 + h0
    elif program_number==9:
        h_des = min(sim_time,60.0)*5.0800 + h0

    # Sinusoidal programs
    elif program_number==10:
        h_des = -7.62*np.sin(np.pi*sim_time/30.0) + h0
    elif program_number==11:
        h_des = 38.10*np.sin(np.pi*sim_time/30.0) + h0
    elif program_number==12:
        h_des = 76.20*np.sin(np.pi*sim_time/30.0) + h0

    return h_des

def _sim_loop(proj, plane, controller, program_nmuber, rng, **kwargs):
    # Extract kwargs
    time = kwargs.get('time', 20.0)
    real_time = kwargs.get('real_time', True)
    turb_mag = kwargs.get('turbulence', 0.0)
    shake = kwargs.get('shake', 0.035)
    h0 = kwargs.get('h', 100.0)
    u_inf0 = kwargs.get('u_inf', 43.4816)
    alpha0 = kwargs.get('alpha', 0.060307)
    omega_theta0 = kwargs.get('omega_theta', 0.0)
    theta0 = kwargs.get('theta', 0.060307)

    # Apply initial state
    p_world = np.array([0.0, 0.0, h0])
    u_world = np.array([u_inf0*np.cos(theta0-alpha0),
                        0.0,
                        u_inf0*np.sin(theta0-alpha0)])
    pitch = theta0
    plane.set_initial_state(pitch = -theta0,
                            omega = (0.0, -omega_theta0, 0.0))

    # Reset the project to its initial state.
    proj.reset()

    # Make structure to hold simulation data
    data = {'time':np.array([]),
            'h':np.array([]),
            'u_inf':np.array([]),
            'alpha':np.array([]),
            'omega_theta':np.array([]),
            'theta':np.array([]),
            'delta':np.array([]),
            'P':np.array([]),
            'h_des':np.array([]),}

    # Run a simulation loop
    mass = _mass(plane)
    dt = proj.simulator.dt
    earth_rot = 0.0
    p = 10*rng.random(3)
    s = 2.0*rng.random(3)-1.0
    o = 120*rng.random(3)
    while proj.simtime <= time:

        # Get the state of the system
        turbulence = _turbulence(turb_mag, proj.simtime, p, s, o)
        state = _state(plane, p_world, u_world + turbulence, pitch)

        # Get the desired altitude
        h_des = _h_des(program_nmuber, proj.simtime, h0)

        # Get the controller inputs
        delta, P = controller(state, h_des)
        delta = min(max(delta, -0.332), 0.384)
        P = min(max(P, 0.0), 1.342e5)

        # Update the data
        data['time'] = np.append(data['time'], proj.simtime)
        data['h']=np.append(data['h'], state['h'])
        data['u_inf'] = np.append(data['u_inf'], state['u_inf'])
        data['alpha'] = np.append(data['alpha'], state['alpha'])
        data['omega_theta']=np.append(data['omega_theta'],state['omega_theta'])
        data['theta'] = np.append(data['theta'], state['theta'])
        data['delta'] = np.append(data['delta'], delta)
        data['P'] = np.append(data['P'], P)
        data['h_des'] = np.append(data['h_des'], h_des)

        # Crash condition
        if state['h'] <= 3.0:
            break

        # Calculate the net lift, drag, and prop force and torque
        F_net, tau_net = _net_aero_force_torque(state, delta, P)
        F_net_world = _plane_to_world(F_net)
        F_g_world = np.array([0.0, 0.0, - mass*_g(state['h'])])
        F_net_world += F_g_world

        # Apply the torque
        plane.apply_torque(_plane_to_world(tau_net))

        # Update the visuals
        if real_time:
            plane.joints['fuselage_to_elevator'].set_state(angle=delta)

            # Position the camera
            cz = shake*(np.mean(data['h'][-100:])-p_world[2])
            proj.visualizer.set_cam_position((0, 1, cz+0.125))
            proj.visualizer.set_cam_target((0, 0, cz))

            # Rotate the earth according to the forward velocity
            proj.visualizer.set_transform('ground', pitch=earth_rot,
                                          scale=(20000, 20000, 20000),
                                          position=(0.0,0.0,-10000-p_world[2]))

        # Take a simulation step
        u_world += (F_net_world / mass) * dt
        p_world += u_world * dt
        pitch += dt * state['omega_theta']
        pitch = ((180*pitch/np.pi+180)%360-180)*np.pi/180
        earth_rot += -(u_world[0] / (p_world[2] + 10000.0)) * dt
        proj.step(real_time=real_time, stable_step=False)

    # Return the collected data
    return data

def run(controller, program_number, **kwargs):
    """
    Makes and runs a condynsate-based simulation of a Cessna 172. The goal
    of this project is to provide a controller that, when called, generates
    elevator deflection and engine power commands that trim the aircraft
    around a desired altitude.

    Parameters
    ----------
    controller : function
        The controller function.
    psi_program : int
        An integer that selects which of the desired altitude programs is run.
        Each program gives a sequence of desired altitudes as a function
        of time that will be passed to the controller. Valid numbers are 0-12.
    **kwargs

    Keyword Args
    ------------
        time : float, optional
            The amount of time to run the simulation in seconds. The default is
            20.
        real_time : boolean, optional
            A boolean flag that indicates if the simulation is run in real time
            with visualization (True) or as fast as possible with no
            visualization (False). Regardless of choice, simulation data is
            still gathered. The default is True
        h : float, optional
            The initial altitude in meters. The default value is 100
        u_inf : float, optional
            The initial airspeed in meters/second. The default value is 43.4816
        alpha : float, optional
            The initial angle of attack in radians. The default value
            is 0.060307
        omega_theta : float, optional
            The initial pitching rate in radians/second. The default value is 0
        theta : float, optional
            The initial pitch angle in radians. The default value is 0.060307
        seed : int, optional
            The seed of the random number generator used for the simulation.
            The default is 2357136050
        turbulence : float, optional
            The magnitude of the turbulent wind in m/s. The default is 0
        shake : float, optional
            The magnitude by which plane accelerations are visualized. The
            default is 0.035

    Returns
    -------
    data : dictionary of array-likes with length n
        The data collected during the simulation.

    """
    # Build the project, run the simulation loop, terminate the project
    proj, plane, rng = _make(**kwargs)
    if kwargs.get('real_time', True):
        _stall(proj)
    data = _sim_loop(proj, plane, controller, program_number, rng, **kwargs)
    proj.terminate()
    return data
