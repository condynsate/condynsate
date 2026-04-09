# -*- coding: utf-8 -*-
"""
This module implements the backend for the airplane project.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""
from time import sleep
from time import time as now
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np
from plane_parameters import Cessna172
from flight_sim import FlightSim

R_PLANET = 637100
DT = 1.0 / 250.0
PARAM = Cessna172()

class _SimData():
    _data : dict

    def __init__(self):
        self._data = {'time':[],
                     'h':[],
                     'V_inf':[],
                     'position':[],
                     'velocity':[],
                     'alpha':[],
                     'beta':[],
                     'omega_psi':[],
                     'omega_theta':[],
                     'omega_phi':[],
                     'psi':[],
                     'theta':[],
                     'phi':[],
                     'delta_e_des':[],
                     'delta_r_des':[],
                     'delta_a_des':[],
                     'P_des':[],
                     'delta_e':[],
                     'delta_r':[],
                     'delta_a':[],
                     'P':[],
                     'prop_rpm':[],
                     'h_des':[],}

    def __dict__(self):
        return dict(self)

    def __iter__(self):
        for (key, value) in self._data.items():
            yield (key, np.array(value))

    def __getitem__(self, key):
        return self._data[key]

    def step(self, telem, h_des):
        self._data['time'].append(float(telem['time']))
        self._data['h'].append(float(telem['h']))
        self._data['V_inf'].append(float(telem['V_inf']))
        self._data['position'].append(tuple(float(p) for p in telem['p_W']))
        self._data['velocity'].append(tuple(float(v) for v in telem['v_CoM']))
        self._data['alpha'].append(float(telem['alpha']))
        self._data['beta'].append(float(telem['beta']))
        self._data['omega_psi'].append(float(telem['omega_psi']))
        self._data['omega_theta'].append(float(telem['omega_theta']))
        self._data['omega_phi'].append(float(telem['omega_phi']))
        self._data['psi'].append(float(telem['psi']))
        self._data['theta'].append(float(telem['theta']))
        self._data['phi'].append(float(telem['phi']))
        self._data['delta_e'].append(float(telem['delta_e']))
        self._data['delta_r'].append(float(telem['delta_r']))
        self._data['delta_a'].append(float(telem['delta_a']))
        self._data['P'].append(float(telem['P']))
        self._data['delta_e_des'].append(float(telem['delta_e_des']))
        self._data['delta_r_des'].append(float(telem['delta_r_des']))
        self._data['delta_a_des'].append(float(telem['delta_a_des']))
        self._data['P_des'].append(float(telem['P_des']))
        self._data['prop_rpm'].append(float(telem['prop_rpm']))
        self._data['h_des'].append(float(h_des))

def _read_kwargs(**kwargs):
    state0 = {'h' : kwargs.get('h', 2000.0),
              'V_inf' : kwargs.get('V_inf', 47.82),
              'alpha' : kwargs.get('alpha', 0.05923),
              'beta' : kwargs.get('beta', 0.0),
              'omega_psi' : kwargs.get('omega_psi', 0.0),
              'omega_theta' : kwargs.get('omega_theta', 0.0),
              'omega_phi' : kwargs.get('omega_phi', 0.0),
              'psi' : kwargs.get('psi', 0.0),
              'theta' : kwargs.get('theta', 0.05923),
              'phi' : kwargs.get('phi', 0.0)}
    input0 = {'delta_e' : kwargs.get('delta_e', 0.06592),
              'delta_r' : kwargs.get('delta_r', 0.0),
              'delta_a' : kwargs.get('delta_a', 0.0),
              'P' : kwargs.get('P', 62160.),}
    kwargs = {'state0' : state0,
              'input0' : input0,
              'dt' : DT,
              'params' : PARAM,
              'r_planet' : R_PLANET,
              'duration' : kwargs.get('time', 20.0),
              'real_time' : kwargs.get('real_time', True),
              'turb_mag' : kwargs.get('turbulence', 0.0),
              'shake' : kwargs.get('shake', 1.0),
              'seed' : kwargs.get('seed', 2357136050),}
    return kwargs

def _load_planet(proj, telem):
    n_repeat = int(np.sqrt((4*np.pi*R_PLANET**2)/5.827e8)//2)*2+1
    tex_paths = [v for k,v in assets.items()
                 if k.startswith('countryside_225sqmi_')]
    tex_paths = sorted(tex_paths)
    proj.visualizer.add_object('ground',
                               assets['sphere_1_center_origin.stl'],
                               scale=(2*R_PLANET,)*3,
                               tex_path=tex_paths,
                               tex_repeat=[n_repeat, n_repeat],
                               emissive_color=(0.15, 0.15, 0.15),
                               position=(0.0, 0.0, -R_PLANET-telem['h']),)

def _load_sky(proj):
    proj.visualizer.add_object('skybox',
                               assets['sphere_1_center_origin.stl'],
                               scale=(R_PLANET*5*2, )*3,
                               tex_path=assets['skybox_day.jpg'],
                               roll=1.5708,
                               position=(0.0, 0.0, -R_PLANET))

def _load_vis_env(proj, telem):
    # Increase render distance to the skybox
    proj.visualizer.set_cam_frustum(far=5.1*R_PLANET)

    # Load the ground and sun
    _load_planet(proj, telem)
    _load_sky(proj)

    # Set the scene lighting
    proj.visualizer.set_background(bottom=(1.0, 1.0, 1.0))
    exp = int(np.ceil(np.log10(R_PLANET)))
    proj.visualizer.set_ptlight_1(on=True, intensity=1, shadow=True,
                            position=(10**exp/1.5, -10**exp/1.5, 10**exp/3.0),
                            distance=10**(exp+1.1))
    proj.visualizer.set_spotlight(on=True, position=(0, 0, -10),
                                  angle=0.51, intensity=0.2, distance=13,
                                  shadow=True)
    proj.visualizer.set_amblight(on=True, intensity=0.5)
    proj.visualizer.set_ptlight_2(on=False)
    proj.visualizer.set_dirnlight(on=False)
    proj.visualizer.set_dirnlight(on=False)

    # Look at the plane
    proj.visualizer.set_cam_position(telem['R_CoM_W']@(-25, 0, -5))
    proj.visualizer.set_cam_target((0, 0, 0))
    proj.visualizer.set_cam_zoom(2.5)

    # Make the grid and axes invisible
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)

    # Update the reflect changes
    proj.refresh_visualizer()

def _set_init_conds(plane, telem):
    # Rotate the prop
    omega = telem['prop_rpm'] * 0.1047
    plane.joints['fuselage_to_nosecone'].set_initial_state(omega=omega)

    # Set the initial control surfaces
    plane.joints['fuselage_to_flaps'].set_initial_state(angle=0)
    plane.joints['fuselage_to_elevator'].set_initial_state(
        angle=telem['delta_e'])
    plane.joints['fuselage_to_rudder'].set_initial_state(
        angle=telem['delta_r'])
    plane.joints['fuselage_to_r_aileron'].set_initial_state(
        angle=telem['delta_a'])
    plane.joints['fuselage_to_l_aileron'].set_initial_state(
        angle=telem['delta_a'])

    # Apply initial state
    plane.set_initial_state(yaw = -telem['psi'],
                            pitch = -telem['theta'],
                            roll = telem['phi'],
                            omega = (telem['omega_phi'],
                                     -telem['omega_theta'],
                                     -telem['omega_psi']))

def _make(**kwargs):
    # Create the project
    proj = Project(keyboard=False, visualizer=kwargs['real_time'],
                   simulator_gravity = (0.,0.,0.),
                   simulator_dt = DT,)

    # Create the flight simulator
    flightsim = FlightSim(**kwargs)
    telem = flightsim.telem()

    # Load the visual environment
    if kwargs['real_time']:
        _load_vis_env(proj, telem)

    # Load the plane, set its initial condition, and look at it
    plane = proj.load_urdf(assets['cessna172.urdf'], fixed=False)
    _set_init_conds(plane, telem)

    # Remove all joint friction
    for joint in plane.joints.values():
        joint.set_dynamics(damping=0.0)

    # Set all air resistance to 0
    for link in plane.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    return proj, plane, flightsim

def _await(proj, t=5.0):
    proj.refresh_visualizer()
    try:
        proj.keyboard.await_press('enter')
    except AttributeError:
        sleep(t)
    proj.refresh_visualizer()

def _h_des(program_number, curr_time, h0):
    h_des = h0

    # Step programs
    if program_number==1:
        h_des = h0 - 25.0 if curr_time > 2.0 else h0
    elif program_number==2:
        h_des = h0 + 75.0 if curr_time > 2.0 else h0
    elif program_number==3:
        h_des = h0 + 300. if curr_time > 2.0 else h0

    # Sequential step programs
    elif program_number==4:
        h_des = (curr_time//5.0) *-30.48/12.0 + h0
    elif program_number==5:
        h_des = (curr_time//5.0) * 152.4/12.0 + h0
    elif program_number==6:
        h_des = (curr_time//5.0) * 304.8/12.0 + h0

    # Linear programs
    elif program_number==7:
        h_des = curr_time*-0.508 + h0
    elif program_number==8:
        h_des = curr_time*2.5400 + h0
    elif program_number==9:
        h_des = curr_time*5.0800 + h0

    # Sinusoidal programs
    elif program_number==10:
        h_des = -7.62*np.sin(np.pi*curr_time/30.0) + h0
    elif program_number==11:
        h_des = 38.10*np.sin(np.pi*curr_time/30.0) + h0
    elif program_number==12:
        h_des = 76.20*np.sin(np.pi*curr_time/30.0) + h0

    return min(max(h_des, 50.0), 4000.0)

def _rotation_state(plane):
    pybullet_state = plane.state
    rotation_state = {'omega_psi' : -pybullet_state.omega_in_body[2],
                      'omega_theta' : -pybullet_state.omega_in_body[1],
                      'omega_phi' : pybullet_state.omega_in_body[0],
                      'psi' : -pybullet_state.ypr[0],
                      'theta' : -pybullet_state.ypr[1],
                      'phi' : pybullet_state.ypr[2],}
    return rotation_state

def _update_vis_env(proj, plane, telem, shake):
    # Set the elevator deflection
    plane.joints['fuselage_to_elevator'].set_state(angle = telem['delta_e'])
    plane.joints['fuselage_to_rudder'].set_state(angle = telem['delta_r'])
    plane.joints['fuselage_to_r_aileron'].set_state(angle = telem['delta_a'])
    plane.joints['fuselage_to_l_aileron'].set_state(angle = telem['delta_a'])

    # Update the prop speed
    omega = telem['prop_rpm'] * 0.1047
    plane.joints['fuselage_to_nosecone'].set_state(omega=omega)

    # Position the camera
    if shake > 0.0:
        p = shake * telem['g_force_W']
        proj.visualizer.set_cam_target(p)

    # Rotate the earth according to the forward velocity,
    # and move the earth according to the altitude
    proj.visualizer.set_transform('ground',
                                  pitch = telem['earth_pitch'],
                                  roll = telem['earth_roll'],
                                  position=(0,0,-R_PLANET-telem['h']))

def _get_keypresses(proj):
    delta_e = 0.0 # Torque applied to outermost ring
    delta_e -= 0.25 * float(proj.keyboard.is_pressed('w'))
    delta_e += 0.25 * float(proj.keyboard.is_pressed('s'))

    delta_r = 0.0 # Torque applied to the middle ring
    delta_r -= 0.25 * float(proj.keyboard.is_pressed('q'))
    delta_r += 0.25 * float(proj.keyboard.is_pressed('e'))

    delta_a = 0.0 # Torque applied to the inner ring
    delta_a -= 0.25 * float(proj.keyboard.is_pressed('a'))
    delta_a += 0.25 * float(proj.keyboard.is_pressed('d'))

    return delta_e, delta_r, delta_a

def _sim_loop(controller, program_num, proj, plane, flightsim, **kwargs):
    # Make structure to hold simulation data
    data = _SimData()

    # Reset the project to its initial state.
    proj.reset()

    # Get the t=0 data
    telem = flightsim.telem()
    h_des = _h_des(program_num, proj.simtime, kwargs['state0']['h'])
    data.step(telem, h_des)

    # Run a simulation loop
    start = now()
    while proj.simtime <= kwargs['duration']:
        # Crash condition (will strike ground in 0.5 seconds)
        if telem['h'] + 0.5*telem['v_W'][2] <= 0.0:
            break

        # Get the controller inputs
        delta_e_des, P_des = controller(telem, h_des)
        # delta_e_des, delta_r_des, delta_a_des = _get_keypresses(proj)

        # Step the simulation. Use the flight sim calculated aero torques
        # to rotate the airplane in the Pybullet engine
        tau_aero_net = flightsim.step(_rotation_state(plane),
                                      delta_e_des,
                                      0.0,
                                      0.0, P_des)
        plane.apply_torque(tau_aero_net)
        proj.step(real_time=kwargs['real_time'], stable_step=False)

        # Get the new telem and desired altitude
        telem = flightsim.telem()
        h_des = _h_des(program_num, proj.simtime, kwargs['state0']['h'])

        # Update the visuals
        if kwargs['real_time']:
            _update_vis_env(proj, plane, telem, kwargs['shake'])

        # Update the data
        data.step(telem, h_des)

    # Return the collected data
    print(f"Simulation took {now() - start:.2f} seconds.")
    return dict(data)

def run(controller, program_num, **kwargs):
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
            The initial altitude in meters. The default value is 2000
        V_inf : float, optional
            The initial indicated airspeed in meters/second. The default
            value is 47.82
        alpha : float, optional
            The initial angle of attack in radians. The default value
            is 0.05923
        omega_theta : float, optional
            The initial pitching rate in radians/second. The default value is 0
        theta : float, optional
            The initial pitch angle in radians. The default value is 0.05923
        delta_e : float, optional
            The initial elevator deflection angle in radians. The default
            value is 0.06592
        P : float, optional
            The initial power setting in KW. The default value is 62160.
        seed : int, optional
            The seed of the random number generator used for the simulation.
            The default is 2357136050
        turbulence : float, optional
            The mean magnitude of the turbulent wind in N. The default is 0
        shake : float, optional
            The magnitude by which plane accelerations are visualized. The
            default is 1.0.

    Returns
    -------
    data : dictionary of array-likes with length n
        The data collected during the simulation.

    """
    # Build the project, run the simulation loop, terminate the project
    kwargs = _read_kwargs(**kwargs)
    proj, plane, flightsim = _make(**kwargs)
    if kwargs['real_time']:
        _await(proj)
    data = _sim_loop(controller, program_num, proj, plane, flightsim, **kwargs)
    proj.terminate()
    return data

def ctrlr(state, h_des):
    m_e = np.array([0.0, 47.82, 0.05923, 0.0, 0.05923])
    n_e = np.array([0.06592, 62160.])
    x_des = np.array([h_des, 0.0, 0.0, 0.0, 0.0])
    K = np.array([[ 8.801e-03,  2.748e-03, -1.720e+00, 1.287e+00,  1.639e+00],
                  [ 2.730e+02,  1.469e+03, -5.856e+03, 1.563e+02,  6.134e+03]])
    m = np.array([state['h'],
                  state['V_inf'],
                  state['alpha'],
                  state['omega_theta'],
                  state['theta'],])
    n = -K@(m-m_e-x_des) + n_e
    return (n[0], n[1])

if __name__ ==  "__main__":
    data = run(ctrlr, 2, real_time=True, time=120)
