# -*- coding: utf-8 -*-
"""
This module implements the backend for the airplane project. In this
project, we use develop FLC/IAS autopilot.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""
from dataclasses import dataclass
from time import sleep
from time import time as now
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

R_PLANET = 6371000
@dataclass()
class _PlaneParams():
    params : dict

    def __init__(self):
        self.params = {}

        # Wing param
        self.params['a0_w'] = 5.011
        self.params['s_w'] = 16.2
        self.params['c_w'] = 1.49
        self.params['ar_w'] = 7.52

        # Horizontal stab param
        self.params['a0_t'] = 4.817
        self.params['s_t'] = 2.00
        self.params['ar_t'] = 6.32

        # Ele param
        self.params['a0_e'] = 5.230
        self.params['s_e'] = 1.35
        self.params['ar_e'] = 9.37

        # Tail and elevator chord
        self.params['c_te'] = 0.942

        # Body parameters
        self.params['s_b'] = 5.59
        self.params['cDf_b'] = 0.095
        self.params['mass'] = 964.0

        # Distance from CoM to wing and tail center of lift
        self.params['r_w'] = 0.111
        self.params['r_te'] = 4.560

        # Input limits
        self.params['delta_mn'] = -0.332
        self.params['delta_mx'] = 0.384
        self.params['delta_rate'] = 0.26
        self.params['P_mx'] = 1.342e5
        self.params['P_rate'] = 3.4e4

        # Powerplant sizing
        self.params['P_rpm_max'] = 2700.0
        self.params['prop_diameter'] = 1.905

    def __getattr__(self, key):
        return self.params[key]
PARAM = _PlaneParams()

class _SimVars():
    state : dict
    _dt : float

    def __init__(self, state0, input0, dt):
        self.state = {}
        self.state['p'] = np.array((0.0, 0.0, state0['h']))
        self.state['u'] = np.array((np.cos(state0['theta']-state0['alpha']),
                                    0.0,
                                    np.sin(state0['theta']-state0['alpha'])))
        self.state['u'] *= state0['u_inf']
        self.state['theta'] = state0['theta']
        self.state['delta'] = input0['delta']
        self.state['P'] = input0['P']
        self.state['earth_pitch'] = 0.0

        self._dt = dt

    def __getattr__(self, key):
        return self.state[key]

    def _to_m180_p180(self, ang_deg):
        return np.rad2deg((np.deg2rad(ang_deg) + np.pi) % (2*np.pi) - np.pi)

    def step(self, F_aero_net, omega_theta, delta_des, P_des):
        F_net = F_aero_net + np.array([0, 0, -_g(self.p[2])*PARAM.mass])

        self.state['p'] += self.state['u'] * self._dt
        self.state['u'] += (F_net / PARAM.mass) * self._dt
        self.state['theta'] += omega_theta * self._dt
        self.state['theta'] = self._to_m180_p180(self.state['theta'])
        d_ep = -self.state['u'][0]*self._dt / (self.state['p'][2] + R_PLANET)
        self.state['earth_pitch'] += d_ep

        clipped_delta_des = min(max(delta_des,PARAM.delta_mn),PARAM.delta_mx)
        d_delta = clipped_delta_des - self.state['delta']
        if abs(d_delta) <= PARAM.delta_rate*self._dt:
            self.state['delta'] += d_delta
        else:
            self.state['delta'] += PARAM.delta_rate*np.sign(d_delta)*self._dt

        clipped_P_des =  min(max(P_des, 0.0), PARAM.P_mx)
        d_P = clipped_P_des - self.state['P']
        if abs(d_P) <= PARAM.P_rate*self._dt:
            self.state['P'] += d_P
        else:
            self.state['P'] += PARAM.P_rate*np.sign(d_P)*self._dt

class _SimData():
    data : dict

    def __init__(self):
        self.data = {'time':[],
                     'h':[],
                     'u_inf':[],
                     'alpha':[],
                     'omega_theta':[],
                     'theta':[],
                     'delta_des':[],
                     'P_des':[],
                     'delta':[],
                     'P':[],
                     'h_des':[],}

    def __dict__(self):
        return dict(self)

    def __iter__(self):
        for (key, value) in self.data.items():
            yield (key, np.array(value))

    def __getitem__(self, key):
        return self.data[key]

    def step(self, t, state, input_cur_des, h_des):
        self.data['time'].append(float(t))
        self.data['h'].append(float(state['h']))
        self.data['u_inf'].append(float(state['u_inf']))
        self.data['alpha'].append(float(state['alpha']))
        self.data['omega_theta'].append(float(state['omega_theta']))
        self.data['theta'].append(float(state['theta']))
        self.data['delta'].append(float(input_cur_des[0]))
        self.data['P'].append(float(input_cur_des[1]))
        self.data['delta_des'].append(float(input_cur_des[2]))
        self.data['P_des'].append(float(input_cur_des[3]))
        self.data['h_des'].append(float(h_des))

def _read_kwargs(**kwargs):
    state0 = {'h' : kwargs.get('h', 2000.0),
              'u_inf' : kwargs.get('u_inf', 47.8184),
              'alpha' : kwargs.get('alpha', 0.05923),
              'omega_theta' : kwargs.get('omega_theta', 0.0) ,
              'theta' : kwargs.get('theta', 0.05923)}
    input0 = {'delta' : kwargs.get('delta', 0.06592),
              'P' : kwargs.get('P', 62159.9),}
    settings = {'state0' : state0,
                'input0' : input0,
                'duration' : kwargs.get('time', 20.0),
                'real_time' : kwargs.get('real_time', True),
                'turb_mag' : kwargs.get('turbulence', 0.0),
                'shake' : kwargs.get('shake', 0.35),
                'seed' : kwargs.get('seed', 2357136050),}
    return settings

def _load_planet(proj, state0):
    n_repeat = int(np.sqrt((4*np.pi*R_PLANET**2)/5.827e8)//2)*2+1
    tex_paths = [v for k,v in assets.items()
                 if k.startswith('countryside_225sqmi_')]
    tex_paths = sorted(tex_paths)
    proj.visualizer.add_object('ground',
                               assets['sphere_1_center_origin.stl'],
                               scale=(2*R_PLANET,)*3,
                               tex_path=tex_paths,
                               tex_wrap=[1000, 1000],
                               tex_repeat=[n_repeat, n_repeat],
                               emissive_color=(0.15, 0.15, 0.15),
                               position=(0.0, 0.0, -R_PLANET-state0['h']),)

def _load_sun(proj):
    for i in range(10):
        proj.visualizer.add_object(f'sun_{i}',
                                   assets['sphere_1_center_origin.stl'],
                                   scale=((5.+.25*i)*0.01745*.1*R_PLANET,)*3,
                                   color=(1.0, 1.0, 1.0),
                                   emissive_color=(1.0, 1.0, 1.0),
                                   opacity=0.5-0.05*i,
                                   position=np.array([.7,-.7,.1])*0.1*R_PLANET)
def _load_vis_env(proj, state0):
    # Increase render distance to the horizon
    dist_to_hori = 1.1*np.sqrt(2*R_PLANET*state0['h'])
    proj.visualizer.set_cam_frustum(far=max(dist_to_hori, 0.1005*R_PLANET))

    # Load the ground and sun
    _load_planet(proj, state0)
    _load_sun(proj)

    # Set the scene lighting
    proj.visualizer.set_background(bottom=(1.0, 1.0, 1.0))
    proj.visualizer.set_ptlight_1(on=True, position=(1.0, -1.0, -12),
                                  intensity=0.05, shadow=True, distance=50)
    proj.visualizer.set_spotlight(on=True, position=(12.6, -12.6, 1.8),
                                  angle=0.35, intensity=1.5, distance=50,
                                  shadow=True)
    proj.visualizer.set_amblight(on=True, intensity=0.6)
    proj.visualizer.set_ptlight_2(on=False)
    proj.visualizer.set_dirnlight(on=False)
    proj.visualizer.set_dirnlight(on=False)

    # Look at the plane
    proj.visualizer.set_cam_position((0, 10, 1.25))
    proj.visualizer.set_cam_target((0, 0, 0))

    # Make the grid and axes invisible
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)

    # Update the reflect changes
    proj.refresh_visualizer()

def _set_init_conds(plane, state0, input0):
    # Rotate the prop at 2500rpm
    plane.joints['fuselage_to_nosecone'].set_initial_state(omega=261.8)

    # Set the initial control surfaces
    plane.joints['fuselage_to_flaps'].set_initial_state(angle=0)
    plane.joints['fuselage_to_r_aileron'].set_initial_state(angle=0)
    plane.joints['fuselage_to_l_aileron'].set_initial_state(angle=0)
    plane.joints['fuselage_to_elevator'].set_initial_state(
        angle=input0['delta'])
    plane.joints['fuselage_to_rudder'].set_initial_state(angle=0)

    # Apply initial state
    plane.set_initial_state(pitch = -state0['theta'] + 0.083, # Model axis cor
                            omega = (0.0, -state0['omega_theta'], 0.0))

def _make(**kwargs):
    # Create the project
    proj = Project(keyboard=False, visualizer=kwargs['real_time'])
    proj.simulator.set_gravity((0., 0., 0.))

    # Load the visual environment
    if kwargs['real_time']:
        _load_vis_env(proj, kwargs['state0'])

    # Load the plane, set its initial condition, and look at it
    plane = proj.load_urdf(assets['cessna172.urdf'], fixed=False)
    _set_init_conds(plane, kwargs['state0'], kwargs['input0'])

    # Remove all friction
    for joint in plane.joints.values():
        joint.set_dynamics(damping=0.0)
    for link in plane.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.0)

    return proj, plane

def _await(proj, t=5.0):
    proj.refresh_visualizer()
    try:
        proj.keyboard.await_press('enter')
    except AttributeError:
        sleep(t)
    proj.refresh_visualizer()

def _g(h):
    # Newtonian gravity
    return 3.986025446e14 / (6.371e6+h)**2

def _rho(h):
    # Ideal gas law applied to barometric formula
    return 1.225*np.exp(-h/10363.)

def _T(h):
    # Standard atmosphere temperature model
    if h < 10000:
        return 288.2 - 0.00649*h
    return 223.3

def _mu(h):
    return 1.458e-6*_T(h)**(1.5) / (_T(h) + 110.4)

def _cL(alpha, a_s, a_0, a_l0):
    # No stall
    if abs(alpha) <= a_s:
        return a_0*(alpha-a_l0)

    # Positive stalling region
    if 0 < alpha < a_s+0.0873:
        a = -32.83*(a_0*(a_s+0.3491)-a_l0*a_0)
        b = 65.66*(a_0*(a_s**2+0.3491*a_s+0.0152)-a_l0*a_0*a_s)
        c = -32.83*(a_0*a_s**2*(a_s+0.3491)-a_l0*a_0*(a_s**2-0.0305))
        return a*alpha**2 + b*alpha + c

    # Negative stalling region
    if -a_s-0.0873 < alpha < 0:
        a = 32.83*(a_0*(a_s+0.3491)+a_l0*a_0)
        b = 65.66*(a_0*(a_s**2+0.3491*a_s+0.0152)+a_l0*a_0*a_s)
        c = 32.83*(a_0*a_s**2*(a_s+0.3491)+a_l0*a_0*(a_s**2-0.0305))
        return a*alpha**2 + b*alpha + c

    # Complete stall
    return 0.0

def _wing_forces(state):
    # Get the reynold's number
    re = _rho(state['h'])*PARAM.c_w*state['u_inf']/_mu(state['h'])

    # Calculate stall angle and 0 lift angle (for NACA2412)
    a_s = 0.0407*np.log(re) - 0.266
    a_l0 = -0.0436 * (np.arctan(re/11880. - 10.52) / np.pi + 0.5)

    # Get the lift and drag
    cL = _cL(state['alpha'], a_s, PARAM.a0_w, a_l0)
    cDi = (PARAM.a0_w*(state['alpha']-a_l0))**2 / (np.pi*PARAM.ar_w)
    cDf = 0.074*re**(-0.2)
    L = 0.5 * _rho(state['h']) * PARAM.s_w * cL * state['u_inf']**2
    D = 0.5 * _rho(state['h']) * PARAM.s_w * (cDi + cDf) * state['u_inf']**2
    return L, D

def _tail_forces(state, delta):
    # Get the reynold's number
    re = _rho(state['h'])*PARAM.c_te*state['u_inf']/_mu(state['h'])

    # Calculate the stall conditions (for inverted NACA2412)
    a_ste = 0.0407*np.log(re) - 0.266
    a_l0te = 0.0436 * (np.arctan(re/11880. - 10.52) / np.pi + 0.5)

    # Get the lift and drag
    cLt = _cL(state['alpha'], a_ste, PARAM.a0_t, a_l0te)
    cDit = (PARAM.a0_t*(state['alpha']-a_l0te))**2 / (np.pi*PARAM.ar_t)
    cLe = _cL(state['alpha']-delta, a_ste, PARAM.a0_e, a_l0te)
    cDie = (PARAM.a0_e*(state['alpha']-delta-a_l0te))**2 / (np.pi*PARAM.ar_e)
    cDfte = 0.074*re**(-0.2)
    half_rho_usq = 0.5*_rho(state['h'])*state['u_inf']**2
    L=(PARAM.s_t*cLt+PARAM.s_e*cLe)*half_rho_usq
    D=(PARAM.s_t*cDit+PARAM.s_e*cDie+(PARAM.s_t+PARAM.s_e)*cDfte)*half_rho_usq
    return L, D

def  _body_forces(state):
    # Get the lift and drag
    L = 0.0
    D = 0.5 * _rho(state['h']) * PARAM.s_b * PARAM.cDf_b * state['u_inf']**2
    return L, D

def _prop_rps(P):
    # Prop RPS as a function of engine power
    # (0% power = 0.25*rps_max)
    # (75% power = 0.925*rps_max)
    # (100% power = rps_max)
    num = 0.0174*(71.65)**(P/PARAM.P_mx)*PARAM.P_rpm_max
    den = (71.65)**(P/PARAM.P_mx) + 3.177
    return num / den

def _eta(u_inf, P):
    # Ideal prop efficiency based on advance ratio for a 20 degree angle prop
    J = u_inf / (_prop_rps(P) * PARAM.prop_diameter)
    if 0 <= J < 0.87:
        return -1.097*J*J + 1.908*J
    if 0.87 <= J <= 1.05:
        return -25.62*J*J + 44.57*J - 18.56
    return 0.0

def _prop_forces(state, P):
    return P*_eta(state['u_inf'], P) / state['u_inf']

def _LD_to_plane(L, D, alpha, theta):
    Lx = -L*np.sin(theta-alpha)
    Lz = -L*np.cos(theta-alpha)
    Dx = -D*np.cos(theta-alpha)
    Dz = D*np.sin(theta-alpha)
    return np.array((Lx, 0.0, Lz)) + np.array((Dx, 0.0, Dz))

def _net_aero_force_torque(state, delta, P):
    # Get the forces
    Lw, Dw = _wing_forces(state)
    Lt, Dt = _tail_forces(state, delta)
    Lb, Db = _body_forces(state)
    T = _prop_forces(state, P)

    # Get the net torque
    tau_y = (-np.cos(state['alpha'])*(Lw*PARAM.r_w + Lt*PARAM.r_te) -
             np.sin(state['alpha'])*(Dw*PARAM.r_w + Dt*PARAM.r_te))

    # Convert to plane coords
    Fw = _LD_to_plane(Lw, Dw, state['alpha'], state['theta'])
    Ft = _LD_to_plane(Lt, Dt, state['alpha'], state['theta'])
    Fb = _LD_to_plane(Lb, Db, state['alpha'], state['theta'])
    Fp = np.array((np.cos(state['theta'])*T, 0.0, -np.sin(state['theta'])*T))
    return Fw + Ft + Fb + Fp, np.array([0.0, tau_y, 0.0])

def _plane_to_world(vec):
    return np.array((vec[0], -vec[1], -vec[2]))

def _state(plane, h, u, theta):
    state = {'h' : h,
             'u_inf' : np.linalg.norm(u),
             'alpha' : theta - np.atan2(u[2], u[0]),
             'omega_theta' : -plane.state.omega_in_body[1],
             'theta' : theta,}
    return state

def _gen_turb_param(magnitude, seed):
    rng = np.random.default_rng(seed=seed)
    period = 7.0*rng.random(3)
    scale = magnitude*(2.0*rng.random(3)-1.0)
    offset = 120.0*rng.random(3)
    return (period, scale, offset)

def _turbulence(turb_param, t):
    tl=[s*(0.25*(np.sin(2.0/p*(t - o)) + np.sin(3.1415/p*(t - o))) + 0.5)
        for p, s, o in zip(*turb_param)]
    ts=[0.07*s*(0.25*(np.sin(10.0/p*(t - o)) + np.sin(15.708/p*(t - o))) + 0.5)
        for p, s, o in zip(*turb_param)]
    return np.array(tl) + np.array(ts)

def _mass(obj):
    mass = 0.0
    for link in obj.links.values():
        mass += link.mass
    return mass

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

def _update_vis_env(proj, plane, simvars, prev_states, shake):
    # Set the elevator deflection
    plane.joints['fuselage_to_elevator'].set_state(angle = simvars.delta)

    # Position the camera
    if shake > 0.0:
        cz = shake*(np.mean(prev_states['h'][-100:])-simvars.p[2])
        proj.visualizer.set_cam_position((0, 10, cz+1.25))
        proj.visualizer.set_cam_target((0, 0, cz))

    # Rotate the earth according to the forward velocity,
    # and move the earth according to the altitude
    proj.visualizer.set_transform('ground',
                                  pitch = simvars.earth_pitch,
                                  scale = (2*R_PLANET, )*3,
                                  position=(0,0,-R_PLANET-simvars.p[2]))

    # Increase render distance to the horizon
    far = max(1.1*np.sqrt(2*R_PLANET*simvars.p[2]), 0.1005*R_PLANET)
    proj.visualizer.set_cam_frustum(far=far)

def _sim_loop(controller, program_nmuber, proj, plane, **kwargs):
    # Generate a set of turbulence parameters based on the selected seed
    turb = _gen_turb_param(kwargs['turb_mag'], kwargs['seed'])

    # Build a structure to track and update all hand updated sim variables
    v = _SimVars(kwargs['state0'], kwargs['input0'], proj.simulator.dt)

    # Make structure to hold simulation data
    data = _SimData()

    # Reset the project to its initial state.
    proj.reset()

    # Run a simulation loop
    start = now()
    while proj.simtime <= kwargs['duration']:

        # Get the state of the system
        state = _state(plane,v.p[2],v.u+_turbulence(turb,proj.simtime),v.theta)
        h_des = _h_des(program_nmuber, proj.simtime, kwargs['state0']['h'])

        # Crash condition (will strike ground in 0.5 seconds)
        if state['h'] + 0.5*v.u[2] <= 0.0:
            break

        # Get the controller inputs
        delta_des, P_des = controller(state, h_des)

        # Update the data
        data.step(proj.simtime, state, (v.delta, v.P, delta_des, P_des), h_des)

        # Calculate the net force and torque on the plane
        F_aero_net, tau_aero_net = _net_aero_force_torque(state, v.delta, v.P)

        # Apply only the torque. We ignore the forces
        # because motion is handled by moving the planet instead of the plane.
        plane.apply_torque(_plane_to_world(tau_aero_net))

        # Update the visuals
        if kwargs['real_time']:
            _update_vis_env(proj, plane, v, data, kwargs['shake'])

        # Take a simulation step
        v.step(_plane_to_world(F_aero_net), state['omega_theta'],
               delta_des, P_des)
        proj.step(real_time=kwargs['real_time'], stable_step=False)

    # Return the collected data
    print(f"Simulation took {now() - start:.2f} seconds.")
    return dict(data)

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
            The initial altitude in meters. The default value is 2000
        u_inf : float, optional
            The initial airspeed in meters/second. The default value is 47.8184
        alpha : float, optional
            The initial angle of attack in radians. The default value
            is 0.05923
        omega_theta : float, optional
            The initial pitching rate in radians/second. The default value is 0
        theta : float, optional
            The initial pitch angle in radians. The default value is 0.05923
        seed : int, optional
            The seed of the random number generator used for the simulation.
            The default is 2357136050
        turbulence : float, optional
            The magnitude of the turbulent wind in m/s. The default is 0
        shake : float, optional
            The magnitude by which plane accelerations are visualized. The
            default is 0.35. Set to 0.0 for free camera movement.

    Returns
    -------
    data : dictionary of array-likes with length n
        The data collected during the simulation.

    """
    # Build the project, run the simulation loop, terminate the project
    settings = _read_kwargs(**kwargs)
    proj, plane = _make(**settings)
    if settings['real_time']:
        _await(proj)
    data = _sim_loop(controller, program_number, proj, plane, **settings)
    proj.terminate()
    return data

def ctrlr(state, h_des):
    m_e = np.array([0.0, 43.4816, 0.060307, 0.0, 0.060307])
    n_e = np.array([0.06878, 56237.45])
    x_des = np.array([h_des, 0.0, 0.0, 0.0, 0.0])
    K = np.array([[ 9.87485821e-03, -2.92274181e-02, -2.59702909e+00,
         6.50978731e+00,  3.57972626e+00],
       [ 6.89177862e+02,  7.37680069e+03, -4.86655469e+01,
        -3.54140719e+02, -4.88447472e+04]])
    m = np.array([state['h'],
                  state['u_inf'],
                  state['alpha'],
                  state['omega_theta'],
                  state['theta'],])
    n = -K@(m-m_e-x_des) + n_e
    return (n[0], n[1])

if __name__ ==  "__main__":
    data = run(ctrlr, 2, time=10.0)
