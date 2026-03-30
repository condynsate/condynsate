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
        self.params['a0_w'] = 5.01
        self.params['s_w'] = 16.2
        self.params['c_w'] = 1.49
        self.params['ar_w'] = 7.52
        self.params['b_w'] = 10.9
        self.params['dihedral_w'] = 0.0218
        self.params['d_eta_d_alpha'] = 0.25 # Downwash angle slope wrt AoA

        # Horizontal stab param
        self.params['a0_t'] = 4.817
        self.params['s_t'] = 2.00
        self.params['ar_t'] = 6.32

        # Ele param
        self.params['a0_e'] = 5.230
        self.params['s_e'] = 1.35
        self.params['ar_e'] = 9.37

        # Horizontal and elevator chord
        self.params['c_te'] = 0.942

        # Combined vertical stab and rudder param
        self.params['a0_v'] = 1.63 # Jone's theory estimate
        self.params['s_v'] = 1.73
        self.params['ar_v'] = 1.04
        self.params['c_v'] = 1.17

        # Body parameters
        self.params['s_b'] = 5.59
        self.params['cDf_b'] = 0.095
        self.params['mass'] = 964.0

        # Distance from CoM to wing and tail center of lift (at 0 AoA)
        # positive behind and above CoM
        self.params['dcL_w'] = 0.156   # Axial distance (wing)
        self.params['hcL_w'] = 0.971   # Vertical distance (wing)
        self.params['dcL_te'] = 4.59   # Axial distance (hori stab + ele)
        self.params['hcL_te'] = -.0288 # Vertical distance (hori stab + ele)
        self.params['dcL_v'] =  4.76   # Axial distance (v stab)
        self.params['hcL_v'] = 0.260   # Vertical distance (v stab)
        self.params['dcL_b'] =  0.608  # Axial distance (fuselage)
        self.params['hcL_b'] = 0.127   # Vertical distance (fuselage)

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

def _RX(x):
    return np.array([[1, 0,          0        ],
                     [0, np.cos(x), -np.sin(x)],
                     [0, np.sin(x),  np.cos(x)]])
def _RY(x):
    return np.array([[ np.cos(x), 0, np.sin(x)],
                     [ 0,         1, 0        ],
                     [-np.sin(x), 0, np.cos(x)]])
def _RZ(x):
    return np.array([[np.cos(x), -np.sin(x), 0],
                     [np.sin(x),  np.cos(x), 0],
                     [0,          0,         1]])
class _SimVars():
    R : dict
    state : dict
    _dt : float

    def __init__(self, state0, input0, dt):
        self._dt = dt

        self.R = {}
        self.R['PF'] = _RY(state0['alpha'])@_RZ(state0['beta'])
        self.R['FP'] = self.R['PF'].T
        r, p, y = state0['phi'], state0['theta'], state0['psi']
        self.R['WP'] = _RX(r)@_RY(np.pi-p)@_RZ(np.pi+y)
        self.R['PW'] = self.R['WP'].T
        self.R['WPl'] = _RX(r+PARAM.dihedral_w)@_RY(np.pi-p)@_RZ(np.pi+y)
        self.R['WPr'] = _RX(r-PARAM.dihedral_w)@_RY(np.pi-p)@_RZ(np.pi+y)
        self.R['PlW'] = self.R['WPl'].T
        self.R['PrW'] = self.R['WPr'].T

        self.state = {}
        self.state['p_W'] = np.array((0.0, 0.0, state0['h']))
        self.state['u_P'] = self.R['FP'] @ (state0['u_inf'], 0.0, 0.0)
        self.state['u_W'] = self.R['PW'] @ self.state['u_P']
        self.state['u_Pl'] = self.R['WPl']@self.state['u_W']
        self.state['u_Pr'] = self.R['WPr']@self.state['u_W']
        self.state['u_inf'] = np.linalg.norm(self.state['u_W'])

        u, v, w = self.state['u_P']
        self.state['alpha'] = np.arctan2(w, u)
        self.state['beta'] = np.arcsin(v / self.state['u_inf'])

        u, v, w = self.state['u_Pl']
        self.state['alpha_l'] = np.arctan2(w, u)
        self.state['beta_l'] = np.arcsin(v / self.state['u_inf'])
        self.R['PlF'] = _RY(self.state['alpha_l'])@_RZ(self.state['beta_l'])
        self.R['FPl'] = self.R['PlF'].T

        u, v, w = self.state['u_Pr']
        self.state['alpha_r'] = np.arctan2(w, u)
        self.state['beta_r'] = np.arcsin(v / self.state['u_inf'])
        self.R['PrF'] = _RY(self.state['alpha_r'])@_RZ(self.state['beta_r'])
        self.R['FPr'] = self.R['PrF'].T

        self.state['psi'] = state0['psi']
        self.state['theta'] = state0['theta']
        self.state['phi'] = state0['phi']
        self.state['delta'] = input0['delta']
        self.state['P'] = input0['P']
        self.state['earth_pitch'] = 0.0
        self.state['earth_roll'] = 0.0

    def __getattr__(self, key):
        return self.state[key]

    def _to_m180_p180(self, ang_deg):
        return np.rad2deg((np.deg2rad(ang_deg) + np.pi) % (2*np.pi) - np.pi)

    def step(self, F_aero_net, state, delta_des, P_des):
        # Get the net force by adding gravity
        F_net = F_aero_net+np.array([0,0,-_g(self.state['p_W'][2])*PARAM.mass])

        # Apply accelerations and velocities
        self.state['u_W'] += (F_net / PARAM.mass) * self._dt
        self.state['u_inf'] = np.linalg.norm(self.state['u_W'])
        self.state['p_W'] += self.state['u_W'] * self._dt
        self.state['psi'] += state['omega_psi'] * self._dt
        self.state['theta'] += state['omega_theta'] * self._dt
        self.state['phi'] += state['omega_phi'] * self._dt
        self.state['theta'] = self._to_m180_p180(self.state['theta'])
        self.state['phi'] = self._to_m180_p180(self.state['phi'])

        # Update the local velocities
        self.state['u_P'] = self.R['WP'] @ self.state['u_W']
        self.state['u_Pl'] = self.R['WPl'] @ self.state['u_W']
        self.state['u_Pr'] = self.R['WPr'] @ self.state['u_W']

        # Update the local flow angles
        u, v, w = self.state['u_P']
        self.state['alpha'] = np.arctan2(w, u)
        self.state['beta'] = np.arcsin(v / self.state['u_inf'])
        u, v, w = self.state['u_Pl']
        self.state['alpha_l'] = np.arctan2(w, u)
        self.state['beta_l'] = np.arcsin(v / self.state['u_inf'])
        u, v, w = self.state['u_Pr']
        self.state['alpha_r'] = np.arctan2(w, u)
        self.state['beta_r'] = np.arcsin(v / self.state['u_inf'])

        # Update the world rotation based on position
        d = -self.state['u_W'][0]*self._dt / (self.state['p_W'][2]+R_PLANET)
        self.state['earth_pitch'] += d
        d = self.state['u_W'][1]*self._dt / (self.state['p_W'][2]+R_PLANET)
        self.state['earth_roll'] += d

        # Update the coordinate transform matrices
        r, p, y = self.state['phi'], self.state['theta'], self.state['psi']
        self.R['WP'] = _RX(r)@_RY(np.pi-p)@_RZ(np.pi+y)
        self.R['WPl'] = _RX(r+PARAM.dihedral_w)@_RY(np.pi-p)@_RZ(np.pi+y)
        self.R['WPr'] = _RX(r-PARAM.dihedral_w)@_RY(np.pi-p)@_RZ(np.pi+y)
        self.R['PW'] = self.R['WP'].T
        self.R['PlW'] = self.R['WPl'].T
        self.R['PrW'] = self.R['WPr'].T
        self.R['PF'] = _RY(self.state['alpha'])@_RZ(self.state['beta'])
        self.R['PlF'] = _RY(self.state['alpha_l'])@_RZ(self.state['beta_l'])
        self.R['PrF'] = _RY(self.state['alpha_r'])@_RZ(self.state['beta_r'])
        self.R['FP'] = self.R['PF'].T
        self.R['FPl'] = self.R['PlF'].T
        self.R['FPr'] = self.R['PrF'].T

        # Update the elevator angle
        d=min(max(delta_des,PARAM.delta_mn),PARAM.delta_mx)-self.state['delta']
        if abs(d) <= PARAM.delta_rate*self._dt:
            self.state['delta'] += d
        else:
            self.state['delta'] += PARAM.delta_rate*np.sign(d)*self._dt

        # Update the power setting
        d = min(max(P_des, 0.0), PARAM.P_mx) - self.state['P']
        if abs(d) <= PARAM.P_rate*self._dt:
            self.state['P'] += d
        else:
            self.state['P'] += PARAM.P_rate*np.sign(d)*self._dt

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
              'u_inf' : kwargs.get('u_inf', 47.82),
              'alpha' : kwargs.get('alpha', 0.05923),
              'beta' : kwargs.get('beta', 0.0),
              'omega_psi' : kwargs.get('omega_psi', 0.0),
              'omega_theta' : kwargs.get('omega_theta', 0.0),
              'omega_phi' : kwargs.get('omega_phi', 0.0),
              'psi' : kwargs.get('psi', 0.0),
              'theta' : kwargs.get('theta', 0.05923),
              'phi' : kwargs.get('phi', 0.0)}
    input0 = {'delta' : kwargs.get('delta', 0.06592),
              'P' : kwargs.get('P', 62160.),}
    settings = {'state0' : state0,
                'input0' : input0,
                'duration' : kwargs.get('time', 20.0),
                'real_time' : kwargs.get('real_time', True),
                'turb_mag' : kwargs.get('turbulence', 0.0),
                'shake' : kwargs.get('shake', 0.5),
                'seed' : kwargs.get('seed', 2357136050),}
    return settings

def _load_planet(proj, state0):
    n_repeat = int(np.sqrt((4*np.pi*R_PLANET**2)/5.827e8)//2)*2+1
    tex_paths = [v for k,v in assets.items()
                 if k.startswith('countryside_225sqmi_28')]
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
    proj.visualizer.set_cam_position((10*np.sin(state0['psi']),
                                      10*np.cos(state0['psi']),
                                      1.25))
    proj.visualizer.set_cam_target((0, 0, 0))

    # Make the grid and axes invisible
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)

    # Update the reflect changes
    proj.refresh_visualizer()

def _set_init_conds(plane, state0, input0):
    # Rotate the prop
    omega = _prop_rps(input0['P']) * np.pi * 2.0
    plane.joints['fuselage_to_nosecone'].set_initial_state(omega=omega)

    # Set the initial control surfaces
    plane.joints['fuselage_to_flaps'].set_initial_state(angle=0)
    plane.joints['fuselage_to_r_aileron'].set_initial_state(angle=0)
    plane.joints['fuselage_to_l_aileron'].set_initial_state(angle=0)
    plane.joints['fuselage_to_elevator'].set_initial_state(
        angle=input0['delta'])
    plane.joints['fuselage_to_rudder'].set_initial_state(angle=0)

    # Apply initial state
    plane.set_initial_state(yaw = -state0['psi'],
                            pitch = -state0['theta'], # Model axis cor
                            roll = state0['phi'],
                            omega = (state0['omega_phi'],
                                     -state0['omega_theta'],
                                     -state0['omega_psi']))

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

    # Remove all joint friction
    for joint in plane.joints.values():
        joint.set_dynamics(damping=0.0)

    # Set linear air resistance to 0 because we track position manually
    # Add some rotational air resistance. Rotations are handled by Pybullet
    for link in plane.links.values():
        link.set_dynamics(linear_air_resistance=0.0,
                          angular_air_resistance=0.4)

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

def _cL(alpha_eff, beta, a_s, a_0, a_l0):

    # No stall
    if abs(alpha_eff) <= a_s:
        # Add a cos beta as a first order crab angle correction term
        return np.cos(beta)*a_0*(alpha_eff-a_l0)

    # Positive stalling region
    if 0 < alpha_eff < a_s+0.0524:
        a = 182.4*a_0*(a_l0-a_s-0.1047)
        b = -364.8*a_0*(a_l0*a_s-a_s**2-0.1047*a_s-0.0027)
        c = 182.4*a_0*(a_l0*(a_s**2-0.0055)-a_s**2*(a_s+0.1047))

        # Add a cos beta as a first order crab angle correction term
        return np.cos(beta)*(a*alpha_eff**2 + b*alpha_eff + c)

    # Negative stalling region
    if -a_s-0.0524 < alpha_eff < 0:
        a = 182.4*a_0*(a_l0+a_s+0.1047)
        b = 364.8*a_0*(a_l0*a_s+a_s**2+0.1047*a_s+0.0027)
        c = 182.4*a_0*(a_l0*(a_s**2-0.0055)+a_s**2*(a_s+0.1047))

        # Add a cos beta as a first order crab angle correction term
        return np.cos(beta)*(a*alpha_eff**2 + b*alpha_eff + c)

    # Complete stall
    return 0.0

def _wing_forces(s):
    # Get the reynold's number
    re = _rho(s.p_W[2])*PARAM.c_w*s.u_inf/_mu(s.p_W[2])

    # Calculate stall angle and 0 lift angle (for NACA2412)
    a_s = min(0.0408*np.log(re) - 0.267, 0.244)
    a_l0 = -0.0436 * (np.arctan(re/11880. - 10.52) / np.pi + 0.5)

    # Get the lift and drag of the left wing
    cL_l = _cL(s.alpha_l, s.beta_l, a_s, PARAM.a0_w, a_l0)
    cDi = (PARAM.a0_w*(s.alpha_l-a_l0))**2 / (np.pi*PARAM.ar_w)
    cDf = 0.074*re**(-0.2)
    L_l = 0.5*_rho(s.p_W[2])*0.5*PARAM.s_w*cL_l*s.u_inf**2
    D_l = 0.5*_rho(s.p_W[2])*0.5*PARAM.s_w*(cDi + cDf)*s.u_inf**2

    # Get the lift and drag of the right wing
    # Note, we bias the right wing to stall first to simulate
    # slight degredation of the right surface compared to the left
    cL_r = _cL(s.alpha_r, s.beta_r, 0.99*a_s, PARAM.a0_w, a_l0)
    cDi = (PARAM.a0_w*(s.alpha_r-a_l0))**2 / (np.pi*PARAM.ar_w)
    cDf = 0.074*re**(-0.2)
    L_r = 0.5*_rho(s.p_W[2])*0.5*PARAM.s_w*cL_r*s.u_inf**2
    D_r = 0.5*_rho(s.p_W[2])*0.5*PARAM.s_w*(cDi + cDf)*s.u_inf**2

    # Convert from lift and drag to world coords
    F_l = s.R['PlW'] @ s.R['FPl'] @ (-D_l, 0.0, -L_l)
    F_r = s.R['PrW'] @ s.R['FPr'] @ (-D_r, 0.0, -L_r)
    return F_l, F_r

def _hori_stab_force(s):
    # Get the reynold's number
    re = _rho(s.p_W[2])*PARAM.c_te*s.u_inf/_mu(s.p_W[2])

    # Calculate the stall conditions (for inverted NACA2412)
    a_ste = min(0.0408*np.log(re) - 0.267, 0.297)
    a_l0te = 0.0436 * (np.arctan(re/11880. - 10.52) / np.pi + 0.5)

    # Get the downwash angle from the wings
    eta = PARAM.d_eta_d_alpha * (s.alpha_l + s.alpha_r)*0.5

    # Get the lift and induced drag of the horizontal stab
    cL_t = _cL(s.alpha - eta, s.beta, a_ste, PARAM.a0_t, a_l0te)
    cDi_t = (PARAM.a0_t*(s.alpha-a_l0te))**2 / (np.pi*PARAM.ar_t)
    L_t = 0.5*_rho(s.p_W[2])*PARAM.s_t*cL_t*s.u_inf**2
    Di_t = 0.5*_rho(s.p_W[2])*PARAM.s_t*cDi_t*s.u_inf**2

    # Get the lift and induced drag of the elevators
    cL_e = _cL(s.alpha - eta - s.delta, s.beta, a_ste, PARAM.a0_e, a_l0te)
    cDi_e = (PARAM.a0_e*(s.alpha-s.delta-a_l0te))**2 / (np.pi*PARAM.ar_e)
    L_e = 0.5*_rho(s.p_W[2])*PARAM.s_t*cL_e*s.u_inf**2
    Di_e = 0.5*_rho(s.p_W[2])*PARAM.s_t*cDi_e*s.u_inf**2

    # Parasitic drag of stab and elevator
    cDf_te = 0.074*re**(-0.2)
    Df_te = 0.5*_rho(s.p_W[2])*(PARAM.s_t+PARAM.s_e)*cDf_te*s.u_inf**2

    # Convert from lift and drag to world coords
    return s.R['PW'] @ s.R['FP'] @ (-(Di_t + Di_e + Df_te), 0.0, -(L_t + L_e))

def _vert_stab_force(s):
    # Get the reynold's number
    re = _rho(s.p_W[2])*PARAM.c_v*s.u_inf/_mu(s.p_W[2])

    # Calculate stall angle and 0 lift angle (for NACA0012)
    a_s = min(0.0408*np.log(re) - 0.267, 0.297)
    a_l0 = 0.0

    # Get the lift and drag of the left wing
    # Note that alpha and beta are flipped for the vert stab
    cL_v = _cL(s.beta, s.alpha, a_s, PARAM.a0_v, a_l0)
    cDi_v = (PARAM.a0_v*(s.beta-a_l0))**2 / (np.pi*PARAM.ar_v)
    cDf_v = 0.074*re**(-0.2)
    L_v = 0.5*_rho(s.p_W[2])*0.5*PARAM.s_v*cL_v*s.u_inf**2
    D_v = 0.5*_rho(s.p_W[2])*0.5*PARAM.s_v*(cDi_v + cDf_v)*s.u_inf**2

    # Convert from lift and drag to world coords
    # Note that for vert stab, lift force is in -y instead of -z
    return s.R['PW'] @ s.R['FP'] @ (-D_v, -L_v, 0.0)

def  _body_force(s):
    # Body lift is treated as though body is low-lift symmetric airfoil
    # with no stall angle
    cL_alpha =  _cL(s.alpha, s.beta, np.inf, 0.5, 0.0)
    cL_beta =  _cL(s.beta, s.alpha, np.inf, 0.5, 0.0)
    L_alpha = 0.5*_rho(s.p_W[2])*PARAM.s_b*cL_alpha*s.u_inf**2
    L_beta = 0.5*_rho(s.p_W[2])*PARAM.s_b*cL_beta*s.u_inf**2

    # We assume that the of the body is independent of wind direction
    D = 0.5 * _rho(s.p_W[2]) * PARAM.s_b * PARAM.cDf_b * s.u_inf**2

    # Convert from lift and drag to world coords
    return s.R['PW'] @ s.R['FP'] @ (-D, -L_beta, -L_alpha)

def _prop_rps(P):
    # Prop RPS as a function of engine power
    # (0% power = 0.25*rps_max)
    # (75% power = 0.925*rps_max)
    # (100% power = rps_max)
    num = 0.0174*(71.65)**(P/PARAM.P_mx)*PARAM.P_rpm_max
    den = (71.65)**(P/PARAM.P_mx) + 3.177
    return num / den

def _eta(u_P, P):
    # Ideal prop efficiency based on advance ratio for a 20 degree angle prop
    J = u_P[0] / (_prop_rps(P) * PARAM.prop_diameter)
    if 0 <= J < 0.87:
        return -1.097*J*J + 1.908*J
    if 0.87 <= J <= 1.05:
        return -25.62*J*J + 44.57*J - 18.56
    return 0.0

def _prop_force(s):
    T = s.P * _eta(s.u_P, s.P) / s.u_inf

    # Convert from plane coords to world coords
    return s.R['PW'] @ (T, 0.0, 0.0)

def _pcL_W(s):
    # Adjust dCL based on AoA for nonsymmetric surfaces (wings, hstab, ele)
    # Assume wings, hstab, and ele are all NACA2412 airfoils
    # dcL moves fore by 10%c at 20 deg and aft by 10% at -20 deg AoA
    ddcLwl = min(max(0.286*PARAM.c_w*s.alpha_l, -.1*PARAM.c_w), .1*PARAM.c_w)
    ddcLwr = min(max(0.286*PARAM.c_w*s.alpha_r, -.1*PARAM.c_w), .1*PARAM.c_w)
    ddcLte = min(max(0.286*PARAM.c_te*s.alpha, -.1*PARAM.c_te), .1*PARAM.c_te)
    rwl_Pl = (-PARAM.dcL_w+ddcLwl, -0.5*PARAM.b_w, -PARAM.hcL_w)
    rwr_Pr = (-PARAM.dcL_w+ddcLwr,  0.5*PARAM.b_w, -PARAM.hcL_w)
    rte_P = (-PARAM.dcL_te+ddcLte, 0.0, -PARAM.hcL_te)
    rv_P = (-PARAM.dcL_v, 0.0, -PARAM.hcL_v)
    rb_P = (-PARAM.dcL_b, 0.0, -PARAM.hcL_b)
    return (s.R['PlW']@rwl_Pl, s.R['PrW']@rwr_Pr,
            s.R['PW']@rte_P, s.R['PW']@rv_P, s.R['PW']@rb_P)

def _tau_LD(s, Fwl, Fwr, Fte, Fv, Fb):
    pcLwl, pcLwr, pcLv, pcLte, pcLb = _pcL_W(s)
    return (np.cross(pcLwl, Fwl) + np.cross(pcLwr, Fwr) +
            np.cross(pcLte, Fte) + np.cross(pcLv, Fv) +
            np.cross(pcLb, Fb))

def _net_aero_force_torque(s):
    # Get the forces
    Fwl, Fwr,  = _wing_forces(s)
    Fte = _hori_stab_force(s)
    Fv = _vert_stab_force(s)
    Fb = _body_force(s)
    Fp = _prop_force(s)
    F_net = Fwl + Fwr + Fte + Fv + Fb + Fp

    # Get the net torque from the lift and drag of the surfaces
    tau_LD = _tau_LD(s, Fwl, Fwr, Fte, Fv, Fb)
    return F_net, tau_LD

def _state(plane, s):
    state = {'h' : s.p_W[2],
             'u_inf' : s.u_inf,
             'alpha' : s.alpha,
             'beta' : s.beta,
             'omega_psi' : -plane.state.omega_in_body[2],
             'omega_theta' : -plane.state.omega_in_body[1],
             'omega_phi' : plane.state.omega_in_body[0],
             'psi' : s.psi,
             'theta' : s.theta,
             'phi' : s.phi,}
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

def _update_vis_env(proj, plane, s, prev_states, shake):
    # Set the elevator deflection
    plane.joints['fuselage_to_elevator'].set_state(angle = s.delta)

    # Update the prop speed
    omega = _prop_rps(s.P)*np.pi*2.0
    plane.joints['fuselage_to_nosecone'].set_state(omega=omega)

    # Position the camera
    if shake > 0.0:
        cz = shake*(np.mean(prev_states['h'][-100:])-s.p_W[2])
        proj.visualizer.set_cam_position((10*np.sin(s.psi),
                                          10*np.cos(s.psi),
                                          cz + 1.25))
        proj.visualizer.set_cam_target((0, 0, cz))

    # Rotate the earth according to the forward velocity,
    # and move the earth according to the altitude
    proj.visualizer.set_transform('ground',
                                  pitch = s.earth_pitch,
                                  roll = s.earth_roll,
                                  scale = (2*R_PLANET, )*3,
                                  position=(0,0,-R_PLANET-s.p_W[2]))

    # Increase render distance to the horizon
    far = max(1.1*np.sqrt(2*R_PLANET*s.p_W[2]), 0.1005*R_PLANET)
    proj.visualizer.set_cam_frustum(far=far)

def _sim_loop(controller, program_nmuber, proj, plane, **kwargs):
    # Generate a set of turbulence parameters based on the selected seed
    turb = _gen_turb_param(kwargs['turb_mag'], kwargs['seed'])

    # Build a structure to track and update all hand updated sim variables
    simvars = _SimVars(kwargs['state0'], kwargs['input0'], proj.simulator.dt)

    # Make structure to hold simulation data
    data = _SimData()

    # Reset the project to its initial state.
    proj.reset()

    # Run a simulation loop
    start = now()
    while proj.simtime <= kwargs['duration']:

        # Get the state of the system
        state = _state(plane, simvars)
        h_des = _h_des(program_nmuber, proj.simtime, kwargs['state0']['h'])

        # Crash condition (will strike ground in 0.5 seconds)
        if state['h'] - 0.5*simvars.u_W[2] <= 0.0:
            break

        # Get the controller inputs
        inputs_des = controller(state, h_des)

        # Update the data
        input_args = (simvars.delta, simvars.P, inputs_des[0], inputs_des[1])
        data.step(proj.simtime, state, input_args, h_des)

        # Calculate the net force and torque on the plane
        F_aero_net, tau_aero_net = _net_aero_force_torque(simvars)
        F_aero_net += _turbulence(turb, proj.simtime)

        # Apply only the torque. We ignore the forces
        # because motion is handled by moving the planet instead of the plane.
        plane.apply_torque(tau_aero_net)

        # Update the visuals
        if kwargs['real_time']:
            _update_vis_env(proj, plane, simvars, data, kwargs['shake'])

        # Take a simulation step
        simvars.step(F_aero_net, state, inputs_des[0], inputs_des[1])
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
            The initial airspeed in meters/second. The default value is 47.82
        alpha : float, optional
            The initial angle of attack in radians. The default value
            is 0.05923
        omega_theta : float, optional
            The initial pitching rate in radians/second. The default value is 0
        theta : float, optional
            The initial pitch angle in radians. The default value is 0.05923
        delta : float, optional
            The initial elevator deflection angle in radians. The default
            value is 0.06592
        P : float, optional
            The initial power setting in KW. The default value is 62160.
        seed : int, optional
            The seed of the random number generator used for the simulation.
            The default is 2357136050
        turbulence : float, optional
            The magnitude of the turbulent wind in N. The default is 0
        shake : float, optional
            The magnitude by which plane accelerations are visualized. The
            default is 0.5. Set to 0.0 for free camera movement.

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
    m_e = np.array([0.0, 47.82, 0.05923, 0.0, 0.05923])
    n_e = np.array([0.06592, 62160.])
    x_des = np.array([h_des, 0.0, 0.0, 0.0, 0.0])
    K = np.array([[ 8.801e-03,  2.748e-03, -1.720e+00, 1.287e+00,  1.639e+00],
                  [ 2.730e+02,  1.469e+03, -5.856e+03, 1.563e+02,  6.134e+03]])
    m = np.array([state['h'],
                  state['u_inf'],
                  state['alpha'],
                  state['omega_theta'],
                  state['theta'],])
    n = -K@(m-m_e-x_des) + n_e
    return (n[0], n[1])

if __name__ ==  "__main__":
    data = run(ctrlr, 0, shake=0, t=20, phi=np.deg2rad(10))
