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

R_PLANET = 637100
@dataclass()
class _PlaneParams():
    params : dict

    def __init__(self):
        self.params = {}

        # Wing param
        self.params['a0_w'] = 5.01          # cL slope wrt AoA [1/rad]
        self.params['s_w'] = 16.2           # Projected top-down area [m^2]
        self.params['c_w'] = 1.49           # Mean aerodynamic chord [m]
        self.params['ar_w'] = 7.52          # Aspect ratio [-]
        self.params['b_w'] = 10.9           # Span [m]
        self.params['dihedral_w'] = 0.0297  # Diheadral angle [rad]
        self.params['d_eta_d_alpha'] = 0.25 # Downwash angle slope wrt AoA [-]

        # Horizontal stab param
        self.params['a0_h'] = 4.817 # cL slope wrt AoA [1/rad]
        self.params['s_h'] = 2.00   # Projected top-down area [m^2]
        self.params['ar_h'] = 6.32  # Aspect ratio [-]

        # Elevator param
        self.params['a0_e'] = 5.230 # cL slope wrt AoA [1/rad]
        self.params['s_e'] = 1.35   # Projected top-down area [m^2]
        self.params['ar_e'] = 9.37  # Aspect ratio [-]

        # Combined horizontal and elevator parameters
        self.params['c_he'] = 0.942 # Mean aerodynamic chord [m]

        # Combined vertical stab and rudder parameters
        # Jone's theory estimate for a0
        self.params['a0_v'] = 1.63 # cL slope wrt AoA [1/rad]
        self.params['s_v'] = 1.73  # Projected side area [m^2]
        self.params['ar_v'] = 1.04 # Aspect ratio [-]
        self.params['c_v'] = 1.17  # Mean aerodynamic chord [m]

        # Body parameters
        self.params['s_b'] = 5.59   # Project side area [m^2]
        self.params['mass'] = 964.0 # Mass [kg]

        # Distance from CoM to wing and tail center of lift (at 0 AoA)
        # positive in front of, to the right of, and below CoM
        self.params['x_w'] = -0.156  # Axial distance (wing) [m]
        self.params['y_w'] = 4.05    # Lateral distance (right wing) [m]
        self.params['z_w'] = -0.971  # Vertical distance (wing) [m]
        self.params['x_he'] = -4.59  # Axial distance (hori stab + ele) [m]
        self.params['y_he'] = 0.0    # Lateral distance (hori stab + ele) [m]
        self.params['z_he'] = 0.0288 # Vertical distance (hori stab + ele) [m]
        self.params['x_v'] = -4.76   # Axial distance (v stab + rud) [m]
        self.params['y_v'] = 0.0     # Lateral distance (v stab + rud) [m]
        self.params['z_v'] = -0.260  # Vertical distance (v stab + rud) [m]
        self.params['x_b'] = -0.608  # Axial distance (fuselage) [m]
        self.params['y_b'] = 0.0     # Lateral distance (fuselage) [m]
        self.params['z_b'] = -0.127  # Vertical distance (fuselage) [m]

        # Input limits
        self.params['delta_mn'] = -0.332 # Max downward deflection of ele [rad]
        self.params['delta_mx'] = 0.384  # Max upward deflection of ele [rad]
        self.params['delta_rate'] = 0.26 # Max deflection rate of ele [rad/s]
        self.params['P_mx'] = 1.342e5    # Max engine power setting [kW]
        self.params['P_rate'] = 3.4e4    # Max engine power setting rate [kW/s]

        # Prop sizing
        self.params['P_rpm_max'] = 2700.0     # RPM of prop at 100% power [rpm]
        self.params['prop_diameter'] = 1.905  # Diameter of prop [m]

        # Moment stability derivatives
        self.params['cnr'] = -0.099 # Yaw damping wrt yaw rate
        self.params['cnp'] = -0.03  # Yaw damping wrt roll rate
        self.params['cmq'] = -12.4  # Pitch damping wrt pitch rate
        self.params['clr'] = 0.096  # Roll damping wrt yaw rate
        self.params['clp'] = -0.47  # Roll damping wrt roll rate

    def __getattr__(self, key):
        return self.params[key]
PARAM = _PlaneParams()

def _RYZ(y, z):
    cy, cz = np.cos(y), np.cos(z)
    sy, sz = np.sin(y), np.sin(z)
    return np.array([[cy*cz, -cy*sz, sy ],
                     [sz,     cz,    0.0],
                     [-sy*cz, sy*sz, cy ]])

def _RXYZ(x, y, z):
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    return np.array([[cy*cz,          -cy*sz,           sy   ],
                     [cx*sz+sx*sy*cz,  cx*cz-sx*sy*sz, -sx*cy],
                     [sx*sz-cx*sy*cz,  sx*cz+cx*sy*sz,  cx*cy]])

def _cross(va, vb, ret_np=True):
    a1, a2, a3 = va
    b1, b2, b3 = vb
    if ret_np:
        return np.array((a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1))
    return (a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1)

def _norm3(v3):
    return np.sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2])

def _g(h):
    # Newtonian gravity
    return 3.986025446e14 / ((6.371e6+h)*(6.371e6+h))

def _rho(h):
    # Ideal gas law applied to barometric formula
    return 1.225*np.exp(-h/10363.)

def _T(h):
    # Standard atmosphere temperature model
    if h < 10000:
        return 288.2 - 0.00649*h
    return 223.3

def _mu(h):
    # Sutherland's law
    return 1.458e-6*_T(h)**(1.5) / (_T(h) + 110.4)

class _SimVars():
    R : dict
    state : dict
    params : dict
    _dt : float

    def __init__(self, state0, input0, dt):
        # Build the empty state and rotation dicts
        self.R = {}
        self.state = {}
        self.params = {}

        # Define the time step
        self._dt = dt

        # Apply the initial euler angles
        self._update_euler_angs(state0)

        # Get the rotation matrices that are determined by euler angles
        # and flow angles at center of mass
        self._update_plane_rot_mats()

        # Set the flow angles at the center of mass and get the
        # relevant rotations
        self.state['alpha'] = state0['alpha']
        self.state['beta'] = state0['beta']
        self.R['CoM_F'] = _RYZ(self.state['alpha'], self.state['beta'])
        self.R['F_CoM'] = self.R['CoM_F'].T

        # Set the initial world velocity and position
        R_F_W = self.R['CoM_W'] @ self.R['F_CoM']
        self.state['u_W'] = R_F_W @ (state0['u_inf'], 0.0, 0.0)
        self.state['p_W'] = np.array( (0.0, 0.0, state0['h']) )

        # Set the initial inputs
        self.state['delta'] = input0['delta']
        self.state['P'] = input0['P']

        # Set the rotations of the planet to 0
        self.state['earth_pitch'] = 0.0
        self.state['earth_roll'] = 0.0

        # Calculate the gravity, density, temperature, and viscosity
        self._update_params()

        # Calculate the local flow velocity at the center of mass
        self._update_velocities()

        # Calculate the effective flow angles at each of the surfaces
        self._update_local_flow_angs()

        # Get the rotations between local surface flow and the plane
        self._update_flow_rot_mats()

    def __getattr__(self, key):
        if key.startswith('R_'):
            return self.R[key[2:]]
        elif key.startswith('param_'):
            return self.params[key[6:]]
        return self.state[key]

    def _update_euler_angs(self, state):
        self.state['omega_psi'] = state['omega_psi']
        self.state['omega_theta'] = state['omega_theta']
        self.state['omega_phi'] = state['omega_phi']
        self.state['psi'] = state['psi']
        self.state['theta'] = state['theta']
        self.state['phi'] = state['phi']

    def _update_plane_rot_mats(self):
        r, p, y = self.state['phi'], self.state['theta'], self.state['psi']
        self.R['W_CoM'] = _RXYZ(r, np.pi-p, np.pi+y)
        self.R['CoM_W'] = self.R['W_CoM'].T
        self.R['W_WL'] = _RXYZ(r+PARAM.dihedral_w, np.pi-p, np.pi+y)
        self.R['WL_W'] = self.R['W_WL'].T
        self.R['W_WR'] = _RXYZ(r-PARAM.dihedral_w, np.pi-p, np.pi+y)
        self.R['WR_W'] = self.R['W_WR'].T
        self.R['W_HE'] = self.R['W_CoM']
        self.R['HE_W'] = self.R['CoM_W']
        self.R['W_V'] = self.R['W_CoM']
        self.R['V_W'] = self.R['CoM_W']
        self.R['W_B'] = self.R['W_CoM']
        self.R['B_W'] = self.R['CoM_W']

    def _apply_force(self, F_aero_net):
        # Get the net force by adding gravity
        F_gravity = np.array([0.0, 0.0, -self.params['g']*PARAM.mass])
        F_net = F_aero_net + F_gravity

        # Apply accelerations and velocities
        self.state['u_W'] += (F_net / PARAM.mass) * self._dt
        self.state['p_W'] += self.state['u_W'] * self._dt

    def _update_inputs(self, delta_des, P_des):
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

    def _update_world_rot(self):
        d = -self.state['u_W'][0]*self._dt / (self.state['p_W'][2]+R_PLANET)
        self.state['earth_pitch'] += d
        d = self.state['u_W'][1]*self._dt / (self.state['p_W'][2]+R_PLANET)
        self.state['earth_roll'] += d

    def _update_params(self):
        self.params['g'] = _g(self.state['p_W'][2])
        self.params['rho'] = _rho(self.state['p_W'][2])
        self.params['T'] = _T(self.state['p_W'][2])
        self.params['mu'] = _mu(self.state['p_W'][2])

    def _update_velocities(self):
        # Update the local flow velocity at the center of mass
        self.state['u_CoM'] = self.R['W_CoM'] @ self.state['u_W']

        # Update the flow velocity and dynamic pressure
        self.state['u_inf'] = _norm3(self.state['u_W'])

    def _update_alpha_beta(self, ):
        # Update the center of mass local flow angles
        self.state['alpha'] = np.arctan2(self.state['u_CoM'][2],
                                         self.state['u_CoM'][0])
        self.state['beta'] = np.arcsin((self.state['u_CoM'][1] /
                                        self.state['u_inf']))

    def _u_LOC(self, O_LOC_CoM, R_CoM_LOC):
        # Get the angular rate of the plane in the plane coordinates
        omega_CoM_CoM = (self.state['omega_phi'],
                         self.state['omega_theta'],
                         self.state['omega_psi'])

        # Get the induced velocity at the origin of the local flow frame
        # in the local coordinates
        i_LOC = R_CoM_LOC @ _cross(omega_CoM_CoM, O_LOC_CoM, ret_np=False)

        # Return the flow velocity in local coordinates
        return R_CoM_LOC @ self.state['u_CoM'] + i_LOC

    def _update_local_flow_angs(self):
        # Get the flow velocity at each of the surfaces in the surfaces'
        # own coordinates
        # Note, h stab, v stab, and body coordinate orientations all match
        # the plane coordinate directions
        u_WL = self._u_LOC((PARAM.x_w, -PARAM.y_w, PARAM.z_w),
                           self.R['W_WL']@self.R['CoM_W'])
        u_WR = self._u_LOC((PARAM.x_w, PARAM.y_w, PARAM.z_w),
                           self.R['W_WR']@self.R['CoM_W'])
        eye3 = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
        u_HE = self._u_LOC((PARAM.x_he, PARAM.y_he, PARAM.z_he), eye3)
        u_V = self._u_LOC((PARAM.x_v, PARAM.y_v, PARAM.z_v), eye3)
        u_B = self._u_LOC((PARAM.x_b, PARAM.y_b, PARAM.z_b), eye3)

        # Save the magnitude of the local surface velocities
        self.state['u_WL'] = _norm3(u_WL)
        self.state['u_WR'] = _norm3(u_WR)
        self.state['u_HE'] = _norm3(u_HE)
        self.state['u_V'] = _norm3(u_V)
        self.state['u_B'] = _norm3(u_B)

        # Update the local flow angles at each of the surfaces
        self.state['alpha_wl'] = np.arctan2(u_WL[2], u_WL[0])
        self.state['beta_wl'] = np.arcsin(u_WL[1] / self.state['u_WL'])
        self.state['alpha_wr'] = np.arctan2(u_WR[2], u_WR[0])
        self.state['beta_wr'] = np.arcsin(u_WR[1] / self.state['u_WR'])
        self.state['alpha_he'] = np.arctan2(u_HE[2], u_HE[0])
        self.state['beta_he'] = np.arcsin(u_HE[1] / self.state['u_HE'])
        self.state['alpha_v'] = np.arctan2(u_V[2], u_V[0])
        self.state['beta_v'] = np.arcsin(u_V[1] / self.state['u_V'])
        self.state['alpha_b'] = np.arctan2(u_B[2], u_B[0])
        self.state['beta_b'] = np.arcsin(u_B[1] / self.state['u_B'])

    def _update_flow_rot_mats(self):
        self.R['CoM_F'] = _RYZ(self.state['alpha'], self.state['beta'])
        self.R['F_CoM'] = self.R['CoM_F'].T
        self.R['WL_FWL'] = _RYZ(self.state['alpha_wl'], self.state['beta_wl'])
        self.R['FWL_WL'] = self.R['WL_FWL'].T
        self.R['WR_FWR'] = _RYZ(self.state['alpha_wr'], self.state['beta_wr'])
        self.R['FWR_WR'] = self.R['WR_FWR'].T
        self.R['HE_FHE'] = _RYZ(self.state['alpha_he'], self.state['beta_he'])
        self.R['FHE_HE'] = self.R['HE_FHE'].T
        self.R['V_FV'] = _RYZ(self.state['alpha_v'], self.state['beta_v'])
        self.R['FV_V'] = self.R['V_FV'].T
        self.R['B_FB'] = _RYZ(self.state['alpha_b'], self.state['beta_b'])
        self.R['FB_B'] = self.R['B_FB'].T

    def step(self, F_aero_net, state, delta_des, P_des):
        # Step 1
        self._update_euler_angs(state)
        self._update_plane_rot_mats()
        self._apply_force(F_aero_net)
        self._update_inputs(delta_des, P_des)
        self._update_world_rot()
        self._update_params()

        # Step 2
        self._update_velocities()

        # Step 3
        self._update_alpha_beta()
        self._update_local_flow_angs()

        # Step 4
        self._update_flow_rot_mats()

class _SimData():
    data : dict

    def __init__(self):
        self.data = {'time':[],
                     'h':[],
                     'u_inf':[],
                     'alpha':[],
                     'beta':[],
                     'omega_psi':[],
                     'omega_theta':[],
                     'omega_phi':[],
                     'psi':[],
                     'theta':[],
                     'phi':[],
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
        self.data['beta'].append(float(state['beta']))
        self.data['omega_psi'].append(float(state['omega_psi']))
        self.data['omega_theta'].append(float(state['omega_theta']))
        self.data['omega_phi'].append(float(state['omega_phi']))
        self.data['psi'].append(float(state['psi']))
        self.data['theta'].append(float(state['theta']))
        self.data['phi'].append(float(state['phi']))
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
                'shake' : kwargs.get('shake', 3.0),
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
                               tex_repeat=[n_repeat, n_repeat],
                               emissive_color=(0.15, 0.15, 0.15),
                               position=(0.0, 0.0, -R_PLANET-state0['h']),)

def _load_sky(proj):
    proj.visualizer.add_object('skybox',
                               assets['sphere_1_center_origin.stl'],
                               scale=(R_PLANET*5*2, )*3,
                               tex_path=assets['skybox_day.jpg'],
                               roll=1.5708,
                               position=(0.0, 0.0, -R_PLANET))

def _load_vis_env(proj, state0):
    # Increase render distance to the skybox
    proj.visualizer.set_cam_frustum(far=5.1*R_PLANET)

    # Load the ground and sun
    _load_planet(proj, state0)
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
    R = _RXYZ(state0['phi'], np.pi-state0['theta'], np.pi+state0['psi'])
    proj.visualizer.set_cam_position(R.T@(-25, 0, -5))
    proj.visualizer.set_cam_target((0, 0, 0))
    proj.visualizer.set_cam_zoom(2.5)

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
    proj = Project(keyboard=False, visualizer=kwargs['real_time'],
                   simulator_gravity = (0.,0.,0.),
                   simulator_dt = 1.0/250.0,)

    # Load the visual environment
    if kwargs['real_time']:
        _load_vis_env(proj, kwargs['state0'])

    # Load the plane, set its initial condition, and look at it
    plane = proj.load_urdf(assets['cessna172.urdf'], fixed=False)
    _set_init_conds(plane, kwargs['state0'], kwargs['input0'])

    # Remove all joint friction
    for joint in plane.joints.values():
        joint.set_dynamics(damping=0.0)

    # Set aii air resistance to 0
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

def _cL(alpha, beta, a_s, a_0, a_l0):
    alpha_abs = abs(alpha)

    # Add random buffeting when within +- 3 degrees of stall
    buffet = 0.0
    if abs(alpha_abs - a_s) <= 0.0524:
        buffet = -0.3 * ((alpha_abs - a_s) + 0.0524) * 9.55 * np.random.rand()

    # No stall
    if alpha_abs <= a_s:
        # Add a cos beta as a first order crab angle correction term
        return np.cos(beta) * a_0 * (alpha - a_l0) + buffet

    # Positive stalling region
    # Assume stall starts at a_s where cL decreases to 50% at 3 deg past a_s
    if 0 < alpha < a_s + 0.0524:
        a = 182.4*a_0*(a_l0-a_s-0.1047)
        b = -364.8*a_0*(a_l0*a_s-a_s**2-0.1047*a_s-0.0027)
        c = 182.4*a_0*(a_l0*(a_s**2-0.0055)-a_s**2*(a_s+0.1047))

        # Add a cos beta as a first order crab angle correction term
        return np.cos(beta) * (a * alpha**2 + b * alpha + c) + buffet

    # Negative stalling region
    # Assume stall starts at -a_s where cL decreases to 50% at 3 deg past -a_s
    if -a_s - 0.0524 < alpha < 0:
        a = 182.4*a_0*(a_l0+a_s+0.1047)
        b = 364.8*a_0*(a_l0*a_s+a_s**2+0.1047*a_s+0.0027)
        c = 182.4*a_0*(a_l0*(a_s**2-0.0055)+a_s**2*(a_s+0.1047))

        # Add a cos beta as a first order crab angle correction term
        return np.cos(beta) * (a * alpha**2 + b * alpha + c) + buffet

    # Complete stall
    # Any angle of attack outside of [-a_s-3deg, a_s+3deg] produces no lift
    return 0.0

def _L_and_D(rho, mu, c, s, ar, a0, typ, h, u_inf, alpha, beta):
    # Get the reynold's number and dynamic pressure
    re = rho * c * u_inf / mu
    q = 0.5 * rho * u_inf * u_inf

    # Calculate stall angle (Assume all airfoils stall at around 17 deg)
    # though very low reynolds number result in lower stall angles all the
    # way to 10 deg
    re_e = min(max(re, 50_000), 1_000_000)
    a_s=min(max(-2.333e-18*re_e**3+3.674e-12*re_e**2-3.499e-7*re_e+0.0086,0),1)
    a_s = a_s*np.deg2rad(7) + np.deg2rad(10)

    # Calculate the 0 lift angle of attack where, for all airfoils, at very
    # low reynold's number, a_l0 is 0, and at normal flight reynold's numbers
    # a_l0 is whatever is defined for a specific airfoil
    a_l0 = min(max(-1.11e-12*re_e*re_e + 2.22e-6*re_e - 0.108, 0.0), 1.0)
    if typ == 'NACA2412':
        a_l0 *= -0.0436
    elif typ == 'NACA2412i':
        a_l0 *= 0.0436
    elif typ == 'NACA0012':
        a_l0 = 0.0
    else:
        a_l0 = 0.0

    # Get the lift and drag coefficients
    cL = _cL(alpha, beta, a_s, a0, a_l0)
    cDi = cL*cL / (np.pi * ar)

    # Turbulent flow of thin plate skin friction
    cDf_skin = 0.074 * re**(-0.2)

    # Form drag is that of streamlined body below stall, then
    # transitions to that of a detached flow, rotated, thin plate after stall
    cDf_form = (0.006 if abs(alpha) < a_s else
                abs(1.8*np.sin(alpha)) - 0.284*np.cos(alpha))

    # Get the total lift and drag
    L = 0.5 * s * cL * q
    D = 0.5 * s * (cDi + cDf_skin + cDf_form) * q
    return L, D

def _wing_forces(s):
    # Get the lift and drag of the left wing (in local left wing flow)
    LD_wl = _L_and_D(s.param_rho, s.param_mu,
                     PARAM.c_w, PARAM.s_w, PARAM.ar_w, PARAM.a0_w, 'NACA2412',
                     s.p_W[2], s.u_WL, s.alpha_wl, s.beta_wl)

    # Get the lift and drag of the right wing (in local right wing flow)
    LD_wr = _L_and_D(s.param_rho, s.param_mu,
                     PARAM.c_w, PARAM.s_w, PARAM.ar_w, PARAM.a0_w, 'NACA2412',
                     s.p_W[2], s.u_WR, s.alpha_wr, s.beta_wr)

    # Convert from local flow coords to world coords
    F_wl_W = s.R_WL_W @ s.R_FWL_WL @ (-LD_wl[1], 0.0, -LD_wl[0])
    F_wr_W = s.R_WR_W @ s.R_FWR_WR @ (-LD_wr[1], 0.0, -LD_wr[0])
    return F_wl_W, F_wr_W

def _hori_stab_force(s):
    # Get the downwash angle induced from the wings
    eta = PARAM.d_eta_d_alpha * 0.5 * (s.alpha_wl + s.alpha_wr)

    # Get the lift and drag of the h stab (in local h stab flow)
    LD_h = _L_and_D(s.param_rho, s.param_mu,
                    PARAM.c_he, PARAM.s_h, PARAM.ar_h, PARAM.a0_h, 'NACA2412i',
                    s.p_W[2], s.u_HE, s.alpha_he-eta, s.beta_he)

    # Get the lift and drag of the elevator (in local h stab flow)
    LD_e = _L_and_D(s.param_rho, s.param_mu,
                    PARAM.c_he, PARAM.s_e, PARAM.ar_e, PARAM.a0_e, 'NACA2412i',
                    s.p_W[2], s.u_HE, s.alpha_he-eta-s.delta, s.beta_he)

    # Convert from local flow coords to world coords
    F_he_W = s.R_HE_W @ s.R_FHE_HE @ (-LD_h[1]-LD_e[1], 0.0, -LD_h[0]-LD_e[0])
    return F_he_W

def _vert_stab_force(s):
    # Get the lift and drag of the v stab (in local v stab flow)
    LD_v = _L_and_D(s.param_rho, s.param_mu,
                    PARAM.c_v, PARAM.s_v, PARAM.ar_v, PARAM.a0_v, 'NACA0012',
                    s.p_W[2], s.u_V, s.beta_v, s.alpha_v)

    # Convert from local flow coords to world coords
    # Note that for vert stab, lift force is in -y instead of -z
    F_v_W = s.R_V_W @ s.R_FV_V @ (-LD_v[1], -LD_v[0], 0.0)
    return F_v_W

def  _body_force(s):
    # Body lift is treated as though body is low-lift symmetric airfoil
    # with no stall angle
    a_0 = 0.5
    a_s = np.inf
    a_l0 = 0.0
    cL_alpha = _cL(s.alpha_b, s.beta_b, a_s, a_0, a_l0)
    cL_beta = _cL(s.beta_b, s.alpha_b, a_s, a_0, a_l0)
    q = 0.5 * s.param_rho * s.u_B * s.u_B
    L_alpha = PARAM.s_b * cL_alpha * q
    L_beta = PARAM.s_b * cL_beta * q

    # Assume body acts approximately like streamline body with cD_tot = 0.045
    D = PARAM.s_b * 0.045 * q

    # Convert from local flow coords to world coords
    # Note that for body, side forces are also generated
    F_b_W = s.R_B_W @ s.R_FB_B @ (-D, -L_beta, -L_alpha)
    return F_b_W

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
    # Convert from plane coords to world coords
    T = s.P * _eta(s.u_CoM, s.P) / s.u_inf
    return s.R_CoM_W @ (T, 0.0, 0.0)

def _r_cL_W(s):
    # Adjust dCL based on AoA for nonsymmetric surfaces (wings, hstab, ele)
    # Assume wings, hstab, and ele are all NACA2412 airfoils
    # x_cL moves fore by 10%c at 20 deg AoA and aft by 10%c at -20 deg AoA
    dx_wl = min(max(0.286*PARAM.c_w*s.alpha_wl, -.1*PARAM.c_w), .1*PARAM.c_w)
    dx_wr = min(max(0.286*PARAM.c_w*s.alpha_wr, -.1*PARAM.c_w), .1*PARAM.c_w)
    dx_he = min(max(0.286*PARAM.c_he*s.alpha_he, -.1*PARAM.c_he),.1*PARAM.c_he)
    r_wl_W = s.R_WL_W @ (PARAM.x_w+dx_wl, -PARAM.y_w,  PARAM.z_w)
    r_wr_W = s.R_WR_W @ (PARAM.x_w+dx_wr,  PARAM.y_w,  PARAM.z_w)
    r_he_W = s.R_HE_W @ (PARAM.x_he+dx_he, PARAM.y_he, PARAM.z_he)
    r_v_W = s.R_V_W @ (PARAM.x_v, PARAM.y_v, PARAM.z_v)
    r_b_W = s.R_B_W @ (PARAM.x_b, PARAM.y_b, PARAM.z_b)
    return (r_wl_W, r_wr_W, r_he_W, r_v_W, r_b_W)

def _tau_LD(s, F_wl_W, F_wr_W, F_he_W, F_v_W, F_b_W):
    r_wl_W, r_wr_W, r_he_W, r_v_W, r_b_W = _r_cL_W(s)
    return (_cross(r_wl_W, F_wl_W) + _cross(r_wr_W, F_wr_W) +
            _cross(r_he_W, F_he_W) + _cross(r_v_W, F_v_W) +
            _cross(r_b_W, F_b_W))

def _net_aero_force_torque(s):
    # Get the forces
    F_wl_W, F_wr_W  = _wing_forces(s)
    F_he_W = _hori_stab_force(s)
    F_v_W = _vert_stab_force(s)
    F_b_W = _body_force(s)
    F_p_W = _prop_force(s)
    F_net_W = F_wl_W + F_wr_W + F_he_W + F_v_W + F_b_W + F_p_W

    # Get the net torque from the lift and drag of the surfaces
    # and from Euler rate-based damping moments
    tau_LD_W = _tau_LD(s, F_wl_W, F_wr_W, F_he_W, F_v_W, F_b_W)
    return F_net_W, tau_LD_W

def _state(plane, s):
    pybullet_state = plane.state
    state = {'h' : s.p_W[2],
             'u_inf' : s.u_inf,
             'alpha' : s.alpha,
             'beta' : s.beta,
             'omega_psi' : -pybullet_state.omega_in_body[2],
             'omega_theta' : -pybullet_state.omega_in_body[1],
             'omega_phi' : pybullet_state.omega_in_body[0],
             'psi' : -pybullet_state.ypr[0],
             'theta' : -pybullet_state.ypr[1],
             'phi' : pybullet_state.ypr[2],}
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
    return np.array((tl[0]+ts[0], tl[1]+ts[1], tl[2]+ts[2]))

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
        cz_a = 400*shake*(prev_states['alpha'][-2:][0]-s.alpha)
        cz_b = 400*shake*(prev_states['beta'][-2:][0]-s.beta)
        cz_h = shake*(np.mean(prev_states['h'][-50:])-s.p_W[2])
        cz_rand = shake*0.0001*(np.random.rand(3)*2-1)
        p = s.R_CoM_W @( 0, -cz_b, -cz_a-cz_h) + cz_rand
        proj.visualizer.set_cam_target(p)

    # Rotate the earth according to the forward velocity,
    # and move the earth according to the altitude
    proj.visualizer.set_transform('ground',
                                  pitch = s.earth_pitch,
                                  roll = s.earth_roll,
                                  position=(0,0,-R_PLANET-s.p_W[2]))

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
        if kwargs['turb_mag'] != 0.0:
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
            default is 3.0. Set to 0.0 for free camera movement.

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
    data = run(ctrlr, 0)
