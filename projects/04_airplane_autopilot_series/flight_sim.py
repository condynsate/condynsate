# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
# pylint: disable=invalid-name
# pylint: disable=pointless-string-statement
"""
This module implements the backend for the airplane project.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

####################################################################################################
#DEPENDENCIES
####################################################################################################
import math
import random
from collections import deque
from plane_parameters import Params, Cessna172
STALL_DEV = math.radians(4) # Rad after stall angle where stall developments ends, full stall starts
EPSILON = 0.001

####################################################################################################
#MATH OPERATION FUNCTIONS
####################################################################################################
def _RYZ(y, z):
    cy, cz = math.cos(y), math.cos(z)
    sy, sz = math.sin(y), math.sin(z)
    return ((cy*cz, -cy*sz, sy ),
            (sz,     cz,    0.0),
            (-sy*cz, sy*sz, cy ))

def _RBW(r, p, y):
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    return ((cp*cy, cy*sp*sr-cr*sy, cr*cy*sp+sr*sy),
            (cp*sy, cr*cy+sp*sr*sy, cr*sp*sy-cy*sr),
            (-sp,   cp*sr,          cp*cr))

def _cross(a, b):
    a0, a1, a2 = a
    b0, b1, b2 = b
    return (a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0)

def _norm3(v3):
    return math.sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2])

def _CoV(R_A_B, v_A):
    return (R_A_B[0][0]*v_A[0] + R_A_B[0][1]*v_A[1] + R_A_B[0][2]*v_A[2],
            R_A_B[1][0]*v_A[0] + R_A_B[1][1]*v_A[1] + R_A_B[1][2]*v_A[2],
            R_A_B[2][0]*v_A[0] + R_A_B[2][1]*v_A[1] + R_A_B[2][2]*v_A[2],)

def __mmij(A, B, i, j):
    return A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j]

def _mmul(A, B):
    return ((__mmij(A, B, 0, 0), __mmij(A, B, 0, 1), __mmij(A, B, 0, 2)),
            (__mmij(A, B, 1, 0), __mmij(A, B, 1, 1), __mmij(A, B, 1, 2)),
            (__mmij(A, B, 2, 0), __mmij(A, B, 2, 1), __mmij(A, B, 2, 2)),)

def _sum(*args):
    return tuple(map(sum, zip(*args)))

def _T(A):
    return tuple(tuple(row) for row in zip(*A))

####################################################################################################
#STANDARD ATMOSPHERIC MODEL FUNCTIONS
####################################################################################################
def _g(h):
    # Newtonian gravity
    return 3.986025446e14 / ((6.371e6+h)*(6.371e6+h))

def _rho(h):
    # Ideal gas law applied to barometric formula
    return 1.225*math.exp(-h/10363.)

def _temperature(h):
    # Standard atmosphere temperature model
    if h < 10000:
        return 288.2 - 0.00649*h
    return 223.3

def _mu(h):
    # Sutherland's law
    return 1.458e-6*_temperature(h)**(1.5) / (_temperature(h) + 110.4)

####################################################################################################
#FLIGHT MODEL CLASS
####################################################################################################
class _FlightModel:
    def __init__(self, params):
        self.p = params

    def _cL(self, angles, slopes, spans, alpha_L0, alpha_s):
        # Check if the surface spans the entire wing (within 1% precision)
        if abs(spans[0]-spans[1]) >= 0.01*spans[0]:
            # If it does not, switch to outboard-inboard lift calculation with
            # the surface always as outboard as possible (like with ailerons)
            cL_o, _, sc_o = self._cL(angles, (slopes[0], slopes[1]*spans[0]/spans[1]),
                                     (spans[1],)*2, alpha_L0, alpha_s)
            cL_i, _, sc_i = self._cL(angles, (slopes[0], 0.0),
                                     (spans[0]-spans[1],)*2, alpha_L0, alpha_s)

            # Effective coefficient of lift
            cL = (cL_o*spans[1] + cL_i*(spans[0]-spans[1])) / spans[0]

            # Center of lift calculation
            y_o = spans[0] - 0.5*spans[1]
            y_i = 0.5*(spans[0] - spans[1])
            try:
                return (cL, (cL_o*spans[1]*y_o + cL_i*(spans[0]-spans[1])*y_i) / (spans[0]*cL),
                        max(sc_o, sc_i))
            except ZeroDivisionError:
                return cL, 0.5*spans[0], max(sc_o, sc_i)

        # Get the no stall coefficient of lift and center of lift
        # Outside of stall, CoL is always half the span
        alpha_eff = angles[0] + slopes[1]*angles[2]
        cL = slopes[0]*(alpha_eff-alpha_L0)

        # No stall
        if abs(alpha_eff) < alpha_s:
            # Add a cos beta as a first order crab angle correction term
            cL = math.cos(angles[1])*cL
            return cL, 0.5*spans[0], 0

        # Add random buffeting while stall is developing
        # abs(alpha_eff) = alpha_s + STALL_DEV  -> Avg buff = -(1/10)*2*pi*(alpha_s + STALL_DEV)
        const = -4*math.pi*(alpha_s+STALL_DEV) / (10 * STALL_DEV)
        buffet = (abs(alpha_eff) - alpha_s) * random.random() * const

        # Positive stall developement region
        if alpha_s <= alpha_eff <= alpha_s + STALL_DEV:
            cL_stalling = cL * (1.0 + 4.0*(alpha_s-alpha_eff) / (10.0*STALL_DEV))
            cL = math.cos(angles[1])*(cL_stalling + buffet)
            return cL, 0.5*spans[0], 1

        # Negative stall developement region
        if -alpha_s - STALL_DEV <= alpha_eff <= -alpha_s:
            cL_stalling = cL * (1.0 + 4.0*(alpha_s+alpha_eff) / (10.0*STALL_DEV))
            cL = math.cos(angles[1])*(cL_stalling + buffet)
            return cL, 0.5*spans[0], 1

        # Complete stall
        return 0.0, 0.5*spans[0], 2

    def _alpha_s(self, re):
        # Calculate stall angle (Assume all airfoils stall at around 16 deg)
        # though very low reynolds number result in lower stall angles
        if re >= 1_000_000:
            return math.radians(16)
        return 0.0350*(math.log(min(max(re, 50_000), 1_000_000)) - 5.827)

    def _alpha_L0(self, re, typ):
        # Calculate the 0 lift angle of attack where, at low reynold's number, a_l0 is 0,
        # and at normal flight reynold's numbers a_l0 is airfoil specific
        if re >= 1_000_000:
            multiplier = 1.0
        elif re <= 50_000:
            return 0.0
        else:
            x = min(max(re, 50_000), 1_000_000)
            multiplier = min(max(0.3338*(math.log(x)-10.82), 0.0), 1.0)

        if typ == 'NACA2412':
            return -math.radians(3.5) * multiplier
        if typ == 'NACA2412i':
            return math.radians(3.5) * multiplier
        if typ == 'NACA0012':
            return 0.0
        return 0.0

    def _L_D(self, atmosphere, wing, state):
        # Get the reynold's number and dynamic pressure
        re = atmosphere['rho'] * wing['c'] * state['V'] / atmosphere['mu']
        q_S = 0.5 * atmosphere['rho'] * state['V'] * state['V'] * wing['S']
        cD = 0.0

        # Get the lift and induced drag
        angles = (state['alpha'], state['beta'], state['delta'])
        slopes = (wing['alpha_0'], wing['tau'])
        spans = (wing['b'], wing['b_s'])
        cL, CoL, stall_cond = self._cL(angles, slopes, spans,
                                       self._alpha_L0(re, wing['typ']), self._alpha_s(re))
        cD += cL*cL / (math.pi * wing['AR'])

        # Skin drag: Turbulent flow of thin plate skin friction
        cD += 0.074 * re**(-0.2)

        # Form drag is that of streamlined body below stall, then
        # transitions to that of a rotated, thin plate after stall
        if stall_cond == 0:
            cD += 0.006
        else:
            cD += max((1.8*abs(math.sin(state['alpha'])) -
                       0.5*math.cos(state['alpha'])), 0.006)

        # Get the total lift and drag
        L = cL * q_S
        D = cD * q_S
        return L, D, CoL

    def _wing_forces(self, sim):
        # Build the arguments for the left wing
        half_wing = {'S' : 0.5 * self.p.S_w,
                     'AR' : self.p.AR_w,
                     'b' : 0.5*self.p.b_w,
                     'b_s' : 0.5*self.p.b_a,
                     'c' : self.p.c_w,
                     'alpha_0' : self.p.alpha_0_w,
                     'tau' : self.p.tau_a,
                     'typ' : self.p.typ_w, }
        state = {'V' : sim.V_wl,
                 'alpha' : sim.alpha_wl,
                 'delta' : sim.delta_a,
                 'beta' : sim.beta_wl}

        # Get the lift and drag of the left wing system (in local left wing flow)
        L, D, CoL_y = self._L_D({'rho' : sim.rho, 'mu' : sim.mu}, half_wing, state)
        CoL_wl_CoM = (self.p.x_w, -CoL_y, self.p.z_w)
        F_wl_CoM = _CoV(sim.R_FWL_CoM, (-D, 0.0, -L))

        # Get the lift and drag of the right wing system (in local right wing flow)
        state = {'V' : sim.V_wr,
                 'alpha' : sim.alpha_wr,
                 'delta' : -sim.delta_a,
                 'beta' : sim.beta_wr}
        L, D, CoL_y = self._L_D({'rho' : sim.rho, 'mu' : sim.mu}, half_wing, state)
        CoL_wr_CoM = (self.p.x_w, CoL_y, self.p.z_w)
        F_wr_CoM = _CoV(sim.R_FWR_CoM, (-D, 0.0, -L))

        # Calculate torques
        tau_wl_CoM = _cross(CoL_wl_CoM, F_wl_CoM)
        tau_wr_CoM = _cross(CoL_wr_CoM, F_wr_CoM)
        return _sum(F_wl_CoM, F_wr_CoM), _sum(tau_wl_CoM, tau_wr_CoM)

    def _h_stab_ele_forces(self, sim):
        # Get the downwash angle induced from the wings
        eta = self.p.d_eta_d_alpha * 0.5 * (sim.alpha_wl + sim.alpha_wr)

        # Build the arguments
        wing = {'S' : self.p.S_h,
                'AR' : self.p.AR_h,
                'b' : self.p.b_h,
                'b_s' : self.p.b_e,
                'c' : self.p.c_h,
                'alpha_0' : self.p.alpha_0_h,
                'tau' : self.p.tau_e,
                'typ' : self.p.typ_h, }
        state = {'V' : sim.V_he,
                 'alpha' : sim.alpha_he - eta,
                 'delta' : -sim.delta_e,
                 'beta' : sim.beta_he}

        # Get the lift and drag of the h stab system (in local h stab flow)
        L, D, _ = self._L_D({'rho' : sim.rho, 'mu' : sim.mu}, wing, state)
        CoL_CoM = (self.p.x_he, 0.0, self.p.z_he)

        # Convert from local flow coords to world coords
        F_CoM = _CoV(sim.R_FHE_CoM, (-D, 0.0, -L))
        return F_CoM, _cross(CoL_CoM, F_CoM)

    def _vert_stab_force(self, sim):
        # Build the arguments
        wing = {'S' : self.p.S_v,
                'AR' : self.p.AR_v,
                'b' : self.p.b_v,
                'b_s' : self.p.b_r,
                'c' : self.p.c_v,
                'alpha_0' : self.p.alpha_0_v,
                'tau' : self.p.tau_r,
                'typ' : self.p.typ_v, }
        state = {'V' : sim.V_v,
                 'alpha' : sim.alpha_v,
                 'delta' : sim.delta_r,
                 'beta' : sim.beta_v}

        # Get the lift and drag of the v stab system (in local v stab flow)
        L, D, _ = self._L_D({'rho' : sim.rho, 'mu' : sim.mu}, wing, state)
        CoL_CoM = (self.p.x_v, 0.0, self.p.z_v)

        # Convert from local flow coords to world coords
        F_CoM = _CoV(sim.R_FV_CoM, (-D, 0.0, -L))
        return F_CoM, _cross(CoL_CoM, F_CoM)

    def  _body_force(self, sim):
        # Body lift is treated as though body is low-lift symmetric airfoil
        # with no stall angle
        q_S = 0.5 * sim.rho * sim.V_b * sim.V_b * self.p.S_b
        L_alpha = 0.1 * sim.alpha_b * q_S
        L_beta = 0.1 * sim.beta_b * q_S

        # Assume body acts approximately like streamline body with cD_tot = 0.045
        D = 0.045 * q_S

        # Convert from local flow coords to world coords
        F_CoM = _CoV(sim.R_FB_CoM, (-D, -L_beta, -L_alpha))
        return F_CoM, _cross((self.p.x_b, 0.0, self.p.z_b), F_CoM)

    def prop_rps(self, sim):
        """
        Returns the propeller speed in revolutions per second.

        Parameters
        ----------
        sim : FlightSim
            A flight simulator from which the engine parameters and
            current engine power are extracted. These are used to calculate
            the propeller speed.

        Returns
        -------
        prop_rps : float
            The propeller speed in revolutions per second.

        """
        # Prop RPS as a function of engine power
        # (0% power = 0.25*rps_max)
        # (75% power = 0.925*rps_max)
        # (100% power = rps_max)
        const = (71.65)**(sim.P / self.p.P_max)
        return (0.0174 * const * self.p.rpm_max) / (const + 3.177)

    def _cT(self, sim, n):
        # Based on monotonic decrease, linear cT(J) curve (J is advance ratio)
        n_max = self.p.rpm_max/60.0
        n_75 = 0.925*self.p.rpm_max/60.0
        cT_max = self.p.T_max / (1.225*n_max**2*self.p.prop_D**4)
        slope = self.p.T_cruise / (0.9652*self.p.V_cruise*n_75*self.p.prop_D**3)
        slope -= cT_max*n_75*self.p.prop_D/self.p.V_cruise
        J = sim.v_CoM[0]/(n*self.p.prop_D)
        return min(max(slope*J + cT_max, 0.0), cT_max)

    def _prop_force(self, sim):
        # Calculate the thrust
        n = self.prop_rps(sim)
        return (sim.rho * n*n * self.p.prop_D**4 * self._cT(sim, n), 0.0, 0.0)

    def net_aero_force_torque(self, sim):
        """
        Gets the net aerodynamic forces and torque on the plane (about the
        center of mass) in world coordinates.

        Parameters
        ----------
        sim : FlightSim
            A flight simulator from which the current flight conditions,
            surface displacements, and engine settings are extracted.

        Returns
        -------
        F_W : 3 tuple of floats
            The net aerodynamic force (N) in world coords.
        tau_W : 3 tuple of floats
            The net aerodynamic torque (Nm) in world coords.
        LD_net : 2 tuple of floats
            The net lift and drag on the plane (referenced to the CoM flow). This
            value is already included in F_W and tau_W.

        """
        # Get the forces and torques
        F_w_CoM, tau_w_CoM  = self._wing_forces(sim)
        F_he_CoM, tau_he_CoM = self._h_stab_ele_forces(sim)
        F_v_CoM, tau_v_CoM = self._vert_stab_force(sim)
        F_b_CoM, tau_b_CoM = self._body_force(sim)
        F_p_CoM = self._prop_force(sim)

        # Calculate the net lift and drag with respect to flow at CoM
        F_surf_CoM = _sum(F_w_CoM, F_he_CoM, F_v_CoM, F_b_CoM)
        DL_net = _CoV(_T(sim.R_F_CoM), F_surf_CoM)

        # Get the net forces and torques
        F_net_CoM = _sum(F_surf_CoM, F_p_CoM)
        tau_net_CoM = _sum(tau_w_CoM, tau_he_CoM, tau_v_CoM, tau_b_CoM)

        # Get the net cL, cD, and cM
        q_S = 0.5 * sim.rho * sim.V_inf * sim.V_inf * self.p.S_w
        cL_net = -DL_net[2] / q_S
        cD_tot = -DL_net[0] / q_S
        cl_tot = tau_net_CoM[0] / (q_S * self.p.c_w)
        cm_tot = tau_net_CoM[1] / (q_S * self.p.c_w)
        cn_tot = tau_net_CoM[2] / (q_S * self.p.c_w)
        coeffs = (cL_net, cD_tot, cl_tot, cm_tot, cn_tot)

        # Change to world coordinates
        return _CoV(sim.R_CoM_W, F_net_CoM), _CoV(sim.R_CoM_W, tau_net_CoM), coeffs

###############################################################################
#FLIGHTSIM CLASS
###############################################################################
class FlightSim:
    """
    A flight simulator that tracks current surface and engine state of an
    airplane as well as the aerodynamic loads.

    Parameters
    ----------
    **kwargs

    Keyword Args
    ------------
    state0 : dictionary of initial state floats, optional
        A dictionary defining the initial state with the keys:
            omega_psi : yaw rate (rad / s). Default = 0
            omega_theta : pitch rate (rad / s). Default = 0
            omega_phi : roll rate (rad / s). Default = 0
            psi : yaw (rad). Default = 0
            theta : pitch (rad). Default = 0
            phi : roll (rad). Default = 0
            alpha : angle of attack (rad). Default = 0
            beta : sideslip angle (rad). Default = 0
            V_inf : Indicated airspeed (m/s). Default = 0
            h : Altitude (m). Default = 0
    input0 : dictionary of initial input floats, optional
        A dictionary defining the initial inputs with the keys:
            delta_e : elevator deflection (rad). Default = 0
            delta_r : rudder deflection (rad). Default = 0
            delta_a : aileron deflection (rad). Default = 0
            P : engine power (Watts). Default = 0
    params : Params, optional
        The airplane parameters. Built by the plane_parameters module.
        The default is Cessna172.
    dt : float, optional
        The simulation time step in seconds. The default is 0.01.
    r_planet : float, optional
        The radius of the planet. The default is 6371000.
    turb_mag : float, optional
        The mean magnitude of the applied turbulence in N. The default is 0.
    seed : float, optional
        The rng seed. The default is 0.

    Attributes
    ----------
    There's a lot, and I don't want to document them right now. It's mostly
    states, inputs, rotation matrices, and time.

    """
    _state : dict
    _atmosphere : dict
    _R : dict
    _params : Params
    _model : _FlightModel
    _t : float

    def __init__(self, **kwargs):
        # Read the kwargs
        state0 = kwargs.get('state0', {'omega_psi' : 0.0,
                                       'omega_theta' : 0.0,
                                       'omega_phi' : 0.0,
                                       'psi' : 0.0,
                                       'theta' : 0.0,
                                       'phi' : 0.0,
                                       'alpha' : 0.0,
                                       'beta' : 0.0,
                                       'V_inf' : 0.0,
                                       'h' : 0.0,})
        input0 = kwargs.get('input0', {'delta_e' : 0.0,
                                       'delta_r' : 0.0,
                                       'delta_a' : 0.0,
                                       'P' : 0.0})
        params = kwargs.get('params', Cessna172())
        dt = kwargs.get('dt', 0.01)
        r_planet = kwargs.get('r_planet', 6371000.0)
        turb_mag = kwargs.get('turb_mag', 0.0)
        seed = kwargs.get('seed', 0)

        # Set
        self._state = {}
        self._atmosphere = {}
        self._R = {}
        self._params = {}
        self._params['plane'] = params
        self._params['r_planet'] = r_planet
        self._params['turbulence'] = self._gen_turb_param(turb_mag, seed)
        self._params['dt'] = dt
        self._t = 0.0
        self._model = _FlightModel(params)

        # Apply the initial euler angles
        self._update_euler_angs(state0)

        # Get the rotation matrices that are determined by euler angles
        # and flow angles at center of mass
        self._update_plane_rot_mats()

        # Set the flow angles at the center of mass and get the
        # relevant rotations
        self._state['alpha'] = state0['alpha']
        self._state['beta'] = state0['beta']
        self._R['F_CoM'] = _RYZ(-self._state['alpha'], self._state['beta'])

        # Set the initial world velocity and position
        v_plane_F = (state0['V_inf'], 0, 0)
        self._state['v_W'] = _CoV(self._R['CoM_W'], _CoV(self._R['F_CoM'], v_plane_F))
        self._state['p_W'] = (0, 0, state0['h'])

        # Set the initial g force on the pilot to 0.0
        self._state['g_force_W'] = (0.0, 0.0, 0.0)

        # Set the initial inputs
        self._state['delta_e'] = input0['delta_e']
        self._state['delta_r'] = input0['delta_r']
        self._state['delta_a'] = input0['delta_a']
        self._state['P'] = input0['P']
        self._state['delta_e_des'] = input0['delta_e']
        self._state['delta_r_des'] = input0['delta_r']
        self._state['delta_a_des'] = input0['delta_a']
        self._state['P_des'] = input0['P']

        # Set the rotations of the planet to 0
        self._state['earth_pitch'] = 0.0
        self._state['earth_roll'] = 0.0

        # Set an initial value for the centers of lift of each surface
        # Note that, for the wings, these values are an estimate and can
        # only be updated accurately by the flight model
        self._state['wl_CoL_CoM'] = (self._params['plane'].x_w,
                                    -0.25*self._params['plane'].b_w,
                                     self._params['plane'].z_w)
        self._state['wr_CoL_CoM'] = (self._params['plane'].x_w,
                                     0.25*self._params['plane'].b_w,
                                     self._params['plane'].z_w)
        self._state['h_CoL_CoM'] = (self._params['plane'].x_he,
                                    0.0,
                                    self._params['plane'].z_he)
        self._state['v_CoL_CoM']  = (self._params['plane'].x_v,
                                     0.0,
                                     self._params['plane'].z_v)
        self._state['b_CoL_CoM']  = (self._params['plane'].x_b,
                                     0.0,
                                     self._params['plane'].z_b)

        # Set the list of previous velocities and accelerations to empty
        # for linear multi-stepping
        self._state['v_hist'] = deque()
        self._state['a_hist'] = deque()

        # Calculate the gravity, density, temperature, and viscosity
        self._update_atmo()

        # Calculate the local flow velocity at the center of mass
        self._update_velocities()

        # Calculate the effective flow angles at each of the surfaces
        self._update_local_flow_angs()

        # Get the rotations between local surface flow and the plane
        self._update_flow_rot_mats()

        # Get the current lift and drag
        _, _, coeffs = self._model.net_aero_force_torque(self)
        self._state['coeffs'] = coeffs

    def __repr__(self):
        state_str = "state = { "
        for key, val in self._state.items():
            try:
                state_str += f"'{key}': {float(val)}, \n{' '*20}"
            except TypeError:
                state_str += f"'{key}': {tuple(float(v) for v in val)}, \n{' '*20}"
        state_str = state_str[:-23]+" }"

        atmo_str = ' '*10 + "atmosphere = { "
        for key, val in self._atmosphere.items():
            atmo_str += f"'{key}': {float(val)}, \n{' '*25}"
        atmo_str = atmo_str[:-28]+" }"

        R_str = ' '*10 + "R = { "
        for key, val in self._R.items():
            rows = val.__str__().split('\n')
            for i, row in enumerate(rows):
                if i == 0:
                    R_str += f"'{key}': {row}\n"
                    continue
                R_str += f"{' '*(len(key)+20)}{row}\n"
            R_str = R_str[:-2] + f", \n{' '*16}"
        R_str = R_str[:-19] + ' }'

        params_str = ' '*10 + self._params['plane'].__str__().split('(')[0] + ' = { '
        ln = len(params_str)
        for key, val in self._params['plane'].params.items():
            try:
                params_str += f"'{key}': {float(val)}, \n{' '*ln}"
            except ValueError:
                params_str += f"'{key}': {val}, \n{' '*ln}"
        params_str = params_str[:-(ln+3)]+" }"

        tot_str = ',\n\n'.join((state_str, atmo_str, R_str, params_str))
        tot_str = 'FlightSim('+tot_str+')'
        return tot_str

    def __getattr__(self, key):
        try:
            return self._state[key]
        except KeyError:
            pass
        try:
            return self._R[key[2:]]
        except KeyError:
            pass
        try:
            return self._atmosphere[key]
        except KeyError:
            pass
        raise AttributeError(f"'FlightSim' object has no attribute '{key}'")

    def _gen_turb_param(self, turbulence_mag, seed):
        rng = random.Random(seed)
        scale = tuple(2.0*(turbulence_mag/math.sqrt(3))*rng.random() for _ in range(3))
        offset = tuple(3600.0*(2.0*rng.random()-1.0) for _ in range(3))
        period = tuple(30.0*rng.random() + 30.0 for _ in range(3))
        return (scale, offset, period)

    def _noise(self, freq_mults, scales, offsets, periods, time):
        vals = []
        for scale, offset, period in zip(scales, offsets, periods):
            val = 0.0
            scale_mult = 1.0
            scale_mult_sum = 0.0
            for freq_mult in freq_mults:
                c = (math.sin(freq_mult*2*math.pi*(time - offset)/period) +
                     math.sin(freq_mult*math.pi*math.pi*(time - offset)/period))
                val += scale_mult*c
                scale_mult_sum += scale_mult
                scale_mult *= 0.5
            vals.append(scale * val / scale_mult_sum)
            scale_mult = 1.0
            scale_mult_sum = 0.0
        return tuple(vals)

    def _get_turbulence(self):
        # Trivial case
        if all(scale==0.0 for scale in self._params['turbulence'][0]):
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        F = self._noise((1.0, 2.0, 10.0),
                        self._params['turbulence'][0],
                        self._params['turbulence'][1],
                        self._params['turbulence'][2],
                        self._t)
        offsets = (0.25*self._params['plane'].x_v,
                   self._params['plane'].y_w,
                   self._params['plane'].z_b)
        r = (F[0]+F[1]+F[2], -F[0]-F[1], -F[1]-F[2])
        r = (math.atan(x/max(self._params['turbulence'][0])) for x in r)
        r = tuple(x*o for x,o in zip(r, offsets))
        tau = _cross(r, F)
        return F, tau

    def _ground_force(self, Fz):
        m = self._params['plane'].mass
        dt = self._params['dt']
        h = self._state['p_W'][2] - 1.171
        vz = self._state['v_W'][2]
        return (0.0, 0.0, max(-m*(h/(dt*dt) + vz/dt) - Fz, 0.0))

    def _update_euler_angs(self, rotation_state):
        self._state['omega_psi'] = rotation_state['omega_psi']
        self._state['omega_theta'] = rotation_state['omega_theta']
        self._state['omega_phi'] = rotation_state['omega_phi']
        self._state['psi'] = rotation_state['psi']
        self._state['theta'] = rotation_state['theta']
        self._state['phi'] = rotation_state['phi']

    def _update_plane_rot_mats(self):
        # Coordinates of center of mass of plane
        self._R['CoM_W'] = _mmul(((1, 0 ,0), (0, -1, 0), (0, 0, -1)),
                                 _RBW(self.phi, self.theta, self.psi))
        self._R['W_CoM'] = _T(self._R['CoM_W'])

        # Orientation of wings and v stab
        # All other surfaces are aligned with CoM of plane
        cos = math.cos(self._params['plane'].dihedral_w)
        sin = math.sin(self._params['plane'].dihedral_w)
        self._R['WL_CoM'] = (( 1.0, 0.0,  0.0 ),
                             ( 0.0, cos, -sin ),
                             ( 0.0, sin,  cos ))
        self._R['WR_CoM'] = (( 1.0, 0.0,  0.0 ),
                             ( 0.0, cos,  sin ),
                             ( 0.0, -sin, cos ))
        self._R['V_CoM'] = (( 1.0,  0.0, 0.0 ),
                            ( 0.0,  0.0, 1.0 ),
                            ( 0.0, -1.0, 0.0 ))
        self._R['CoM_WL'] = _T(self._R['WL_CoM'])
        self._R['CoM_WR'] = _T(self._R['WR_CoM'])
        self._R['CoM_V']  = _T(self._R['V_CoM'])
        self._R['WL_W'] = _mmul(self._R['CoM_W'], self._R['WL_CoM'])
        self._R['WR_W'] = _mmul(self._R['CoM_W'], self._R['WR_CoM'])
        self._R['V_W'] =  _mmul(self._R['CoM_W'], self._R['V_CoM'])

    def _apply_force(self, F_net):
        # Calculate the net acceleration
        dt = self._params['dt']
        acceleration = tuple(F/self._params['plane'].mass for F in F_net)

        # Linear multistep (AB2)
        try:
            self._state['p_W'] = tuple(p + 1.5*dt*v - 0.5*dt*vm1 for p, v, vm1 in
                  zip(self._state['p_W'], self._state['v_W'], self._state['v_hist'][-1]))

            self._state['v_W'] = tuple(v + 1.5*dt*a - 0.5*dt*am1 for v, a, am1 in
                  zip(self._state['v_W'], acceleration, self._state['a_hist'][-1]))

        # If not enough stored previous states, fall back to Euler stepping
        except IndexError:
            self._state['p_W'] = tuple(p+v*dt
                                       for p,v in zip(self._state['p_W'], self._state['v_W']))
            self._state['v_W'] = tuple(v+a*dt
                                       for v,a in zip(self._state['v_W'], acceleration))


        # Store the current acceleration for linear multi-step
        self._state['v_hist'].append(self._state['v_W'])
        self._state['a_hist'].append(acceleration)

        # Only keep just enough previous states for the order of linear-multistep
        # being used
        if len(self._state['v_hist']) > 1:
            self._state['v_hist'].popleft()
            self._state['a_hist'].popleft()

        # Update the gforce sensor
        self._state['g_force_W'] = tuple(-a/9.81+1.0*(i==2)
                                         for i,a in enumerate(acceleration))

    def _update_inputs(self, delta_e_des, delta_r_des, delta_a_des, P_des):
        # Store the desired values
        self._state['delta_e_des'] = delta_e_des
        self._state['delta_r_des'] = delta_r_des
        self._state['delta_a_des'] = delta_a_des
        self._state['P_des'] = P_des

        # Update the elevator angle
        clipped_delta_e = min(max(delta_e_des, self._params['plane'].delta_e_min),
                              self._params['plane'].delta_e_max)
        d_delta_e = clipped_delta_e - self._state['delta_e']
        if abs(d_delta_e) <= self._params['plane'].delta_e_rate * self._params['dt']:
            self._state['delta_e'] += d_delta_e
        else:
            sign = math.copysign(1, d_delta_e)
            d_delta_e = self._params['plane'].delta_e_rate*sign*self._params['dt']
            self._state['delta_e'] += d_delta_e

        # Update the rudder angle
        clipped_delta_r = min(max(delta_r_des, -self._params['plane'].delta_r_max),
                              self._params['plane'].delta_r_max)
        d_delta_r = clipped_delta_r - self._state['delta_r']
        if abs(d_delta_r) <= self._params['plane'].delta_r_rate * self._params['dt']:
            self._state['delta_r'] += d_delta_r
        else:
            sign = math.copysign(1, d_delta_r)
            d_delta_r = self._params['plane'].delta_r_rate*sign*self._params['dt']
            self._state['delta_r'] += d_delta_r

        # Update the aileron angle
        clipped_delta_a = min(max(delta_a_des, self._params['plane'].delta_a_min),
                              self._params['plane'].delta_a_max)
        d_delta_a = clipped_delta_a - self._state['delta_a']
        if abs(d_delta_a) <= self._params['plane'].delta_a_rate * self._params['dt']:
            self._state['delta_a'] += d_delta_a
        else:
            sign = math.copysign(1, d_delta_a)
            d_delta_a = self._params['plane'].delta_a_rate*sign*self._params['dt']
            self._state['delta_a'] += d_delta_a

        # Update the power setting
        d_P = min(max(P_des, 0.0), self._params['plane'].P_max) - self._state['P']
        if abs(d_P) <= self._params['plane'].P_rate * self._params['dt']:
            self._state['P'] += d_P
        else:
            sign = math.copysign(1, d_P)
            self._state['P'] += self._params['plane'].P_rate*sign*self._params['dt']

    def _update_world_rot(self):
        d_pitch = -self._state['v_W'][0]/(self._state['p_W'][2]+self._params['r_planet'])
        self._state['earth_pitch'] += d_pitch * self._params['dt']
        d_roll = self._state['v_W'][1]/(self._state['p_W'][2]+self._params['r_planet'])
        self._state['earth_roll'] += d_roll * self._params['dt']

    def _update_atmo(self):
        h = self._state['p_W'][2]
        self._atmosphere['g'] = _g(h)
        self._atmosphere['rho'] = _rho(h)
        self._atmosphere['T'] = _temperature(h)
        self._atmosphere['mu'] = _mu(h)

    def _update_velocities(self):
        # Update the local flow velocity at the center of mass
        self._state['v_CoM'] = _CoV(self._R['W_CoM'], self._state['v_W'])

        # Update the flow velocity and dynamic pressure
        self._state['V_inf'] = _norm3(self._state['v_W'])

    def _update_alpha_beta(self):
        # Update the center of mass local flow angles
        v_CoM = self._state['v_CoM']
        self._state['alpha'] = math.atan2(v_CoM[2], v_CoM[0])
        self._state['beta'] = math.asin((v_CoM[1] / self._state['V_inf']))

    def _update_local_flow_angs(self):
        # Get the rotation rate of the plane's center of mass about the world
        omega_CoM_CoM = (self._state['omega_phi'],
                         self._state['omega_theta'],
                         self._state['omega_psi'])

        # Calculate the induced velocity at each of the surfaces
        v_wl_WL = _CoV(self._R['CoM_WL'],
                       _sum(self._state['v_CoM'],
                            _cross(omega_CoM_CoM, self._state['wl_CoL_CoM'])))
        v_wr_WR = _CoV(self._R['CoM_WR'],
                       _sum(self._state['v_CoM'],
                            _cross(omega_CoM_CoM, self._state['wr_CoL_CoM'])))
        v_v_V   = _CoV(self._R['CoM_V'],
                       _sum(self._state['v_CoM'],
                            _cross(omega_CoM_CoM, self._state['v_CoL_CoM'])))
        v_he_HE = _sum(self._state['v_CoM'],
                       _cross(omega_CoM_CoM, self._state['h_CoL_CoM']))
        v_b_B   = _sum(self._state['v_CoM'],
                       _cross(omega_CoM_CoM, self._state['b_CoL_CoM']))

        # Save the magnitude of the local surface velocities
        self._state['V_wl'] = max(_norm3(v_wl_WL), EPSILON)
        self._state['V_wr'] = max(_norm3(v_wr_WR), EPSILON)
        self._state['V_he'] = max(_norm3(v_he_HE), EPSILON)
        self._state['V_v'] = max(_norm3(v_v_V), EPSILON)
        self._state['V_b'] = max(_norm3(v_b_B), EPSILON)

        # Update the local flow angles at each of the surfaces
        self._state['alpha_wl'] = math.atan2(v_wl_WL[2], v_wl_WL[0])
        self._state['beta_wl'] = math.asin(v_wl_WL[1] / self._state['V_wl'])
        self._state['alpha_wr'] = math.atan2(v_wr_WR[2], v_wr_WR[0])
        self._state['beta_wr'] = math.asin(v_wr_WR[1] / self._state['V_wr'])
        self._state['alpha_he'] = math.atan2(v_he_HE[2], v_he_HE[0])
        self._state['beta_he'] = math.asin(v_he_HE[1] / self._state['V_he'])
        self._state['alpha_v'] = math.atan2(v_v_V[2], v_v_V[0])
        self._state['beta_v'] = math.asin(v_v_V[1] / self._state['V_v'])
        self._state['alpha_b'] = math.atan2(v_b_B[2], v_b_B[0])
        self._state['beta_b'] = math.asin(v_b_B[1] / self._state['V_b'])

    def _update_flow_rot_mats(self):
        self._R['F_CoM']   = _RYZ(-self._state['alpha'],
                                   self._state['beta'])
        self._R['FWL_CoM'] = _mmul(self._R['WL_CoM'],
                                   _RYZ(-self._state['alpha_wl'],
                                         self._state['beta_wl']))
        self._R['FWR_CoM'] = _mmul(self._R['WR_CoM'],
                                   _RYZ(-self._state['alpha_wr'],
                                         self._state['beta_wr']))
        self._R['FHE_CoM'] = _RYZ(-self._state['alpha_he'],
                                   self._state['beta_he'])
        self._R['FB_CoM']  = _RYZ(-self._state['alpha_b'],
                                   self._state['beta_b'])
        self._R['FV_CoM']  = _mmul(self._R['V_CoM'],
                                   _RYZ(-self._state['alpha_v'],
                                         self._state['beta_v']))

    def step(self, rotation_state, delta_e_des, delta_r_des, delta_a_des, P_des):
        """
        Takes a single time step. This simulation steps updates all linear
        translations, velocities, and accelerations, but does not handle and
        rotations.

        Parameters
        ----------
        rotation_state : dictionary of the current rotation state of the plane
            The keys are:
                omega_psi : float, yaw rate (rad/s)
                omega_theta : float, pitch rate (rad/s)
                omega_phi : float, roll rate (rad/s)
                psi : float, yaw (rad/s)
                theta : float, pitch (rad/s)
                phi : float, roll (rad/s)
        delta_e_des : float
            The desired elevator deflection in radians.
        delta_r_des : float
            The desired rudder deflection in radians.
        delta_a_des : float
            The desired aileron deflection in radians.
        P_des : float
            The desired engine power setting in Watts.

        Returns
        -------
        tau_aero_net : 3 tuple of floats
            The net aerodynamic torque acting on the plane.

        """
        # Step 0
        self._update_euler_angs(rotation_state)
        self._update_plane_rot_mats()

        # Step 1
        F_aero_net, tau_aero_net, coeffs = self._model.net_aero_force_torque(self)
        F_turbulence, tau_turbulence = self._get_turbulence()
        F_gravity = (0, 0, -self._atmosphere['g']*self._params['plane'].mass)
        F_ground = self._ground_force(F_aero_net[2]+F_turbulence[2]+F_gravity[2])
        F_net = _sum(F_aero_net, F_turbulence, F_gravity, F_ground)
        tau_net = _sum(tau_aero_net, tau_turbulence)
        self._state['coeffs'] = coeffs

        # Step 2
        self._apply_force(F_net)

        # Step 3
        self._update_inputs(delta_e_des, delta_r_des, delta_a_des, P_des)
        self._update_world_rot()
        self._update_atmo()

        # Step 2
        self._update_velocities()

        # Step 3
        self._update_alpha_beta()
        self._update_local_flow_angs()

        # Step 4
        self._update_flow_rot_mats()

        # Step 5
        self._t += self._params['dt']

        return tau_net

    def telem(self):
        """
        Returns the current plane telemetry.

        Returns
        -------
        telem : dictionary
            time : float, the current time in seconds.
            h : float, the current altitude in m
            V_inf : float, the current indicated airspeed in m/s
            p_W : 3 tuple of floats, the current world coords position in m
            v_W : 3 tuple of floats, the current world coords velocity in m/s
            g_force_W : 3 tuple of floats, the current world coords g force in m/s2
            v_CoM : 3 tuple of floats, the current plane coords velocity in meters/s
            R_CoM_W : 3x3 array-like of floats, Rotation from plane coords to world coords.
            alpha : float, the current angle of attack in radians
            beta : float, the current sideslip angle in radians
            omega_psi : float, the current yaw rate in rad/s
            omega_theta : float, the current pitch rate in rad/s
            omega_phi : float, the current roll rate in rad/s
            psi : float, the current yaw rate in rad
            theta : float, the current pitch rate in rad
            phi : float, the current roll rate in rad
            delta_e : float, the current elevator deflection in radians
            delta_r : float, the current rudder deflection in radians
            delta_a : float, the current aileron deflection in radians
            P : float, the current engine power in Watts
            delta_e_des : float, the desired elevator deflection in radians
            delta_r_des : float, the desired rudder deflection in radians
            delta_a_des : float, the desired aileron deflection in radians
            P_des : float, the desired engine power in Watts
            prop_rpm : float, the current propeller speed in rev/min
            earth_pitch : float, the current pitch of the planet required to
                          simulation longitudinal movement
            earth_roll : float, the current roll of the planet required to
                          simulation lateral movement
        """
        telem = {'time' : self._t,
                 'h' : self._state['p_W'][2],
                 'V_inf' : self._state['V_inf'],
                 'p_W' : self._state['p_W'],
                 'v_W' : self._state['v_W'],
                 'g_force_W' : self._state['g_force_W'],
                 'v_CoM' : self._state['v_CoM'],
                 'R_CoM_W' : self._R['CoM_W'],
                 'alpha' : self._state['alpha'],
                 'beta' : self._state['beta'],
                 'omega_psi' : self._state['omega_psi'],
                 'omega_theta' : self._state['omega_theta'],
                 'omega_phi' : self._state['omega_phi'],
                 'psi' : self._state['psi'],
                 'theta' : self._state['theta'],
                 'phi' : self._state['phi'],
                 'delta_e' : self._state['delta_e'],
                 'delta_r' : self._state['delta_r'],
                 'delta_a' : self._state['delta_a'],
                 'P' : self._state['P'],
                 'delta_e_des' : self._state['delta_e_des'],
                 'delta_r_des' : self._state['delta_r_des'],
                 'delta_a_des' : self._state['delta_a_des'],
                 'P_des' : self._state['P_des'],
                 'prop_rpm' : 60.0*self._model.prop_rps(self),
                 'cL_tot' : self._state['coeffs'][0],
                 'cD_tot' : self._state['coeffs'][1],
                 'cl_tot' : self._state['coeffs'][2],
                 'cm_tot' : self._state['coeffs'][3],
                 'cn_tot' : self._state['coeffs'][4],
                 'earth_pitch' : self._state['earth_pitch'],
                 'earth_roll' : self._state['earth_roll'],
                 'rho' : self._atmosphere['rho'],}
        return telem
