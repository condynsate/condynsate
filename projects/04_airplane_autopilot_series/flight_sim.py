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

###############################################################################
#DEPENDENCIES
###############################################################################
import math
import random
from plane_parameters import Params, Cessna172
STALL_DEV = 5.0 * (math.pi/180.0) # Rad from stall angle where stall developments starts

###############################################################################
#MATH OPERATION FUNCTIONS
###############################################################################
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

###############################################################################
#STANDARD ATMOSPHERIC MODEL FUNCTIONS
###############################################################################
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

###############################################################################
#FLIGHT MODEL CLASS
###############################################################################
class _FlightModel:
    def __init__(self, params):
        self.p = params

    def _cL(self, angles, slopes, spans, alpha_L0, alpha_s):
        # Read the args
        alpha = angles[0]
        beta = angles[1]
        delta = angles[2]
        alpha_0 = slopes[0]
        tau = slopes[1]
        b = spans[0]
        b_s = spans[1]

        # Check if the surface spans the entire wing (within 1% precision)
        if abs(b-b_s) >= 0.01*b:
            # If it does not, switch to outboard-inboard lift calculation with
            # the surface always as outboard as possible (like with ailerons)
            slopes_o = (alpha_0, (b/b_s)*tau)
            spans_o = (b_s, b_s)
            cL_o, _ = self._cL(angles, slopes_o, spans_o, alpha_L0, alpha_s)
            slopes_i = (alpha_0, 0.0)
            spans_i = (b-b_s, b-b_s)
            cL_i, _ = self._cL(angles, slopes_i, spans_i, alpha_L0, alpha_s)

            # Effective coefficient of lift
            cL = (cL_o*b_s + cL_i*(b-b_s)) / b

            # Center of lift calculation
            y_o = b - 0.5*b_s
            y_i = 0.5*(b - b_s)
            try:
                CoL = (cL_o*b_s*y_o + cL_i*(b-b_s)*y_i) / (b*cL)
            except ZeroDivisionError:
                CoL = 0.5*b
            return cL, CoL

        # Get the effective angle of attack
        alpha_eff = alpha + tau*delta
        alpha_eff_abs = abs(alpha_eff)

        # Get the no stall coefficient of lift and center of lift
        cL = alpha_0*(alpha_eff-alpha_L0)
        CoL = 0.5*b

        # Add random buffeting while stall is developing
        # At stall, this gives an average cL buffet of -0.4661
        # Which is 1/20 of the lift produced by an infinite wing at alpha_s
        buffet = 0.0
        if alpha_eff_abs >= alpha_s - STALL_DEV:
            buffet = (alpha_eff_abs-alpha_s+STALL_DEV) * (-144/20*alpha_s*random.random())

        # No stall (angle of attack is more than 5 degrees below stall angle)
        if alpha_eff_abs < alpha_s - STALL_DEV:
            # Add a cos beta as a first order crab angle correction term
            cL = math.cos(beta)*(cL + buffet)
            return cL, CoL

        # Positive stall developement region
        if alpha_s - STALL_DEV <= alpha_eff <= alpha_s:
            cL_stalling = -1/(2*STALL_DEV)*cL*(alpha_eff - alpha_s - STALL_DEV)
            cL = math.cos(beta)*(cL_stalling + buffet)
            return cL, CoL

        # Negative stall developement region
        if -alpha_s <= alpha_eff <= -alpha_s + STALL_DEV:
            cL_stalling = 1/(2*STALL_DEV)*cL*(alpha_eff + alpha_s + STALL_DEV)
            cL = math.cos(beta)*(cL_stalling + buffet)
            return cL, CoL

        # Complete stall
        # Any angle of attack outside of [-a_s, a_s] produces no lift
        return 0.0, CoL

    def _alpha_s(self, re):
        # Calculate stall angle (Assume all airfoils stall at around 17 deg)
        # though very low reynolds number result in lower stall angles all the
        # way to 10 deg
        x = min(max(re, 50_000), 1_000_000)

        # Fast return for high Reynold's num
        if x == 1_000_000:
            return 0.2966693209247442 # 16.9979 degreees

        alpha_s = -2.333e-18*x*x*x + 3.674e-12*x*x - 3.499e-7*x + 0.0086
        alpha_s = min(max(alpha_s,0),1)*math.radians(7) + math.radians(10)
        return alpha_s

    def _alpha_L0(self, re, typ):
        # Calculate the 0 lift angle of attack where, for all airfoils, at very
        # low reynold's number, a_l0 is 0, and at normal flight reynold's numbers
        # a_l0 is whatever is defined for a specific airfoil
        x = min(max(re, 50_000), 1_000_000)

        # Fast calculation for high Reynold's num
        if x == 1_000_000:
            alpha_L0 = 1.0
        else:
            alpha_L0 = -1.11e-12*x*x + 2.22e-6*x - 0.108
            alpha_L0 = min(max(alpha_L0, 0.0), 1.0)
        if typ == 'NACA2412':
            alpha_L0 *= -0.0436
        elif typ == 'NACA2412i':
            alpha_L0 *= 0.0436
        elif typ == 'NACA0012':
            alpha_L0 = 0.0
        else:
            alpha_L0 = 0.0
        return alpha_L0

    def _L_D(self, atmosphere, wing, state):
        # Get the reynold's number and dynamic pressure
        re = atmosphere['rho'] * wing['c'] * state['V'] / atmosphere['mu']
        q_S = 0.5 * atmosphere['rho'] * state['V'] * state['V'] * wing['S']

        # Calculate stall and 0 lift angles
        alpha_L0 = self._alpha_L0(re, wing['typ'])
        alpha_s = self._alpha_s(re)

        # Get the lift and induced drag coefficients
        angles = (state['alpha'], state['beta'], state['delta'])
        slopes = (wing['alpha_0'], wing['tau'])
        spans = (wing['b'], wing['b_s'])
        cL, CoL = self._cL(angles, slopes, spans, alpha_L0, alpha_s)
        cDi = cL*cL / (math.pi * wing['AR'])

        # Turbulent flow of thin plate skin friction
        cDf_skin = 0.074 * re**(-0.2)

        # Form drag is that of streamlined body below stall, then
        # transitions to that of a rotated, thin plate after stall
        if cL != 0.0:
            cDf_form = 0.006
        else:
            cDf_form = (abs(1.8*math.sin(state['alpha'])) -
                        0.284*math.cos(state['alpha']))
            cDf_form = 0.05*max(cDf_form, 0.0)

        # Get the total lift and drag
        L = cL * q_S
        D = (cDi + cDf_skin + cDf_form) * q_S
        return L, D, CoL

    def _wing_forces(self, sim):
        # Build the arguments
        atmosphere = {'rho' : sim.rho,
                      'mu' : sim.mu}
        half_wing = {'S' : 0.5 * self.p.S_w,
                     'AR' : self.p.AR_w,
                     'b' : 0.5*self.p.b_w,
                     'b_s' : 0.5*self.p.b_a,
                     'c' : self.p.c_w,
                     'alpha_0' : self.p.alpha_0_w,
                     'tau' : self.p.tau_a,
                     'typ' : self.p.typ_w, }
        state_wl = {'V' : sim.V_wl,
                    'alpha' : sim.alpha_wl,
                    'delta' : sim.delta_a,
                    'beta' : sim.beta_wl}
        state_wr = {'V' : sim.V_wr,
                    'alpha' : sim.alpha_wr,
                    'delta' : -sim.delta_a,
                    'beta' : sim.beta_wr}

        # Get the lift and drag of the left wing system (in local left wing flow)
        L_wl, D_wl, CoL_y_wl = self._L_D(atmosphere, half_wing, state_wl)
        CoL_wl_CoM = (self.p.x_w, -CoL_y_wl, self.p.z_w)

        # Get the lift and drag of the right wing system (in local right wing flow)
        L_wr, D_wr, CoL_y_wr = self._L_D(atmosphere, half_wing, state_wr)
        CoL_wr_CoM = (self.p.x_w, CoL_y_wl, self.p.z_w)

        # Convert from local flow coords to world coords
        F_wl_CoM = _CoV(sim.R_FWL_CoM, (-D_wl, 0.0, -L_wl))
        F_wr_CoM = _CoV(sim.R_FWR_CoM, (-D_wr, 0.0, -L_wr))

        # Calculate torques
        tau_wl_CoM = _cross(CoL_wl_CoM, F_wl_CoM)
        tau_wr_CoM = _cross(CoL_wr_CoM, F_wr_CoM)
        return _sum(F_wl_CoM, F_wr_CoM), _sum(tau_wl_CoM, tau_wr_CoM)

    def _h_stab_ele_forces(self, sim):
        # Get the downwash angle induced from the wings
        eta = self.p.d_eta_d_alpha * 0.5 * (sim.alpha_wl + sim.alpha_wr)

        # Build the arguments
        atmosphere = {'rho' : sim.rho,
                      'mu' : sim.mu}
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
        L, D, _ = self._L_D(atmosphere, wing, state)
        CoL_CoM = (self.p.x_he, 0.0, self.p.z_he)

        # Convert from local flow coords to world coords
        F_CoM = _CoV(sim.R_FHE_CoM, (-D, 0.0, -L))

        # Calculate torque
        tau_CoM = _cross(CoL_CoM, F_CoM)

        # Convert from local flow coords to world coords
        return F_CoM, tau_CoM

    def _vert_stab_force(self, sim):
        # Build the arguments
        atmosphere = {'rho' : sim.rho,
                      'mu' : sim.mu}
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
        L, D, _ = self._L_D(atmosphere, wing, state)
        CoL_CoM = (self.p.x_v, 0.0, self.p.z_v)

        # Convert from local flow coords to world coords
        F_CoM = _CoV(sim.R_FV_CoM, (-D, 0.0, -L))

        # Calculate torque
        tau_CoM = _cross(CoL_CoM, F_CoM)

        # Convert from local flow coords to world coords
        return F_CoM, tau_CoM

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

        # Calculate torque
        tau_CoM = _cross((self.p.x_b, 0.0, self.p.z_b), F_CoM)
        return F_CoM, tau_CoM

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
        num = 0.0174 * const * self.p.rpm_max
        den = const + 3.177
        return num / den

    def _cT(self, sim, n):
        # Based on monotonic decrease, linear cT(J) curve
        n_max = self.p.rpm_max/60.0
        n_75 = 0.925*self.p.rpm_max/60.0
        D = self.p.prop_D
        V_c = self.p.V_cruise
        b = self.p.T_max / (1.225*n_max**2*D**4)
        m = self.p.T_cruise/(1.225*V_c*n_75*D**3) - b*n_75*D/V_c
        J = sim.v_CoM[0] / (n*self.p.prop_D)
        return min(max(m*J + b, 0.0), b)

    def _prop_force(self, sim):
        # Calculate the thrust
        n = self.prop_rps(sim)
        T = sim.rho * n**2 * self.p.prop_D**4 * self._cT(sim, n)

        # Convert from plane coords to world coords
        return (T, 0.0, 0.0)

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
        F_W
            The net aerodynamic force (N).
        tau_W
            The net aerodynamic torque (Nm).

        """
        # Get the forces and torques
        F_w_CoM, tau_w_CoM  = self._wing_forces(sim)
        F_he_CoM, tau_he_CoM = self._h_stab_ele_forces(sim)
        F_v_CoM, tau_v_CoM = self._vert_stab_force(sim)
        F_b_CoM, tau_b_CoM = self._body_force(sim)
        F_p_CoM = self._prop_force(sim)

        # Get the net forces and torques
        F_net_CoM = _sum(F_w_CoM, F_he_CoM, F_v_CoM, F_b_CoM, F_p_CoM)
        tau_net_CoM = _sum(tau_w_CoM, tau_he_CoM, tau_v_CoM, tau_b_CoM)

        # Change to world coordinates
        return _CoV(sim.R_CoM_W, F_net_CoM), _CoV(sim.R_CoM_W, tau_net_CoM)

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

        # Calculate the gravity, density, temperature, and viscosity
        self._update_atmo()

        # Calculate the local flow velocity at the center of mass
        self._update_velocities()

        # Calculate the effective flow angles at each of the surfaces
        self._update_local_flow_angs()

        # Get the rotations between local surface flow and the plane
        self._update_flow_rot_mats()

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
            R_str = R_str[:-2] + f', \n{' '*16}'
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

    def _update_euler_angs(self, rotation_state):
        self._state['omega_psi'] = rotation_state['omega_psi']
        self._state['omega_theta'] = rotation_state['omega_theta']
        self._state['omega_phi'] = rotation_state['omega_phi']
        self._state['psi'] = rotation_state['psi']
        self._state['theta'] = rotation_state['theta']
        self._state['phi'] = rotation_state['phi']

    def _update_plane_rot_mats(self):
        # Coordinates of center of mass of plane
        r, p, y = self.phi, self.theta, self.psi
        self._R['CoM_W'] = _mmul(((1, 0 ,0), (0, -1, 0), (0, 0, -1)), _RBW(r, p, y))
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
        # Apply accelerations and velocities
        dt = self._params['dt']
        acceleration = tuple(F/self._params['plane'].mass for F in F_net)
        self._state['v_W'] = tuple(v+a*dt for v,a in zip(self._state['v_W'], acceleration))
        self._state['p_W'] = tuple(p+v*dt for p,v in zip(self._state['p_W'], self._state['v_W']))

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
        clipped_delta_a = min(max(delta_a_des, -self._params['plane'].delta_a_max),
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
        # in the plane's coordinate's
        omega_CoM_CoM = (self._state['omega_phi'],
                         self._state['omega_theta'],
                         self._state['omega_psi'])

        # Calculate the induced velocity at each of the surfaces from the
        # plane's rotation in surface coords then get the total flow velocity
        # at each surface in the surface coords.
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
        self._state['V_wl'] = _norm3(v_wl_WL)
        self._state['V_wr'] = _norm3(v_wr_WR)
        self._state['V_he'] = _norm3(v_he_HE)
        self._state['V_v'] = _norm3(v_v_V)
        self._state['V_b'] = _norm3(v_b_B)

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
        rotations. To handle rotations, the net torque acting on the plane is
        returned.

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
        delta_r_des : TYPE
            The desired rudder deflection in radians.
        delta_a_des : TYPE
            The desired aileron deflection in radians.
        P_des : TYPE
            The desired engine power setting in Watts.

        Returns
        -------
        tau_aero_net : 3 tuple of floats
            The net aerodynamic torque acting on the plane.

        """
        # Step 0
        F_aero_net, tau_aero_net = self._model.net_aero_force_torque(self)
        F_turbulence, tau_turbulence = self._get_turbulence()
        F_gravity = (0, 0, -self._atmosphere['g']*self._params['plane'].mass)
        F_net = _sum(F_aero_net, F_gravity, F_turbulence)
        tau_net = _sum(tau_aero_net, tau_turbulence)

        # Step 1
        self._update_euler_angs(rotation_state)
        self._update_plane_rot_mats()

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
                 'earth_pitch' : self._state['earth_pitch'],
                 'earth_roll' : self._state['earth_roll'],}
        return telem
