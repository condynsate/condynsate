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
import numpy as np
from plane_parameters import Params, Cessna172

###############################################################################
#MATH OPERATION FUNCTIONS
###############################################################################
def _RYZ(y, z):
    cy, cz = math.cos(y), math.cos(z)
    sy, sz = math.sin(y), math.sin(z)
    return np.array(((cy*cz, -cy*sz, sy ),
                     (sz,     cz,    0.0),
                     (-sy*cz, sy*sz, cy )))

def _RBW(r, p, y):
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    return np.array(((cp*cy, cy*sp*sr-cr*sy, cr*cy*sp+sr*sy),
                     (cp*sy, cr*cy+sp*sr*sy, cr*sp*sy-cy*sr),
                     (-sp,   cp*sr,          cp*cr)))

def _cross(a, b):
    a0, a1, a2 = a
    b0, b1, b2 = b
    return (a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0)

def _norm3(v3):
    return math.sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2])

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

    def _cL(self, alpha, beta, alpha_0, alpha_L0, alpha_s):
        alpha_abs = abs(alpha)

        # Add random buffeting when within +- 3 degrees of stall
        buffet = 0.0
        if abs(alpha_abs - alpha_s) <= 0.0524:
            buffet = -0.3*((alpha_abs-alpha_s)+0.0524)*9.55*random.random()

        # No stall
        if alpha_abs <= alpha_s:
            # Add a cos beta as a first order crab angle correction term
            return math.cos(beta) * alpha_0 * (alpha - alpha_L0) + buffet
            # return alpha_0 * (alpha - alpha_L0) + buffet

        # Positive stalling region
        # Assume stall starts at a_s. cL decreases to 50% at 3 deg past a_s
        if 0 < alpha < alpha_s + 0.0524:
            assq = alpha_s*alpha_s
            a = 182*alpha_0*(alpha_L0 - alpha_s - 0.105)
            b =-365*alpha_0*(alpha_L0*alpha_s - assq - 0.105*alpha_s - 0.003)
            c = 182*alpha_0*(alpha_L0*(assq - 0.006) - assq*(alpha_s + 0.105))

            # Add a cos beta as a first order crab angle correction term
            return math.cos(beta) * (a * alpha*alpha + b * alpha + c) + buffet
            # return (a * alpha*alpha + b * alpha + c) + buffet

        # Negative stalling region
        # Assume stall starts at -a_s. cL decreases to 50% at 3 deg past -a_s
        if -alpha_s - 0.0524 < alpha < 0:
            assq = alpha_s*alpha_s
            a = 182*alpha_0*(alpha_L0 + alpha_s + 0.105)
            b = 365*alpha_0*(alpha_L0*alpha_s + assq + 0.105*alpha_s + 0.003)
            c = 182*alpha_0*(alpha_L0*(assq - 0.006) + assq*(alpha_s + 0.105))

            # Add a cos beta as a first order crab angle correction term
            return math.cos(beta) * (a * alpha*alpha + b * alpha + c) + buffet
            # return (a * alpha*alpha + b * alpha + c) + buffet

        # Complete stall
        # Any angle of attack outside of [-a_s-3deg, a_s+3deg] produces no lift
        return 0.0

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
        cL = self._cL(state['alpha'], state['beta'], wing['alpha_0'], alpha_L0, alpha_s)
        cDi = cL*cL / (math.pi * wing['AR'])

        # Turbulent flow of thin plate skin friction
        cDf_skin = 0.074 * re**(-0.2)

        # Form drag is that of streamlined body below stall, then
        # transitions to that of a rotated, thin plate after stall
        if abs(state['alpha']) < alpha_s:
            cDf_form = 0.006
        else:
            cDf_form = (abs(1.8*math.sin(state['alpha'])) -
                        0.284*math.cos(state['alpha']))

        # Get the total lift and drag
        L = cL * q_S
        D = (cDi + cDf_skin + cDf_form) * q_S
        return L, D

    def _wing_forces(self, sim):
        # Build the arguments
        atmosphere = {'rho' : sim.rho,
                      'mu' : sim.mu}
        wing = {'c' : self.p.c_wa,
                'S' : 0.5 * self.p.S_w, # half due to seperated wings calc
                'AR' : self.p.AR_w,
                'alpha_0' : self.p.alpha_0_w,
                'typ' : self.p.typ_wa}
        aileron = {'c' : self.p.c_wa,
                   'S' : self.p.S_a,
                   'AR' : self.p.AR_a,
                   'alpha_0' : self.p.alpha_0_a,
                   'typ' : self.p.typ_wa}
        state_wl = {'V' : sim.V_wl,
                    'alpha' : sim.alpha_wl,
                    'beta' : sim.beta_wl}
        state_wr = {'V' : sim.V_wr,
                    'alpha' : sim.alpha_wr,
                    'beta' : sim.beta_wr}
        state_al = {'V' : sim.V_wl,
                    'alpha' : sim.alpha_wl + sim.delta_a,
                    'beta' : sim.beta_wl}
        state_ar = {'V' : sim.V_wr,
                    'alpha' : sim.alpha_wr - sim.delta_a,
                    'beta' : sim.beta_wr}

        # Get the lift and drag of the left wing (in local left wing flow)
        LD_wl = self._L_D(atmosphere, wing, state_wl)

        # Get the lift and drag of the left aileron (in local left wing flow)
        LD_al = self._L_D(atmosphere, aileron, state_al)

        # Get the lift and drag of the right wing (in local right wing flow)
        LD_wr = self._L_D(atmosphere, wing, state_wr)

        # Get the lift and drag of the right state_wl (in local right wing flow)
        LD_ar = self._L_D(atmosphere, aileron, state_ar)

        # Convert from local flow coords to world coords
        F_wl_CoM = sim.R_FWL_CoM @ (-LD_wl[1]-LD_al[1], 0.0, -LD_wl[0]-LD_al[0])
        F_wr_CoM = sim.R_FWR_CoM @ (-LD_wr[1]-LD_ar[1], 0.0, -LD_wr[0]-LD_ar[0])
        return F_wl_CoM, F_wr_CoM

    def _h_stab_ele_forces(self, sim):
        # Get the downwash angle induced from the wings
        eta = self.p.d_eta_d_alpha * 0.5 * (sim.alpha_wl + sim.alpha_wr)

        # Build the arguments
        atmosphere = {'rho' : sim.rho,
                      'mu' : sim.mu}
        h_stab = {'c' : self.p.c_he,
                  'S' : self.p.S_h,
                  'AR' : self.p.AR_h,
                  'alpha_0' : self.p.alpha_0_h,
                  'typ' : self.p.typ_he}
        ele = {'c' : self.p.c_he,
               'S' : self.p.S_e,
               'AR' : self.p.AR_e,
               'alpha_0' : self.p.alpha_0_e,
               'typ' : self.p.typ_he}
        state_h = {'V' : sim.V_he,
                   'alpha' : sim.alpha_he - eta,
                   'beta' : sim.beta_he}
        state_e = {'V' : sim.V_he,
                   'alpha' : sim.alpha_he - eta - sim.delta_e,
                   'beta' : sim.beta_he}

        # Get the lift and drag of the h stab (in local h stab flow)
        LD_h = self._L_D(atmosphere, h_stab, state_h)

        # Get the lift and drag of the elevator (in local h stab flow)
        LD_e = self._L_D(atmosphere, ele, state_e)

        # Convert from local flow coords to world coords
        return sim.R_FHE_CoM @ (-LD_h[1]-LD_e[1], 0.0, -LD_h[0]-LD_e[0])

    def _vert_stab_force(self, sim):
        # Build the arguments
        atmosphere = {'rho' : sim.rho,
                      'mu' : sim.mu}
        v_stab = {'c' : self.p.c_vr,
                  'S' : self.p.S_v,
                  'AR' : self.p.AR_v,
                  'alpha_0' : self.p.alpha_0_v,
                  'typ' : self.p.typ_vr}
        rud = {'c' : self.p.c_vr,
               'S' : self.p.S_r,
               'AR' : self.p.AR_r,
               'alpha_0' : self.p.alpha_0_r,
               'typ' : self.p.typ_vr}
        state_v = {'V' : sim.V_v,
                   'alpha' : sim.alpha_v,
                   'beta' : sim.beta_v}
        state_r = {'V' : sim.V_v,
                   'alpha' : sim.alpha_v + sim.delta_r,
                   'beta' : sim.beta_v}

        # Get the lift and drag of the v stab (in local v stab flow)
        LD_v = self._L_D(atmosphere, v_stab, state_v)

        # Get the lift and drag of the rudder (in local v stab flow)
        LD_r = self._L_D(atmosphere, rud, state_r)

        # Convert from local flow coords to world coords
        return sim.R_FV_CoM @ (-LD_v[1]-LD_r[1], 0.0, -LD_v[0]-LD_r[0])

    def  _body_force(self, sim):
        # Body lift is treated as though body is low-lift symmetric airfoil
        # with no stall angle
        alpha_0 = 0.1
        alpha_L0 = 0.0
        alpha_s = math.inf
        cL_alpha=self._cL(sim.alpha_b, sim.beta_b, alpha_0, alpha_L0, alpha_s)
        cL_beta=self._cL(sim.beta_b, sim.alpha_b, alpha_0, alpha_L0, alpha_s)
        q = 0.5 * sim.rho * sim.V_b * sim.V_b
        S_b = self.p.S_b
        L_alpha = S_b * cL_alpha * q
        L_beta = S_b * cL_beta * q

        # Assume body acts approximately like streamline body with cD_tot = 0.045
        D = S_b * 0.045 * q

        # Convert from local flow coords to world coords
        # Note that for body, side forces are also generated
        return sim.R_FB_CoM @ (-D, -L_beta, -L_alpha)

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

    def _r_CoM2CoL_CoM(self, sim):
        # Adjust dCL based on AoA for nonsymmetric surfaces (wings, hstab, ele)
        # Assume wings, hstab, and ele are all NACA2412 airfoils
        # x_cL moves fore by 10%c at 20 deg AoA and aft by 10%c at -20 deg AoA
        c_wa = self.p.c_wa
        dx_wl = 0.286*c_wa*sim.alpha_wl
        dx_wl = min(max(dx_wl, -.1*c_wa), .1*c_wa)
        dx_wr = 0.286*c_wa*sim.alpha_wr
        dx_wr = min(max(dx_wr, -.1*c_wa), .1*c_wa)
        c_he = self.p.c_he
        dx_he = 0.286*c_he*sim.alpha_he
        dx_he = min(max(dx_he, -.1*c_he),.1*c_he)

        x_w = self.p.x_w
        y_w = self.p.y_w
        z_w = self.p.z_w
        r_wl_CoM = (x_w+dx_wl, -y_w,  z_w)
        r_wr_CoM = (x_w+dx_wr,  y_w,  z_w)
        r_he_CoM = (self.p.x_he+dx_he, self.p.y_he, self.p.z_he)
        r_v_CoM = (self.p.x_v, self.p.y_v, self.p.z_v)
        r_b_CoM = (self.p.x_b, self.p.y_b, self.p.z_b)

        return (r_wl_CoM, r_wr_CoM, r_he_CoM, r_v_CoM, r_b_CoM)

    def _tau_LD(self, sim, forces_CoM):
        rs_CoM = self._r_CoM2CoL_CoM(sim)
        taus_CoM = (_cross(r, F) for (r,F) in zip(rs_CoM, forces_CoM))
        return tuple(map(sum, zip(*taus_CoM)))

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
        # Get the forces
        F_wl_CoM, F_wr_CoM  = self._wing_forces(sim)
        F_he_CoM = self._h_stab_ele_forces(sim)
        F_v_CoM = self._vert_stab_force(sim)
        F_b_CoM = self._body_force(sim)
        F_p_CoM = self._prop_force(sim)
        F_net_CoM = F_wl_CoM+F_wr_CoM+F_he_CoM+F_v_CoM+F_b_CoM+F_p_CoM

        # Get the net torque from the lift and drag of the surfaces
        # and from Euler rate-based damping moments
        forces_CoM = (F_wl_CoM, F_wr_CoM, F_he_CoM, F_v_CoM, F_b_CoM)
        rs_CoM = self._r_CoM2CoL_CoM(sim)
        taus_CoM = tuple(_cross(r, F) for (r,F) in zip(rs_CoM, forces_CoM))
        tau_LD_CoM = tuple(map(sum, zip(*taus_CoM)))

        return sim.R_CoM_W @ F_net_CoM, sim.R_CoM_W @ tau_LD_CoM

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
        self._state['v_W'] = self._R['CoM_W'] @ self._R['F_CoM'] @ v_plane_F
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
        rng = np.random.default_rng(seed=seed)
        scale = 2.0 * (turbulence_mag / math.sqrt(3)) * rng.random(3)
        offset = 3600.0 * (2.0*rng.random(3)-1.0)
        period = 30.0*rng.random(3) + 30.0
        return (scale, offset, period)

    def _noise(self, freq_mults, scales, offsets, periods, time):
        # Trivial case
        if all(scales==0.0):
            return (0.0, 0.0, 0.0)

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
        return vals

    def _get_turbulence(self):
        F_turbulence = self._noise((1.0, 2.0, 10.0),
                                   self._params['turbulence'][0],
                                   self._params['turbulence'][1],
                                   self._params['turbulence'][2],
                                   self._t)
        return F_turbulence

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
        self._R['CoM_W'] = ((1, 0 ,0), (0, -1, 0), (0, 0, -1)) @ _RBW(r, p, y)
        self._R['W_CoM'] = self._R['CoM_W'].T

        # Orientation of wings and v stab
        # All other surfaces are aligned with CoM of plane
        cos = math.cos(self._params['plane'].dihedral_w)
        sin = math.sin(self._params['plane'].dihedral_w)
        self._R['WL_CoM'] = np.array([[ 1.0, 0.0,  0.0 ],
                                      [ 0.0, cos, -sin ],
                                      [ 0.0, sin,  cos ]])
        self._R['WR_CoM'] = np.array([[ 1.0, 0.0,  0.0 ],
                                      [ 0.0, cos,  sin ],
                                      [ 0.0, -sin, cos ]])
        self._R['V_CoM'] = np.array([[ 1.0,  0.0, 0.0 ],
                                     [ 0.0,  0.0, 1.0 ],
                                     [ 0.0, -1.0, 0.0 ]])
        self._R['CoM_WL'] = self._R['WL_CoM'].T
        self._R['CoM_WR'] = self._R['WR_CoM'].T
        self._R['CoM_V']  = self._R['V_CoM'].T
        self._R['WL_W'] = self._R['CoM_W'] @ self._R['WL_CoM']
        self._R['WR_W'] = self._R['CoM_W'] @ self._R['WR_CoM']
        self._R['V_W'] =  self._R['CoM_W'] @ self._R['V_CoM']

    def _apply_force(self, F_aero_net, F_turbulence):
        # Get the net force by adding gravity
        F_gravity = (0.0, 0.0, -self.g*self._params['plane'].mass)
        F_net = F_aero_net + F_turbulence + F_gravity

        # Apply accelerations and velocities
        acceleration = F_net / self._params['plane'].mass
        self._state['v_W'] += (F_net / self._params['plane'].mass) * self._params['dt']
        self._state['p_W'] += self._state['v_W'] * self._params['dt']

        # Update the gforce sensor
        self._state['g_force_W'] = -acceleration/9.81 + (0.0, 0.0, 1.0)

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
        self._state['v_CoM'] = self._R['W_CoM'] @ self._state['v_W']

        # Update the flow velocity and dynamic pressure
        self._state['V_inf'] = _norm3(self._state['v_W'])

    def _update_alpha_beta(self):
        # Update the center of mass local flow angles
        v_CoM = self._state['v_CoM']
        self._state['alpha'] = math.atan2(v_CoM[2], v_CoM[0])
        self._state['beta'] = math.asin((v_CoM[1] / self._state['V_inf']))

    def _update_local_flow_angs(self):
        # Get the origins of the center of lift for each surface
        O_WL_CoM = (self._params['plane'].x_w,
                   -self._params['plane'].y_w,
                    self._params['plane'].z_w)
        O_WR_CoM = (self._params['plane'].x_w,
                    self._params['plane'].y_w,
                    self._params['plane'].z_w)
        O_HE_CoM = (self._params['plane'].x_he,
                    self._params['plane'].y_he,
                    self._params['plane'].z_he)
        O_V_CoM  = (self._params['plane'].x_v,
                    self._params['plane'].y_v,
                    self._params['plane'].z_v)
        O_B_CoM  = (self._params['plane'].x_b,
                    self._params['plane'].y_b,
                    self._params['plane'].z_b)

        # Get the rotation rate of the plane's center of mass about the world
        # in the plane's coordinate's
        omega_CoM_CoM = (self._state['omega_phi'],
                         self._state['omega_theta'],
                         self._state['omega_psi'])

        # Calculate the induced velocity at each of the surfaces from the
        # plane's rotation in surface coords then get the total flow velocity
        # at each surface in the surface coords.
        v_wl_WL = self._R['CoM_WL'] @ (self._state['v_CoM'] + _cross(omega_CoM_CoM, O_WL_CoM))
        v_wr_WR = self._R['CoM_WR'] @ (self._state['v_CoM'] + _cross(omega_CoM_CoM, O_WR_CoM))
        v_v_V   = self._R['CoM_V']  @ (self._state['v_CoM'] + _cross(omega_CoM_CoM, O_V_CoM))
        v_he_HE = self._state['v_CoM'] + _cross(omega_CoM_CoM, O_HE_CoM)
        v_b_B   = self._state['v_CoM'] + _cross(omega_CoM_CoM, O_B_CoM)

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
        self._R['FWL_CoM'] = self._R['WL_CoM'] @ _RYZ(-self._state['alpha_wl'],
                                                       self._state['beta_wl'])
        self._R['FWR_CoM'] = self._R['WR_CoM'] @ _RYZ(-self._state['alpha_wr'],
                                                       self._state['beta_wr'])
        self._R['FHE_CoM'] = _RYZ(-self._state['alpha_he'],
                                   self._state['beta_he'])
        self._R['FB_CoM']  = _RYZ(-self._state['alpha_b'],
                                   self._state['beta_b'])
        self._R['FV_CoM']  = self._R['V_CoM'] @ _RYZ(-self._state['alpha_v'],
                                                      self._state['beta_v'])

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
        F_turbulence = self._get_turbulence()

        # # Step 1
        self._update_euler_angs(rotation_state)
        self._update_plane_rot_mats()
        self._apply_force(F_aero_net, F_turbulence)
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

        return tau_aero_net

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
