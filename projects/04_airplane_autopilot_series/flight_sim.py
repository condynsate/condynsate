# -*- coding: utf-8 -*-
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
def RYZ(y, z):
    cy, cz = math.cos(y), math.cos(z)
    sy, sz = math.sin(y), math.sin(z)
    return np.array(((cy*cz, -cy*sz, sy ),
                     (sz,     cz,    0.0),
                     (-sy*cz, sy*sz, cy )))

def RXYZ(x, y, z):
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    return np.array(((cy*cz,          -cy*sz,           sy   ),
                     (cx*sz+sx*sy*cz,  cx*cz-sx*sy*sz, -sx*cy),
                     (sx*sz-cx*sy*cz,  sx*cz+cx*sy*sz,  cx*cy)))

def cross(va, vb):
    a1, a2, a3 = va
    b1, b2, b3 = vb
    return (a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1)

def norm3(v3):
    return math.sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2])

###############################################################################
#STANDARD ATMOSPHERIC MODEL FUNCTIONS
###############################################################################
def g(h):
    # Newtonian gravity
    return 3.986025446e14 / ((6.371e6+h)*(6.371e6+h))

def rho(h):
    # Ideal gas law applied to barometric formula
    return 1.225*math.exp(-h/10363.)

def temperature(h):
    # Standard atmosphere temperature model
    if h < 10000:
        return 288.2 - 0.00649*h
    return 223.3

def mu(h):
    # Sutherland's law
    return 1.458e-6*temperature(h)**(1.5) / (temperature(h) + 110.4)

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

        # Positive stalling region
        # Assume stall starts at a_s. cL decreases to 50% at 3 deg past a_s
        if 0 < alpha < alpha_s + 0.0524:
            assq = alpha_s*alpha_s
            a = 182*alpha_0*(alpha_L0 - alpha_s - 0.105)
            b =-365*alpha_0*(alpha_L0*alpha_s - assq - 0.105*alpha_s - 0.003)
            c = 182*alpha_0*(alpha_L0*(assq - 0.006) - assq*(alpha_s + 0.105))

            # Add a cos beta as a first order crab angle correction term
            return math.cos(beta) * (a * alpha*alpha + b * alpha + c) + buffet

        # Negative stalling region
        # Assume stall starts at -a_s. cL decreases to 50% at 3 deg past -a_s
        if -alpha_s - 0.0524 < alpha < 0:
            assq = alpha_s*alpha_s
            a = 182*alpha_0*(alpha_L0 + alpha_s + 0.105)
            b = 365*alpha_0*(alpha_L0*alpha_s + assq + 0.105*alpha_s + 0.003)
            c = 182*alpha_0*(alpha_L0*(assq - 0.006) + assq*(alpha_s + 0.105))

            # Add a cos beta as a first order crab angle correction term
            return math.cos(beta) * (a * alpha*alpha + b * alpha + c) + buffet

        # Complete stall
        # Any angle of attack outside of [-a_s-3deg, a_s+3deg] produces no lift
        return 0.0

    def _alpha_s(self, re):
        # Calculate stall angle (Assume all airfoils stall at around 17 deg)
        # though very low reynolds number result in lower stall angles all the
        # way to 10 deg
        x = min(max(re, 50_000), 1_000_000)
        alpha_s = -2.333e-18*x*x*x + 3.674e-12*x*x - 3.499e-7*x + 0.0086
        alpha_s = min(max(alpha_s,0),1)*math.radians(7) + math.radians(10)
        return alpha_s

    def _alpha_L0(self, re, typ):
        # Calculate the 0 lift angle of attack where, for all airfoils, at very
        # low reynold's number, a_l0 is 0, and at normal flight reynold's numbers
        # a_l0 is whatever is defined for a specific airfoil
        x = min(max(re, 50_000), 1_000_000)
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
        q = 0.5 * atmosphere['rho'] * state['V'] * state['V']

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
        L = 0.5 * wing['S'] * cL * q
        D = 0.5 * wing['S'] * (cDi + cDf_skin + cDf_form) * q
        return L, D

    def _wing_forces(self, sim):
        # Build the arguments
        atmosphere = {'rho' : sim.rho,
                      'mu' : sim.mu}
        wing = {'c' : self.p.c_w,
                'S' : self.p.S_w,
                'AR' : self.p.AR_w,
                'alpha_0' : self.p.alpha_0_w,
                'typ' : self.p.typ_w}
        state_wl = {'V' : sim.V_wl,
                    'alpha' : sim.alpha_wl,
                    'beta' : sim.beta_wl}
        state_wr = {'V' : sim.V_wr,
                    'alpha' : sim.alpha_wr,
                    'beta' : sim.beta_wr}

        # Get the lift and drag of the left wing (in local left wing flow)
        LD_wl = self._L_D(atmosphere, wing, state_wl)

        # Get the lift and drag of the right wing (in local right wing flow)
        LD_wr = self._L_D(atmosphere, wing, state_wr)

        # Convert from local flow coords to world coords
        F_wl_W = sim.R_WL_W @ sim.R_FWL_WL @ (-LD_wl[1], 0.0, -LD_wl[0])
        F_wr_W = sim.R_WR_W @ sim.R_FWR_WR @ (-LD_wr[1], 0.0, -LD_wr[0])
        return F_wl_W, F_wr_W

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
        F_he_FHE = (-LD_h[1]-LD_e[1], 0.0, -LD_h[0]-LD_e[0])
        return sim.R_HE_W @ sim.R_FHE_HE @ F_he_FHE

    def _vert_stab_force(self, sim):
        # Build the arguments
        atmosphere = {'rho' : sim.rho,
                      'mu' : sim.mu}
        v_stab = {'c' : self.p.c_v,
                  'S' : self.p.S_v,
                  'AR' : self.p.AR_v,
                  'alpha_0' : self.p.alpha_0_v,
                  'typ' : self.p.typ_v}
        state_v = {'V' : sim.V_v,
                   'alpha' : sim.alpha_v,
                   'beta' : sim.beta_v}

        # Get the lift and drag of the v stab (in local v stab flow)
        LD_v = self._L_D(atmosphere, v_stab, state_v)

        # Convert from local flow coords to world coords
        # Note that for vert stab, lift force is in -y instead of -z
        return sim.R_V_W @ sim.R_FV_V @ (-LD_v[1], -LD_v[0], 0.0)

    def  _body_force(self, sim):
        # Body lift is treated as though body is low-lift symmetric airfoil
        # with no stall angle
        alpha_0 = 0.5
        alpha_L0 = 0.0
        alpha_s = math.inf
        cL_alpha=self._cL(sim.alpha_b, sim.beta_b, alpha_0, alpha_L0, alpha_s)
        cL_beta=self._cL(sim.beta_b, sim.alpha_b, alpha_0, alpha_L0, alpha_s)
        q = 0.5 * sim.rho * sim.V_b * sim.V_b
        L_alpha = self.p.S_b * cL_alpha * q
        L_beta = self.p.S_b * cL_beta * q

        # Assume body acts approximately like streamline body with cD_tot = 0.045
        D = self.p.S_b * 0.045 * q

        # Convert from local flow coords to world coords
        # Note that for body, side forces are also generated
        return sim.R_B_W @ sim.R_FB_B @ (-D, -L_beta, -L_alpha)

    def prop_rps(self, sim):
        # Prop RPS as a function of engine power
        # (0% power = 0.25*rps_max)
        # (75% power = 0.925*rps_max)
        # (100% power = rps_max)
        const = (71.65)**(sim.P / self.p.P_max)
        num = 0.0174 * const * self.p.rpm_max
        den = const + 3.177
        return num / den

    def _prop_efficiency(self, sim):
        # Ideal prop efficiency based on advance ratio for a 20 degree angle prop
        J = sim.v_CoM[0] / (self.prop_rps(sim)*self.p.prop_D)
        if 0 <= J < 0.87:
            return -1.097*J*J + 1.908*J
        if 0.87 <= J <= 1.05:
            return -25.62*J*J + 44.57*J - 18.56
        return 0.0

    def _prop_force(self, sim):
        # Convert from plane coords to world coords
        T = sim.P * self._prop_efficiency(sim) / sim.V_inf
        return sim.R_CoM_W @ (T, 0.0, 0.0)

    def _r_CoM2CoL_W(self, sim):
        # Adjust dCL based on AoA for nonsymmetric surfaces (wings, hstab, ele)
        # Assume wings, hstab, and ele are all NACA2412 airfoils
        # x_cL moves fore by 10%c at 20 deg AoA and aft by 10%c at -20 deg AoA
        dx_wl = 0.286*self.p.c_w*sim.alpha_wl
        dx_wl = min(max(dx_wl, -.1*self.p.c_w), .1*self.p.c_w)
        dx_wr = 0.286*self.p.c_w*sim.alpha_wr
        dx_wr = min(max(dx_wr, -.1*self.p.c_w), .1*self.p.c_w)
        dx_he = 0.286*self.p.c_he*sim.alpha_he
        dx_he = min(max(dx_he, -.1*self.p.c_he),.1*self.p.c_he)

        r_wl_CoM = (self.p.x_w+dx_wl, -self.p.y_w,  self.p.z_w)
        r_wr_CoM = (self.p.x_w+dx_wr,  self.p.y_w,  self.p.z_w)
        r_he_CoM = (self.p.x_he+dx_he, self.p.y_he, self.p.z_he)
        r_v_CoM = (self.p.x_v, self.p.y_v, self.p.z_v)
        r_b_CoM = (self.p.x_b, self.p.y_b, self.p.z_b)

        return (sim.R_CoM_W @ r_wl_CoM, sim.R_CoM_W @ r_wr_CoM,
                sim.R_CoM_W @ r_he_CoM, sim.R_CoM_W @ r_v_CoM,
                sim.R_CoM_W @ r_b_CoM)

    def _tau_LD(self, sim, forces):
        rs = self._r_CoM2CoL_W(sim)
        lst = (cross(r, F) for (r,F) in zip(rs, forces))
        return tuple(map(sum, zip(*lst)))

    def net_aero_force_torque(self, sim):
        # Get the forces
        F_wl_W, F_wr_W  = self._wing_forces(sim)
        F_he_W = self._h_stab_ele_forces(sim)
        F_v_W = self._vert_stab_force(sim)
        F_b_W = self._body_force(sim)
        F_p_W = self._prop_force(sim)
        F_net_W = F_wl_W + F_wr_W + F_he_W + F_v_W + F_b_W + F_p_W

        # Get the net torque from the lift and drag of the surfaces
        # and from Euler rate-based damping moments
        forces = (F_wl_W, F_wr_W, F_he_W, F_v_W, F_b_W)
        tau_LD_W = self._tau_LD(sim, forces)
        return F_net_W, tau_LD_W

###############################################################################
#FLIGHTSIM CLASS
###############################################################################
class FlightSim:
    _state : dict
    _atmosphere : dict
    _R : dict
    _params : Params
    _model : _FlightModel
    _dt : float
    _t : float
    _r_planet : float
    _turb_param : list

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
        self._dt = dt
        self._t = 0.0
        self._params = params
        self._model = _FlightModel(params)
        self._r_planet = r_planet
        self._turb_param = self._gen_turb_param(turb_mag, seed)

        # Apply the initial euler angles
        self._update_euler_angs(state0)

        # Get the rotation matrices that are determined by euler angles
        # and flow angles at center of mass
        self._update_plane_rot_mats()

        # Set the flow angles at the center of mass and get the
        # relevant rotations
        self._state['alpha'] = state0['alpha']
        self._state['beta'] = state0['beta']
        self._R['CoM_F'] = RYZ(self.alpha, self.beta)
        self._R['F_CoM'] = self.R_CoM_F.T

        # Set the initial world velocity and position
        self._state['v_W'] = self.R_CoM_W@self.R_F_CoM@(state0['V_inf'],0,0)
        self._state['p_W'] = (0, 0, state0['h'])

        # Set the initial inputs
        self._state['delta_e'] = input0['delta_e']
        self._state['P'] = input0['P']
        self._state['delta_e_des'] = input0['delta_e']
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

        params_str = ' '*10 + self._params.__str__().split('(')[0] + ' = { '
        ln = len(params_str)
        for key, val in self._params.params.items():
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
        vals = []
        for scale, offset, period in zip(scales, offsets, periods):
            val = 0.0
            scale_mult = 1.0
            scale_mult_sum = 0.0
            for freq_mult in freq_mults:
                c1 = math.sin(freq_mult*2*math.pi*(time - offset)/period)
                c2 = math.sin(freq_mult*math.pi*math.pi*(time - offset)/period)
                val += scale_mult*(c1 + c2)
                scale_mult_sum += scale_mult
                scale_mult *= 0.5
            vals.append(scale * val / scale_mult_sum)
            scale_mult = 1.0
            scale_mult_sum = 0.0
        return vals

    def _get_turbulence(self):
        F_turbulence = self._noise((1, 2.0, 10.0),
                                   self._turb_param[0],
                                   self._turb_param[1],
                                   self._turb_param[2],
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
        r, p, y = self.phi, self.theta, self.psi

        self._R['W_CoM'] = RXYZ(r, math.pi-p, math.pi+y)
        self._R['W_WL'] = RXYZ(r+self._params.dihedral_w, 3.14159-p, 3.14159+y)
        self._R['W_WR'] = RXYZ(r-self._params.dihedral_w, 3.14159-p, 3.14159+y)
        self._R['CoM_W'] = self.R_W_CoM.T
        self._R['WL_W'] = self.R_W_WL.T
        self._R['WR_W'] = self.R_W_WR.T

        # Alias the identical ones
        self._R['W_HE'] = self.R_W_CoM
        self._R['W_V'] =  self.R_W_CoM
        self._R['W_B'] =  self.R_W_CoM
        self._R['HE_W'] = self.R_CoM_W
        self._R['V_W'] =  self.R_CoM_W
        self._R['B_W'] =  self.R_CoM_W

    def _apply_force(self, F_aero_net, F_turbulence):
        # Get the net force by adding gravity
        F_gravity = (0.0, 0.0, -self.g*self._params.mass)
        F_net = F_aero_net + F_turbulence + F_gravity

        # Apply accelerations and velocities
        self._state['v_W'] += (F_net / self._params.mass) * self._dt
        self._state['p_W'] += self.v_W * self._dt

    def _update_inputs(self, delta_e_des, P_des):
        # Store the desired values
        self._state['delta_e_des'] = delta_e_des
        self._state['P_des'] = P_des

        # Update the elevator angle
        clipped_delta_e = min(max(delta_e_des, self._params.delta_e_min),
                              self._params.delta_e_max)
        d_delta_e = clipped_delta_e - self.delta_e
        if abs(d_delta_e) <= self._params.delta_e_rate * self._dt:
            self._state['delta_e'] += d_delta_e
        else:
            sign = math.copysign(1, d_delta_e)
            d_delta_e = self._params.delta_e_rate*sign*self._dt
            self._state['delta_e'] += d_delta_e

        # Update the power setting
        d_P = min(max(P_des, 0.0), self._params.P_max) - self.P
        if abs(d_P) <= self._params.P_rate * self._dt:
            self._state['P'] += d_P
        else:
            sign = math.copysign(1, d_P)
            self._state['P'] += self._params.P_rate*sign*self._dt

    def _update_world_rot(self):
        d_pitch = -self.v_W[0] * self._dt / (self.p_W[2] + self._r_planet)
        self._state['earth_pitch'] += d_pitch
        d_roll = self.v_W[1] * self._dt / (self.p_W[2] + self._r_planet)
        self._state['earth_roll'] += d_roll

    def _update_atmo(self):
        h = self.p_W[2]
        self._atmosphere['g'] = g(h)
        self._atmosphere['rho'] = rho(h)
        self._atmosphere['T'] = temperature(h)
        self._atmosphere['mu'] = mu(h)

    def _update_velocities(self):
        # Update the local flow velocity at the center of mass
        self._state['v_CoM'] = self.R_W_CoM @ self.v_W

        # Update the flow velocity and dynamic pressure
        self._state['V_inf'] = norm3(self.v_W)

    def _update_alpha_beta(self):
        # Update the center of mass local flow angles
        self._state['alpha'] = math.atan2(self.v_CoM[2], self.v_CoM[0])
        self._state['beta'] = math.asin((self.v_CoM[1] / self.V_inf))

    def _update_local_flow_angs(self):
        # Get the origins of the center of lift for each surface
        O_WL_CoM = (self._params.x_w, -self._params.y_w,  self._params.z_w)
        O_WR_CoM = (self._params.x_w,  self._params.y_w,  self._params.z_w)
        O_HE_CoM = (self._params.x_he, self._params.y_he, self._params.z_he)
        O_V_CoM  = (self._params.x_v,  self._params.y_v,  self._params.z_v)
        O_B_CoM  = (self._params.x_b,  self._params.y_b,  self._params.z_b)

        # Get the rotation of each surface in plane CoM coordinates
        # HE, V, and B are all not rotated
        R_CoM_WL = self.R_W_WL@self.R_CoM_W
        R_CoM_WR = self.R_W_WR@self.R_CoM_W

        # Get the rotation rate of the plane's center of mass about the world
        # in the plane's coordinate's
        omega_CoM_CoM = (self.omega_phi, self.omega_theta, self.omega_psi)

        # Calculate the induced velocity at each of the surfaces from the
        # plane's rotation in surface coords
        vi_wl_WL = R_CoM_WL @ cross(omega_CoM_CoM, O_WL_CoM)
        vi_wr_WR = R_CoM_WR @ cross(omega_CoM_CoM, O_WR_CoM)
        vi_he_HE = cross(omega_CoM_CoM, O_HE_CoM)
        vi_v_V   = cross(omega_CoM_CoM, O_V_CoM)
        vi_b_B   = cross(omega_CoM_CoM, O_B_CoM)

        # Get the total flow velocity at each surface in the surface coords
        v_wl_WL = R_CoM_WL @ self.v_CoM + vi_wl_WL
        v_wr_WR = R_CoM_WL @ self.v_CoM + vi_wr_WR
        v_he_HE = self.v_CoM + vi_he_HE
        v_v_V   = self.v_CoM + vi_v_V
        v_b_B   = self.v_CoM + vi_b_B

        # Save the magnitude of the local surface velocities
        self._state['V_wl'] = norm3(v_wl_WL)
        self._state['V_wr'] = norm3(v_wr_WR)
        self._state['V_he'] = norm3(v_he_HE)
        self._state['V_v'] = norm3(v_v_V)
        self._state['V_b'] = norm3(v_b_B)

        # Update the local flow angles at each of the surfaces
        self._state['alpha_wl'] = math.atan2(v_wl_WL[2], v_wl_WL[0])
        self._state['beta_wl'] = math.asin(v_wl_WL[1] / self.V_wl)
        self._state['alpha_wr'] = math.atan2(v_wr_WR[2], v_wr_WR[0])
        self._state['beta_wr'] = math.asin(v_wr_WR[1] / self.V_wr)
        self._state['alpha_he'] = math.atan2(v_he_HE[2], v_he_HE[0])
        self._state['beta_he'] = math.asin(v_he_HE[1] / self.V_he)
        #BECAUSE V STAB ROTATED, ALPHA AND BETA ARE FLIPPED
        self._state['alpha_v'] = math.asin(v_v_V[1] / self.V_v)
        self._state['beta_v'] = math.atan2(v_v_V[2], v_v_V[0])
        self._state['alpha_b'] = math.atan2(v_b_B[2], v_b_B[0])
        self._state['beta_b'] = math.asin(v_b_B[1] / self.V_b)

    def _update_flow_rot_mats(self):
        self._R['CoM_F'] = RYZ(self.alpha, self.beta)
        self._R['WL_FWL'] = RYZ(self.alpha_wl, self.beta_wl)
        self._R['WR_FWR'] = RYZ(self.alpha_wr, self.beta_wr)
        self._R['HE_FHE'] = RYZ(self.alpha_he, self.beta_he)
        self._R['V_FV'] = RYZ(self.alpha_v, self.beta_v)
        self._R['B_FB'] = RYZ(self.alpha_b, self.beta_b)

        self._R['F_CoM'] = self.R_CoM_F.T
        self._R['FWL_WL'] = self.R_WL_FWL.T
        self._R['FWR_WR'] = self.R_WR_FWR.T
        self._R['FHE_HE'] = self.R_HE_FHE.T
        self._R['FV_V'] = self.R_V_FV.T
        self._R['FB_B'] = self.R_B_FB.T

    def step(self, rotation_state, delta_e_des, P_des):
        # Step 0
        F_aero_net, tau_aero_net = self._model.net_aero_force_torque(self)
        F_turbulence = self._get_turbulence()

        # # Step 1
        self._update_euler_angs(rotation_state)
        self._update_plane_rot_mats()
        self._apply_force(F_aero_net, F_turbulence)
        self._update_inputs(delta_e_des, P_des)
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
        self._t += self._dt

        return tau_aero_net

    def telem(self):
        telem = {'time' : self._t,
                 'h' : self.p_W[2],
                 'V_inf' : self.V_inf,
                 'p_W' : self.p_W,
                 'v_W' : self.v_W,
                 'v_CoM' : self.v_CoM,
                 'R_CoM_W' : self.R_CoM_W,
                 'alpha' : self.alpha,
                 'beta' : self.beta,
                 'omega_psi' : self.omega_psi,
                 'omega_theta' : self.omega_theta,
                 'omega_phi' : self.omega_phi,
                 'psi' : self.psi,
                 'theta' : self.theta,
                 'phi' : self.phi,
                 'delta_e' : self.delta_e,
                 'P' : self.P,
                 'delta_e_des' : self.delta_e_des,
                 'P_des' : self.P_des,
                 'engine_rpm' : 60.0*self._model.prop_rps(self),
                 'earth_pitch' : self.earth_pitch,
                 'earth_roll' : self.earth_roll,}
        return telem
