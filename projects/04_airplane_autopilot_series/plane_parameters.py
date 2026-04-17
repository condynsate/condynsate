# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
# pylint: disable=invalid-name
# pylint: disable=pointless-string-statement
"""
This module implements the parameters for a Cessna 172 airplane.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""
from dataclasses import dataclass
import math

@dataclass()
class Params():
    """
    The empty design parameters class.

    Parameters
    ----------
    None

    Attributes
    ----------
    params : dictionary of parameters
        An empty dictionary.

    """
    params : dict

    def __init__(self):
        self.params = {}

    def __getattr__(self, key):
        return self.params[key]

@dataclass()
class Cessna172(Params):
    """
    Design parameters of a Cessna 172 airplane.

    Parameters
    ----------
    None

    Attributes
    ----------
    params : dictionary of parameters
        The design parameters of a Cessna 172

    """

    def __init__(self):
        super().__init__()

        # Wing param (Gross : left + right + ailerons)
        self.params['S_w'] = 16.2           # Projected top-down area [m^2]
        self.params['c_w'] = 1.49           # Mean aerodynamic chord [m]
        self.params['typ_w'] = 'NACA2412'   # Airfoil type of wing
        self.params['dihedral_w'] = 0.0303  # Diheadral angle [rad]
        self.params['d_eta_d_alpha'] = 0.25 # Downwash angle slope wrt AoA [-]

        # Aileron param (Gross : left + right)
        self.params['S_a'] = 1.362          # Projected top-down area [m^2]
        self.params['c_a'] = 0.312          # Mean aerodynamic chord [m]
        self.params['delta_a_max'] = 0.262  # Max mag deflection of ail [rad]
        self.params['delta_a_rate'] = 0.26  # Max deflection rate of ail [rad/s]

        # Horizontal stab param (Gross : wing + surface)
        self.params['S_h'] = 3.50           # Projected top-down area [m^2]
        self.params['c_h'] = 1.02          # Mean aerodynamic chord [m]
        self.params['typ_h'] = 'NACA2412i'  # Airfoil type of h stab

        # Elevator param
        self.params['S_e'] = 1.37           # Projected top-down area [m^2]
        self.params['delta_e_min'] = -0.332 # Max downward deflection of ele [rad]
        self.params['delta_e_max'] = 0.384  # Max upward deflection of ele [rad]
        self.params['delta_e_rate'] = 0.26  # Max deflection rate of ele [rad/s]

        # Vertical stab parameters (Gross : wing + surface)
        self.params['S_v'] = 2.16         # Projected side area [m^2]
        self.params['c_v'] = 1.42         # Mean aerodynamic chord [m]
        self.params['typ_v'] = 'NACA2012' # Airfoil type of v stab

        # Rudder parameters
        self.params['S_r'] = 0.846          # Projected side area [m^2]
        self.params['delta_r_max'] = 0.279  # Max mag deflection of rud [rad]
        self.params['delta_r_rate'] = 0.26  # Max deflection rate of rud [rad/s]

        # Body parameters
        self.params['S_b'] = 5.59   # Project side area [m^2]
        self.params['mass'] = 964.0 # Mass [kg]

        # Distance from CoM to center of lift (at 0 AoA)
        # positive in front of and below CoM
        self.params['x_w'] = -0.156  # Axial distance (wing) [m]
        self.params['z_w'] = -0.971  # Vertical distance (wing) [m]
        self.params['x_he'] = -4.59  # Axial distance (hori stab + ele) [m]
        self.params['z_he'] = 0.0288 # Vertical distance (hori stab + ele) [m]
        self.params['x_v'] = -4.81   # Axial distance (v stab + rud) [m]
        self.params['z_v'] = -0.281  # Vertical distance (v stab + rud) [m]
        self.params['x_b'] = -0.608  # Axial distance (fuselage) [m]
        self.params['z_b'] = -0.127  # Vertical distance (fuselage) [m]

        # Powerplant sizing
        self.params['P_max'] = 1.342e5  # Max engine power setting [kW]
        self.params['P_rate'] = 3.4e4   # Max engine power setting rate [kW/s]
        self.params['rpm_max'] = 2700.0 # RPM of prop at 100% power [rpm]
        self.params['prop_D'] = 1.905   # Diameter of prop [m]
        self.params['T_max'] = 2447     # Max thrust (v=0.0, P=100%, STP) [N]
        self.params['T_cruise'] = 1557  # Thrust @ cruise (v=V_cruise, P=75%, STP) [N]
        self.params['V_cruise'] = 64.82 # Cruise velocity (P=75%, STP)[m/s]

        # Calculated params
        self._calculate_other_params()

    def _b_AR_alpha_0_tau(self, S_w, c_w, S_s, c_s):
        # Geometric consistency
        b_w = S_w / c_w
        AR_w = b_w**2 / S_w
        b_s = S_s / c_s

        # Lifting line, TAT, oswald correction factor = 0.85
        alpha_0 = 2*math.pi*AR_w / (AR_w + 2.353)

        # Effectiveness of surface (Equal to 1 when 100% of gross wing is surf)
        tau = math.sqrt(S_s / S_w)
        return (b_w, AR_w, alpha_0), (b_s, tau)

    def _calculate_other_params(self):
        # Get the full length wing parameters
        wing, surf = self._b_AR_alpha_0_tau(self.params['S_w'], self.params['c_w'],
                                            self.params['S_a'], self.params['c_a'])
        self.params['b_w'] = wing[0]       # Span [m]
        self.params['AR_w'] = wing[1]      # Aspect ratio [-]
        self.params['alpha_0_w'] = wing[2] # cL slope wrt AoA [1/rad]
        self.params['b_a'] = surf[0]       # Span [m]
        self.params['tau_a'] = surf[1]     # Surface effectiveness [-]

        # Calculated h stab and elevator
        # Assign c_e so that b_e == b_h
        self.params['c_e'] = self.params['c_h']*self.params['S_e']/self.params['S_h']
        wing, surf = self._b_AR_alpha_0_tau(self.params['S_h'], self.params['c_h'],
                                            self.params['S_e'], self.params['c_e'])
        self.params['b_h'] = wing[0]       # Span [m]
        self.params['AR_h'] = wing[1]      # Aspect ratio [-]
        self.params['alpha_0_h'] = wing[2] # cL slope wrt AoA [1/rad]
        self.params['b_e'] = surf[0]       # Span [m]
        self.params['tau_e'] = surf[1]     # Surface effectiveness [-]

        # Calculated v stab and rudder
        # Assign c_r so that b_r == b_v
        self.params['c_r'] = self.params['c_v']*self.params['S_r']/self.params['S_v']
        wing, surf = self._b_AR_alpha_0_tau(self.params['S_v'], self.params['c_v'],
                                            self.params['S_r'], self.params['c_r'])
        self.params['b_v'] = wing[0]       # Span [m]
        self.params['AR_v'] = wing[1]      # Aspect ratio [-]
        self.params['alpha_0_v'] = wing[2] # cL slope wrt AoA [1/rad]
        self.params['b_r'] = surf[0]       # Span [m]
        self.params['tau_r'] = surf[1]     # Surface effectiveness [-]
