# -*- coding: utf-8 -*-
"""
This module implements the parameters for a Cessna 172 airplane.
"""
"""
© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""
from dataclasses import dataclass

@dataclass()
class Params():
    params : dict

    def __init__(self):
        self.params = {}

    def __getattr__(self, key):
        return self.params[key]

@dataclass()
class Cessna172(Params):

    def __init__(self):
        super().__init__()

        # Wing param (Params for combined left right wing system)
        self.params['alpha_0_w'] = 5.01     # cL slope wrt AoA [1/rad]
        self.params['S_w'] = 14.5           # Projected top-down area [m^2]
        self.params['AR_w'] = 7.52          # Aspect ratio [-]
        self.params['dihedral_w'] = 0.0297  # Diheadral angle [rad]
        self.params['d_eta_d_alpha'] = 0.25 # Downwash angle slope wrt AoA [-]

        # Aileron param (Param for single aileron not both ailerons)
        self.params['alpha_0_a'] = 4.77 # cL slope wrt AoA [1/rad]
        self.params['S_a'] = 0.850      # Projected top-down area [m^2]
        self.params['AR_a'] = 7.17      # Aspect ratio [-]

        # Combined wing and aileron params
        self.params['c_wa'] = 1.49           # Mean aerodynamic chord [m]
        self.params['typ_wa'] = 'NACA2412'   # Airfoil type of wing

        # Horizontal stab param
        self.params['alpha_0_h'] = 4.60  # cL slope wrt AoA [1/rad]
        self.params['S_h'] = 2.00        # Projected top-down area [m^2]
        self.params['AR_h'] = 6.32       # Aspect ratio [-]

        # Elevator param
        self.params['alpha_0_e'] = 5.80  # cL slope wrt AoA [1/rad]
        self.params['S_e'] = 1.35        # Projected top-down area [m^2]
        self.params['AR_e'] = 9.37       # Aspect ratio [-]

        # Combined horizontal and elevator parameters
        self.params['c_he'] = 0.942         # Mean aerodynamic chord [m]
        self.params['typ_he'] = 'NACA2412i' # Airfoil type of h stab + ele

        # Vertical stab parameters
        self.params['alpha_0_v'] = 1.95 # cL slope wrt AoA [1/rad]
        self.params['S_v'] = 1.04       # Projected side area [m^2]
        self.params['AR_v'] = 1.37      # Aspect ratio [-]

        # Rudder parameters
        self.params['alpha_0_r'] = 3.67 # cL slope wrt AoA [1/rad]
        self.params['S_r'] = 0.690      # Projected side area [m^2]
        self.params['AR_r'] = 3.55      # Aspect ratio [-]

        # Combined v stab and rudder parameters
        self.params['c_vr'] = 1.31         # Mean aerodynamic chord [m]
        self.params['typ_vr'] = 'NACA2012' # Airfoil type of v stab + rudder

        # Body parameters
        self.params['S_b'] = 5.59   # Project side area [m^2]
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

        # Elevator limits
        self.params['delta_e_min'] = -0.332 # Max downward deflection of ele [rad]
        self.params['delta_e_max'] = 0.384  # Max upward deflection of ele [rad]
        self.params['delta_e_rate'] = 0.26  # Max deflection rate of ele [rad/s]

        # Rudder limits
        self.params['delta_r_max'] = 0.349  # Max mag deflection of rud [rad]
        self.params['delta_r_rate'] = 0.26  # Max deflection rate of rud [rad/s]

        # Ailerion limits
        self.params['delta_a_max'] = 0.349  # Max mag deflection of ail [rad]
        self.params['delta_a_rate'] = 0.26  # Max deflection rate of ail [rad/s]

        # Powerplant sizing
        self.params['P_max'] = 1.342e5  # Max engine power setting [kW]
        self.params['P_rate'] = 3.4e4   # Max engine power setting rate [kW/s]
        self.params['rpm_max'] = 2700.0 # RPM of prop at 100% power [rpm]
        self.params['prop_D'] = 1.905   # Diameter of prop [m]
        self.params['T_max'] = 2447     # Max thrust (v=0.0, P=100%, STP) [N]
        self.params['T_cruise'] = 1557  # Thrust @ cruise (v=V_cruise, P=75%, STP) [N]
        self.params['V_cruise'] = 64.82 # Cruise velocity (P=75%, STP)[m/s]
