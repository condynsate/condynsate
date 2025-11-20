# -*- coding: utf-8 -*-
"""
This module provides the dataclasses which simulator objects use.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
from dataclasses import dataclass
import numpy as np
import condynsate.misc.transforms as t

###############################################################################
#BODY CLASS
###############################################################################
@dataclass(frozen=True)
class BodyState():
    """
    Stores state information about the base of a body. Has 7 attributes:
        position : 3 tuple of floats
            The (x,y,z) position in world coordinates.
        orientation : 4 tuple of floats
            The wxyz quaternion representation of the orientation in world
            coordinates.
        ypr : 3 tuple of floats
            The (z-y'-x' Tait–Bryan) Euler angles in radians ordered as
            (yaw, pitch, roll).
        velocity : 3 tuple of floats
            The (x,y,z) velocity in world coordinates.
        omega : 3 tuple of floats
            The (x,y,z) angular velocity in world coordinates.
        velocity_in_body : 3 tuple of floats
            The (x,y,z) velocity in body coordinates.
        omega_in_body : 3 tuple of floats
            The (x,y,z) angular velocity in body coordinates.

    Parameters
    ----------
    **kwargs :
        State information with the following acceptable keys
        position : 3 tuple of floats, optional
            The XYZ position in world coordinates.
            The default is (0., 0., 0.)
        orientation : 4 tuple of floats, optional
            The wxyz quaternion representation of the orientation in world
            coordinates. The default is (1., 0., 0., 0.)
        velocity : 3 tuple of floats, optional
            The XYZ velocity in either world or body coordinates. Body
            coordinates are defined based on objects orientation.
            The default is (0., 0., 0.)
        omega : 3 tuple of floats, optional
            The XYZ angular velocity in either world or body coordinates.
            Body coordinates are defined based on objects orientation.
            The default is (0., 0., 0.)
        body : bool, optional
            Whether velocity and omega are being set in world or body
            coordinates. The default is False
    """
    position: tuple
    orientation: tuple
    ypr: tuple
    velocity: tuple
    omega: tuple
    velocity_in_body: tuple
    omega_in_body: tuple

    def __init__(self, **kwargs):
        # Read kwargs
        body = kwargs.get('body', False)
        position = kwargs.get('position', (0.0, 0.0, 0.0))
        orientation = kwargs.get('orientation', (1.0, 0.0, 0.0, 0.0))
        velocity = kwargs.get('velocity', (0.0, 0.0, 0.0))
        omega = kwargs.get('omega', (0.0, 0.0, 0.0))

        # Set states
        self._set_position(position)
        self._set_orientation(orientation)
        self._set_ypr()
        if body:
            self._set_velocity_in_body(velocity)
            self._set_omega_in_body(omega)
        else:
            self._set_velocity(velocity)
            self._set_omega(omega)
        self._set_body_vels()

    def _set_position(self, position):
        p0 = tuple(float(p) for p in position)
        super().__setattr__('position', p0)

    def _set_orientation(self, orientation):
        q0 = np.array([q for i,q in enumerate(orientation) if i < 4])
        q0 = tuple((q0 / np.linalg.norm(q0)).tolist())
        super().__setattr__('orientation', q0)

    def _set_velocity(self, velocity):
        v0 = tuple(float(v) for v in velocity)
        super().__setattr__('velocity', v0)

    def _set_velocity_in_body(self, velocity):
        Rbw = t.Rbw_from_wxyz(self.orientation)
        v0 = tuple(t.va_to_vb(Rbw, velocity).tolist())
        super().__setattr__('velocity', v0)

    def _set_omega(self, omega):
        o0 = tuple(float(w) for w in omega)
        super().__setattr__('omega', o0)

    def _set_omega_in_body(self, omega):
        Rbw = t.Rbw_from_wxyz(self.orientation)
        o0 = tuple(t.va_to_vb(Rbw, omega).tolist())
        super().__setattr__('omega', o0)

    def _set_ypr(self):
        ypr = tuple(float(e) for e in t.euler_from_wxyz(self.orientation))
        super().__setattr__('ypr', ypr)

    def _set_body_vels(self):
        Rbw = t.Rbw_from_wxyz(self.orientation)
        Rwb = t.Rab_to_Rba(Rbw)
        vb = tuple(t.va_to_vb(Rwb, self.velocity).tolist())
        super().__setattr__('velocity_in_body', vb)
        ob = tuple(t.va_to_vb(Rwb, self.omega).tolist())
        super().__setattr__('omega_in_body', ob)

###############################################################################
#JOINT CLASS
###############################################################################
@dataclass(frozen=True)
class JointState():
    """
    Stores state information about a joint.

    Parameters
    ----------
    **kwargs :
        State information with the following acceptable keys
        angle : float, optional
            The angle of the joint about the joint axis. The default is 0.
        omega : float, optional
            The angular velocity of the joint about the joint axis.
            The default is 0.
    """
    angle: float
    omega: float

    def __init__(self, **kwargs):
        self._set_angle(kwargs.get('angle', 0.0))
        self._set_omega(kwargs.get('omega', 0.0))

    def _set_angle(self, angle):
        super().__setattr__('angle', float(angle))

    def _set_omega(self, omega):
        super().__setattr__('omega', float(omega))

###############################################################################
#LINK CLASS
###############################################################################
@dataclass(frozen=True)
class LinkState(BodyState):
    """
    Stores state information about a link. Has 7 attributes:
        position : 3 tuple of floats
            The (x,y,z) position in world coordinates.
        orientation : 4 tuple of floats
            The wxyz quaternion representation of the orientation in world
            coordinates.
        ypr : 3 tuple of floats
            The (z-y'-x' Tait–Bryan) Euler angles in radians ordered as
            (yaw, pitch, roll).
        velocity : 3 tuple of floats
            The (x,y,z) velocity in world coordinates.
        omega : 3 tuple of floats
            The (x,y,z) angular velocity in world coordinates.
        velocity_in_body : 3 tuple of floats
            The (x,y,z) velocity in body coordinates.
        omega_in_body : 3 tuple of floats
            The (x,y,z) angular velocity in body coordinates.

    Parameters
    ----------
    **kwargs :
        State information with the following acceptable keys
        position : 3 tuple of floats, optional
            The XYZ position in world coordinates.
            The default is (0., 0., 0.)
        orientation : 4 tuple of floats, optional
            The wxyz quaternion representation of the orientation in world
            coordinates. The default is (1., 0., 0., 0.)
        velocity : 3 tuple of floats, optional
            The XYZ velocity in either world or body coordinates. Body
            coordinates are defined based on objects orientation.
            The default is (0., 0., 0.)
        omega : 3 tuple of floats, optional
            The XYZ angular velocity in either world or body coordinates.
            Body coordinates are defined based on objects orientation.
            The default is (0., 0., 0.)
        body : bool, optional
            Whether velocity and omega are being set in world or body
            coordinates. The default is False
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
