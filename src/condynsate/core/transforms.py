# -*- coding: utf-8 -*-
"""
This modules provides transform functions that are used by the simulator.

@author: Grayson
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import numpy as _np

###############################################################################
#TRANSFORM FUNCTIONS
###############################################################################
def xyzw_from_vecs(vec1, vec2):
    """
    Calculates a JPL (xyzw) quaternion representing the transformation of the
    the vec1 vector to the vec2 vector.

    Parameters
    ----------
    vec1 : array-like, shape(3,)
        The initial vector.
    vec2 : array-like, shape(3,)
        The vector to which the transformation is calculated.

    Returns
    -------
    xyzw : array-like, shape(4,)
        The JPL quaternion (xyzw) the takes the vec1 vector to the
        vec2 vector (without scaling). dirn(vec2) = dirn(xyzw*vec1)

    """
    # Convert to numpy array
    arr1 = _np.array(vec1)
    arr2 = _np.array(vec2)

    # Calculate the norm of vec
    mag1 = _np.linalg.norm(arr1)
    mag2 = _np.linalg.norm(arr2)

    # If either magnitude is 0, no rotation can be found.
    if mag1==0. or mag2==0.:
        return _np.array([0., 0., 0., 1.])

    # If the magnitude is not zero, get the direction of vec
    dirn1 = arr1/mag1
    dirn2 = arr2/mag2

    # If the vec is exactly 180 degrees away, set the 180 deg quaternion
    if (dirn2==-1*dirn1).all():
        return _np.array([0.5*_np.sqrt(2), -0.5*_np.sqrt(2), 0., 0.])

    # If the vec is some other relative orientation, calculate it
    q_xyz = _np.cross(dirn1, dirn2)
    q_w = 1.0 + _np.dot(dirn1, dirn2)
    xyzw = _np.append(q_xyz, q_w)
    xyzw = xyzw/_np.linalg.norm(xyzw)
    return xyzw

def xyzw_to_wxyz(xyzw):
    """
    Converts a JPL quaternion (xyzw) to a Hamilton quaternion (wxyz)

    Parameters
    ----------
    xyzw : array-like, size (4,)
        A JPL quaternion to be converted.

    Returns
    -------
    wxyz : array-like, size (4,)
        The Hamilton representation of the input JPL quaterion

    """
    return _np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

def wxyz_to_xyzw(wxyz):
    """
    Converts a Hamilton quaternion (wxyz) to a JPL quaternion (xyzw)

    Parameters
    ----------
    wxyz : array-like, size (4,)
        A Hamilton quaternion to be converted.

    Returns
    -------
    xyzw : array-like, size (4,)
        The JPL representation of the input Hamilton quaterion.

    """
    return _np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

def xyzw_mult(q1, q2):
    """
    Gets the resultant JPL quaternion (xyzw) that arises from first
    applying the q1 (xyzw) rotation then applying the q2 (xyzw) rotation.

    Parameters
    ----------
    q1 : array-like, shape(4,)
        The first xyzw quaternion applied.
    q2 : array-like, shape(4,)
        The second xyzw quaternion applied.

    Returns
    -------
    q3 : array-like, shape(4,)
        The resultant transformation from first doing the q1 transformation
        then doing the q2 transformation. Given in JPL form (xyzw).

    """
    q3_wxyz = wxyz_mult(xyzw_to_wxyz(q1), xyzw_to_wxyz(q2))
    return wxyz_to_xyzw(q3_wxyz)

def wxyz_mult(q1, q2):
    """
    Gets the resultant Hamilton quaternion (wxyz) that arises from first
    applying the q1 (wxyz) rotation then applying the q2 (wxyz) rotation.

    Parameters
    ----------
    q1 : array-like, shape(4,)
        The first wxyz quaternion applied.
    q2 : array-like, shape(4,)
        The second wxyz quaternion applied.

    Returns
    -------
    q3 : array-like, shape(4,)
        The resultant transformation from first doing the q1 transformation
        then doing the q2 transformation. Given in Hamilton form (wxyz).

    """
    a1 = q1[0]
    b1 = q1[1]
    c1 = q1[2]
    d1 = q1[3]
    a2 = q2[0]
    b2 = q2[1]
    c2 = q2[2]
    d2 = q2[3]
    q3w = a2*a1 - b2*b1 - c2*c1 - d2*d1
    q3x = a2*b1 + b2*a1 + c2*d1 - d2*c1
    q3y = a2*c1 - b2*d1 + c2*a1 + d2*b1
    q3z = a2*d1 + b2*c1 - c2*b1 + d2*a1
    return _np.array([q3w, q3x, q3y, q3z])

def wxyz_from_euler(yaw, pitch, roll):
    """
    Converts (Tait–Bryan) Euler Angles to a Hamilton quaternion (wxyz)

    Parameters
    ----------
    yaw : float
        The yaw angle in rad.
    pitch : float
        The pitch angle in rad.
    roll : float
        The roll angle in rad.

    Returns
    -------
    wxyz_quaternion : array-like, shape (4,)
        The Hamilton quaternion representation of the input Euler angles.

    """
    cr = _np.cos(roll * 0.5)
    sr = _np.sin(roll * 0.5)
    cp = _np.cos(pitch * 0.5)
    sp = _np.sin(pitch * 0.5)
    cy = _np.cos(yaw * 0.5)
    sy = _np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return _np.array([w, x, y, z])

def euler_from_wxyz(wxyz):
    """
    Converts a Hamilton quaternion (wxyz) to (Tait–Bryan) Euler Angles.

    Parameters
    ----------
    wxyz : array-like, shape (4,)
        The wxyz quaternion being converted.

    Returns
    -------
    yaw : float
        The equivalent yaw angle in rad.
    pitch : float
        The equivalent pitch angle in rad.
    roll : float
        The equivalent roll angle in rad.

    """
    w = wxyz[0]
    x = wxyz[1]
    y = wxyz[2]
    z = wxyz[3]
    roll = _np.arctan2(2.*(w*x+y*z), 1.-2.*(x*x+y*y))
    p1 = _np.sqrt(1.+2.*(w*y-x*z))
    p2 = _np.sqrt(1.-2.*(w*y-x*z))
    pitch = 2.*_np.arctan2(p1, p2) - 0.5*_np.pi
    yaw = _np.arctan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    return yaw, pitch, roll

def Rbw_from_wxyz(wxyz):
    """
    Converts a wxyz quaternion into a rotation matrix

    Parameters
    ----------
    wxyz : 4vector of floats
        The wxyz quaternion being converted.

    Returns
    -------
    Rbw : 3x3 matrix
        The equivalent rotation matrix.

    """
    # Trivial case
    if all(_np.isclose(wxyz, [1., 0., 0., 0.])):
        return _np.eye(3)

    # Ensure the quat's norm is greater than 0
    s = _np.linalg.norm(wxyz)
    if s == 0.0:
        return _np.eye(4)
    s = s**-2

    # Extract the values from Q
    qr = wxyz[0]
    qi = wxyz[1]
    qj = wxyz[2]
    qk = wxyz[3]

    # First row of the rotation matrix
    r00 = 1. - 2.*s*(qj*qj + qk*qk)
    r01 = 2.*s*(qi*qj - qk*qr)
    r02 = 2.*s*(qi*qk + qj*qr)

    # Second row of the rotation matrix
    r10 = 2.*s*(qi*qj + qk*qr)
    r11 = 1. - 2.*s*(qi*qi + qk*qk)
    r12 = 2.*s*(qj*qk - qi*qr)

    # Third row of the rotation matrix
    r20 = 2.*s*(qi*qk - qj*qr)
    r21 = 2.*s*(qj*qk + qi*qr)
    r22 = 1. - 2.*s*(qi*qi + qj*qj)

    # Build the rotation matrix
    Rbw = _np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])
    return Rbw

def Rbw_from_euler(yaw, pitch, roll):
    """
    Gets the orientation of a body in world coordinates from the (Tait–Bryan)
    Euler Angles of the body.

    Parameters
    ----------
    yaw : float
        The yaw angle in rad.
    pitch : float
        The pitch angle in rad.
    roll : float
        The roll angle in rad.

    Returns
    -------
    R_ofC_inW : array-like, shape(3,3)
        The orientation of the body in world coordinates. This rotation matrix
        takes vectors in body coordinates to world coordinates .

    """
    cx = _np.cos(roll)
    sx = _np.sin(roll)
    cy = _np.cos(pitch)
    sy = _np.sin(pitch)
    cz = _np.cos(yaw)
    sz = _np.sin(yaw)
    Rx = _np.array([[1., 0., 0.],
                    [0., cx, -sx],
                    [0., sx, cx]])
    Ry = _np.array([[cy, 0., sy],
                    [0., 1., 0.],
                    [-sy, 0., cy]])
    Rz = _np.array([[cz, -sz, 0.],
                    [sz, cz, 0.],
                    [0., 0., 1.]])
    return Rz@Ry@Rx

def Rab_to_Rba(Rab):
    """
    Takes a rotation cosine matrix of the a frame in b coords to the rotation
    cosine of the b frame in a coords.

    Parameters
    ----------
    Rab : valid rotation matrix, shape(3,3)
        The rotation cosine matrix of the a frame in b coords.

    Returns
    -------
    Rba : valid rotation matrix, shape(3,3)
        The rotation cosine matrix of the b frame in a coords.

    """
    return _np.array(Rab).T

def Oab_to_Oba(Rab, Oab):
    """
    Takes the origin of frame a in b coords to the origin of frame b in a
    coords.

    Parameters
    ----------
    Rab : valid rotation matrix, shape(3,3)
        The rotation cosine matrix of the a frame in b coords.
    Oab : array, shape(3,)
        The origin of frame a in b coords.

    Returns
    -------
    Oba : array, shape(3,)
        The origin of frame b in a coords.

    """
    return -Rab_to_Rba(Rab) @ _np.array(Oab)

def va_to_vb(Rab, va):
    """
    Based on a relative cosine matrix, takes vecotrs in frame a coords to
    frame b coords.

    Parameters
    ----------
    Rab : valid rotation matrix, shape(3,3)
        The rotation cosine matrix of the a frame in b coords.
    va : array, shape(3,)
        A 3 vector in frame a.

    Returns
    -------
    vb : array, shape(3,)
        The same 3 vector in frame b coords.

    """
    return _np.array(Rab) @ _np.array(va)

def pa_to_pb(Rab, Oab, pa):
    """
    Based on a relative cosine matrix and origin vector, takes a point in frame
    a coords to frame b coords.

    Parameters
    ----------
    Rab : valid rotation matrix, shape(3,3)
        The rotation cosine matrix of the a frame in b coords.
    Oab : array, shape(3,)
        The origin of frame a in b coords.
    pa : array, shape(3,)
        A 3D point in frame a.

    Returns
    -------
    pb : array, shape(3,)
        The same 3D point in b coords.

    """
    return _np.array(Rab) @ _np.array(pa) + _np.array(Oab)
