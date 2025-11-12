# -*- coding: utf-8 -*-
"""
This modules provides transform functions that are used by the simulator.

@author: Grayson
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import numpy as np

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
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)

    # Calculate the norm of vec
    mag1 = np.linalg.norm(arr1)
    mag2 = np.linalg.norm(arr2)

    # If either magnitude is 0, no rotation can be found.
    if mag1==0. or mag2==0.:
        return np.array([0., 0., 0., 1.])

    # If the magnitude is not zero, get the direction of vec
    dirn1 = arr1/mag1
    dirn2 = arr2/mag2

    # If the vec is exactly 180 degrees away, set the 180 deg quaternion
    if (dirn2==-1*dirn1).all():
        return np.array([0.5*np.sqrt(2), -0.5*np.sqrt(2), 0., 0.])

    # If the vec is some other relative orientation, calculate it
    q_xyz = np.cross(dirn1, dirn2)
    q_w = 1.0 + np.dot(dirn1, dirn2)
    xyzw = np.append(q_xyz, q_w)
    xyzw = xyzw/np.linalg.norm(xyzw)
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
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

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
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

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
    return np.array([q3w, q3x, q3y, q3z])

def wxyz_from_euler(yaw, pitch, roll):
    """
    Converts Euler angles to a Hamilton quaternion (wxyz)

    Parameters
    ----------
    yaw : float
        The roll angle in rad.
    pitch : float
        The pitch angle in rad.
    roll : float
        The yaw angle in rad.

    Returns
    -------
    wxyz_quaternion : array-like, shape (4,)
        The Hamilton quaternion representation of the input Euler angles.

    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

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
    # Ensure the quat's norm is greater than 0
    s = np.linalg.norm(wxyz)
    if s == 0.0:
        return np.eye(4)
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
    Rbw = np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])
    return Rbw

def Rbw_from_euler(yaw, pitch, roll):
    """
    Gets the orientation of a body in world coordinates from the euler angles
    of the body.

    Parameters
    ----------
    yaw : float
        The roll angle in rad.
    pitch : float
        The pitch angle in rad.
    roll : float
        The yaw angle in rad.

    Returns
    -------
    R_ofC_inW : array-like, shape(3,3)
        The orientation of the body in world coordinates. This rotation matrix
        takes vectors in body coordinates to world coordinates .

    """
    cx = np.cos(roll)
    sx = np.sin(roll)
    cy = np.cos(pitch)
    sy = np.sin(pitch)
    cz = np.cos(yaw)
    sz = np.sin(yaw)
    Rx = np.array([[1., 0., 0.],
                   [0., cx, -sx],
                   [0., sx, cx]])
    Ry = np.array([[cy, 0., sy],
                   [0., 1., 0.],
                   [-sy, 0., cy]])
    Rz = np.array([[cz, -sz, 0.],
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
    return np.array(Rab).T

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
    return -Rab_to_Rba(Rab) @ np.array(Oab)

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
    return np.array(Rab) @ np.array(va)

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
    return np.array(Rab) @ np.array(pa) + np.array(Oab)
