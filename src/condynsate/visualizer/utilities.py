"""
This module provides utilities functions used by the Visualizer class.
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import os
import numpy as np

###############################################################################
#ARGUMENT CHECKING FUNCTIONS
###############################################################################
def is_num(arg):
    """
    Ensures that an argument is a number.

    Parameters
    ----------
    arg : TYPE
        The argument being tested.

    Returns
    -------
    is_num : bool
        A Boolean flag that indicates if arg is valid.

    """
    # If float castable, not inf, and not nan, is a number
    try:
        f = float(arg)
        return (not np.isinf(f)) and (not np.isnan(f))

    # If something went wrong, is not a number
    except (TypeError, ValueError):
        return False

def is_nvector(arg, n):
    """
    Ensures that an argument is a nvector of numbers.

    Parameters
    ----------
    arg : TYPE
        The argument being tested.
    n : int
        The desired length of vector.

    Returns
    -------
    is_nvec : bool
        A Boolean flag that indicates if arg is valid.

    """
    try:
        iter(arg) # Ensure iterable
        if len(arg) != n: # Ensure of length 3
            raise TypeError('Arg of wrong length')

        # Ensure each arg is number
        return all(is_num(a) for a in arg)

    # If something went wrong, arg is not a 3vector
    except TypeError:
        return False

def name_valid(name):
    """
    Ensures that a name tuple is valid. Does not check if the object is 
    actually a member of the scene.

    Parameters
    ----------
    name : string or tuple of strings
        A list of strings defining the name of a scene object as well
        as its position in the scene heirarchy. For example, 
        ('foo', 'bar') refers to the object at the scene location
        /Scene/foo/bar while 'baz' refers to the object at scene location
        /Scene/baz

    Returns
    -------
    name_valid : bool
        A Boolean flag that indicates if the name tuple is a valid name tuple.

    """
    if isinstance(name, (tuple, list, np.ndarray)):
        if not all(isinstance(n, str) for n in name):
            return False
    elif not isinstance(name, str):
        return False
    return True

def path_valid(path):
    """
    Checks if a given file path is valid.

    Parameters
    ----------
    path : string
        The file path being validated.

    Returns
    -------
    is_valid : bool
        True if valid, else false.

    """
    if not  isinstance(path, str):
        return False
    split = list(os.path.split(path))
    try:
        if split[0] == '':
            split[0] = '.'
        if not split[1] in os.listdir(split[0]):
            return False
    except FileNotFoundError:
        return False
    return True

###############################################################################
#TRANSFORMATION FUNCTIONS
###############################################################################
def from_quat(wxyz_quat):
    """
    Converts a wxyz quaternion into a 4x4 homogeneous transform matrix

    Parameters
    ----------
    wxyz_quat : 4vector of floats
        The wxyz quaternion being converted.

    Returns
    -------
    R : 4X4 matrix
        The equivalent homogeneous transform matrix.

    """               
    # Extract the values from Q
    qr = wxyz_quat[0]
    qi = wxyz_quat[1]
    qj = wxyz_quat[2]
    qk = wxyz_quat[3]
    s = np.linalg.norm(wxyz_quat)**(-2)
     
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
    R = np.array([[r00, r01, r02, 0.0],
                  [r10, r11, r12, 0.0],
                  [r20, r21, r22, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    return R

def from_ypr(yaw, pitch, roll):
    """
    Converts intrinsic yaw, pitch, and roll angles in degrees to the
    equivalient homogeneous rotation matrix.

    Parameters
    ----------
    yaw : float
        The intrinsic yaw angle in degrees. Defined about the object's 
        intrinsic Z axis.
    pitch : float
        The intrinsic pitch angle in degrees. Defined about the object's 
        intrinsic Y axis.
    roll : float
        The intrinsic roll angle in degrees. Defined about the object's 
        intrinsic X axis.

    Returns
    -------
    R : 4X4 matrix
        The equivalent homogeneous transform matrix.

    """
    # Build yaw transform
    y = yaw*np.pi / 180
    cy = np.cos(y)
    sy = np.sin(y)
    Y = np.array([[ cy, -sy,  0.,  0.],
                  [ sy,  cy,  0.,  0.],
                  [ 0.,  0.,  1.,  0.],
                  [ 0.,  0.,  0.,  1.]])

    # Build pitch transform
    p = pitch*np.pi / 180
    cp = np.cos(p)
    sp = np.sin(p)
    P = np.array([[ cp,  0.,  sp,  0.],
                  [ 0.,  1.,  0.,  0.],
                  [-sp,  0.,  cp,  0.],
                  [ 0.,  0.,  0.,  1.]])

    # Build roll transform
    r = roll*np.pi / 180
    cr = np.cos(r)
    sr = np.sin(r)
    R = np.array([[ 1.,  0.,  0.,  0.],
                  [ 0.,  cr, -sr,  0.],
                  [ 0.,  sr,  cr,  0.],
                  [ 0.,  0.,  0.,  1.]])

    # Build rotation matrix
    R = Y @ P @ R
    return R

def homogeneous_transform(translation, wxyz_quat, yaw, pitch, roll, scale):
    """
    Builds a homogeneous cooridinate transform matrix representing the 
    equivalent transform described by a translation, wxyz quaternion, yaw
    pitch, roll, and scale arguments. The transforms are applied in the order
    scaling, wxyz quaternion rotation, YPR rotation, translation such that the
    resultant homogeneous matrix is given by
    H = T @ R_y @ R_p @ R_r @ R_quat @ S. Does not validate inputs.

    Parameters
    ----------
    translation : 3vector of floats
        A 3 vector defining the extrinsic translation to apply.
    wxyz_quat : 4vector of floats
        A 4 vector defining the extrinsic rotation to apply.
    yaw : float
        The intrinsic yaw (degrees) about the objects Z axis to apply.
    pitch : float
        The intrinsic pitch (degrees) about the objects Y axis to apply.
    roll : float
        The intrinsic roll (degrees) about the objects X axis to apply.
    scale : 3vector of floats
        The extrinsic scaling to apply.

    Returns
    -------
    H : 4X4 matrix
        The equivalent homogeneous coordinate transformation matrix.

    """
    # Build the scaling matrix
    S = np.eye(4)
    for i,s in enumerate(scale):
        S[i,i] = s
    
    # Build the rotation matrix
    R_quat = from_quat(wxyz_quat)
    R_ypr = from_ypr(yaw, pitch, roll)
    
    # Build the translation matrix
    T = np.eye(4)
    for i,t in enumerate(translation):
        T[i,3] = t
    
    # Combine the translation, rotation, and scaling matrices
    H = T @ R_ypr @ R_quat @ S
    return H

###############################################################################
#SCENE PATH MANIPULATION
###############################################################################
def get_scene_path(name):
    """
    Converts a list of strings to a formatted scene path.

    Parameters
    ----------
    name : string or tuple of strings
        A list of strings defining the name and position of a scene element 
        as well  in the scene heirarchy. For example, 
        ('foo', 'bar') refers to /Scene/foo/bar while 'baz' refers to
        /Scene/baz

    Returns
    -------
    scene_path : String
        The formatted scene path.

    """
    # Get the scene path
    if isinstance(name, (tuple, list, np.ndarray)):
        scene_path = '/'+'/'.join(name)
    else:
        scene_path = '/'+name
    return scene_path