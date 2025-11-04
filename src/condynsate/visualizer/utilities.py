"""
This module provides utilities functions used by the Visualizer class.
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import os
import numpy as np
from warnings import warn

###############################################################################
#ARGUMENT CHECKING FUNCTIONS
###############################################################################
def is_instance(arg, typ, arg_name=None):
    """
    Returns True if arg is type typ. Else, False.

    Parameters
    ----------
    arg
        The variable being tested.
    typ
        The type against which arg is compared.
    arg_name : String, optional
        The name of the argument. When not None, a warning will be output if
        function returns false. The default is None

    Returns
    -------
    bool
        If arg is type typ.

    """
    # Check arg is not None.
    if arg is None:
        if not arg_name is None:
            msg = f"{arg_name} cannot be None."
            warn(msg)
        return False
        
    # Check arg is correct type
    if not isinstance(arg, typ):
        if not arg_name is None:
            msg = f"{arg_name} must be type {typ}."
            warn(msg)
        return False
    
    # All tests passed
    return True

def is_num(arg, arg_name=None):
    """
    Returns True if arg is float castable and not inf and not nan. Else, False.

    Parameters
    ----------
    arg
        The variable being tested.
    arg_name : String, optional
        The name of the argument. When not None, a warning will be output if
        function returns false. The default is None

    Returns
    -------
    bool
        If arg is float castable and not inf and not nan.

    """
    # Check if float castable
    try:
        float(arg)
    except (TypeError, ValueError):
        if not arg_name is None:
            msg = f"{arg_name} must be castable to <class 'float'>."
            warn(msg)
        return False
    
    # Check if not inf and not nan
    is_inf_or_nan = np.isinf(float(arg)) or np.isnan(float(arg))
    if is_inf_or_nan and not arg_name is None:
        msg = f"{arg_name} cannot be inf or nan."
        warn(msg)
    return not is_inf_or_nan

def is_nvector(arg, n, arg_name=None):
    """
    Returns True if arg is n-vector of non-inf, non-nan, float castables.
    Else, False.

    Parameters
    ----------
    arg
        The variable being tested.
    n : int
        The required length of arg.
    arg_name : String, optional
        The name of the argument. When not None, a warning will be output if
        function returns false. The default is None

    Returns
    -------
    bool
        If arg is n-vector of non-inf, non-nan, float castables.

    """
    # Ensure iterable
    try:
        iter(arg) 
    except TypeError:
        if not arg_name is None:
            msg = f"{arg_name} must be iterable."
            warn(msg)
        return False
    
    # Ensure of length 3
    if len(arg) != n:
        if not arg_name is None:
            msg = f"{arg_name} must be length {n}."
            warn(msg)
        return False

    # Ensure each arg is number
    all_num = all(is_num(a) for a in arg)
    if not all_num and not arg_name is None:
        msg = (f"Elements of {arg_name} must be non-inf, "
               "non-nan, float castables.")
        warn(msg)
    return all_num

def name_valid(arg, arg_name=None):
    """
    True if arg string or tuple of strings, else False.

    Parameters
    ----------
    arg
        The variable being tested.
    arg_name : String, optional
        The name of the argument. When not None, a warning will be output if
        function returns false. The default is None

    Returns
    -------
    bool
        If arg string or tuple of strings.

    """
    # Tuple of strings case
    if isinstance(arg, (tuple, list, np.ndarray)):
        if not all(isinstance(name, str) for name in arg):
            if not arg_name is None:
                msg = f"When {arg_name} is tuple, must be tuple of strings."
                warn(msg)
            return False
        
    # String only case
    elif not isinstance(arg, str):
        if not arg_name is None:
            msg = f"{arg_name} must be tuple of strings or string."
            warn(msg)
        return False
    
    # All tests passed
    return True

def path_valid(arg, ftype=None, arg_name=None):
    """
    True if arg is path string that points to a valid file. Else False.

    Parameters
    ----------
    arg
        The variable being tested.
    ftype : None, String, or tuple of Strings, optional
        The list of valid file extensions the file pointed to can have. When 
        None, the file may have any extension. The default is None
    arg_name : String, optional
        The name of the argument. When not None, a warning will be output if
        function returns false. The default is None

    Returns
    -------
    bool
        If arg is a path string that points to a valid file.

    """
    # Check if is string
    if not isinstance(arg, str):
        if not arg_name is None:
            msg = f"{arg_name} must be a string."
            warn(msg)
        return False

    # Check if file is in dirpath
    split = list(os.path.split(arg))
    try:
        if split[0] == '':
            split[0] = '.'
        if not split[1] in os.listdir(split[0]):
            if not arg_name is None:
                msg = f"The file pointed to by {arg_name} does not exist."
                warn(msg)
            return False

    # Check if file exists
    except FileNotFoundError:
        if not arg_name is None:
            msg = f"The parent file pointed to by {arg_name} does not exist."
            warn(msg)
        return False
    
    # Check file extension
    if not ftype is None and not arg.endswith(ftype):
        if not arg_name is None:
            msg = f"The file pointed to by {arg_name} must be type {ftype}."
            warn(msg)
        return False
            
    # All cases true
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
    # Ensure the quat's norm is greater than 0
    s = np.linalg.norm(wxyz_quat)
    if s == 0.0:
        return np.eye(4)
    s = s**-2
    
    # Extract the values from Q
    qr = wxyz_quat[0]
    qi = wxyz_quat[1]
    qj = wxyz_quat[2]
    qk = wxyz_quat[3]
     
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