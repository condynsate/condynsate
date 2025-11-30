# -*- coding: utf-8 -*-
"""
This module provides utilities functions used by the Visualizer class.
"""
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import os
import numpy as np
from warnings import warn
from condynsate.misc.transforms import (Rbw_from_euler, Rbw_from_wxyz)

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
            warn(msg, UserWarning)
        return False

    # Check arg is correct type
    if not isinstance(arg, typ):
        if not arg_name is None:
            msg = f"{arg_name} must be type {typ}."
            warn(msg, UserWarning)
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
            warn(msg, UserWarning)
        return False

    # Check if not inf and not nan
    is_inf_or_nan = np.isinf(float(arg)) or np.isnan(float(arg))
    if is_inf_or_nan and not arg_name is None:
        msg = f"{arg_name} cannot be inf or nan."
        warn(msg, UserWarning)
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
            warn(msg, UserWarning)
        return False

    # Ensure of length 3
    if len(arg) != n:
        if not arg_name is None:
            msg = f"{arg_name} must be length {n}."
            warn(msg, UserWarning)
        return False

    # Ensure each arg is number
    all_num = all(is_num(a) for a in arg)
    if not all_num and not arg_name is None:
        msg = (f"Elements of {arg_name} must be non-inf, "
               "non-nan, float castables.")
        warn(msg, UserWarning)
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
                warn(msg, UserWarning)
            return False

    # String only case
    elif not isinstance(arg, str):
        if not arg_name is None:
            msg = f"{arg_name} must be tuple of strings or string."
            warn(msg, UserWarning)
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
            warn(msg, UserWarning)
        return False

    # Check if file is in dirpath
    split = list(os.path.split(arg))
    try:
        if split[0] == '':
            split[0] = '.'
        if not split[1] in os.listdir(split[0]):
            if not arg_name is None:
                msg = f"The file pointed to by {arg_name} does not exist."
                warn(msg, UserWarning)
            return False

    # Check if file exists
    except FileNotFoundError:
        if not arg_name is None:
            msg = f"The parent file pointed to by {arg_name} does not exist."
            warn(msg, UserWarning)
        return False

    # Check file extension
    if not ftype is None and not arg.endswith(ftype):
        if not arg_name is None:
            msg = f"The file pointed to by {arg_name} must be type {ftype}."
            warn(msg, UserWarning)
        return False

    # All cases true
    return True

###############################################################################
#TRANSFORMATION FUNCTIONS
###############################################################################
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
        The intrinsic yaw (radian) about the objects Z axis to apply.
    pitch : float
        The intrinsic pitch (radian) about the objects Y axis to apply.
    roll : float
        The intrinsic roll (radian) about the objects X axis to apply.
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
    R_quat = np.eye(4)
    R_quat[0:3, 0:3] = Rbw_from_wxyz(wxyz_quat)
    R_ypr = np.eye(4)
    R_ypr[0:3, 0:3] = Rbw_from_euler(yaw, pitch, roll)

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
