# -*- coding: utf-8 -*-
"""
This module provides utilities functions used to render videos from zlib
compressed frame data.
"""
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import os
import zlib
import cv2
import numpy as np
from condynsate.exceptions import InvalidNameException
from condynsate.misc.exception_handling import ESC

###############################################################################
#VIDEO RENDERING FUNCTIONS
###############################################################################
def _get_valid_name_(file_name, file_container):
    """
    Checks a file name for validity (no invalid characters, does not exist
    in the current working directory). If invalid, appends an int, up to
    99 to the end of filename it validify.

    Parameters
    ----------
    file_name : string
        The desired file name being validated.
    file_container : string
        The container type for the file.

    Raises
    ------
    InvalidNameException
        Raised when name is invalid or cannot be validified.

    Returns
    -------
    valid_file : string
        The validated file name in format "valid_file_name.file_container".

    """
    illegal_chars = ("<", ">", ":", "|", "?", "*", ".", "\"", "\'",)
    if any(illegal_char in file_name for illegal_char in illegal_chars):
        msg = "May not include the characters {}."
        msg = msg.format(r' '.join(illegal_chars))
        raise InvalidNameException(msg, (file_name, -1))

    if len(file_name) == 0:
        msg = "Must be longer than 0 characters."
        msg = msg.format(file_name)
        raise InvalidNameException(msg, (file_name, -1))

    valid_file = f"{file_name}.{file_container}"
    if not valid_file in os.listdir(os.path.abspath(os.getcwd())):
        return valid_file

    max_append = 99
    for i in range(max_append):
        valid_file = f"{file_name}_{i+1:02}.{file_container}"
        if not valid_file in os.listdir(os.path.abspath(os.getcwd())):
            return valid_file

    msg = "Too many files already exist with the same name."
    msg = msg.format(file_name)
    raise InvalidNameException(msg, (file_name, max_append))

def _get_valid_name(file_name, file_container):
    """
    Attempts to convert a file name into a valid file name (no invalid
    characters, does not exist in the current working directory). If it
    cannot do this automatically, prompts the user to input a new name.

    Parameters
    ----------
    file_name : string
        The desired file name to be validated.
    file_container : string
        The container type for the file.

    Returns
    -------
    valid_file : string
        The validated file name in format "valid_file_name.file_container".

    """
    try:
        if not isinstance(file_name, str):
            raise TypeError("file_name must be instance of string.")
        valid_name = _get_valid_name_(file_name, file_container)
        return valid_name
    except InvalidNameException as e:
        bars = ''.join(['-']*80)
        print(ESC.fail(bars), flush=True)
        print(ESC.fail(type(e).__name__), flush=True)
        s = 'Cannot save the video to the file name: \"{}\"'
        print(ESC.warning(s.format(file_name)), flush=True)
        print(ESC.warning(e.message), flush=True)
        s = 'Enter new name (do not include file extension): '
        file_name = input(s)
        print(ESC.fail(bars), flush=True)
        return _get_valid_name(file_name, file_container)
    except TypeError as e:
        bars = ''.join(['-']*80)
        print(ESC.fail(bars), flush=True)
        print(ESC.fail(type(e).__name__), flush=True)
        s = 'The file name entered was not a string'
        print(ESC.warning(s.format(file_name)), flush=True)
        print(ESC.warning(e.message), flush=True)
        s = 'Enter new name (do not include file extension): '
        file_name = input(s)
        print(ESC.fail(bars), flush=True)
        return _get_valid_name(file_name, file_container)

def _get_fps_and_frames(frames, frame_times):
    """
    Given a set of collected frames, calculates the required FPS to
    display all frames at constant fps and doubles frames, where necessary,
    to maintain that FPS.

    Parameters
    ----------
    frames: list of tuples of (image bytes, image.shape)
        The zlib compressed frame data. Each frame should a tuple of the zlib
        compressed m x n x 3 numpy array of dtype np.uint8 and its shape in
        the form (m, n, 3).
    frame_times : list of floats
        The times in seconds at which each frame was recorded.

    Returns
    -------
    vid_fps : float
        The fps at which the frames are to be played back.
    vid_frames : list of tuples of form (bytes, (int, int, int))
        The zlib compressed video frames and their shapes.

    """
    # Get the fps for the video
    min_dt = min(np.diff(frame_times))
    max_fps = 1.0 / min_dt
    vid_fps = np.ceil(max_fps/5.0)*5.0  # Round up to nearest 5
    vid_fps = float(np.clip(vid_fps, 20.0, 120.0)) # Clip the fps

    # Get the video frame times spaced by the fps
    vid_frame_times = np.arange(0.0, max(frame_times), 1.0/vid_fps)

    # Match each captured frame to a video frame time
    cap_frame_idxs = [np.argmin(abs(t-vid_frame_times)) for t in frame_times]

    # Make the video frames by keying the captured frames via their
    # frame_idxs and then copying the previous video frame to fill unkeyed
    # video frames
    vid_frames = [None, ] * len(vid_frame_times)
    for i, cap_frame_idx in enumerate(cap_frame_idxs):
        vid_frames[cap_frame_idx] = frames[i]
    prev_cap_frame = vid_frames[0]
    for i in range(len(vid_frame_times)):
        if not vid_frames[i] is None:
            prev_cap_frame = vid_frames[i]
        else:
            vid_frames[i] = prev_cap_frame
    return vid_fps, vid_frames

def _make_video(vid_fps, vid_frames, name):
    """
    Takes a set of compressed frames and a target FPS and renders them
    to animator_video.mp4 with the h.264 codec.

    Parameters
    ----------
    vid_fps : float
        The fps at which the frames are to be played back.
    vid_frames : list of tuples of form (bytes, (int, int, int))
        The zlib compressed video frames and their shapes.
    name : string
        The name of the file to save to.

    Returns
    -------
    ret_code : int
        0 if successful, -1 if something went wrong.

    """
    # Get the video frame size (scale up to 1080p)
    cap_frame_sizes = np.array([f[1] for f in vid_frames])
    cap_frame_height  = max(cap_frame_sizes[:,0])
    cap_frame_width  = max(cap_frame_sizes[:,1])
    scale = 1080/cap_frame_height
    vid_frame_height = 1080
    vid_frame_width = int(np.round(scale * cap_frame_width))
    vid_frame_width += (vid_frame_width)%2 # Make sure even size
    vid_size = (vid_frame_width, vid_frame_height)

    # Get a valid directory name
    fname = _get_valid_name(name, 'mp4')

    # Make the video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(fname, fourcc, vid_fps, vid_size)
    for f in vid_frames:
        dat = zlib.decompress(f[0])
        img = np.frombuffer(dat, np.uint8).reshape(f[1])
        img = img[:, :, ::-1] # Convert RGB image to BGR

        # Upscale to 1080p
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

        # Ensure proper shape
        if not img.shape[0] == vid_size[1]:
            n_rows_add = vid_size[1] - img.shape[0]
            rows_shape = (n_rows_add, img.shape[1], img.shape[2])
            rows = np.zeros(rows_shape, dtype=np.uint8)
            img = np.concatenate((img, rows), axis=0)
        if not img.shape[1] == vid_size[0]:
            n_cols_add = vid_size[0] - img.shape[1]
            cols_shape = (img.shape[0], n_cols_add, img.shape[2])
            cols = np.zeros(cols_shape, dtype=np.uint8)
            img = np.concatenate((img, cols), axis=1)

        # Write the frame
        out.write(img)
    out.release()
    return 0

def save_recording(frames, frame_times, name='recording'):
    """
    Converts frames and frame times to a constant FPS video.

    Parameters
    ----------
    frames: list of tuples of (image bytes, image.shape)
        The zlib compressed frame data. Each frame should a tuple of the zlib
        compressed m x n x 3 numpy array of dtype np.uint8 and its shape in
        the form (m, n, 3).
    frame_times : list of floats
        The times in seconds at which each frame was recorded.
    name : string
        The name of the file to save the video to.

    Returns
    -------
    ret_code : int
        0 if successful, -1 if something went wrong.

    """
    try:
        vid_fps, vid_frames = _get_fps_and_frames(frames, frame_times)
        return _make_video(vid_fps, vid_frames, name)
    except Exception:
        return -1
