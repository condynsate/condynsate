# -*- coding: utf-8 -*-
"""
This module provides the Animator class which is used to draw data to a plot
in real time.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
from warnings import warn
from copy import copy
import zlib
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from condynsate.misc import save_recording
from condynsate.animator.figure import Figure
from condynsate.animator.subplots import (Lineplot, Barchart)

###############################################################################
#ANIMATOR CLASS
###############################################################################
class Animator():
    """
    The animator main class. Creates, modifies, and displays lineplots and
    barcharts in real time according to user inputs

    Parameters
    ----------
    frame_rate : float, optional
        The upper limit of the allowed frame rate in frames per second.
        When set, the animator will not update faster than this speed.
        When none, the animator will update each time refresh is called.
        The default is 20.

    record : bool, optional
        A boolean flag that indicates if the animator should be recorded. If
        True, all frames from the start function call to the terminate function
        call are recorded. After the terminate function call, these frames are
        saved with h.264 and outputs in an MP4 container. The saved file name
        has the form animator_video.mp4

    Attributes
    ----------
    frame_delta : float
        The time, in seconds, between each animator frame update.
    record : bool
        A boolean flag that indicates if the animator is recording or not.
    is_running : bool
        A boolean flag that indicates if the animator is running or not.

    """
    def __init__(self, frame_rate=20, record=False):
        """
        Constructor func.
        """
        # Calculate time between frames
        if not frame_rate is None:
            self.frame_delta = 1.0 / frame_rate
        else:
            self.frame_delta = 0.0

        # Recording support
        self.record = record
        self._frames = []
        self._frame_ticks = []

        # Track the number of subplots
        self._n_plots = 0

        # Track the figure and the subplots
        self._THREADED = True
        self._figure = None
        self._plots = []
        self._started = False
        self._last_refresh = cv2.getTickCount()

        # GUI support
        self._root = tk.Tk()
        self._root.title("condynsate Animator")
        self._panel = None

    def __del__(self):
        """
        Deconstructor func.
        """
        self.terminate()

    @property
    def is_running(self):
        """
        True if the animator GUI is open and the animation loop is running,
        False else.

        Returns
        -------
        bool
            If running.

        """
        return copy(self._started)

    def add_lineplot(self, n_lines, **kwargs):
        """
        Adds a lineplot to the animator window. Neither the lineplot nor the
        window appear until the user calls the start function.

        Parameters
        ----------
        n_lines : int
            The number of lines that are drawn on the plot. Must be integer
            between [1, 16].
        **kwargs

        Keyword Args
        ------------
        x_lim : [float, float], optional
            The limits to apply to the x axis of the plot. A value of None
            will apply automatically updating limits to the corresponding
            bound of the axis. For example [None, 10.] will fix the upper
            bound to exactly 10, but the lower bound will freely change to
            show all data.The default is [None, None].
        y_lim : [float, float], optional
            The limits to apply to the y axis of the plot. A value of None
            will apply automatically updating limits to the corresponding
            bound of the axis. For example [None, 10.] will fix the upper
            bound to exactly 10, but the lower bound will freely change to
            show all data.The default is [None, None].
        h_zero_line : boolean, optional
            A boolean flag that indicates whether a horizontal line will be
            drawn on the y=0 line. The default is false
        v_zero_line : boolean, optional
            A boolean flag that indicates whether a vertical line will be
            drawn on the x=0 line. The default is false
        tail : int or tuple of ints optional
            Specifies how many data points are used to draw a line. Only
            the most recently added data points are kept. Any data points
            added more than tail data points ago are discarded and not
            plotted. When tuple, must have length n_lines. A value less
            than or equal to 0 means that no data is ever discarded and all
            data points added to the animator will be drawn. The default
            is -1.
        title : string, optional
            The title of the plot. Will be written above the plot when
            rendered. The default is None.
        x_label : string, optional
            The label to apply to the x axis. Will be written under the
            plot when rendered. The default is None.
        y_label : string, optional
            The label to apply to the y axis. Will be written to the
            left of the plot when rendered. The default is None.
        label : string or tuple of strings, optional
            The label applied to each artist. The labels are shown in a
            legend in the top right of the plot. When tuple, must have
            length n_lines. When None, no labels are made. The default
            is None.
        color : matplotlib color string or tuple of color strings, optional
            The color each artist draws in. When tuple, must have length
            n_lines. The default is 'black'.
        line_width : float or tuple of floats, optional
            The line weigth each artist uses. When tuple, must have length
            n_lines. The default is 1.5.
        line_style : line style string or tuple of ls strings, optional
            The line style each artist uses. When tuple, must have length
            n_lines. The default is 'solid'. Select from 'solid', 'dashed',
            'dashdot', or 'dotted'.

        Raises
        ------
        RuntimeError
            If cannot add another subplot or the animator was already running.
            Can only add up to 16 subplots total.

        ValueError
            If n_lines is not an int or is less than 1 or greater than 16.

        Returns
        -------
        lines_ids : hex or tuple of hex
            A unique identifier that allows the user to address each line
            in the lineplot. For example, if n_lines = 3, the tuple will have
            length three; however, if n_lines = 1, the returned value will be
            the hex id of the only line (not a tuple).

        """
        self._assert_not_started()
        self._assert_n_artists_valid(n_lines)
        self._assert_can_add_subplot()

        # Add an empty lineplot to the plot data list
        self._n_plots += 1
        plot_data = {'subplot_ind' : self._n_plots - 1,
                     'artist_inds' : np.arange(n_lines).tolist(),
                     'type' : 'Lineplot',
                     'n_artists' : n_lines,
                     'threaded' : self._THREADED,
                     'kwargs' : kwargs,
                     'Subplot' : None,}
        self._plots.append(plot_data)

        # Return line artist ids that identify line and subplot
        lines_ids = tuple(hex(16 * plot_data['subplot_ind'] + a_ind)
                          for a_ind in plot_data['artist_inds'])
        if len(lines_ids) == 1:
            lines_ids = lines_ids[0]
        return lines_ids

    def add_barchart(self, n_bars, **kwargs):
        """
        Adds a barchart to the animator window. Neither the barchart nor the
        window appear until the user calls the start function.

        Parameters
        ----------
        n_bars : int
            The number of bars on the chart. Must be integer between [1, 16].
        **kwargs

        Keyword Args
        ------------
        x_lim : [float, float], optional
            The limits to apply to the x axis of the plot. A value of None
            will apply automatically updating limits to the corresponding
            bound of the axis. For example [None, 10.] will fix the upper
            bound to exactly 10, but the lower bound will freely change to
            show all data.The default is [None, None].
        y_lim : [float, float], optional
            The limits to apply to the y axis of the plot. A value of None
            will apply automatically updating limits to the corresponding
            bound of the axis. For example [None, 10.] will fix the upper
            bound to exactly 10, but the lower bound will freely change to
            show all data.The default is [None, None].
        h_zero_line : boolean, optional
            A boolean flag that indicates whether a horizontal line will be
            drawn on the y=0 line. The default is false
        v_zero_line : boolean, optional
            A boolean flag that indicates whether a vertical line will be
            drawn on the x=0 line. The default is false
        title : string, optional
            The title of the plot. Will be written above the plot when
            rendered. The default is None.
        x_label : string, optional
            The label to apply to the x axis. Will be written under the
            plot when rendered. The default is None.
        y_label : string, optional
            The label to apply to the y axis. Will be written to the left
            of the plot when rendered. The default is None.
        label : string or tuple of strings, optional
            The label applied to each bar. The labels are shown in a legend
            in the top right of the chart. When tuple, must have length
            n_bars. When None, no labels are made. The default is None.
        color : matplotlib color string or tuple of color strings, optional
            The color of each bar. When tuple, must have length
            n_bars. The default is 'blue'.

        Raises
        ------
        RuntimeError
            If cannot add another subplot or the animator was already running.
            Can only add up to 16 subplots total.

        ValueError
            If n_bars is not an int or is less than 1 or greater than 16.

        Returns
        -------
        bar_ids : tuple of hex
            A unique identifier that allows the user to address each bar
            in the barchart. For example, if n_bars = 3, the tuple will have
            length three; however, if n_bars = 1, the returned value will be
            the hex id of the only bar (not a tuple).

        """
        self._assert_not_started()
        self._assert_n_artists_valid(n_bars)
        self._assert_can_add_subplot()

        # Add an empty barchart to the plot data list
        self._n_plots += 1
        plot_data = {'subplot_ind' : self._n_plots - 1,
                     'artist_inds' : np.arange(n_bars).tolist(),
                     'type' : 'Barchart',
                     'n_artists' : n_bars,
                     'threaded' : self._THREADED,
                     'kwargs' : kwargs,
                     'Subplot' : None,}
        self._plots.append(plot_data)

        # Return line artist ids that identify line and subplot
        bar_ids = tuple(hex(16 * plot_data['subplot_ind'] + a_ind)
                        for a_ind in plot_data['artist_inds'])
        if len(bar_ids) == 1:
            bar_ids = bar_ids[0]
        return bar_ids

    def start(self):
        """
        Starts the animator. Creates a new window and begins displaying live
        subplot data to it. Please ensure to call the terminate function
        when done to ensure all child threads are killed.

        Raises
        ------
        RuntimeError
            If something goes wrong while attempting to start the animator.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if self._started:
            m = "Start failed because the animator is already started."
            warn(m, UserWarning)
            return -1

        try:
            # Make the figure
            self._figure = Figure(max(self._n_plots,1),threaded=self._THREADED)

        except Exception as e:
            self.terminate()
            err = "Something went wrong while building the figure."
            raise RuntimeError(err) from e

        try:
            # Make each subplot
            for subplot_ind, subplot in enumerate(self._plots):

                # Make a lineplot
                if (subplot['type']).lower() == 'lineplot':
                    self._make_lineplot(subplot_ind)

                # Make a barchart
                elif (subplot['type']).lower() == 'barchart':
                    self._make_barchart(subplot_ind)

        except Exception as e:
            self.terminate()
            err = "Something went wrong while building the subplots."
            raise RuntimeError(err) from e


        # Indicate that threads are running
        self._started = True

        # Open the viewing window
        self._panel = tk.Label(self._root)
        self._panel.pack(side="bottom", fill="both", expand="yes")
        self.refresh()
        return 0

    def refresh(self):
        """
        Updates the animator GUI with the most recently drawn figure. Must
        be called regularly to maintain responsivness of GUI.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.


        """
        # Get elapsed time since refresh
        dt = (cv2.getTickCount() - self._last_refresh)/cv2.getTickFrequency()

        # If not enough time has passed, only refresh the responsiveness.
        if dt < self.frame_delta:
            self._root.update_idletasks()
            self._root.update()
            return 0

        # Get the current image, draw it to screen, update last frame time
        image = self._figure.get_image()
        if not self._panel is None:
            imagetk = ImageTk.PhotoImage(Image.fromarray(image))
            self._panel.configure(image=imagetk)
            self._panel.image = imagetk
        self._root.update_idletasks()
        self._root.update()
        self._last_refresh = cv2.getTickCount()

        # If recording, save the current image
        if self.record:
            self._frames.append((zlib.compress(image), image.shape))
            self._frame_ticks.append(self._last_refresh)

        return 0

    def barchart_set_value(self, bar_id, value):
        """
        Set's a bar's value.

        Parameters
        ----------
        bar_id : hex string
            The id of the bar whose value is being set.
        value : float
            The value to which the bar is set.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Sanitization
        if not self._started:
            m="barchart_set_value failed because animator not started."
            warn(m, UserWarning)
            return -1
        if not self._sanitize_barchart(bar_id):
            m="barchart_set_value failed because bar_id is invalid."
            warn(m, UserWarning)
            return -1
        if not self._is_number(value):
            m="barchart_set_value failed because value is invalid."
            warn(m, UserWarning)
            return -1

        # Extract the subplot_ind and bar_ind from the bar_id
        subplot_ind = int(bar_id, 16)//16
        bar_ind = int(bar_id, 16) % 16

        # Set value
        self._plots[subplot_ind]['Subplot'].set_value(value, bar_ind)

        # Refresh the viewer
        self.refresh()
        return 0

    def lineplot_append_point(self, line_id, x_val, y_val):
        """
        Appends a single y versus x data point to the end of a line.

        Parameters
        ----------
        line_id : hex string
            The id of the line to which a point is appended.
        x_val : float
            The x coordinate of the data point being appended.
        y_val : float
            The y coordinate of the data point being appended.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Sanitization
        if not self._started:
            m="lineplot_append_point failed because animator not started."
            warn(m, UserWarning)
            return -1
        if not self._sanitize_lineplot(line_id):
            m="lineplot_append_point failed because line_id is invalid."
            warn(m, UserWarning)
            return -1
        if not self._is_number(x_val):
            m="lineplot_append_point failed because x_val is invalid."
            warn(m, UserWarning)
            return -1
        if not self._is_number(y_val):
            m="lineplot_append_point failed because y_val is invalid."
            warn(m, UserWarning)
            return -1

        # Extract the subplot_ind and line_ind from the line_id
        subplot_ind = int(line_id, 16)//16
        line_ind = int(line_id, 16) % 16

        # Append point
        self._plots[subplot_ind]['Subplot'].append_point(x_val,y_val,line_ind)

        # Refresh the viewer
        self.refresh()
        return 0

    def lineplot_set_data(self, line_id, x_vals, y_vals):
        """
        Plots y_vals versus x_vals.

        Parameters
        ----------
        line_id : hex string
            The id of the line on which that data are plotted.
        x_vals : list of floats
            A list of x coordinates of the points being plotted.
        y_vals : list of floats
            A list of y coordinates of the points being plotted.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Sanitization
        if not self._started:
            m="lineplot_set_data failed because animator not started."
            warn(m, UserWarning)
            return -1
        if not self._sanitize_lineplot(line_id):
            m="lineplot_set_data failed because line_id is invalid."
            warn(m, UserWarning)
            return -1
        if not self._is_number_list(x_vals):
            m="lineplot_set_data failed because x_vals is invalid."
            warn(m, UserWarning)
            return -1
        if not self._is_number_list(y_vals):
            m="lineplot_set_data failed because y_vals is invalid."
            warn(m, UserWarning)
            return -1

        # Extract the subplot_ind and line_ind from the line_id
        subplot_ind = int(line_id, 16)//16
        line_ind = int(line_id, 16) % 16

        # Set data
        self._plots[subplot_ind]['Subplot'].set_data(x_vals, y_vals, line_ind)

        # Refresh the viewer
        self.refresh()
        return 0

    def reset(self):
        """
        Resets all data on all subplots and the recording, if applicable.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Ensure the animator is already started
        if not self._started:
            m="reset_all failed because animator not started."
            warn(m, UserWarning)
            return -1

        # Reset all subplot data
        for subplot in self._plots:
            subplot['Subplot'].reset_data()

        # Reset the recording data
        self._frames = []
        self._frame_ticks = []

        # Refresh the viewer
        self.refresh()
        return 0

    def terminate(self):
        """
        Terminates and removes all subplots from the animator. Should be called
        when done with Animator.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Attempt to terminate each subplot
        for subplot in self._plots:
            try:
                subplot['Subplot'].terminate()
            except Exception:
                pass

        # Attempt to terminate the figure
        try:
            self._figure.terminate()
        except Exception:
            pass

        # Destroy all open windows
        if not self._root is None:
            self._root.destroy()
            self._root = None
        self._panel = None

        # Save recording
        if self.record and len(self._frames) > 1:
            print('Saving animator recording...')
            # Convert frame ticks to frame times
            frame_times = np.array(self._frame_ticks, dtype=float)
            frame_times /= cv2.getTickFrequency()
            frame_times -= frame_times[0]
            save_recording(self._frames, frame_times, 'animator')

        # Reset locals
        self._frames = []
        self._frame_ticks = []
        self._n_plots = 0
        self._figure = None
        self._plots = []
        self._started = False
        self._last_refresh = cv2.getTickCount()
        return 0

    def _assert_not_started(self):
        """
        Asserts that the animator is not yet started.

        Raises
        ------
        RuntimeError
            If the animator is already started.

        Returns
        -------
        None.

        """
        # Ensure that animator is not started
        if self._started:
            self.terminate()
            err = ("Cannot add more subplots after start function is called."
                   " Terminating animator")
            raise RuntimeError(err)

    def _assert_n_artists_valid(self, n_artists):
        """
        Asserts the number of artists being added is valid. Checks if
        is int between 1 and 16.

        Parameters
        ----------
        n_artists : int
            Candidate number of artists to add.

        Raises
        ------
        RuntimeError
            If n_artists is invalid.

        Returns
        -------
        None.

        """
        # Ensure n_lines is int > 0 and int <= 99
        if not self._is_int(n_artists) or n_artists <= 0 or n_artists >= 17:
            err = ("When adding artists to subplots, must be integer "
                   "between 1 and 16, inclusive.")
            raise ValueError(err)

    def _assert_can_add_subplot(self):
        """
        Asserts can add another subplot. There can only be up to 16 subplots
        total.

        Raises
        ------
        RuntimeError
            If cannot add another subplot.

        Returns
        -------
        None.

        """
        # Ensure that another plot can be added
        if self._n_plots >= 17:
            err = "Cannot include more than 16 plots."
            raise RuntimeError(err)

    def _make_lineplot(self, subplot_ind):
        """
        Creates and starts a Lineplot object.

        Parameters
        ----------
        subplot_ind : int
            The index of the lineplot being created.

        Returns
        -------
        None.

        """
        # Build the lineplot
        axes = self._figure.get_axes()[subplot_ind]
        fig_lock = self._figure.get_lock()
        n_lines = self._plots[subplot_ind]['n_artists']
        threaded = self._plots[subplot_ind]['threaded']
        kwargs = self._plots[subplot_ind]['kwargs']
        subplot = Lineplot(axes, fig_lock, n_lines, threaded, **kwargs)

        # Add the lineplot and figure to the plot data structure
        self._plots[subplot_ind]['Subplot'] = subplot

    def _make_barchart(self, subplot_ind):
        """
        Creates and starts a barchart object.

        Parameters
        ----------
        subplot_ind : int
            The index of the barchart being created.

        Returns
        -------
        None.

        """
        # Build the barchart
        axes = self._figure.get_axes()[subplot_ind]
        fig_lock = self._figure.get_lock()
        n_bars = self._plots[subplot_ind]['n_artists']
        threaded = self._plots[subplot_ind]['threaded']
        kwargs = self._plots[subplot_ind]['kwargs']
        subplot = Barchart(axes, fig_lock, n_bars, threaded, **kwargs)

        # Add the barchart and figure to the plot data structure
        self._plots[subplot_ind]['Subplot'] = subplot

    def _is_valid_id(self, artist_id):
        """
        Ensures that a candidate artist_id is a valid hex string.

        Parameters
        ----------
        artist_id : type
            Candidate artist_id.

        Returns
        -------
        bool
            True if valid, false if invalid.

        """
        # Check the type
        try:
            artist_int = int(artist_id, 16)
        except Exception:
            return False

        # Ensure between 0 and 255
        if artist_int < 0 or artist_int > 255:
            return False

        # All conditions met, is valid artist_id
        return True

    def _is_subplot_ind(self, artist_id):
        """
        Checks that the artist id refers to a valid subplot.

        Parameters
        ----------
        artist_id : hex string
            id being inspected.

        Returns
        -------
        bool
            True if valid, false if invalid.

        """
        # Check the upper and lower bounds
        artist_int = int(artist_id, 16)
        subplot_ind = artist_int//16
        lb_ok = subplot_ind >= 0
        ub_ok = subplot_ind < len(self._plots)
        if not lb_ok or not ub_ok:
            return False

        # Make sure the indexed subplot is type Subplot
        sp_ok = type(self._plots[subplot_ind]['Subplot']).__module__
        sp_ok = sp_ok == 'condynsate.animator.subplots'
        if not sp_ok:
            return False
        return True

    def _is_barchart(self, artist_id):
        """
        Checks that the artist id refers to a barchart.

        Parameters
        ----------
        artist_id : hex string
            id being inspected.

        Returns
        -------
        bool
            True if barchart, false else.

        """
        subplot_ind = int(artist_id, 16)//16
        typ = self._plots[subplot_ind]['type']
        if not typ.lower() == 'barchart':
            return False
        return True

    def _is_lineplot(self, artist_id):
        """
        Checks that the artist id refers to a lineplot.

        Parameters
        ----------
        artist_id : hex string
            id being inspected.

        Returns
        -------
        bool
            True if lineplot, false else.

        """
        subplot_ind = int(artist_id, 16)//16
        typ = self._plots[subplot_ind]['type']
        if not typ.lower() == 'lineplot':
            return False
        return True

    def _artist_ind_ok(self, artist_id):
        """
        Checks that the artist id refers to a valid artist.

        Parameters
        ----------
        artist_id : hex string
            id being inspected.

        Returns
        -------
        bool
            True if valid artist, false else.

        """
        subplot_ind = int(artist_id, 16)//16
        artist_ind = int(artist_id, 16) % 16

        # Check the upper and lower bounds
        lb_ok = artist_ind >= 0
        ub_ok = artist_ind < self._plots[subplot_ind]['n_artists']
        if not lb_ok or not ub_ok:
            return False
        return True

    def _is_int(self, val):
        """
        Checks if a value is a built-in integer or numpy integer.

        Parameters
        ----------
        value : type
            Candidate value.

        Returns
        -------
        bool
            True if int, else false.

        """
        ints = (np.int64, np.int32, np.int16, np.int8,
                np.uint64, np.uint64, np.uint64, np.uint64, int, )
        if isinstance(val, ints):
            return True
        return False

    def _is_float(self, val):
        """
        Checks if a value is a built-in float or numpy float. nan does not
        count as a valid float. inf does not count as a valid float.

        Parameters
        ----------
        val : type
            Candidate value.

        Returns
        -------
        bool
            True if float, else false.

        """
        floats = (np.float64, np.float32, np.float16, float, )
        if isinstance(val, floats) and not np.isnan(val) and not np.isinf(val):
            return True
        return False

    def _is_number(self, val):
        """
        Checks if a candidate value is a built-in int, or a numpy int, or
        a built-in float, or a numpy float. nan does not count as a valid
        float. inf does not count as a valid float.

        Parameters
        ----------
        val : type
            Candidate value.

        Returns
        -------
        bool
            True if number, else false.

        """
        if self._is_int(val) or self._is_float(val):
            return True
        return False

    def _is_number_list(self, vals):
        """
        Checks if candidate values are list or numpy.ndarray of numbers.
        Numbers are considered as built-in int, or a numpy int, or
        a built-in float, or a numpy float. nan does not count as a valid
        float. inf does not count as a valid float.

        Parameters
        ----------
        vals : type
            Candidate vals.

        Returns
        -------
        bool
            True if list or array of numbers, else false.

        """
        if isinstance(vals, (list, np.ndarray)):
            for val in vals:
                if not self._is_number(val):
                    return False
            return True
        return False

    def _sanitize_barchart(self, bar_id):
        """
        Checks if an artist id
            0: is a valid id hex string between 0 and 255, and
            1: refers to a subplot that exists, and
            2: refers to a subplot that is a barchart, and
            3: refers to an artist that exists.

        Parameters
        ----------
        bar_id : hex string
            The candidate bar_id.

        Returns
        -------
        bool
            True if all conditions are met, else false.

        """
        if not self._is_valid_id(bar_id):
            return False
        cond_1 =  self._is_subplot_ind(bar_id)
        cond_2 = self._is_barchart(bar_id)
        cond_3 = self._artist_ind_ok(bar_id)
        return cond_1 and cond_2 and cond_3

    def _sanitize_lineplot(self, line_id):
        """
        Checks if an artist id
            0: is a valid id hex string between 0 and 255, and
            1: refers to a subplot that exists, and
            2: refers to a subplot that is a lineplot, and
            3: refers to an artist that exists.

        Parameters
        ----------
        line_id : hex string
            The candidate line_id.

        Returns
        -------
        bool
            True if all conditions are met, else false.

        """
        if not self._is_valid_id(line_id):
            return False
        cond_1 =  self._is_subplot_ind(line_id)
        cond_2 = self._is_lineplot(line_id)
        cond_3 = self._artist_ind_ok(line_id)
        return cond_1 and cond_2 and cond_3
