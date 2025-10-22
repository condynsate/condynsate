###############################################################################
#DEPENDENCIES
###############################################################################
import os
import zlib
import signal
from warnings import warn
import cv2
import numpy as np
from condynsate.animator.figure import Figure
from condynsate.animator.subplots import (Lineplot, Barchart)
from condynsate.exceptions import InvalidNameException
from condynsate.misc.exception_handling import ESC


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
        The default is None.

    record : bool, optional
        A boolean flag that indicates if the animator should be recorded. If
        True, all frames from the start function call to the terminate function
        call are recorded. After the terminate function call, these frames are
        saved with h.264 and outputs in an MP4 container. The saved file name
        has the form animator_video.mp4
    """
    def __init__(self, frame_rate=None, record=False):
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
        self._frame_times = []

        # Track the number of subplots
        self._n_plots = 0

        # Track the figure and the subplots
        self._figure = None
        self._plots = []
        self._started = False
        self._last_refresh = cv2.getTickCount()

        # Constants
        self._WINDOW_NAME = 'condynsate Animator'
        self._THREADED = True

        # Asynch listen for script exit
        signal.signal(signal.SIGTERM, self._sig_handler)
        signal.signal(signal.SIGINT, self._sig_handler)


    def __del__(self):
        """
        Deconstructor func.
        """
        self.terminate()


    def _sig_handler(self, sig, frame):
        """
        Handles script termination events so the keyboard listener exits
        gracefully.

        Parameters
        ----------
        sig : int
            The signal number.
        frame : signal.frame object
            The current stack frame.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        warn("Interrupt or termination signal detected. Terminating animator.")
        return self.terminate()


    def add_lineplot(self, n_lines, **kwargs):
        """
        Adds a lineplot to the animator window. Neither the lineplot nor the
        window appear until the user calls the start function.

        Parameters
        ----------
        n_lines : int
            The number of lines that are drawn on the plot. Must be integer
            between [1, 16].
        **kwargs: dict
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
        lines_ids : list of hex
            A unique identifier that allows the user to address each line
            in the lineplot. For example, if n_lines = 3, the list will have
            length three.

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
        lines_ids = [hex(16 * plot_data['subplot_ind'] + a_ind)
                     for a_ind in plot_data['artist_inds']]
        return lines_ids


    def add_barchart(self, n_bars, **kwargs):
        """
        Adds a barchart to the animator window. Neither the barchart nor the
        window appear until the user calls the start function.

        Parameters
        ----------
        n_bars : int
            The number of bars on the chart. Must be integer between [1, 16].
        **kwargs: dict
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
        bar_ids : list of hex
            A unique identifier that allows the user to address each bar
            in the barchart. For example, if n_bars = 3, the list will have
            length three.

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
        bar_ids = [hex(16 * plot_data['subplot_ind'] + a_ind)
                   for a_ind in plot_data['artist_inds']]
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
            warn("Start failed because the animator is already started.")
            return -1

        try:
            # Make the figure
            self._figure = Figure(self._n_plots, threaded=self._THREADED)

        except Exception as e:
            self.terminate()
            err = "Something went wrong while building the figure."
            raise RuntimeError(err) from e

        try:
            # Make each subplot
            for subplot_ind in range(len(self._plots)):

                # Make a lineplot
                if (self._plots[subplot_ind]['type']).lower() == 'lineplot':
                    self._make_lineplot(subplot_ind)

                # Make a barchart
                elif (self._plots[subplot_ind]['type']).lower() == 'barchart':
                    self._make_barchart(subplot_ind)

        except Exception as e:
            self.terminate()
            err = "Something went wrong while building the subplots."
            raise RuntimeError(err) from e


        # Indicate that threads are running
        self._started = True

        # Open the viewing window
        cv2.namedWindow(self._WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
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
            cv2.waitKey(1)
            return 0

        # If enough time has passed, update the GUI
        try:
            # Get the current image, draw it to screen, update last frame time
            image = cv2.cvtColor(self._figure.get_image(), cv2.COLOR_BGR2RGB)
            cv2.imshow(self._WINDOW_NAME, image)
            cv2.waitKey(1)
            self._last_refresh = cv2.getTickCount()

            # If recording, save the current image
            if self.record:
                self._frames.append((zlib.compress(image), image.shape))
                self._frame_times.append(self._last_refresh)

            return 0
        except Exception:
            return -1


    def barchart_set_value(self, value, bar_id):
        """
        Set's a bar's value.

        Parameters
        ----------
        value : float
            The value to which the bar is set.
        bar_id : hex string
            The id of the bar whose value is being set.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Sanitization
        if not self._started:
            warn("barchart_set_value failed because animator not started.")
            return -1
        if not self._sanitize_barchart(bar_id):
            warn("barchart_set_value failed because bar_id is invalid.")
            return -1
        if not self._is_number(value):
            warn("barchart_set_value failed because value is invalid.")
            return -1

        # Extract the subplot_ind and bar_ind from the bar_id
        subplot_ind = int(bar_id, 16)//16
        bar_ind = int(bar_id, 16) % 16

        # Set value
        self._plots[subplot_ind]['Subplot'].set_value(value, bar_ind)

        # Refresh the viewer
        self.refresh()
        return 0


    def lineplot_append_point(self, x_val, y_val, line_id):
        """
        Appends a single y versus x data point to the end of a line.

        Parameters
        ----------
        x_val : float
            The x coordinate of the data point being appended.
        y_val : float
            The y coordinate of the data point being appended.
        line_id : hex string
            The id of the line to which a point is appended.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Sanitization
        if not self._started:
            warn("lineplot_append_point failed because animator not started.")
            return -1
        if not self._sanitize_lineplot(line_id):
            warn("lineplot_append_point failed because line_id is invalid.")
            return -1
        if not self._is_number(x_val):
            warn("lineplot_append_point failed because x_val is invalid.")
            return -1
        if not self._is_number(y_val):
            warn("lineplot_append_point failed because y_val is invalid.")
            return -1

        # Extract the subplot_ind and line_ind from the line_id
        subplot_ind = int(line_id, 16)//16
        line_ind = int(line_id, 16) % 16

        # Append point
        self._plots[subplot_ind]['Subplot'].append_point(x_val,y_val,line_ind)

        # Refresh the viewer
        self.refresh()
        return 0

    def lineplot_set_data(self, x_vals, y_vals, line_id):
        """
        Plots y_vals versus x_vals.

        Parameters
        ----------
        x_vals : list of floats
            A list of x coordinates of the points being plotted.
        y_vals : list of floats
            A list of y coordinates of the points being plotted.
        line_id : hex string
            The id of the line on which that data are plotted.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Sanitization
        if not self._started:
            warn("lineplot_set_data failed because animator not started.")
            return -1
        if not self._sanitize_lineplot(line_id):
            warn("lineplot_set_data failed because line_id is invalid.")
            return -1
        if not self._is_number_list(x_vals):
            warn("lineplot_set_data failed because x_vals is invalid.")
            return -1
        if not self._is_number_list(y_vals):
            warn("lineplot_set_data failed because y_vals is invalid.")
            return -1

        # Extract the subplot_ind and line_ind from the line_id
        subplot_ind = int(line_id, 16)//16
        line_ind = int(line_id, 16) % 16

        # Set data
        self._plots[subplot_ind]['Subplot'].set_data(x_vals, y_vals, line_ind)

        # Refresh the viewer
        self.refresh()
        return 0


    def reset_all(self):
        """
        Resets all data on all subplots.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Ensure the animator is already started
        if not self._started:
            warn("reset_all failed because animator not started.")
            return -1

        # Reset all subplot data
        for subplot_ind in range(len(self._plots)):
            self._plots[subplot_ind]['Subplot'].reset_data()

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
        for subplot_ind in range(len(self._plots)):
            try:
                self._plots[subplot_ind]['Subplot'].terminate()
            except Exception:
                pass

        # Attempt to terminate the figure
        try:
            self._figure.terminate()
        except Exception:
            pass

        # Destroy all open windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Save recording
        if self.record and len(self._frames) > 1:
            self._save_recording()

        # Reset locals
        self._frames = []
        self._frame_times = []
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
        INTS = [np.int64, np.int32, np.int16, np.int8,
                np.uint64, np.uint64, np.uint64, np.uint64, int]
        if type(val) in INTS:
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
        FLOATS = [np.float64, np.float32, np.float16, float]
        if type(val) in FLOATS and not np.isnan(val) and not np.isinf(val):
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
        if isinstance(vals,list) or isinstance(vals,np.ndarray):
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


    def _get_valid_name_(self, file_name, file_container):
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

        valid_file = '{}.{}'.format(file_name, file_container)
        if not valid_file in os.listdir(os.path.abspath(os.getcwd())):
            return valid_file

        max_append = 99
        for i in range(max_append):
            valid_file = '{}_{:02}.{}'.format(file_name, i+1, file_container)
            if not valid_file in os.listdir(os.path.abspath(os.getcwd())):
                return valid_file

        msg = "Too many files already exist with the same name."
        msg = msg.format(file_name)
        raise InvalidNameException(msg, (file_name, max_append))


    def _get_valid_name(self, file_name, file_container):
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
            valid_name = self._get_valid_name_(file_name, file_container)
            return valid_name
        except InvalidNameException as e:
            bars = ''.join(['-']*80)
            print(ESC.fail(bars), flush=True)
            print(ESC.fail(type(e).__name__), flush=True)
            s = 'Cannot save the animator video to the file name: \"{}\"'
            print(ESC.warning(s.format(file_name)), flush=True)
            print(ESC.warning(e.message), flush=True)
            s = 'Enter new name (do not include file extension): '
            file_name = input(s)
            print(ESC.fail(bars), flush=True)
            return self._get_valid_name(file_name, file_container)


    def _get_fps_and_frames(self):
        """
        Given a set of collected frames, calculates the required FPS to
        display all frames at constant fps and doubles frames, where necessary,
        to maintain that FPS.

        Returns
        -------
        vid_fps : float
            The fps at which the frames are to be played back.
        vid_frames : list of tuples of form (bytes, (int, int, int))
            The zlib compressed video frames and their shapes.

        """
        # Calculate the times in seconds at which each frame was captured
        cap_t = np.array(self._frame_times)/cv2.getTickFrequency()
        cap_t -= cap_t[0]

        # Get the fps for the video
        min_dt = min(np.diff(cap_t))
        max_fps = 1.0 / min_dt
        vid_fps = np.ceil(max_fps/5.0)*5.0  # Round up to nearest 5
        vid_fps = float(np.clip(vid_fps, 20.0, 120.0)) # Clip the fps

        # Get the video frame times spaced by the fps
        vid_frame_times = np.arange(0.0, max(cap_t), 1.0/vid_fps)

        # Match each captured frame to a video frame time
        cap_frame_idxs = [np.argmin(abs(t-vid_frame_times)) for t in cap_t]

        # Make the video frames by keying the captured frames via their
        # frame_idxs and then copying the previous video frame to fill unkeyed
        # video frames
        vid_frames = [None, ] * len(vid_frame_times)
        for i, cap_frame_idx in enumerate(cap_frame_idxs):
            vid_frames[cap_frame_idx] = self._frames[i]
        prev_cap_frame = vid_frames[0]
        for i in range(len(vid_frame_times)):
            if not vid_frames[i] is None:
                prev_cap_frame = vid_frames[i]
            else:
                vid_frames[i] = prev_cap_frame

        return vid_fps, vid_frames


    def _make_video(self, vid_fps, vid_frames):
        """
        Takes a set of compressed frames and a target FPS and renders them
        to animator_video.mp4 with the h.264 codec.

        Parameters
        ----------
        vid_fps : float
            The fps at which the frames are to be played back.
        vid_frames : list of tuples of form (bytes, (int, int, int))
            The zlib compressed video frames and their shapes.

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
        fname = self._get_valid_name('animator_video', 'mp4')

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


    def _save_recording(self):
        """
        Converts frames and frame times to a constant FPS video.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        try:
            vid_fps, vid_frames = self._get_fps_and_frames()
            return self._make_video(vid_fps, vid_frames)
        except Exception:
            return -1
