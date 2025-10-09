###############################################################################
#DEPENDENCIES
###############################################################################
import time
from warnings import warn
from figure import Figure
from subplots import (Lineplot, Barchart)
import cv2
import numpy as np


###############################################################################
#ANIMATOR CLASS
###############################################################################
class Animator():
    """
    The animator main class. Creates, modifies, and displays lineplots and
    barcharts in real time according to user inputs

    Parameters
    ----------
    None.
    """
    def __init__(self):
        """
        Constructor func.
        """
        # Track the number of subplots
        self._n_plots = 0

        # Track the figure and the subplots
        self._figure = None
        self._plots = []
        self._started = False

        # Constants
        self._WINDOW_NAME = 'condynsate Animator'
        self._THREADED = True


    def __del__(self):
        """
        Deconstructor func.
        """
        self.terminate()


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


    def refresh(self):
        try:
            image = cv2.cvtColor(self._figure.get_image(), cv2.COLOR_BGR2RGB)
            cv2.imshow(self._WINDOW_NAME, image)
            cv2.waitKey(1)
            return 0
        except:
            return -1


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
            0 if started properly, -1 if something went wrong.

        """
        if self._started:
            warn("Start failed because the animator is already started.")
            return -1

        try:
            # Make the figure
            self._figure = Figure(self._n_plots, threaded=self._THREADED)

        except:
            self.terminate()
            err = ("Something went wrong while building the figure.")
            raise RuntimeError(err)

        try:
            # Make each subplot
            for subplot_ind in range(len(self._plots)):

                # Make a lineplot
                if (self._plots[subplot_ind]['type']).lower() == 'lineplot':
                    self._make_lineplot(subplot_ind)

                # Make a barchart
                elif (self._plots[subplot_ind]['type']).lower() == 'barchart':
                    self._make_barchart(subplot_ind)

        except:
            self.terminate()
            err = ("Something went wrong while building the subplots.")
            raise RuntimeError(err)


        # Indicate that threads are running
        self._started = True

        # Open the viewing window
        cv2.namedWindow(self._WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        self.refresh()

        return 0


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
        except:
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
        sp_ok = sp_ok == 'subplots'
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
        if type(vals) is list or type(vals) is np.ndarray:
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
            0 if value set successfully, -1 if something went wrong.

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
            0 if data point appended successfully, -1 if something went wrong.

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
            0 if data set successfully, -1 if something went wrong.

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
            0 if reset successfully, -1 if something went wrong.

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
        Terminates and removes all subplots from the animator. Must be called
        after start function is called to prevent orphaned children threads.

        Returns
        -------
        ret_code : int
            0 if reset successfully, -1 if something went wrong.

        """
        # Attempt to terminate each subplot
        for subplot_ind in range(len(self._plots)):
            try:
                self._plots[subplot_ind]['Subplot'].terminate()
            except:
                pass

        # Attempt to terminate the figure
        try:
            self._figure.terminate()
        except:
            pass

        # Destroy all open windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Reset number of subplots
        self._n_plots = 0

        # Reset figure and the subplots
        self._figure = None
        self._plots = []
        self._started = False

        return 0


###############################################################################
#TESTING DONE IN MAIN LOOP
###############################################################################
if __name__ == "__main__":
    animator = Animator()
    lines_1 = animator.add_lineplot(n_lines = 2, color = ['r', 'b'])
    bars_1 = animator.add_barchart(n_bars = 4, color = ['k', 'r', 'g', 'b'])
    animator.start()
    for i in range(100):
        animator.lineplot_append_point(i, np.random.rand(), lines_1[0])
        time.sleep(0.5)
    animator.terminate()
