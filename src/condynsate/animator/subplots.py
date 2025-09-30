###############################################################################
#DEPENDENCIES
###############################################################################
from threading import (Thread, Lock)
import sys
from copy import copy
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
FONT_SIZE = 7

###############################################################################
#Functions
###############################################################################
def _parse_arg(arg, default, arg_str, n):
    """
    Parses an argument into a tuple of length n.

    Parameters
    ----------
    arg : add_plot argument
        The argument
    default : variable typing
        The default value of arg
    arg_str : string
        The string name of the argument.
    n : int
        The length of tuple to which the argument is parsed.

    Raises
    ------
    TypeError
        The argument cannot be parsed.

    Returns
    -------
    arg :

    """
    arg_prime = copy(arg)
    if not isinstance(arg, list) and not isinstance(arg, tuple):
        if arg is None:
            arg = default
        arg = [arg,]*n
    arg = list(arg)
    if len(arg) != n:
        err = "Could not parse {}: {} to tuple of {} arguments"
        raise TypeError(err.format(arg_str, arg_prime, n))
    return arg


###############################################################################
#LINE PLOT CLASS
###############################################################################
class Lineplot():
    """
    Functionality for line plots. Stores all line plot data, draws the axes in
    a child thread based on the user set data.

    Parameters
    ----------
    axes : matplotlib.axes
        The axes on which the lineplot lives.
    n_lines : int, optional
        The number of lines that are drawn on the plot. The default value is
        1.
    threaded : bool, optional
        A boolean flag that indicates whether plot redrawing is threaded or
        not. MAKE SURE TO CALL TERMINATE FUNCTION WHEN DONE WITH LINEPLOT IF
        THREADED FLAG IS SET TO TRUE. The default is False.
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
            Specifies how many data points are used to draw a line. Only the
            most recently added data points are kept. Any data points added
            more than tail data points ago are discarded and not plotted. When
            tuple, must have length n_lines. A value less than or equal to 0
            means that no data is ever discarded and all data points added to
            the animator will be drawn. The default is -1.
        title : string, optional
            The title of the plot. Will be written above the plot when
            rendered. The default is None.
        x_label : string, optional
            The label to apply to the x axis. Will be written under the plot
            when rendered. The default is None.
        y_label : string, optional
            The label to apply to the y axis. Will be written to the left of
            the plot when rendered. The default is None.
        label : string or tuple of strings, optional
            The label applied to each artist. The labels are shown in a legend
            in the top right of the plot. When tuple, must have length
            n_lines. When None, no labels are made. The default is None.
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

    """
    def __init__(self, axes, n_lines=1, threaded=False, **kwargs):
        """
        Constructor method.
        """
        # Create a mutex lock to synchronize the drawing thread (child) and the
        # setting thread (parent)
        self._LOCK = Lock()

        # Store the axes on which the plot lives
        self._axes = axes

        # Set default values
        self.options = {'n_lines': n_lines,
                        'x_lim': [None, None],
                        'y_lim': [None, None],
                        'h_zero_line': False,
                        'v_zero_line': False,
                        'tail': [-1,]*n_lines,}
        self.labels = {'title': None,
                       'x_label': None,
                       'y_label': None,
                       'label': [None,]*self.options['n_lines'],}
        self.style = {'color': ['black',]*self.options['n_lines'],
                      'line_width': [1.5,]*self.options['n_lines'],
                      'line_style': ['solid',]*self.options['n_lines'],}
        self.data = {'x': [[],]*self.options['n_lines'],
                     'y': [[],]*self.options['n_lines'],}

        # Update the default values with kwargs
        self._apply_kwargs(kwargs)

        # Apply all settings to the axes on which the plot lives.
        self._apply_settings_2_axes()

        # Create the plot artists
        self._lines = self._make_line_artists()

        # The redraw flag tells when something on the axes has been
        # updated and therefore the axes must be redrawn
        self._need_redraw = [False,]*n_lines

        # Threaded operations:
        self._THREADED = threaded
        if self._THREADED:

            # The done flag tells the drawer thread when to stop.
            self._done = False

            # Start the drawing thread
            self._thread = Thread(target=self._drawer_loop)
            self._thread.daemon = True
            self._thread.start()


    def __del__(self):
        """
        Constructor method.
        """
        self.terminate()


    def _apply_kwargs(self, kwargs):
        """
        Updates default values with kwargs.

        Parameters
        ----------
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
                The label to apply to the y axis. Will be written to the left
                of the plot when rendered. The default is None.
            label : string or tuple of strings, optional
                The label applied to each artist. The labels are shown in a
                legend in the top right of the plot. When tuple, must have
                length n_lines. When None, no labels are made. The default is
                None.
            color : matplotlib color string or tuple of color strings, optional
                The color each artist draws in. When tuple, must have length
                n_lines. The default is 'black'.
            line_width : float or tuple of floats, optional
                The line weigth each artist uses. When tuple, must have length
                n_lines. The default is 1.5.
            line_style : line style string or tuple of ls strings, optional
                The line style each artist uses. When tuple, must have length
                n_lines. The default is 'solid'. Select from 'solid',
                'dashed', 'dashdot', or 'dotted'.

        Returns
        -------
        None.

        """
        # Valid kwargs
        valid = ('x_lim', 'y_lim', 'h_zero_line', 'v_zero_line',
                 'tail', 'title', 'x_label', 'y_label', 'label',
                 'color', 'line_width', 'line_style')

        # kwargs that need parsed
        need_parse = {'tail': self.options['tail'],
                      'label': self.labels['label'],
                      'color': self.style['color'],
                      'line_width': self.style['line_width'],
                      'line_style': self.style['line_style']}

        # Apply kwargs
        for kwarg in kwargs:
            # Check kwarg validity
            if not kwarg in valid:
                warn = "{} is not a recognized kwarg. Continuing..."
                warnings.warn(warn.format(kwarg), RuntimeWarning)
                sys.stderr.flush()
                continue

            # Parse to n_lines if needed
            if kwarg in need_parse:
                kwargs[kwarg] = _parse_arg(kwargs[kwarg], need_parse[kwarg],
                                           kwarg, self.options['n_lines'])

            # Update values
            if kwarg in self.options:
                self.options[kwarg] = kwargs[kwarg]
            elif kwarg in self.labels:
                self.labels[kwarg] = kwargs[kwarg]
            elif kwarg in self.style:
                self.style[kwarg] = kwargs[kwarg]


    def _apply_settings_2_axes(self):
            """
            Applies all settings to the axes on which the plot lives.

            Parameters
            ----------
            None.

            Returns
            -------
            None.

            """
            # Clear the axis
            self._axes.clear()

            # Set the labels
            self._axes.set_title(self.labels['title'], fontsize=FONT_SIZE+1)
            self._axes.set_xlabel(self.labels['x_label'], fontsize=FONT_SIZE)
            self._axes.set_ylabel(self.labels['y_label'], fontsize=FONT_SIZE)

            # Set the tick mark size
            self._axes.tick_params(axis='both', which='major',
                                  labelsize=FONT_SIZE)
            self._axes.tick_params(axis='both', which='minor',
                                  labelsize=FONT_SIZE)

            # Add the zero lines
            if self.options['h_zero_line']:
                self._axes.axhline(y=0, xmin=0, xmax=1,
                                  alpha=0.75, lw=0.75, c='k')
            if self.options['v_zero_line']:
                self._axes.axvline(x=0, ymin=0, ymax=1,
                                  alpha=0.75, lw=0.75, c='k')

            # Set the extents of the axes on which the plot lives
            self._set_axes_extents()


    def _set_axes_extents(self):
        """
        Sets the extents of the axes on which the plot lives.
        For each axis, if either user set limit is set to None, the extent of
        that side of the axis is set by the data range.

        Returns
        -------
        None.

        """
        # Aquire mutex lock to read self.data
        with self._LOCK:

            # Get the range data ranges. None if no data.
            x_range = self._get_data_range(self.data['x'])
            y_range = self._get_data_range(self.data['y'])

        # Calculate the x extents
        x_extents = (self._get_l_extent(self.options['x_lim'][0], x_range),
                     self._get_u_extent(self.options['x_lim'][1], x_range))

        # Calculate the y extents
        y_extents = (self._get_l_extent(self.options['y_lim'][0], y_range),
                     self._get_u_extent(self.options['y_lim'][1], y_range))

        # Set the extents
        self._axes.set_xlim(x_extents[0], x_extents[1])
        self._axes.set_ylim(y_extents[0], y_extents[1])


    def _get_data_range(self, artist_data_list):
        """
        Gets the data range.

        Parameters
        ----------
        artist_data_list : list of lists of floats
            The list of each artist's data.

        Returns
        -------
        rng : 2 tuple of floats or 2 tuple of None
            (min , max) if more than 1 data point found. (None, None) else.

        """
        # Get the range of the data list. If there is not enough data points,
        # range is None
        rng = (None, None)

        # Go through all values in all data of entire list. Extract min and
        # max values
        mn = np.inf
        mx = -np.inf
        for artist_data in artist_data_list:
            for datum in artist_data:
                if datum < mn:
                    mn = copy(datum)
                if datum > mx:
                    mx = copy(datum)

        # If more than 1 data point was found, set the range
        if not mn == np.inf and not mx == -np.inf and not mn == mx:
            rng = (mn, mx)

        return rng


    def _get_l_extent(self, lower_user_limit, data_range):
        """
        Get the lower extent of the axis based on the user set limits and the
        data range.

        Parameters
        ----------
        lower_user_limit : float or None
            The lower user set limit. None if no limit is set.
        data_range : 2 tuple of floats or 2 tuple of None
            (min data value, max data value). (None, None) if no data

        Returns
        -------
        lower_extent : float
            The calculated lower extent.

        """
        if not lower_user_limit is None:
            # user set value takes priority
            lower_extent = float(copy(lower_user_limit))

        elif not any([datum is None for datum in data_range]):
           # If no user value, but there is data, set based on data range
           lower_extent = data_range[0] - 0.05*(data_range[1] - data_range[0])

        else:
            # If no user set value or data, set default extent to 0.0
            lower_extent = 0.0

        return lower_extent


    def _get_u_extent(self, upper_user_limit, data_range):
        """
        Get the upper extent of the axis based on the user set limits and the
        data range.

        Parameters
        ----------
        upper_user_limit : float or None
            The upper user set limit. None if no limit is set.
        data_range : 2 tuple of floats or 2 tuple of None
            (min data value, max data value). (None, None) if no data

        Returns
        -------
        upper_extent : float
            The calculated upper extent.

        """
        if not upper_user_limit is None:
            # user set value takes priority
            upper_extent = float(copy(upper_user_limit))

        elif not any([datum is None for datum in data_range]):
           # If no user value, but there is data, set based on data range
           upper_extent = data_range[1] + 0.05*(data_range[1] - data_range[0])

        else:
            # If no user set value or data, set default extent to 1.0
            upper_extent = 1.0

        return upper_extent


    def _make_line_artists(self):
        """
        Makes one line artist for every n_lines. Sets the current data.
        Applies style and label options.

        Returns
        -------
        lines : list of matplotlib.lines.Line2D
            An ordered list of each line artist.

        """
        # Make one line artist for every n_artist
        lines = []
        for line_ind in range(self.options['n_lines']):

            # Extract style and label options
            kwargs = {'c' : self.style['color'][line_ind],
                      'lw' : self.style['line_width'][line_ind],
                      'ls' : self.style['line_style'][line_ind],
                      'label' : self.labels['label'][line_ind]}

            # Artists that have a tail length less than infinite
            # need a head marker.
            if self.options['tail'][line_ind] > 0:
                kwargs['ms'] = 2.5*self.style['line_width'][line_ind]
                kwargs['marker'] = 'o'

            # Aquire mutex lock to read self.data
            with self._LOCK:

                # Make a line artist. Set the current data. Apply style and
                # label options
                line, = self._axes.plot(self.data['x'][line_ind],
                                        self.data['x'][line_ind],
                                        **kwargs)
                lines.append(line)

        return lines


    def _drawer_loop(self):
        """
        Runs a drawer loop that continuously calls redraw_plot until the done
        flag is set to True

        Returns
        -------
        None.

        """
        # Continuously redraw plot
        while True:
            self.redraw_plot()

            # Aquire mutex lock to read flag
            with self._LOCK:

                # If done flag is set, end drawer loop
                if self._done:
                    break

            # Remove CPU strain by sleeping for a little bit
            time.sleep(0.01)


    def append_point(self, x_point, y_point, line_ind=0):
        """
        Appends a single data point to the end of one artist's data.

        Parameters
        ----------
        x_point : float
            The x coordinate of the data point being appended.
        y_point : float
            The y coordinate of the data point being appended.
        line_ind : int, optional
            The line index whose plot data is being updated. Does not need
            to be changed if the plot only has one line. The default value
            is 0.

        Returns
        -------
        None.

        """
        # Aquire mutex lock to set self.data and flag
        with self._LOCK:
            # Append the datum
            self.data['x'][line_ind].append(copy(x_point))
            self.data['y'][line_ind].append(copy(y_point))

            # Tell the drawer that the axes must be redrawn
            self._need_redraw[line_ind] = True

        # If unthreaded version is used, synchronously redraw plot
        if not self._THREADED:
            self.redraw_plot()


    def reset_data(self):
        """
        Clears all data from plot.

        Returns
        -------
        None.

        """
        for line_ind in range(self.options['n_lines']):
            self.set_data([], [], line_ind=line_ind)


    def set_data(self, x_data, y_data, line_ind=0):
        """
        Sets one artist's plot data to new values.

        Parameters
        ----------
        x_data : list of floats
            The plot's new x data points.
        y_data : list of floats
            The plot's new y data points.
        line_ind : int, optional
            The line index whose plot data is being updated. Does not need
            to be changed if the plot only has one line. The default value
            is 0.

        Returns
        -------
        None.

        """
        # Aquire mutex lock to set self.data and flag
        with self._LOCK:
            # Set the new data
            self.data['x'][line_ind] = copy(x_data)
            self.data['y'][line_ind] = copy(y_data)

            # Tell the drawer that the axes must be redrawn
            self._need_redraw[line_ind] = True

        # If unthreaded version is used, synchronously redraw plot
        if not self._THREADED:
            self.redraw_plot()


    def redraw_plot(self):
        """
        Redraws all artists in plot. Resizes axes.

        Returns
        -------
        None.

        """
        # Aquire mutex lock to read flag
        with self._LOCK:

            # Determine which aritist inds need redrawn
            line_inds = [i for i, f in enumerate(self._need_redraw) if f]

        # Redraw each artist that needs redrawn
        for line_ind in line_inds:
            self._redraw_line(line_ind)

        # Update the axes extents. We only need to do this step if the
        # user has not set at least one limit and at least one artist was
        # redrawn
        update_x_extents = any([x is None for x in self.options['x_lim']])
        update_y_extents = any([y is None for y in self.options['y_lim']])
        update_lines = len(line_inds) > 0
        if update_x_extents or update_y_extents and update_lines:
            self._set_axes_extents()


    def _redraw_line(self, line_ind):
        """
        Redraws a single line.

        Parameters
        line_ind
        artist_ind : int
            The index of the line being redrawn.

        Returns
        -------
        None.

        """
        # Aquire the line artist
        line = self._lines[line_ind]

        # Aquire mutex lock to read self.data and set flag
        with self._LOCK:

            # Update the line artist's data
            line.set_data(self.data['x'][line_ind],
                          self.data['y'][line_ind])

            # If there is a head marker, move it to the new head
            if self.options['tail'][line_ind] > 0:
                line.set_markevery((len(self.data['x'][line_ind])-1, 1))

            # Note that the artist has been redrawn
            self._need_redraw[line_ind] = False


    def terminate(self):
        """
        Terminate the drawer thread (if it exists). MAKE SURE TO CALL THIS
        WHEN DONE WITH LINEPLOT IF THREADED FLAG IS SET TO TRUE.

        Returns
        -------
        None.

        """
        if self._THREADED:
            with self._LOCK:
                self._done = True
            self._thread.join()


###############################################################################
#BAR CHART CLASS
###############################################################################
class Barchart():
    """
    Functionality for bar charts. Stores all bar chart data, draws the axes in
    a child thread based on the user set data.

    Parameters
    ----------
    axes : matplotlib.axes
        The axes on which the barchart lives.
    n_bars : int, optional
        The number of bars on the chart. The default value is
        1.
    threaded : bool, optional
        A boolean flag that indicates whether chart redrawing is threaded or
        not. MAKE SURE TO CALL TERMINATE FUNCTION WHEN DONE WITH BARCHART IF
        THREADED FLAG IS SET TO TRUE. The default is False.
    **kwargs: dict
        x_lim : [float, float], optional
            The limits to apply to the x axis of the plot. A value of None
            will apply automatically updating limits to the corresponding
            bound of the axis. For example [None, 10.] will fix the upper
            bound to exactly 10, but the lower bound will freely change to
            show all data.The default is [None, None].
        v_zero_line : boolean, optional
            A boolean flag that indicates whether a vertical line will be
            drawn on the x=0 line. The default is False.
        title : string, optional
            The title of the chart. Will be written above the chart when
            rendered. The default is None.
        x_label : string, optional
            The label to apply to the x axis. Will be written under the chart
            when rendered. The default is None.
        y_label : string, optional
            The label to apply to the y axis. Will be written to the left of
            the chart when rendered. The default is None.
        label : string or tuple of strings, optional
            The label applied to each bar. The labels are shown in a legend
            in the top right of the chart. When tuple, must have length
            n_bars. When None, no labels are made. The default is None.
        color : matplotlib color string or tuple of color strings, optional
            The color of each bar. When tuple, must have length
            n_bars. The default is 'blue'.

    """
    def __init__(self, axes, n_bars=1, threaded=False, **kwargs):
        """
        Constructor method.
        """
        # Create a mutex lock to synchronize the drawing thread (child) and the
        # setting thread (parent)
        self._LOCK = Lock()

        # Store the axes on which the chart lives
        self._axes = axes

        # Set default values
        self.options = {'n_bars': n_bars,
                        'x_lim': [None, None],
                        'v_zero_line': False,}
        self.labels = {'title': None,
                       'x_label': None,
                       'y_label': None,
                       'label': ['Bar {}'.format(i+1) for i in range(n_bars)],}
        self.style = {'color': ['black',]*n_bars,}
        self.values = [0.0,]*n_bars

        # Update the default values with kwargs
        self._apply_kwargs(kwargs)

        # Apply all settings to the axes on which the chart lives.
        self._apply_settings_2_axes()

        # Create the plot artists
        self._bars = self._make_bar_artists()

        # The redraw flag tells when something on the axes has been
        # updated and therefore the axes must be redrawn
        self._need_redraw = [False,]*n_bars

        # Threaded operations:
        self._THREADED = threaded
        if self._THREADED:

            # The done flag tells the drawer thread when to stop.
            self._done = False

            # Start the drawing thread
            self._thread = Thread(target=self._drawer_loop)
            self._thread.daemon = True
            self._thread.start()


    def __del__(self):
        """
        Constructor method.
        """
        self.terminate()


    def _apply_kwargs(self, kwargs):
        """
        Updates default values with kwargs.

        Parameters
        ----------
        **kwargs: dict
            x_lim : [float, float], optional
                The limits to apply to the x axis of the plot. A value of None
                will apply automatically updating limits to the corresponding
                bound of the axis. For example [None, 10.] will fix the upper
                bound to exactly 10, but the lower bound will freely change to
                show all data.The default is [None, None].
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
                The label applied to each artist. The labels are shown in a
                legend in the top right of the plot. When tuple, must have
                length n_artists. When None, default labels are made.
                The default is None.
            color : matplotlib color string or tuple of color strings, optional
                The color each artist draws in. When tuple, must have length
                n_artists. The default is 'black'.

        Returns
        -------
        None.

        """
        # Valid kwargs
        valid = ('x_lim', 'v_zero_line', 'title',
                 'x_label', 'y_label', 'label', 'color')

        # kwargs that need parsed
        need_parse = {'label': self.labels['label'],
                      'color': self.style['color'],}

        # Apply kwargs
        for kwarg in kwargs:
            # Check kwarg validity
            if not kwarg in valid:
                warn = "{} is not a recognized kwarg. Continuing..."
                warnings.warn(warn.format(kwarg), RuntimeWarning)
                sys.stderr.flush()
                continue

            # Parse to n_bars if needed
            if kwarg in need_parse:
                kwargs[kwarg] = _parse_arg(kwargs[kwarg], need_parse[kwarg],
                                           kwarg, self.options['n_bars'])

            # Update values
            if kwarg in self.options:
                self.options[kwarg] = kwargs[kwarg]
            elif kwarg in self.labels:
                self.labels[kwarg] = kwargs[kwarg]
            elif kwarg in self.style:
                self.style[kwarg] = kwargs[kwarg]


    def _apply_settings_2_axes(self):
            """
            Applies all settings to the axes on which the chart lives.

            Parameters
            ----------
            None.

            Returns
            -------
            None.

            """
            # Clear the axis
            self._axes.clear()

            # Set the labels
            self._axes.set_title(self.labels['title'], fontsize=FONT_SIZE+1)
            self._axes.set_xlabel(self.labels['x_label'], fontsize=FONT_SIZE)
            self._axes.set_ylabel(self.labels['y_label'], fontsize=FONT_SIZE)

            # Set the tick mark size
            self._axes.tick_params(axis='both', which='major',
                                  labelsize=FONT_SIZE)
            self._axes.tick_params(axis='both', which='minor',
                                  labelsize=FONT_SIZE)

            # Add the zero line
            if self.options['v_zero_line']:
                self._axes.axvline(x=0, ymin=0, ymax=1,
                                  alpha=0.75, lw=0.75, c='k')

            # Set the extents of the axes on which the chart lives
            self._set_axes_extents()


    def _set_axes_extents(self):
        """
        Sets the extents of the axes on which the chart lives.
        If either user set limit is set to None, the extent of
        that side of the axis is set by the data range. The y extents are
        always set automatically to include each bar in the chart.

        Returns
        -------
        None.

        """
        # Aquire mutex lock to read self.data
        with self._LOCK:

            # Get the range data ranges. None if no data.
            x_range = self._get_data_range(self.values)

        # Calculate the x extents
        x_extents = (self._get_l_extent(self.options['x_lim'][0], x_range),
                     self._get_u_extent(self.options['x_lim'][1], x_range))

        # Set the extents
        self._axes.set_xlim(x_extents[0], x_extents[1])


    def _get_data_range(self, bar_data):
        """
        Gets the data range.

        Parameters
        ----------
        bar_data : mixed list of floats and None
            The list of each bar's value.

        Returns
        -------
        rng : 2 tuple of floats or 2 tuple of None
            (min , max) if more than 1 data point found. (None, None) else.

        """
        # Get the range of the data list. If there is not enough data points,
        # range is None
        rng = (None, None)

        # Go through all values. Extract min and max
        mn = np.inf
        mx = -np.inf
        for datum in bar_data:
            if not datum is None:
                if datum < mn:
                    mn = copy(datum)
                if datum > mx:
                    mx = copy(datum)

        # If more than 1 data point was found, set the range
        if not mn == np.inf and not mx == -np.inf and not mn == mx:
            rng = (mn, mx)

        return rng


    def _get_l_extent(self, lower_user_limit, data_range):
        """
        Get the lower extent of the axis based on the user set limits and the
        data range.

        Parameters
        ----------
        lower_user_limit : float or None
            The lower user set limit. None if no limit is set.
        data_range : 2 tuple of floats or 2 tuple of None
            (min data value, max data value). (None, None) if no data

        Returns
        -------
        lower_extent : float
            The calculated lower extent.

        """
        if not lower_user_limit is None:
            # user set value takes priority
            lower_extent = float(copy(lower_user_limit))

        elif not any([datum is None for datum in data_range]):
           # If no user value, but there is data, set based on data range
           lower_extent = data_range[0] - 0.05*(data_range[1] - data_range[0])

        else:
            # If no user set value or data, set default extent to 0.0
            lower_extent = 0.0

        return lower_extent


    def _get_u_extent(self, upper_user_limit, data_range):
        """
        Get the upper extent of the axis based on the user set limits and the
        data range.

        Parameters
        ----------
        upper_user_limit : float or None
            The upper user set limit. None if no limit is set.
        data_range : 2 tuple of floats or 2 tuple of None
            (min data value, max data value). (None, None) if no data

        Returns
        -------
        upper_extent : float
            The calculated upper extent.

        """
        if not upper_user_limit is None:
            # user set value takes priority
            upper_extent = float(copy(upper_user_limit))

        elif not any([datum is None for datum in data_range]):
           # If no user value, but there is data, set based on data range
           upper_extent = data_range[1] + 0.05*(data_range[1] - data_range[0])

        else:
            # If no user set value or data, set default extent to 1.0
            upper_extent = 1.0

        return upper_extent


    def _make_bar_artists(self):
        """
        Makes one bar artist for every n_bars. Sets the current data.
        Applies style and label options.

        Returns
        -------
        bars : list of matplotlib.patches.Rectangle
            The bar artists.

        """

        # Extract style options
        kwargs = {'color' : self.style['color'],
                  'edgecolor' : ['k',]*self.options['n_bars'],
                  'linewidth' : [1.25,]*self.options['n_bars'],
                  'align' : 'center',}

        # Aquire mutex lock to read self.values
        with self._LOCK:

            # Make bar artists. Set the current data. Apply style and
            # label options
            container = self._axes.barh(self.labels['label'],
                                        self.values, **kwargs)

        # Extract bar artists from the container
        bars = [artist for artist in container]
        return bars


    def _drawer_loop(self):
        """
        Runs a drawer loop that continuously calls redraw_chart until the done
        flag is set to True

        Returns
        -------
        None.

        """
        # Continuously redraw plot
        while True:
            self.redraw_chart()

            # Aquire mutex lock to read flag
            with self._LOCK:

                # If done flag is set, end drawer loop
                if self._done:
                    break

            # Remove CPU strain by sleeping for a little bit
            time.sleep(0.01)


    def set_value(self, value, bar_ind=0):
        """
        Set's a bar's value.

        Parameters
        ----------
        value : float
            The value to which the bar is set
        bar_ind : int, optional
            The bar index whose value is set. Does not need
            to be changed if the chart only has one bar. The default value
            is 0.

        Returns
        -------
        None.

        """
        # Aquire mutex lock to set self.values and flag
        with self._LOCK:
            # Set the value
            self.values[bar_ind] = float(copy(value))

            # Tell the drawer that the axes must be redrawn
            self._need_redraw[bar_ind] = True

        # If unthreaded version is used, synchronously redraw plot
        if not self._THREADED:
            self.redraw_plot()


    def reset_data(self):
        """
        Clears all data from chart.

        Returns
        -------
        None.

        """
        for bar_ind in range(self.option['n_bars']):
            self.set_value(0.0, bar_ind)


    def redraw_chart(self):
        """
        Redraws all bars in chart. Resizes axes.

        Returns
        -------
        None.

        """
        # Aquire mutex lock to read flag
        with self._LOCK:

            # Determine which aritist inds need redrawn
            bar_inds = [i for i, f in enumerate(self._need_redraw) if f]

        # Redraw each artist that needs redrawn
        for bar_ind in bar_inds:
            self._redraw_bar(bar_ind)

        # Update the axes extents. We only need to do this step if the
        # user has not set at least one limit and at least one bar was
        # redrawn
        update_x_extents = any([x is None for x in self.options['x_lim']])
        update_bars = len(bar_inds) > 0
        if update_x_extents and update_bars:
            self._set_axes_extents()


    def _redraw_bar(self, bar_ind):
        """
        Redraws a single bar.

        Parameters
        ----------
        bar_ind : int
            The index of the bar being redrawn.

        Returns
        -------
        None.

        """
        # Aquire the artist
        bar = self._bars[bar_ind]

        # Aquire mutex lock to read self.values and set flag
        with self._LOCK:

            # Update the line artist's data
            bar.set_width(self.values[bar_ind])

            # Note that the artist has been redrawn
            self._need_redraw[bar_ind] = False


    def terminate(self):
        """
        Terminate the drawer thread (if it exists). MAKE SURE TO CALL THIS
        WHEN DONE WITH LINEPLOT IF THREADED FLAG IS SET TO TRUE.

        Returns
        -------
        None.

        """
        if self._THREADED:
            with self._LOCK:
                self._done = True
            self._thread.join()


if __name__ == "__main__":
        (n_rows, n_cols) = (2, 1)
        fig_res = 300 * n_rows
        fig_dpi = 150
        fig_AR = 1.6*(n_cols/n_rows)
        fig_size = (fig_AR*fig_res/fig_dpi, fig_res/fig_dpi)
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi, frameon=True,
                         facecolor="w")


        axes_list = []
        for i in range(n_rows*n_cols):
                axes_list.append(fig.add_subplot(n_rows, n_cols, i+1))


        lineplot = Lineplot(axes_list[0], n_lines=2, threaded=True,
                            color=['r', 'b'])
        for i in range(100):
            lineplot.append_point(i, np.random.rand(), line_ind=0)
        lineplot.set_data(np.arange(50,75), np.random.rand(25)*2+2, line_ind=1)

        barchart = Barchart(axes_list[1], n_bars=2, threaded=True,
                            color=['r', 'b'], x_lim=[-1.0, 15.0],
                            v_zero_line=True)
        barchart.set_value(10, bar_ind=0)
        barchart.set_value(-0.5, bar_ind=1)

        lineplot.terminate()
        barchart.terminate()
