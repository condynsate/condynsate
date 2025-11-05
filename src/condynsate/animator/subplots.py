"""
The _Subplot classes provide functionality for drawing specifc types of 
subplots to a matplotlib figure to be used and displayed by the Animator 
module.
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import sys
import time
import warnings
from copy import copy
from threading import (Thread, Lock)
import numpy as np
FONT_SIZE = 7

###############################################################################
#SUBPLOT SUPER CLASS
###############################################################################
class _Subplot():
    """
    Parent class for all subplots. Provides functions for parsing kwargs,
    setting kwargs, updating axes' settings, and handling the redraw thread
    (though it does not implement the redraw function).

    Parameters
    ----------
    axes : matplotlib.axes
        The axes on which the subplot lives.
    fig_lock : _thread.lock
        A mutex lock for the fig on which the axes are drawn.
    n_artists : int, optional
        The number of artists that are drawn on the plot. The default value is
        1.
    threaded : bool, optional
        A boolean flag that indicates whether plot redrawing is threaded or
        not. MAKE SURE TO CALL TERMINATE FUNCTION WHEN DONE WITH LINEPLOT IF
        THREADED FLAG IS SET TO TRUE. The default is False.
    """
    def __init__(self, axes, fig_lock, n_artists=1, threaded=False):
        """
        Constructor method.
        """
        # Create a mutex lock to synchronize the drawing thread (child) and the
        # setting thread (parent)
        self._LOCK = Lock()

        # Save mutex lock used to synchronize drawing to figure axes
        self._FIG_LOCK = fig_lock

        # Store the axes on which the plot lives
        self._axes = axes

        # Store the threading flag
        self._THREADED = threaded
        self._done = False
        self._thread = None

        # Dictionaries to hold user set subplot options
        self.options = {'axes' : {},
                        'labels' : {},
                        'artists' : {}}

        # Set the default values
        self.options['axes']['n_artists'] = n_artists
        self.options['axes']['x_lim'] = [None, None]
        self.options['axes']['y_lim'] = [None, None]
        self.options['axes']['h_zero_line'] = False
        self.options['axes']['v_zero_line'] = False
        self.options['labels']['title'] = None
        self.options['labels']['x_label'] = None
        self.options['labels']['y_label'] = None

        # Create a structure for the data
        self.data = {}

        # The redraw flag tells when something on the axes has been
        # updated and therefore the axes must be redrawn
        self._need_redraw = [False,]*n_artists

    def __del__(self):
        """
        Deconstructor method.
        """
        self.terminate()

    def _parse_2_n(self, arg, arg_str, n):
        """
        Parses an argument into a tuple of length n.

        Parameters
        ----------
        arg : add_plot argument
            The argument being parsed to length n.
        arg_str : string
            The string name of the argument being parsed.
        n : int
            The length of tuple to which the argument is parsed.

        Raises
        ------
        ValueError
            The argument cannot be parsed.

        Returns
        -------
        arg : list of length n
            The parsed argument.

        """
        arg_prime = copy(arg)
        if not isinstance(arg, list) and not isinstance(arg, tuple):
            arg = [arg,]*n
        arg = list(arg)
        if len(arg) != n:
            err = "Could not parse {}: {} to tuple of {} arguments"
            raise ValueError(err.format(arg_str, arg_prime, n))
        return arg

    def _parse_artist_options(self):
        """
        Parses each of the artist kwargs so that they are lists with length
        n_artists.

        Returns
        -------
        None.

        """
        # Parse each artist option to length n_artists
        for kwarg in self.options['artists']:
            val = self.options['artists'][kwarg]
            name = kwarg
            n = self.options['axes']['n_artists']
            parsed_val = self._parse_2_n(val, name, n)
            self.options['artists'][kwarg] = parsed_val

    def _apply_kwargs(self, kwargs):
        """
        Updates default values with kwargs. Parses those which are required.

        Parameters
        ----------
        kwargs: dict
            the args being parsed and set.

        Returns
        -------
        None.

        """
        # Apply kwargs
        for kwarg in kwargs:
            in_opts = kwarg in self.options['axes']
            in_labs = kwarg in self.options['labels']
            in_arts = kwarg in self.options['artists']
            if not in_opts and not in_labs and not in_arts:
                msg = f"{kwarg} is not a recognized kwarg."
                warnings.warn(msg)
                sys.stderr.flush()
                continue

            # Update values
            if in_opts:
                self.options['axes'][kwarg] = kwargs[kwarg]
            elif in_labs:
                self.options['labels'][kwarg] = kwargs[kwarg]
            elif in_arts:
                self.options['artists'][kwarg] = kwargs[kwarg]

        # Parse each artist kwarg so that it is a list with length n_artists.
        self._parse_artist_options()

    def _apply_settings(self):
        """
        Applies all settings to the axes on which the plot lives.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # Extract options
        title = self.options['labels']['title']
        x_label = self.options['labels']['x_label']
        y_label = self.options['labels']['y_label']
        hline = self.options['axes']['h_zero_line']
        vline = self.options['axes']['v_zero_line']

        # Aquire mutex lock to draw to figure axes
        with self._FIG_LOCK:

            # Clear the axis
            self._axes.clear()

            # Set the labels
            if not title is None:
                self._axes.set_title(title, fontsize=FONT_SIZE+1)
            if not x_label is None:
                self._axes.set_xlabel(x_label, fontsize=FONT_SIZE)
            if not y_label is None:
                self._axes.set_ylabel(y_label, fontsize=FONT_SIZE)

            # Set the tick mark size
            self._axes.tick_params(axis='both', which='major',
                                  labelsize=FONT_SIZE)
            self._axes.tick_params(axis='both', which='minor',
                                  labelsize=FONT_SIZE)

            # Add the zero lines
            if hline:
                self._axes.axhline(y=0, alpha=0.75, lw=0.75, c='k')
            if vline:
                self._axes.axvline(x=0, alpha=0.75, lw=0.75, c='k')

    def _get_l_extent(self, lower_limit, data_range):
        """
        Get the lower extent of the axis based on the user set limits and the
        data range.

        Parameters
        ----------
        lower_limit : float or None
            The lower user set limit. None if no limit is set.
        data_range : 2 tuple of floats or 2 tuple of None
            (min data value, max data value). (None, None) if no data

        Returns
        -------
        lower_extent : float
            The calculated lower extent.

        """
        if not lower_limit is None:
            # limit value takes priority
            lower_extent = float(copy(lower_limit))

        elif not any(datum is None for datum in data_range):
            # If no user value, but there is data, set based on data range
            lower_extent = data_range[0] - 0.05*(data_range[1] - data_range[0])

        else:
            # If no user set value or data, set default extent to 0.0
            lower_extent = 0.0
        return lower_extent

    def _get_u_extent(self, upper_limit, data_range):
        """
        Get the upper extent of the axis based on the user set limits and the
        data range.

        Parameters
        ----------
        upper_limit : float or None
            The upper user set limit. None if no limit is set.
        data_range : 2 tuple of floats or 2 tuple of None
            (min data value, max data value). (None, None) if no data

        Returns
        -------
        upper_extent : float
            The calculated upper extent.

        """
        if not upper_limit is None:
            # upper limit takes priority
            upper_extent = float(copy(upper_limit))

        elif not any(datum is None for datum in data_range):
            # If no user value, but there is data, set based on data range
            upper_extent = data_range[1] + 0.05*(data_range[1] - data_range[0])

        else:
            # If no user set value or data, set default extent to 1.0
            upper_extent = 1.0
        return upper_extent

    def _get_ranges(self):
        """
        Gets the ranges of all data in self.data.

        Returns
        -------
        ranges : dictionary of 2tuples
            A dictionary of the ranges of each key in the self.data structure.
            (None, None) if not enough data.

        """
        # Set the default ranges to [None, None] for each key in self.data
        ranges = {}
        with self._LOCK:
            keys = copy(list(self.data.keys()))
        for key in keys:
            ranges[key] = [None, None]

        # For each key in the data, find the min and max among all the
        # artists
        for key in keys:
            min_val = np.inf
            max_val = -np.inf

            # Aquire mutext lock to read data
            with self._LOCK:

                # Go through each artist's data extracting range
                for artist_ind in range(len(self.data[key])):
                    if len(self.data[key][artist_ind]) <= 0:
                        continue
                    artist_min = min(self.data[key][artist_ind])
                    artist_max = max(self.data[key][artist_ind])
                    if artist_min < min_val:
                        min_val = copy(artist_min)
                    if artist_max > max_val:
                        max_val = copy(artist_max)

            # Set the ranges for the current data key
            if min_val < np.inf and not min_val == max_val:
                ranges[key][0] = copy(min_val)
            if max_val > -np.inf and not min_val == max_val:
                ranges[key][1] = copy(max_val)

        # Return the ranges
        for key in ranges:
            ranges[key] = tuple(ranges[key])
        return ranges

    def _update_x_extent(self, x_range = (None, None)):
        """
        Sets the extents of the x axis.

        Parameters
        ----------
        x_range : tuple of 2 floats, optional
            The (min, max) value of the x data. If no data, (None, None). The
            default is (None, None).
        Returns
        -------
        None.

        """
        # Calculate the new x extents to set
        lower = self._get_l_extent(self.options['axes']['x_lim'][0], x_range)
        upper = self._get_u_extent(self.options['axes']['x_lim'][1], x_range)
        x_extents = (lower, upper)

        # Aquire mutex lock to set figure axes' extents
        with self._FIG_LOCK:
            self._axes.set_xlim(x_extents[0], x_extents[1])

    def _update_y_extent(self, y_range = (None, None)):
        """
        Sets the extents of the y axis.

        Parameters
        ----------
        y_range : tuple of 2 floats, optional
            The (min, max) value of the y data. If no data, (None, None). The
            default is (None, None).
        Returns
        -------
        None.

        """
        # Calculate the new x extents to set
        lower = self._get_l_extent(self.options['axes']['y_lim'][0], y_range)
        upper = self._get_u_extent(self.options['axes']['y_lim'][1], y_range)
        y_extents = (lower, upper)

        # Aquire mutex lock to set figure axes' extents
        with self._FIG_LOCK:
            self._axes.set_ylim(y_extents[0], y_extents[1])

    def redraw(self):
        """
        Placeholder for the redraw function. Must be redefined for each
        child class to function for the specific child.

        Returns
        -------
        None.

        """

    def _drawer_loop(self):
        """
        Runs a drawer loop that continuously calls redraw until the done
        flag is set to True.

        Returns
        -------
        None.

        """
        # Continuously redraw
        while True:
            self.redraw()

            # Aquire mutex lock to read flag
            with self._LOCK:

                # If done flag is set, end drawer loop
                if self._done:
                    break

            # Remove CPU strain by sleeping for a little bit
            time.sleep(0.01)

    def _start(self):
        """
        Starts the drawing thread.

        Returns
        -------
        None.

        """
        # Threaded operations:
        if self._THREADED:

            # Start the drawing thread
            self._thread = Thread(target=self._drawer_loop)
            self._thread.daemon = True
            self._thread.start()

    def terminate(self):
        """
        Terminate the drawer thread (if it exists). MAKE SURE TO CALL THIS
        WHEN DONE IF THREADED FLAG IS SET TO TRUE.

        Returns
        -------
        None.

        """
        if self._THREADED:
            with self._LOCK:
                self._done = True
            if not self._thread is None:
                self._thread.join()

###############################################################################
#LINE PLOT CLASS
###############################################################################
class Lineplot(_Subplot):
    """
    Functionality for line plots. Stores all line plot data, draws the axes in
    a child thread based on the user set data.

    Parameters
    ----------
    axes : matplotlib.axes
        The axes on which the lineplot lives.
    fig_lock : _thread.lock
        A mutex lock for the fig on which the axes are drawn.
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
    def __init__(self, axes, fig_lock, n_lines=1, threaded=False, **kwargs):
        """
        Constructor method.
        """
        # Initialize the subplot super class
        super().__init__(axes, fig_lock, n_lines, threaded)

        # Set the default lineplot specific artist options
        self.options['artists']['tail'] = -1
        self.options['artists']['label'] = None
        self.options['artists']['color'] = 'black'
        self.options['artists']['line_width'] = 1.5
        self.options['artists']['line_style'] = 'solid'

        # Add x and y data to self.data
        self.data['x'] = np.array([[],]*n_lines).tolist()
        self.data['y'] = np.array([[],]*n_lines).tolist()

        # Apply the user set kwargs
        self._apply_kwargs(kwargs)

        # Apply all settings to the axes on which the plot lives.
        self._apply_settings()

        # Set the x and y extents of the axes
        self._update_extents()

        # Create the plot artists
        self._lines = self._make_lines()

        # Start the drawing thread
        self._start()

    def _update_extents(self):
        """
        Updates the x and y axes' extents.

        Returns
        -------
        None.

        """
        # Get the data ranges
        ranges = self._get_ranges()

        # Update the plot extents
        self._update_x_extent(ranges['x'])
        self._update_y_extent(ranges['y'])

    def _make_lines(self):
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
        for line_ind in range(self.options['axes']['n_artists']):

            # Extract style and label options
            kwargs = {'c' : self.options['artists']['color'][line_ind],
                      'lw' : self.options['artists']['line_width'][line_ind],
                      'ls' : self.options['artists']['line_style'][line_ind],
                      'label' : self.options['artists']['label'][line_ind]}

            # Artists that have a tail length greater than 0
            # need a head marker.
            if self.options['artists']['tail'][line_ind] > 0:
                kwargs['ms'] = 2.5*kwargs['lw']
                kwargs['marker'] = 'o'

            # Aquire mutex lock to read self.data
            with self._LOCK:

                # Aquire mutex lock to draw to figure axes
                with self._FIG_LOCK:

                    # Make a line artist. Set the current data. Apply style and
                    # label options
                    line, = self._axes.plot(self.data['x'][line_ind],
                                            self.data['x'][line_ind],
                                            **kwargs)

                lines.append(line)

        # Add a legend if needed
        if not all(l is None for l in self.options['artists']['label']):
            self._axes.legend(loc='upper right', fontsize=FONT_SIZE-1,
                              frameon=True, fancybox=True, shadow=False,
                              ncols=1)
        return lines

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
            self.redraw()

    def reset_data(self):
        """
        Clears all data from plot.

        Returns
        -------
        None.

        """
        for line_ind in range(self.options['axes']['n_artists']):
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
            self.redraw()

    def redraw(self):
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
        update_x = any(x is None for x in self.options['axes']['x_lim'])
        update_y = any(y is None for y in self.options['axes']['y_lim'])
        update_lines = len(line_inds) > 0
        if update_x or update_y and update_lines:
            self._update_extents()

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

            # Aquire mutex lock to draw to figure axes
            with self._FIG_LOCK:
                # Alias line data
                x_dat = self.data['x'][line_ind]
                y_dat = self.data['y'][line_ind]
                tail = self.options['artists']['tail'][line_ind]

                # Set the line data if there is a tail
                if tail > 0:
                    line.set_data(x_dat[-tail:], y_dat[-tail:])
                    line.set_markevery((len(x_dat[-tail:])-1, 1))

                # Set the line data if there is no tail
                else:
                    line.set_data(x_dat, y_dat)

            # Note that the artist has been redrawn
            self._need_redraw[line_ind] = False

###############################################################################
#BAR CHART CLASS
###############################################################################
class Barchart(_Subplot):
    """
    Functionality for bar charts. Stores all bar chart data, draws the axes in
    a child thread based on the user set data.

    Parameters
    ----------
    axes : matplotlib.axes
        The axes on which the barchart lives.
    fig_lock : _thread.lock
        A mutex lock for the fig on which the axes are drawn.
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
            The label to apply to the x axis. Will be written under the plot
            when rendered. The default is None.
        y_label : string, optional
            The label to apply to the y axis. Will be written to the left of
            the plot when rendered. The default is None.
        label : string or tuple of strings, optional
            The label applied to each bar. The labels are shown in a legend
            in the top right of the chart. When tuple, must have length
            n_bars. When None, no labels are made. The default is None.
        color : matplotlib color string or tuple of color strings, optional
            The color of each bar. When tuple, must have length
            n_bars. The default is 'blue'.

    """
    def __init__(self, axes, fig_lock, n_bars=1, threaded=False, **kwargs):
        """
        Constructor method.
        """
        # Initialize the subplot super class
        super().__init__(axes, fig_lock, n_bars, threaded)

        # Set the default lineplot specific artist options
        self.options['artists']['label'] = [f'Bar {i+1}'
                                            for i in range(n_bars)]
        self.options['artists']['color'] = 'blue'

        # Add x data to self.data
        self.data['x'] = np.array([[0.0],]*n_bars).tolist()

        # Apply the user set kwargs
        self._apply_kwargs(kwargs)

        # Apply all settings to the axes on which the plot lives.
        self._apply_settings()

        # Set the x extents
        self._update_extents()

        # Create the plot artists
        self._bars = self._make_bars()

        # Start the drawing thread
        self._start()

    def _update_extents(self):
        """
        Updates the x and y axes' extents.

        Returns
        -------
        None.

        """
        # Get the data ranges
        ranges = self._get_ranges()

        # Update the plot extents
        self._update_x_extent(ranges['x'])

    def _make_bars(self):
        """
        Makes one bar artist for every n_bars. Sets the current data.
        Applies style and label options.

        Returns
        -------
        bars : list of matplotlib.patches.Rectangle
            The bar artists.

        """

        # Extract style options
        kwargs = {'color' : self.options['artists']['color'],
                  'edgecolor' : ['k',]*self.options['axes']['n_artists'],
                  'linewidth' : [1.25,]*self.options['axes']['n_artists'],
                  'align' : 'center',}

        # Aquire mutex lock to read self.values
        with self._LOCK:

            # Aquire mutex lock to draw to figure axes
            with self._FIG_LOCK:

                # Make bar artists. Set the current data. Apply style and
                # label options
                values = [v[-1] for v in self.data['x']]
                container = self._axes.barh(self.options['artists']['label'],
                                            values, **kwargs)

        # Extract bar artists from the container
        bars = list(container)
        return bars

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
            self.data['x'][bar_ind].append(float(copy(value)))

            # Tell the drawer that the axes must be redrawn
            self._need_redraw[bar_ind] = True

        # If unthreaded version is used, synchronously redraw plot
        if not self._THREADED:
            self.redraw()

    def reset_data(self):
        """
        Clears all data from chart.

        Returns
        -------
        None.

        """
        with self._LOCK:
            n_bars = self.options['axes']['n_artists']
            self.data['x'] = np.array([[0.0],]*n_bars).tolist()

    def redraw(self):
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
        update_x = any(x is None for x in self.options['axes']['x_lim'])
        update_bars = len(bar_inds) > 0
        if update_x and update_bars:
            self._update_extents()

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
        bar_artist = self._bars[bar_ind]

        # Aquire mutex lock to read self.data and set flag
        with self._LOCK:

            # Aquire mutex lock to draw to figure axes
            with self._FIG_LOCK:

                # Update the line artist's data
                bar_artist.set_width(self.data['x'][bar_ind][-1])

            # Note that the artist has been redrawn
            self._need_redraw[bar_ind] = False
