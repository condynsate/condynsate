###############################################################################
#DEPENDENCIES
###############################################################################
import sys
import warnings
import traceback
from figure import Figure
from subplots import (Lineplot, Barchart)
import numpy as np


###############################################################################
#ANIMATOR CLASS
###############################################################################
class Animator():
    def __init__(self, threaded=False):
        self._n_plots = 0
        self._plots = []
        self._figure = None
        self._axes_list = []
        self._THREADED = threaded


    def __del__(self):
        self.terminate()


    def add_lineplot(self, n_lines=1, **kwargs):
        # Ensure n_lines is int > 0
        if not type(n_lines) is int or n_lines <= 0:
            self.terminate()
            err = "Argument n_lines must be type int > 0."
            raise TypeError(err)

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

        return plot_data['subplot_ind'], plot_data['artist_inds']


    def add_barchart(self, n_bars=1, **kwargs):
        # Ensure n_bars is int > 0
        if not type(n_bars) is int or n_bars <= 0:
            self.terminate()
            err = "Argument n_bars must be type int > 0."
            raise TypeError(err)

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

        return plot_data['subplot_ind'], plot_data['artist_inds']


    def start(self):
        try:
            # Make the figure
            self._figure = Figure(self._n_plots, threaded=self._THREADED)

        except:
            self.terminate()
            print("Something went wrong while building the figure.",
                  flush=True)
            traceback.print_exc()
            return -1

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
            print("Something went wrong while building the subplots",
                  flush=True)
            traceback.print_exc()
            return -1

        return 0


    def _make_lineplot(self, subplot_ind):
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
        # Build the barchart
        axes = self._figure.get_axes()[subplot_ind]
        fig_lock = self._figure.get_lock()
        n_bars = self._plots[subplot_ind]['n_artists']
        threaded = self._plots[subplot_ind]['threaded']
        kwargs = self._plots[subplot_ind]['kwargs']
        subplot = Barchart(axes, fig_lock, n_bars, threaded, **kwargs)

        # Add the barchart and figure to the plot data structure
        self._plots[subplot_ind]['Subplot'] = subplot


    def _is_subplot_ind(self, subplot_ind):
        # Check the type
        type_ok = type(subplot_ind) is int
        if not type_ok:
            warn = "subplot_ind must be type int."
            warnings.warn(warn, RuntimeWarning)
            sys.stderr.flush()
            return False

        # Check the upper and lower bounds
        lb_ok = subplot_ind >= 0
        ub_ok = subplot_ind < len(self._plots)
        if not lb_ok or not ub_ok:
            warn = "Invalid subplot_ind. Must be in range [0, {}]"
            warnings.warn(warn.format(len(self._plots)-1), RuntimeWarning)
            sys.stderr.flush()
            return False

        # Make sure the indexed subplot is type Subplot
        sp_ok = type(self._plots[subplot_ind]['Subplot']).__module__
        sp_ok = sp_ok == 'subplots'
        if not sp_ok:
            warn = "The subplot at ind {} is not built or an invalid type."
            warnings.warn(warn.format(subplot_ind), RuntimeWarning)
            sys.stderr.flush()
            return False

        return True


    def _is_barchart(self, subplot_ind):
        typ = self._plots[subplot_ind]['type']
        if not typ.lower() == 'barchart':
            warn = "Subplot {} is not a Barchart. It is a {}."
            warnings.warn(warn.format(subplot_ind, typ), RuntimeWarning)
            sys.stderr.flush()
            return False
        return True


    def _is_lineplot(self, subplot_ind):
        typ = self._plots[subplot_ind]['type']
        if not typ.lower() == 'lineplot':
            warn = "Subplot {} is not a Lineplot. It is a {}."
            warnings.warn(warn.format(subplot_ind, typ), RuntimeWarning)
            sys.stderr.flush()
            return False
        return True


    def _artist_ind_ok(self, subplot_ind, art_ind):
        # Get the artist type
        if self._plots[subplot_ind]['type'].lower() == 'barchart':
            art_typ = "bar_ind"
        elif self._plots[subplot_ind]['type'].lower() == 'lineplot':
            art_typ = "line_ind"

        # Check the type
        type_ok = type(art_ind) is int
        if not type_ok:
            warn = "{} must be type int.".format(art_typ)
            warnings.warn(warn, RuntimeWarning)
            sys.stderr.flush()
            return False

        # Check the upper and lower bounds
        lb_ok = art_ind >= 0
        ub_ok = art_ind < self._plots[subplot_ind]['n_artists']
        if not lb_ok or not ub_ok:
            warn = "Invalid {}. Must be in range [0, {}]"
            warn = warn.format(art_typ,self._plots[subplot_ind]['n_artists']-1)
            warnings.warn(warn, RuntimeWarning)
            sys.stderr.flush()
            return False

        return True


    def _is_number(self, value):
        ints = [np.int64, np.int32, np.int16, np.int8,
                 np.uint64, np.uint64, np.uint64, np.uint64, int]
        floats = [np.float64, np.float32, np.float16, float]
        if type(value) in ints:
            return True
        if type(value) in floats and not np.isnan(value):
            return True
        warn = "{} is not a number.".format(value)
        warnings.warn(warn, RuntimeWarning)
        sys.stderr.flush()
        return False


    def _is_number_list(self, values):
        if type(values) is list or type(values) is np.ndarray:
            for value in values:
                if not self._is_number(value):
                    return False
            return True
        warn = "{} is not a list or array.".format(values)
        warnings.warn(warn, RuntimeWarning)
        sys.stderr.flush()
        return False


    def _sanitize_barchart(self, subplot_ind, bar_ind):
        cond_1 =  self._is_subplot_ind(subplot_ind)
        cond_2 = self._is_barchart(subplot_ind)
        cond_3 = self._artist_ind_ok(subplot_ind, bar_ind)
        return cond_1 and cond_2 and cond_3


    def _sanitize_line_plot(self, subplot_ind, line_ind):
        cond_1 =  self._is_subplot_ind(subplot_ind)
        cond_2 = self._is_lineplot(subplot_ind)
        cond_3 = self._artist_ind_ok(subplot_ind, line_ind)
        return cond_1 and cond_2 and cond_3


    def barchart_set_value(self, value, subplot_ind, bar_ind):
        # Sanitization
        if not self._sanitize_barchart(subplot_ind, bar_ind):
            return -1
        if not self._is_number(value):
            return -1

        # Set value
        self._plots[subplot_ind]['Subplot'].set_value(value, bar_ind)
        return 0


    def lineplot_append_point(self, x_val, y_val, subplot_ind, line_ind):
        # Sanitization
        if not self._sanitize_line_plot(subplot_ind, line_ind):
            return -1
        if not self._is_number(x_val):
            return -1
        if not self._is_number(y_val):
            return -1

        # Append point
        self._plots[subplot_ind]['Subplot'].append_point(x_val,y_val,line_ind)


    def lineplot_set_data(self, x_vals, y_vals, subplot_ind, line_ind):
        # Sanitization
        if not self._sanitize_line_plot(subplot_ind, line_ind):
            return -1
        if not self._is_number_list(x_vals):
            return -1
        if not self._is_number_list(y_vals):
            return -1

        # Set data
        self._plots[subplot_ind]['Subplot'].set_data(x_vals, y_vals, line_ind)


    def reset(self, subplot_ind):
        # Sanitization
        if not self._is_subplot_ind(subplot_ind):
            return -1

        self._plots[subplot_ind]['Subplot'].reset_data()
        return 0


    def reset_all(self):
        for subplot_ind in range(len(self._plots)):
            self._plots[subplot_ind]['Subplot'].reset_data()
        return 0


    def terminate(self):
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

        return 0


###############################################################################
#TESTING DONE IN MAIN LOOP
###############################################################################
if __name__ == "__main__":
    animator = Animator(threaded = True)
    sp1, a1 = animator.add_lineplot(n_lines = 2, color = ['r', 'b'])
    sp2, a2 = animator.add_barchart(n_bars = 4, color = ['k', 'r', 'g', 'b'])
    animator.start()
    animator.barchart_set_value(3, sp2, a2[0])
    animator.barchart_set_value(4, sp2, a2[1])
    animator.barchart_set_value(5, sp2, a2[2])
    animator.barchart_set_value(6, sp2, a2[3])
    animator.lineplot_append_point(0, 0, sp1, a1[0])
    animator.lineplot_append_point(10, 1, sp1, a1[0])
    animator.lineplot_set_data(np.arange(5), np.random.rand(5), sp1, a1[1])
    # animator.reset(sp1)
    # animator.reset_all()
    animator.terminate()
