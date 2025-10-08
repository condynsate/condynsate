###############################################################################
#DEPENDENCIES
###############################################################################
import time
from copy import deepcopy as copy
from threading import (Thread, Lock)
import numpy as np
import matplotlib.pyplot as plt
MAX_N_ROWS = 2


###############################################################################
#FIGURE CLASS
###############################################################################
class Figure():
    """
    Figure class. Creates and draws to a matplotlib figure.

    Parameters
    ----------
    n : int
        The number of subplots in the figure.
    threaded : bool, optional
        A boolean flag that indicates whether figure redrawing is threaded or
        not. MAKE SURE TO CALL TERMINATE FUNCTION WHEN DONE WITH LINEPLOT IF
        THREADED FLAG IS SET TO TRUE. The default is False.
    """
    def __init__(self, n, threaded=False):
        # Make a lock for the figure
        self._LOCK = Lock()

        # Get the figure shape and make it and its axes
        self.shape = self._get_shape(n)
        self._fig, self._axes_list = self._make()

        # Draw the current figure
        self._img = None
        self.redraw()

        # Start the drawing thread
        self._THREADED = threaded
        self._done = False
        self._start()


    def __del__(self):
        """
        Deconstructor method.
        """
        self.terminate()


    def _get_shape(self, n):
        """
        Calculates the figure shape given total number of subplots.

        Parameters
        ----------
        n : int
            The number of subplots.

        Returns
        -------
        dim : tuple, shape(2)
            The dimensions of the figure.

        """
        n_rows = min(n, MAX_N_ROWS)
        n_cols = int(np.ceil(n / MAX_N_ROWS))
        return (n_rows, n_cols)


    def _make(self):
        """
        Makes a figure and adds axes according to the figure shape.

        Returns
        -------
        fig : matplotlib.figure
            The created figure.
        axes : list of matplotlib.axes._axes
            Each of the axes added to the figure in order.

        """
        # Make the figure
        res = 240 * self.shape[0]
        height = 2.0 * self.shape[0]
        AR = 1.6*(self.shape[1]/self.shape[0]) # Widescreen AR
        size = (AR*height, height)
        dpi = res/height
        fig = plt.figure(figsize=size, dpi=dpi, frameon=True,
                         facecolor="w")

        # Add axes to the figure
        axes_list = []
        for i in range(self.shape[0]*self.shape[1]):
            axes = fig.add_subplot(self.shape[0], self.shape[1], i+1)
            axes_list.append(axes)

        # General figure formatting
        fig.tight_layout()

        return fig, axes_list


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


    def get_axes(self):
        """
        Gets the ordered list of axes.

        Returns
        -------
        axes_list : list of matplotlib.axes._axes
            The ordered axes list.

        """
        return self._axes_list


    def get_lock(self):
        """
        Gets the figure's mutex lock.

        Returns
        -------
        lock : threading.Lock
            The figure's mutex lock.

        """
        return self._LOCK


    def redraw(self):
        """
        Redraws the figure to self._img. Thread safe self._fig and self._img
        read and write.

        Returns
        -------
        None.

        """
        # Aquire mutex lock to interact with figure and self._img
        with self._LOCK:
            self._fig.canvas.draw() # Draw on the canvas

            # Get the shape of the canvas image
            img_shape = self._fig.canvas.get_width_height()
            img_shape = (img_shape[1], img_shape[0], 4)

            # Aquire the img on the canvas
            img_buf = self._fig.canvas.tostring_argb()
            img = np.frombuffer(img_buf, dtype=np.uint8)
            img = img.reshape(img_shape)[:,:,1:]

            # Make a copy of the canvas image
            self._img = img.copy()

            # Make sure the image has height and width divisible by 2 for
            # h264 codex (will be implemented later). This is done by adding
            # an extra white row and/or column if needed
            if img_shape[0]%2 != 0:
                extra_row = 255*np.ones((1, img_shape[1], 3), dtype=np.uint8)
                self._img = np.concatenate((self._img, extra_row), axis=0)
            if img_shape[1]%2 != 0:
                extra_col = 255*np.ones((img_shape[0], 1, 3), dtype=np.uint8)
                self._img = np.concatenate((self._img, extra_col), axis=1)


    def get_image(self):
        """
        Gets the current figure image.

        Returns
        -------
        image : numpy array (H, W, 3)
            The current (R,G,B) figure canvas image.

        """
        # Aquire lock to copy image
        with self._LOCK:
            image = copy(self._img)
        return image


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
            self._thread.join()


###############################################################################
#TESTING DONE IN MAIN LOOP
###############################################################################
if __name__ == "__main__":
    test = Figure(n=2, threaded=True)
    test.redraw()
    test.terminate()
