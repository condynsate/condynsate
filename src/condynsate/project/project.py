# -*- coding: utf-8 -*-
"""
This module provides the Project class which is the primary interface with
which users interact when using condynsate.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import time
import signal
from warnings import warn
from condynsate.simulator import Simulator
from condynsate.visualizer import Visualizer
from condynsate.animator import Animator
from condynsate.keyboard import Keyboard

###############################################################################
#PROJECT CLASS
###############################################################################
class Project:
    """
    The Project class ties together a Simulator, Visualizer, Animator, and
    Keyboard into class.

    Parameters
    ----------
    **kwargs

    Keyword Args
    ------------
    simulator_gravity : 3 tuple of floats, optional
        The gravity vector used in the simulation. The default value is
        (0.0, 0.0, -9.81).
    simulator_dt : float, optional
        The finite time step size used by the simulator. If set too
        small, can result in visualizer, simulator desynch. Too small
        is determined by the number of total links in the simulation.
        The default value is 0.01.
    visualizer : bool, optional
        A boolean flag that indicates if the project should include a
        visualizer. This visualizer provides a 3D rendering of the
        simulation state. The default is True.
    visualizer_frame_rate : bool, optional
        The frame rate of the visualizer. When None, attempts to run at
        unlimited. This is not reccomended because it can cause
        communication bottlenecks that cause slow downs. The default
        value is 45.
    visualizer_record : bool, optional
        A boolean flag that indicates if the visualizer will record.
        True, all frames from the start function call to the terminate
        function call are recorded. After the terminate function call,
        these frames are saved with h.264 and outputs in an MP4
        container. The saved file name has the form visualizer.mp4.
        The default is False.
    animator : bool, optional
        A boolean flag that indicates if the project should include an
        animator. This animator provides real-time 2D plotting.
        The default is False.
    animator_frame_rate : float, optional
        The upper limit of the allowed frame rate in frames per second.
        When set, the animator will not update faster than this speed.
        The default is 15.0
    animator_record : bool, optional
        A boolean flag that indicates if the animator should be
        recorded. If True, all frames from the start function call to
        the terminate function call are recorded. After the terminate
        function call, these frames are saved with h.264 and outputs in
        an MP4 container. The saved file name has the form
        animator.mp4. The default is False.

    Attributes
    ----------
    simulator : condynsate.Simulator
        The instance of the condynsate.Simulator class used by this project.
    visualizer : condynsate.Visualizer or None
        The instance of the condynsate.Visualizer class used by this project.
        None if this project uses no visualizer.
    animator : condynsate.Animator or None
        The instance of the condynsate.Animator class used by this project.
        None if this project uses no animator.
    keyboard : condynsate.Keyboard or None
        The instance of the condynsate.Keyboard class used by this project.
        None if this project uses no keyboard.
    bodies : List of condynsate.simulator.objects.Body
        All bodies loaded into the project via the load_urdf fnc.
    simtime : float
        The current simulation time in seconds.

    """
    def __init__(self, **kwargs):
        # Asynch listen for script exit
        signal.signal(signal.SIGTERM, self._sig_handler)
        signal.signal(signal.SIGINT, self._sig_handler)

        # Build the simulator, visualizer, animator, and keyboard
        gravity = kwargs.get('simulator_gravity', (0.0, 0.0, -9.81))
        dt = kwargs.get('simulator_dt', 0.01)
        self._simulator = Simulator(gravity=gravity, dt=dt)
        self._visualizer = None
        self._animator = None
        self._keyboard = None
        if kwargs.get('visualizer', True):
            frame_rate = kwargs.get('visualizer_frame_rate', 60.0)
            record = kwargs.get('visualizer_record', False)
            self._visualizer = Visualizer(frame_rate=frame_rate, record=record)
        if kwargs.get('animator', False):
            frame_rate = kwargs.get('animator_frame_rate', 15.0)
            record = kwargs.get('animator_record', False)
            self._animator = Animator(frame_rate=frame_rate, record=record)
        if kwargs.get('keyboard', False):
            self._keyboard = Keyboard()

    def __del__(self):
        """
        Deconstructor method.

        """
        self.terminate()

    def _sig_handler(self, sig, frame):
        """
        Handles script termination events so the simulator, visualizer,
        animator, and keyboard exit gracefully.

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
        m = "Interrupt or termination signal detected."
        warn(m, UserWarning)
        self.terminate()

    @property
    def simulator(self):
        """
        The instance of the condynsate.Simulator class used by this project.
        """
        return self._simulator

    @property
    def visualizer(self):
        """
        The instance of the condynsate.Visualizer class used by this project.
        None if this project uses no visualizer.
        """
        return self._visualizer

    @property
    def animator(self):
        """
        The instance of the condynsate.Animator class used by this project.
        None if this project uses no animator.
        """
        return self._animator

    @property
    def keyboard(self):
        """
        The instance of the condynsate.Keyboard class used by this project.
        None if this project uses no keyboard.
        """
        return self._keyboard

    @property
    def bodies(self):
        """
        All bodies loaded into the project via the load_urdf fnc in the
        form of a list of condynsate.simulator.objects.Body instances.
         """
        return self._simulator.bodies

    @property
    def simtime(self):
        """ The current simulation time in seconds as a float. """
        return self._simulator.time

    def load_urdf(self, path, **kwargs):
        """
        Loads a body defined by a .URDF file (https://wiki.ros.org/urdf) into
        the project.

        Parameters
        ----------
        path : string
            The path pointing to the .URDF file that defines the body.
        **kwargs

        Keyword Args
        ------------
        fixed : boolean, optional
            A flag that indicates if the body is fixed (has 0 DoF) or free
            (has 6 DoF).

        Returns
        -------
        body : condynsate.core.objects.Body
            The body added to the simulation. This retured object facilitates
            user interaction with the body and its joints and links.

        """
        body = self._simulator.load_urdf(path, **kwargs)
        self.refresh_visualizer()
        return body

    def reset(self):
        """
        Resets the simulator, visualizer (if there is one), and animator
        (if there is one). If the animator is not already running, starts the
        animator then resets it

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        ret_code = self._simulator.reset()
        if not self._visualizer is None:
            self._visualizer.reset()
        if not self._animator is None:
            if not self._animator.is_running:
                ret_code += self._animator.start()
            else:
                ret_code += self._animator.reset()
        time.sleep(0.1)
        ret_code += self.refresh_visualizer()
        ret_code += self.refresh_animator()
        return max(-1, ret_code)

    def step(self, real_time=True, stable_step=True):
        """
        Takes a single simulation step and updates the visualizer to reflect
        the new simulator state.

        Parameters
        ----------
        real_time : bool, optional
            A boolean flag that indicates whether the step is to be taken in
            real time (True) or as fast as possible (False).
            The default is True.
        stable_step : bool, optional
            Boolean flag that indicates the type of real time stepping that
            should be used. The default is False.

            When real_time is False, this flag is ignored.

            When real_time is True and stable_step is False, the time of the
            first step() call since instantiation or reset is noted. Then, at
            every subsequent step() call, the function will sleep until it has
            been exactly dt*(n-1) seconds since the noted epoch. dt is the
            simulator time step and n is the number of times step() has been
            called since instantiation or reset. This guarantees a more
            accurate total run time, but less stable frame rates, especially
            if, at any point while running a simulation, there is a long pause
            between step() calls.

            When real_time and stable_step are True, the function will sleep
            until the duration since the last time step() was called is equal
            to the time step of the simulation. This guarantees a more stable
            frame rate, but less accurate total run time.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if self._simulator.step(real_time=real_time,
                                stable_step=stable_step) != 0:
            return -1
        self.refresh_visualizer()
        return 0

    def refresh_visualizer(self):
        """
        Refreshes the visualizer to synchronize it to the current simulator
        state. This is automatically called by load_urdf, reset, step,
        await_keypress, and await_anykeys. Therefore, its use cases are limited
        to when bodies are modified outside of the main simulation loop.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if self._visualizer is None:
            # Even if there is no visualizer, we need to make sure
            # to clear the visual_data buffer, otherwise it will
            # grow indefinitely
            for body in self.bodies:
                body.clear_visual_buffer()
            return -1
        for body in self.bodies:
            for d in body.visual_data:
                self._visualizer.add_object(**d)
                self._visualizer.set_transform(**d)
                self._visualizer.set_material(**d)
        return 0

    def refresh_animator(self):
        """
        Refreshes the animator GUI to keep it responsive. This is automatically
        called by step, await_keypress, and await_anykeys so the
        animator will remain responsive during those calls.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if self._animator is None:
            return -1
        return self._animator.refresh()

    def await_keypress(self, key_str, timeout=None):
        """
        Blocks progress until the user presses a desired key.

        Parameters
        ----------
        key_str : string
            The key to be detected. May be a lowercase letter, a digit, a
            special character, or a punctuation mark that does not use shift on
            a QWERTY keyboard.
            The special keys are as follows:
            "esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
            "f11", "f12", "print_screen", "scroll_lock", "pause", "backspace",
            "insert", "home", "page_up", "num_lock", "tab", "delete", "end",
            "page_down", "caps_lock", "enter", "up", "left", "down", "right"
            The punctuation marks that do not use shift on a QWERTY keyboard
            are as follows:
            "`", "-", "=", "[", "]", "\", ";", "'", ",", ".", "/"
            The following modifiers can also be used:
            "shift", "alt", "ctrl", "cmd".
            Modifiers are added with the following format:
            "shift+a", "ctrl+a", "alt+a", "shift+ctrl+alt+a", etc.
        timeout : float, optional
            The timeout time at which the function will stop blocking and exit.
            The default is None. When None, this function will not timeout.

        Raises
        ------
        AttributeError:
            If the project has no keyboard.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if self._keyboard is None:
            raise AttributeError('Cannot await_keypress, no keyboard.')
        print(f"Press {key_str} to continue.")
        start = time.time()
        while True:
            if not timeout is None and time.time()-start > timeout:
                print("Timed out.")
                return -1
            if self._keyboard.is_pressed(key_str):
                print("Continuing.")
                return 0
            self.refresh_visualizer()
            self.refresh_animator()
            time.sleep(0.01)

    def await_anykeys(self, timeout=None):
        """
        Blocks progress until the user presses any key or any set of keys.
        Returns which key or keys were pressed.

        Parameters
        ----------
        timeout : float, optional
            The timeout time at which the function will stop blocking and exit.
            The default is None. When None, this function will not timeout.

        Raises
        ------
        AttributeError:
            If the project has no keyboard.

        Returns
        -------
        keys : list
            A list of which keys were pressed.

        """
        if self._keyboard is None:
            raise AttributeError('Cannot await_anykey, no keyboard.')
        start = time.time()
        while True:
            if not timeout is None and time.time()-start > timeout:
                print("Timed out.")
                return []
            pressed = self._keyboard.get_pressed()
            if len(pressed) > 0:
                return pressed
            self.refresh_visualizer()
            self.refresh_animator()
            time.sleep(0.01)

    def terminate(self):
        """
        Gracefully terminates the project and all children threads.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        ret_code = self._simulator.terminate()
        if not self._visualizer is None:
            ret_code += self._visualizer.terminate()
            self._visualizer = None
        if not self._animator is None:
            ret_code += self._animator.terminate()
            self._animator = None
        if not self._keyboard is None:
            ret_code += self._keyboard.terminate()
            self._keyboard = None
        return max(-1, ret_code)
