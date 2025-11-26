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
    """
    def __init__(self, **kwargs):
        # Asynch listen for script exit
        signal.signal(signal.SIGTERM, self._sig_handler)
        signal.signal(signal.SIGINT, self._sig_handler)

        # Build the simulator, visualizer, animator, and keyboard
        gravity = kwargs.get('simulator_gravity', (0.0, 0.0, -9.81))
        dt = kwargs.get('simulator_dt', 0.01)
        self.simulator = Simulator(gravity=gravity, dt=dt)
        self.visualizer = None
        self.animator = None
        self.keyboard = None
        if kwargs.get('visualizer', False):
            record = kwargs.get('visualizer_record', False)
            def_fr = 30.0 if record else 60.0
            frame_rate = kwargs.get('visualizer_frame_rate', def_fr)
            self.visualizer = Visualizer(frame_rate=frame_rate, record=record)
        if kwargs.get('animator', False):
            frame_rate = kwargs.get('animator_frame_rate', 15.0)
            record = kwargs.get('animator_record', False)
            self.animator = Animator(frame_rate=frame_rate, record=record)
        if kwargs.get('keyboard', False):
            self.keyboard = Keyboard()

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
    def bodies(self):
        return self.simulator.bodies

    @property
    def simtime(self):
        return self.simulator.time

    def load_urdf(self, path, **kwargs):
        body = self.simulator.load_urdf(path, **kwargs)
        self.refresh_visualizer()
        return body

    def reset(self):
        ret_code = self.simulator.reset()
        if not self.visualizer is None:
            self.visualizer.reset()
        ret_code += self.refresh_visualizer()
        if not self.animator is None:
            if not self.animator.is_running():
                ret_code += self.animator.start()
            else:
                ret_code += self.animator.reset()
        return max(-1, ret_code)

    def step(self, real_time=True, stable_step=True):
        if self.simulator.step(real_time=real_time,
                               stable_step=stable_step) != 0:
            return -1
        self.refresh_visualizer()
        return 0

    def refresh_visualizer(self):
        if self.visualizer is None:
            # Even if there is no visualizer, we need to make sure
            # to clear the visual_data buffer, otherwise it will
            # grow indefinitely
            for body in self.bodies:
                body.clear_visual_buffer()
            return -1
        for body in self.bodies:
            for d in body.visual_data:
                self.visualizer.add_object(**d)
                self.visualizer.set_transform(**d)
                self.visualizer.set_material(**d)
        return 0

    def refresh_animator(self):
        if self.animator is None:
            return -1
        return self.animator.refresh()

    def await_keypress(self, key_str, timeout=None):
        if self.keyboard is None:
            raise(AttributeError('Cannot await_keypress, no keyboard.'))
        print(f"Press {key_str} to continue.")
        start = time.time()
        while True:
            if not timeout is None and time.time()-start > timeout:
                print("Timed out.")
                return -1
            if self.keyboard.is_pressed(key_str):
                print("Continuing.")
                return 0
            self.refresh_visualizer()
            self.refresh_animator()
            time.sleep(0.01)

    def await_anykeys(self, timeout=None):
        if self.keyboard is None:
            raise(AttributeError('Cannot await_anykey, no keyboard.'))
        start = time.time()
        while True:
            if not timeout is None and time.time()-start > timeout:
                print("Timed out.")
                return []
            pressed = self.keyboard.get_pressed()
            if len(pressed) > 0:
                return pressed
            self.refresh_visualizer()
            self.refresh_animator()
            time.sleep(0.01)

    def terminate(self):
        ret_code = self.simulator.terminate()
        if not self.visualizer is None:
            ret_code += self.visualizer.terminate()
            self.visualizer = None
        if not self.animator is None:
            ret_code += self.animator.terminate()
            self.animator = None
        if not self.keyboard is None:
            ret_code += self.keyboard.terminate()
            self.keyboard = None
        return max(-1, ret_code)
