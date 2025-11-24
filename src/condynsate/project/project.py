# -*- coding: utf-8 -*-
"""
This module provides the Project class which is the primary interface with
which users interact when using condynsate.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
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
            frame_rate = kwargs.get('visualizer_frame_rate', 45.0)
            record = kwargs.get('visualizer_record', False)
            self.visualizer = Visualizer(frame_rate=frame_rate, record=record)
        if kwargs.get('animator', False):
            frame_rate = kwargs.get('animator_frame_rate', 5.0)
            record = kwargs.get('animator_record', False)
            self.animator = Animator(frame_rate=frame_rate, record=record)
        if kwargs.get('keyboard', False):
            self.keyboard = Keyboard()

    def __del__(self):
        """
        Deconstructor method.

        """
        pass

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
        warn(m)
        self.terminate()

    @property
    def bodies(self):
        return self.simulator.bodies

    @property
    def time(self):
        return self.simulator.time

    def load_urdf(self, path, **kwargs):
        body = self.simulator.load_urdf(path, **kwargs)
        self.refresh_visualizer()
        return body

    def reset(self):
        ret_code = self.simulator.reset()
        ret_code += self.refresh_visualizer()
        if not self.animator is None:
            if not self.animator.is_running():
                ret_code += self.animator.start()
            else:
                ret_code += self.animator.reset_all()
        return max(-1, ret_code)

    def step(self, real_time=True):
        if self.simulator.step(real_time=real_time) != 0:
            return -1
        self.refresh_visualizer()
        return 0

    def refresh_visualizer(self):
        if self.visualizer is None:
            return -1
        for body in self.bodies:
            for d in body.visual_data:
                self.visualizer.add_object(**d)
                self.visualizer.set_transform(**d)
                self.visualizer.set_material(**d)
        return 0

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
