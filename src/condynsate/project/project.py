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

        # Track all bodies loaded in project
        self.bodies = []

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

    def load_urdf(self, path, **kwargs):
        self.bodies.append(self.simulator.load_urdf(path, **kwargs))
        if not self.visualizer is None:
            for d in self.bodies[-1].visual_data:
                self.visualizer.add_object(**d)
        return self.bodies[-1]

    def reset(self):
        # Reset the simulator
        self.simulator.reset()

        # Redraw all bodies to the visualizer
        self.refresh_visualizer()

        # Either start (if not started) or reset the animator
        if not self.animator is None:
            if not self.animator.is_running():
                self.animator.start()
            else:
                self.animator.reset_all()

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
                self.visualizer.set_transform(**d)
                self.visualizer.set_material(**d)
        return 0

    def terminate(self):
        sim_code = self.simulator.terminate()
        vis_code = 0
        ani_code = 0
        key_code = 0
        if not self.visualizer is None:
            vis_code = self.visualizer.terminate()
            self.visualizer = None
        if not self.animator is None:
            ani_code = self.animator.terminate()
            self.animator = None
        if not self.keyboard is None:
            key_code = self.keyboard.terminate()
            self.keyboard = None
        return max(sim_code, vis_code, ani_code, key_code)
