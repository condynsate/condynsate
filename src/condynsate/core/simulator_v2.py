# -*- coding: utf-8 -*-
"""
This module provides the simulator class which is used to run physics
simulations using the PyBullet package.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import time
import pybullet
from pybullet_utils import bullet_client as bc
from condynsate.core.objects import Body

###############################################################################
#SIMULATOR CLASS
###############################################################################
class Simulator():
    """
    """
    def __init__(self, gravity=(0.0, 0.0, -9.81), dt=0.01):
        # Start engine and client in direct mode (no visualization)
        self._client = bc.BulletClient(connection_mode=pybullet.GUI)
        self.dt = dt
        client_params = {
                        'fixedTimeStep' : self.dt,
                        'numSubSteps' : 4,
                        'restitutionVelocityThreshold' : 0.05,
                        'enableFileCaching' : 0,
                         }
        self._client.setPhysicsEngineParameter(**client_params)
        self.set_gravity(gravity)
        self._bodies = []
        self._prev_step = float('-inf')
        self.time = 0.0

    def __del__(self):
        """
        Deconstructor method.

        """
        self.terminate()

    def set_gravity(self, gravity):
        """
        Sets the acceleration due to gravity

        Parameters
        ----------
        gravity : array-like, shape (3,)
            The graavity vector in world coordinates with metric units.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        self._client.setGravity(gravity[0], gravity[1], gravity[2])
        return 0

    def load_urdf(self, path, **kwargs):
        """
        Loads a body defined by a .URDF file (https://wiki.ros.org/urdf) into
        the simulator.

        Parameters
        ----------
        path : string
            The path pointing to the .URDF file that defines the body.
        **kwargs
            Additional arguments for the body. Valid keys are
            fixed : boolean, optional
                A flag that indicates if the body is fixed (has 0 DoF) or free
                (has 6 DoF).

        Returns
        -------
        body : condynsate.core.objects.Body
            The body added to the simulation. This retured object facilitates
            user interaction with the body and its joints and links.

        """
        self._bodies.append(Body(self._client, path, **kwargs))
        return self._bodies[-1]

    def step(self, real_time=True):
        """
        Takes a single simulation step.

        Parameters
        ----------
        real_time : bool, optional
            A boolean flag that indicates whether the step is to be taken in
            real time (True) or as fast as possible (False). When True, the
            function will sleep until the duration since the last time step()
            was called is exactly equal to the time step of the simulation.
            The default is True.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        sleep_duration = self.dt - (time.monotonic() - self._prev_step)
        try:
            time.sleep(sleep_duration)
        except (OverflowError, ValueError):
            pass
        self._client.stepSimulation()
        self.time += self.dt
        if real_time:
            self._prev_step = time.monotonic()
        return 0

    def reset(self):
        """
        Resets the simulation and all bodies loaded in the simulation to the
        initial state.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        self._prev_step = float('-inf')
        self.time = 0.0
        for body in self._bodies:
            body.reset()
        return 0

    def terminate(self):
        """
        Terminates the simulator.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if self._client.isConnected():
            self._client.disconnect()
            return 0
        return -1
