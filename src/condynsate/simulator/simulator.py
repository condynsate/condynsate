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
from warnings import warn
import pybullet
from pybullet_utils import bullet_client as bc
from condynsate.simulator.objects import Body

###############################################################################
#SIMULATOR CLASS
###############################################################################
class Simulator():
    """
    The Simulator class handles running the physics simulation.

    Parameters
    ----------
    **kwargs
        The optional arguments provided to build the simulator. Valid keys
        include:
            gravity : 3 tuple of floats, optional
                The gravity vector used in the simulation. The default value is
                (0.0, 0.0, -9.81).
            dt : float, optional
                The finite time step size used by the simulator. If set too
                small, can result in visualizer, simulator desynch. Too small
                is determined by the number of total links in the simulation.
                The default value is 0.01.

    """
    def __init__(self, **kwargs):
        # Start engine and client in direct mode (no visualization)
        self._client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.dt = kwargs.get('dt', 0.01)
        client_params = {
                        'fixedTimeStep' : self.dt,
                        'numSubSteps' : 4,
                        'restitutionVelocityThreshold' : 0.05,
                        'enableFileCaching' : 0,
                         }
        self._client.setPhysicsEngineParameter(**client_params)
        self.set_gravity(kwargs.get('gravity', (0.0, 0.0, -9.81)))
        self.bodies = []
        self._prev_end_t = float('-inf')
        self._prev_dt_act = float('inf')
        self._epoch = float('-inf')
        self.time = 0.0
        self.step_num = 0

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
        self.bodies.append(Body(self._client, path, **kwargs))
        return self.bodies[-1]

    def step(self, real_time=True, stable_step=False):
        """
        Takes a single simulation step.

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
        start_t = time.monotonic()
        if real_time:
            if self._epoch==float('-inf'):
                self._epoch = start_t
            if stable_step:
                # Attempting to correct for the amount of time stepping and
                # rendering take by recording the actual dts and correcting
                # from mean
                error = self._prev_dt_act - self.dt
                duration = self.dt - error + self._prev_end_t - start_t
            else:
                duration = self.step_num*self.dt + self._epoch - start_t
            try:
                time.sleep(duration)
            except (OverflowError, ValueError):
                pass

        # Attempt a step (might fail if the server is disconnected)
        try:
            self._client.stepSimulation()
        except pybullet.error:
            warn('Cannot complete action because simulator is stopped.')
            return -1

        self.time += self.dt
        self.step_num += 1
        end_t = time.monotonic()
        self._prev_dt_act = end_t - self._prev_end_t
        self._prev_end_t = end_t
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
        self._prev_end_t = float('-inf')
        self._prev_dt_act = float('inf')
        self._epoch = float('-inf')
        self.time = 0.0
        self.step_num = 0
        for body in self.bodies:
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
