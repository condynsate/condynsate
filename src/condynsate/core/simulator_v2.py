# -*- coding: utf-8 -*-
"""
This module provides the simulator class which is used to run physics
simulations using the PyBullet package.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
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
        self.set_gravity(gravity)
        client_params = {
                        'fixedTimeStep' : dt,
                        'numSubSteps' : 4,
                        'restitutionVelocityThreshold' : 0.05,
                        'enableFileCaching' : 0,
                         }
        self._client.setPhysicsEngineParameter(**client_params)

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
        return Body(self._client, path, **kwargs)

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
