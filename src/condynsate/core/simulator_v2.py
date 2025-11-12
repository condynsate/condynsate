# -*- coding: utf-8 -*-
"""
This module provides the simulator class which is used to run physics
simulations using the PyBullet package.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
from dataclasses import dataclass
import pybullet
from pybullet_utils import bullet_client as bc

###############################################################################
#SIMULATOR OBJECT STATE DATACLASS
###############################################################################
@dataclass
class _State():
    """
    Stores state information.
    """
    position: tuple = (0.0, 0.0, 0.0)
    velocity: tuple = (0.0, 0.0, 0.0)
    orientation: tuple = (1.0, 0.0, 0.0, 0.0)
    omega: tuple = (0.0, 0.0, 0.0)
    
    def set_position(self, position):
        self.position = tuple(p for i,p in enumerate(position) if i < 3)
        
    def set_position(self, position):
        self.position = tuple(p for i,p in enumerate(position) if i < 3)
        
    def set_position(self, position):
        self.position = tuple(p for i,p in enumerate(position) if i < 3)
            
        def set_position(self, position):
            self.position = tuple(p for i,p in enumerate(position) if i < 3)

@dataclass 
class Sim_Obj_State(_State):
    """
    Stores current state information about simulator object.
    """

@dataclass
class Sim_Obj_Init_State(_State):
    """
    Stores initial state information about simulator object.
    """
    
###############################################################################
#SIMULATOR OBJECT CLASS
###############################################################################
class Sim_Obj():
    
    def __init__(self):
        pass

###############################################################################
#SIMULATOR CLASS
###############################################################################
class Simulator():

    def __init__(self, gravity=(0.0, 0.0, -9.81), dt=0.01):
        # Start engine and client in direct mode (no visualization)
        self._client = bc.BulletClient(connection_mode=pybullet.DIRECT)
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
