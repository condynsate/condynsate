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
from warnings import warn
import numpy as np
import pybullet
from pybullet_utils import bullet_client as bc
import condynsate.core.transforms as t

###############################################################################
#SIMULATOR OBJECT CLASS
###############################################################################
@dataclass
class Base_State():
    """
    Stores state information about the base of an object.

    Parameters
    ----------
    **kwargs :
        State information with the following acceptable keys
        position : 3 tuple of floats, optional
            The XYZ position in world coordinates.
            The default is (0., 0., 0.)
        orientation : 4 tuple of floats, optional
            The wxyz quaternion representation of the orientation in world
            coordinates. The default is (1., 0., 0., 0.)
        velocity : 3 tuple of floats, optional
            The XYZ velocity in either world or body coordinates. Body
            coordinates are defined based on objects orientation.
            The default is (0., 0., 0.)
        omega : 3 tuple of floats, optional
            The XYZ angular velocity in either world or body coordinates.
            Body coordinates are defined based on objects orientation.
            The default is (0., 0., 0.)
        body : bool, optional
            Whether velocity and omega are being set in world or body
            coordinates. The default is False
    """
    _position: tuple = (0.0, 0.0, 0.0)
    _orientation: tuple = (1.0, 0.0, 0.0, 0.0)
    _velocity: tuple = (0.0, 0.0, 0.0)
    _omega: tuple = (0.0, 0.0, 0.0)

    def __init__(self, **kwargs):
        body = kwargs.get('body', False)
        if 'position' in kwargs:
            self.position = kwargs['position']
        if 'orientation' in kwargs:
            self.orientation = kwargs['orientation']
        if 'velocity' in kwargs:
            if body:
                self.set_velocity_in_body(kwargs['velocity'])
            else:
                self.velocity = kwargs['velocity']
        if 'omega' in kwargs:
            if body:
                self.set_omega_in_body(kwargs['omega'])
            else:
                self.omega = kwargs['omega']

    @property
    def position(self):
        """ XYZ position world coordinates """
        return self._position

    @property
    def orientation(self):
        """ WXYZ quaternion orientation in world coordinates """
        return self._orientation

    @property
    def velocity(self):
        """ XYZ velocity in world coordinates """
        return self._velocity

    @property
    def omega(self):
        """ XYZ angular velocity in world coordinates """
        return self._omega

    @position.setter
    def position(self, position):
        try:
            self._position = tuple(float(p) for p in position)
        except (TypeError, ValueError):
            warn(f"Unable to set position to {position}.")

    @orientation.setter
    def orientation(self, orientation):
        try:
            q_0 = np.array([q for i,q in enumerate(orientation) if i < 4])
            self._orientation = tuple((q_0 / np.linalg.norm(q_0)).tolist())
        except (TypeError, ValueError):
            warn(f"Unable to set orientation to {orientation}.")
        except ZeroDivisionError:
            warn(f"Unable to set orientation to {orientation}, magnitude 0.")

    @velocity.setter
    def velocity(self, velocity):
        try:
            self._velocity = tuple(float(v) for v in velocity)
        except (TypeError, ValueError):
            warn(f"Unable to set velocity to {velocity}.")

    @omega.setter
    def omega(self, omega):
        try:
            self._omega = tuple(float(w) for w in omega)
        except (TypeError, ValueError):
            warn(f"Unable to set omega to {omega}.")

    def get_velocity_in_body(self):
        """
        Gets the velocity in body coordinates.

        Returns
        -------
        velocity_in_body : 3 tuple of floats
            The velocity in body coordinates as defined by the orientation.

        """
        Rbw = t.Rbw_from_wxyz(self.orientation)
        Rwb = t.Rab_to_Rba(Rbw)
        return tuple(t.va_to_vb(Rwb, self.velocity).tolist())

    def get_omega_in_body(self):
        """
        Gets the angular velocity in body coordinates.

        Returns
        -------
        omega_in_body : 3 tuple of floats
            The angular velocity in body coordinates as defined by the
            orientation.

        """
        Rbw = t.Rbw_from_wxyz(self.orientation)
        Rwb = t.Rab_to_Rba(Rbw)
        return tuple(t.va_to_vb(Rwb, self.omega).tolist())

    def set_velocity_in_body(self, velocity):
        """
        Sets the velocity where the passed velocity argumnet is defined in
        body coordinates as defined by the orientation.

        Parameters
        ----------
        velocity : 3 tuple of floats
            The velocity to set in body coordinates.

        Returns
        -------
        None.

        """
        try:
            Rbw = t.Rbw_from_wxyz(self.orientation)
            self._velocity = tuple(t.va_to_vb(Rbw, velocity).tolist())
        except (TypeError, ValueError):
            warn(f"Unable to set velocity to {velocity}.")

    def set_omega_in_body(self, omega):
        """
        Sets the velocity where the passed velocity argumnet is defined in
        body coordinates as defined by the orientation.

        Parameters
        ----------
        velocity : 3 tuple of floats
            The velocity to set in body coordinates.

        Returns
        -------
        None.

        """
        try:
            Rbw = t.Rbw_from_wxyz(self.orientation)
            self._omega = tuple(t.va_to_vb(Rbw, omega).tolist())
        except (TypeError, ValueError):
            warn(f"Unable to set omega to {omega}.")

class Sim_Obj():
    """
    """
    def __init__(self, client, path, **kwargs):
        self._client = client
        self._init_base_state = Base_State(**kwargs)
        self._id = self._load_urdf(path, **kwargs)
        self._links, self._joints = self._build_links_and_joints()

    def _load_urdf(self, urdf_path, **kwargs):
        # Use implicit cylinder for collision and physics calculation
        # Specifies to the engine to use the inertia from the urdf file
        f1 = self._client.URDF_USE_IMPLICIT_CYLINDER
        f2 = self._client.URDF_USE_INERTIA_FROM_FILE

        # Load the URDF
        flags = f1 | f2
        basePosition = self._init_base_state.position
        baseOrientation = self._init_base_state.orientation
        useFixedBase = kwargs.get('fixed', False)
        urdf_id = self._client.loadURDF(urdf_path,
                                        flags=flags,
                                        basePosition=basePosition,
                                        baseOrientation=baseOrientation,
                                        useFixedBase=useFixedBase)

        # Set the base velocity of the URDF
        linearVelocity = self._init_base_state.velocity
        angularVelocity = self._init_base_state.omega
        self._client.resetBaseVelocity(objectUniqueId=urdf_id,
                                       linearVelocity=linearVelocity,
                                       angularVelocity=angularVelocity)

        return urdf_id

    def _build_links_and_joints(self):
        for joint_id in range(self._client.getNumJoints(self._id)):
            info = self._client.getJointInfo(self._id, joint_id)

    def set_base_state(self, **kwargs):
        """
        Sets the base state of the object.

        Parameters
        ----------
        **kwargs
            State information with the following acceptable keys
            position : 3 tuple of floats, optional
                The XYZ position in world coordinates.
                The default is (0., 0., 0.)
            yaw : float, optional
                The (Tait–Bryan) yaw angle of the object in radians.
            pitch : float, optional
                The (Tait–Bryan) pitch angle of the object in radians.
            roll : float, optional
                The (Tait–Bryan) roll angle of the object in radians.
            velocity : 3 tuple of floats, optional
                The XYZ velocity in either world or body coordinates. Body
                coordinates are defined based on object's orientation.
                The default is (0., 0., 0.)
            omega : 3 tuple of floats, optional
                The XYZ angular velocity in either world or body coordinates.
                Body coordinates are defined based on object's orientation.
                The default is (0., 0., 0.)
            body : bool, optional
                Whether velocity and omega are being set in world or body
                coordinates. The default is False

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Update the stored base state
        if 'position' in kwargs:
            self._base_state.position = kwargs['position']
        if 'yaw' in kwargs or 'pitch' in kwargs or 'roll' in kwargs:
            y0, p0, r0 = t.euler_from_wxyz(self._base_state.orientation)
            yaw = kwargs.get('yaw', y0)
            pitch = kwargs.get('pitch', p0)
            roll = kwargs.get('roll', r0)
            orientation = t.wxyz_from_euler(yaw, pitch, roll)
            self._base_state.orientation = orientation
        if 'velocity' in kwargs:
            if kwargs.get('body', False):
                self._base_state.set_velocity_in_body(kwargs['velocity'])
            else:
                self._base_state.velocity = kwargs['velocity']
        if 'omega' in kwargs:
            if kwargs.get('body', False):
                self._base_state.set_omega_in_body(kwargs['omega'])
            else:
                self._base_state.omega = kwargs['omega']

        # Send the updated state to the physics client
        posObj = self._base_state.position
        ornObj = self._base_state.orientation
        self._client.resetBasePositionAndOrientation(bodyUniqueId=self._id,
                                                     posObj=posObj,
                                                     ornObj=ornObj)
        linearVelocity = self._base_state.velocity
        angularVelocity = self._base_state.omega
        self._client.resetBaseVelocity(objectUniqueId=self._id,
                                       linearVelocity=linearVelocity,
                                       angularVelocity=angularVelocity)
        return 0

    def get_base_state(self, body=False):
        """
        Gets the base state of the object.

        Parameters
        ----------
        body : bool, optional
            A boolean flag that determines if the fetch velocity and angular
            velocity states are in body coordinate (True) or world coordinates
            (False). The default is False.

        Returns
        -------
        base_state : dict
            A Dictionary containing the state information. The keys are
            position, yaw, pitch, roll, velocity, and omega.

        """
        position = self._base_state.position
        yaw, pitch, roll = t.euler_from_wxyz(self._base_state.orientation)
        if body:
            velocity = self._base_state.get_velocity_in_body()
            omega = self._base_state.get_omega_in_body()
        else:
            velocity = self._base_state.velocity
            omega = self._base_state.omega
        base_state = {'position' : position,
                      'yaw' : yaw,
                      'pitch' : pitch,
                      'roll' : roll,
                      'velocity' : velocity,
                      'omega' : omega,}
        return base_state

###############################################################################
#JOINT CLASS
###############################################################################
@dataclass
class Joint_State():
    """
    Stores state information about the base of an object.

    Parameters
    ----------
    **kwargs :
        State information with the following acceptable keys
        angle : float, optional
            The angle of the joint about the joint axis. The default is 0.
        omega : float, optional
            The angular velocity of the joint about the joint axis.
            The default is 0.
    """
    _angle: float = 0.0
    _omega: float = 0.0

    def __init__(self, **kwargs):
        if 'angle' in kwargs:
            self.angle = kwargs['angle']
        if 'omega' in kwargs:
            self.omega = kwargs['omega']

    @property
    def angle(self):
        """ The joint angle about the joint axis """
        return self._angle

    @property
    def omega(self):
        """ The joint angular velocity about the joint axis """
        return self._omega

    @angle.setter
    def angle(self, angle):
        try:
            self._angle = float(angle)
        except (TypeError, ValueError):
            warn(f"Unable to set angle to {angle}.")

    @omega.setter
    def omega(self, omega):
        try:
            self._omega = float(omega)
        except (TypeError, ValueError):
            warn(f"Unable to set omega to {omega}.")

class Joint:
    """
    """
    def __init__(self, client):
        self._client = client

    def set_dynamics(self, **kwargs):
        args = {}
        if 'joint_resistance' in kwargs:
            args['jointDamping'] = kwargs['joint_resistance']
        if 'max_omega' in kwargs:
            args['maxJointVelocity'] = kwargs['max_omega']

        if 'mass' in kwargs:
            args['mass'] = kwargs['mass']
        if 'lateral_contact_friction' in kwargs:
            args['lateralFriction'] = kwargs['lateral_contact_friction']
        if 'spinning_contact_friction' in kwargs:
            args['spinningFriction'] = kwargs['spinning_contact_friction']
        if 'rolling_contact_friction' in kwargs:
            args['rollingFriction'] = kwargs['rolling_contact_friction']
        if 'bounciness' in kwargs:
            args['restitution'] = kwargs['bounciness']
        if 'linear_air_resistance' in kwargs:
            args['linearDamping'] = kwargs['linear_air_resistance']
        if 'angular_air_resistance' in kwargs:
            args['angularDamping'] = kwargs['angular_air_resistance']
        if 'contact_damping' in kwargs:
            args['contactDamping'] = kwargs['contact_damping']
        if 'contact_stiffness' in kwargs:
            args['contactStiffness'] = kwargs['contact_stiffness']

        urdf_id = 0
        link_id = 0
        self._client.changeDynamics(urdf_id, link_id, **args)

        return 0

        # mode = self.engine.VELOCITY_CONTROL
        # self.engine.setJointMotorControlArray(urdf_id,
        #                                       joint_id,
        #                                       mode,
        #                                       forces=[0])

        # self.engine.enableJointForceTorqueSensor(urdf_id,
        #                                          joint_id,
        #                                          enable_sensor)

###############################################################################
#SIMULATOR CLASS
###############################################################################
class Simulator():
    """
    """
    def __init__(self, gravity=(0.0, 0.0, -9.81), dt=0.01):
        # Start engine and client in direct mode (no visualization)
        self._client = bc.BulletClient(connection_mode=pybullet.GUI)
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
        # Convert the euler angles to quaternion
        yaw = kwargs.get('yaw', 0.0)
        pitch = kwargs.get('pitch', 0.0)
        roll = kwargs.get('roll', 0.0)
        kwargs['orientation'] = t.wxyz_from_euler(yaw, pitch, roll)

        # Build the object and return it
        obj = Sim_Obj(self._client, path, **kwargs)
        return obj

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
