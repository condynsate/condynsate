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
import numpy as np
import pybullet
from pybullet_utils import bullet_client as bc
import condynsate.core.transforms as t

###############################################################################
#SIMULATOR OBJECT CLASS
###############################################################################
@dataclass(frozen=True)
class BaseState():
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
    position: tuple
    orientation: tuple
    ypr: tuple
    velocity: tuple
    omega: tuple
    velocity_in_body: tuple
    omega_in_body: tuple

    def __init__(self, **kwargs):
        # Read kwargs
        body = kwargs.get('body', False)
        position = kwargs.get('position', (0.0, 0.0, 0.0))
        orientation = kwargs.get('orientation', (1.0, 0.0, 0.0, 0.0))
        velocity = kwargs.get('velocity', (0.0, 0.0, 0.0))
        omega = kwargs.get('omega', (0.0, 0.0, 0.0))

        # Set states
        self._set_position(position)
        self._set_orientation(orientation)
        self._set_ypr()
        if body:
            self._set_velocity_in_body(velocity)
            self._set_omega_in_body(omega)
        else:
            self._set_velocity(velocity)
            self._set_omega(omega)
        self._set_body_vels()

    def _set_position(self, position):
        p0 = tuple(float(p) for p in position)
        super().__setattr__('position', p0)

    def _set_orientation(self, orientation):
        q0 = np.array([q for i,q in enumerate(orientation) if i < 4])
        q0 = tuple((q0 / np.linalg.norm(q0)).tolist())
        super().__setattr__('orientation', q0)

    def _set_velocity(self, velocity):
        v0 = tuple(float(v) for v in velocity)
        super().__setattr__('velocity', v0)

    def _set_velocity_in_body(self, velocity):
        Rbw = t.Rbw_from_wxyz(self.orientation)
        v0 = tuple(t.va_to_vb(Rbw, velocity).tolist())
        super().__setattr__('velocity', v0)

    def _set_omega(self, omega):
        o0 = tuple(float(w) for w in omega)
        super().__setattr__('omega', o0)

    def _set_omega_in_body(self, omega):
        Rbw = t.Rbw_from_wxyz(self.orientation)
        o0 = tuple(t.va_to_vb(Rbw, omega).tolist())
        super().__setattr__('omega', o0)

    def _set_ypr(self):
        ypr = tuple(float(e) for e in t.euler_from_wxyz(self.orientation))
        super().__setattr__('ypr', ypr)

    def _set_body_vels(self):
        Rbw = t.Rbw_from_wxyz(self.orientation)
        Rwb = t.Rab_to_Rba(Rbw)
        vb = tuple(t.va_to_vb(Rwb, self.velocity).tolist())
        super().__setattr__('velocity_in_body', vb)
        ob = tuple(t.va_to_vb(Rwb, self.omega).tolist())
        super().__setattr__('omega_in_body', ob)

class Body():
    """
    """
    def __init__(self, client, path, **kwargs):
        self._client = client
        self._init_base_state = BaseState()
        self._id = self._load_urdf(path, **kwargs)
        self.links, self.joints = self._build_links_and_joints()

    def _load_urdf(self, urdf_path, **kwargs):
        # Use implicit cylinder for collision and physics calculation
        # Specifies to the engine to use the inertia from the urdf file
        f1 = self._client.URDF_USE_IMPLICIT_CYLINDER
        f2 = self._client.URDF_USE_INERTIA_FROM_FILE

        # Get the default initial state
        basePosition = self._init_base_state.position
        baseOrientation = self._init_base_state.orientation
        baseOrientation = t.xyzw_from_wxyz(baseOrientation)
        linearVelocity = self._init_base_state.velocity
        angularVelocity = self._init_base_state.omega

        # Load the URDF with default initial conditions
        flags = f1 | f2
        useFixedBase = kwargs.get('fixed', False)
        urdf_id = self._client.loadURDF(urdf_path,
                                        flags=flags,
                                        basePosition=basePosition,
                                        baseOrientation=baseOrientation,
                                        useFixedBase=useFixedBase)
        self._client.resetBaseVelocity(objectUniqueId=urdf_id,
                                       linearVelocity=linearVelocity,
                                       angularVelocity=angularVelocity)

        return urdf_id

    def _build_links_and_joints(self):
        base_link, body = self._client.getBodyInfo(self._id)
        base_link = base_link.decode('UTF-8')
        body = body.decode('UTF-8')
        links = {base_link : -1}
        joints = {}
        for joint_id in range(self._client.getNumJoints(self._id)):
            info = self._client.getJointInfo(self._id, joint_id)
            joint_name = info[1].decode('UTF-8')
            child_link = info[12].decode('UTF-8')
            joints[joint_name] = Joint(self, joint_name, joint_id)
            links[child_link] = Link(self, child_link, joint_id)
        return links, joints

    @property
    def initial_base_state(self):
        """ The initial base state of the object. """
        return self._init_base_state

    @property
    def base_state(self):
        """ The base state of the object. """
        # Get the base states
        pos, ornObj = self._client.getBasePositionAndOrientation(self._id)
        ori = t.wxyz_from_xyzw(ornObj)
        vel, omg = self._client.getBaseVelocity(self._id)

        # Compile and return
        base_state = BaseState(position=pos,
                               orientation=ori,
                               velocity=vel,
                               omega=omg)
        return base_state

    def set_initial_base_state(self, **kwargs):
        """
        Sets the initial base state of the object. When the simulation is
        reset, this object will be reset to this base state.

        Parameters
        ----------
        **kwargs
            State information with the following acceptable keys
            position : 3 tuple of floats, optional
                The XYZ position in world coordinates.
                The default is (0., 0., 0.)
            yaw : float, optional
                The (z-y'-x' Tait–Bryan) yaw angle of the object in radians.
            pitch : float, optional
                The (z-y'-x' Tait–Bryan) pitch angle of the object in radians.
            roll : float, optional
                The (z-y'-x' Tait–Bryan) roll angle of the object in radians.
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
        # Convert the euler angles to quaternion
        yaw = kwargs.get('yaw', 0.0)
        pitch = kwargs.get('pitch', 0.0)
        roll = kwargs.get('roll', 0.0)
        kwargs['orientation'] = t.wxyz_from_euler(yaw, pitch, roll)

        # Set the initial base state
        self._init_base_state = BaseState(**kwargs)

        # Set the base state to the initial state
        args = {
                'position' : self._init_base_state.position,
                'velocity' : self._init_base_state.velocity,
                'orientation' : self._init_base_state.orientation,
                'omega' : self._init_base_state.omega,
                'body' : kwargs.get('body', False)
                }
        return self.set_base_state(**args)

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
                The (z-y'-x' Tait–Bryan) yaw angle of the object in radians.
            pitch : float, optional
                The (z-y'-x' Tait–Bryan) pitch angle of the object in radians.
            roll : float, optional
                The (z-y'-x' Tait–Bryan) roll angle of the object in radians.
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
        # Get the current state of the body
        base_state = self.base_state
        ypr0 = base_state.ypr

        # Get the new position, if not defined, default to current position
        posObj = kwargs.get('position', base_state.position)

        # Get the new orientation, if not defined, default to current
        # orientation (Tait-Bryan angle-wise)
        yaw = kwargs.get('yaw', ypr0[0])
        pitch = kwargs.get('pitch', ypr0[1])
        roll = kwargs.get('roll', ypr0[2])
        ornObj = t.xyzw_from_wxyz(t.wxyz_from_euler(yaw, pitch, roll))

        # Velocities in body coords
        if kwargs.get('body', False):
            # Body to world rotation matrix
            Rbw = t.Rbw_from_euler(yaw, pitch, roll)

            # Linear velocity
            vel = kwargs.get('velocity', None)
            if vel is None:
                linearVelocity = base_state.velocity
            else:
                linearVelocity = tuple(t.va_to_vb(Rbw, vel).tolist())

            # Angular velocity
            omg = kwargs.get('omega', None)
            if omg is None:
                angularVelocity = base_state.omega
            else:
                angularVelocity = tuple(t.va_to_vb(Rbw, omg).tolist())

        # Velocities in world coords
        else:
            linearVelocity = kwargs.get('velocity', base_state.velocity)
            angularVelocity = kwargs.get('omega', base_state.omega)

        # Send the updated state to the physics client
        self._client.resetBasePositionAndOrientation(bodyUniqueId=self._id,
                                                     posObj=posObj,
                                                     ornObj=ornObj)
        self._client.resetBaseVelocity(objectUniqueId=self._id,
                                       linearVelocity=linearVelocity,
                                       angularVelocity=angularVelocity)
        return 0

###############################################################################
#JOINT CLASS
###############################################################################
@dataclass(frozen=True)
class JointState():
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
    angle: float
    omega: float

    def __init__(self, **kwargs):
        self._set_angle(kwargs.get('angle', 0.0))
        self._set_omega(kwargs.get('omega', 0.0))

    def _set_angle(self, angle):
        super().__setattr__('angle', float(angle))

    def _set_omega(self, omega):
        super().__setattr__('omega', float(omega))

class Joint:
    """
    """
    def __init__(self, sim_obj, name, idx):
        self.name = name
        self._client = sim_obj._client
        self._body_id = sim_obj._id
        self._id = idx
        self._init_state = JointState()
        self._set_defaults()

    def _set_defaults(self):
        # Set the default dynamics
        default_dyanamics = {'joint_resistance' : 0.0,
                             'max_omega' : 1000.0}
        self.set_dynamics(**default_dyanamics)

        # Disable joint position and velocity control
        mode = self._client.POSITION_CONTROL
        self._client.setJointMotorControlArray(self._body_id,
                                               [self._id, ],
                                               mode,
                                               forces=[0.0, ])
        mode = self._client.VELOCITY_CONTROL
        self._client.setJointMotorControlArray(self._body_id,
                                               [self._id, ],
                                               mode,
                                               forces=[0.0, ])

        # Disbale the force and torque sensor
        self._client.enableJointForceTorqueSensor(self._body_id,
                                                  self._id,
                                                  enableSensor=False)

        # Set to default initial state
        angle = self._init_state.angle
        omega = self._init_state.omega
        self.set_state(angle=angle, omega=omega)

    @property
    def initial_state(self):
        """ The initial state of the joint. """
        return self._init_state

    @property
    def state(self):
        """ The current state of the joint. """
        angle,omega,_,_ = self._client.getJointState(self._body_id, self._id)
        joint_state = JointState(angle=angle, omega=omega)
        return joint_state

    def set_dynamics(self, **kwargs):
        """
        Set the joint resistance (damping) and the maximum joint angular
        velocity.

        Parameters
        ----------
        **kwargs
            Dynamics information with the following acceptable keys
            joint_resistance : float, optional
                The resistance (damping) of the joint about the joint axis.
            max_omega : float, optional
                The maximum allowed angular velocity of the joint about the
                joint axis.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        args = {}
        try:
            if 'joint_resistance' in kwargs:
                args['jointDamping'] = float(kwargs['joint_resistance'])
            if 'max_omega' in kwargs:
                args['maxJointVelocity'] = float(kwargs['max_omega'])
        except (TypeError, ValueError):
            return -1
        self._client.changeDynamics(self._body_id, self._id, **args)
        return 0

    def set_initial_state(self, **kwargs):
        """
        Sets the initial state of the joint. When the simulation is reset
        the joint will be reset to this value

        Parameters
        ----------
        **kwargs
            Joint state information with the following acceptable keys
            angle : float, optional
                The (angle in radians) of the joint about the joint axis.
            omega : float, optional
                The angular velocity (angle in radians / second) of the joint
                about the joint axis.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        self._init_state = JointState(**kwargs)
        return self.set_state(angle=self._init_state.angle,
                              omega=self._init_state.omega)

    def set_state(self, **kwargs):
        """
        Sets the current state of the joint.

        Parameters
        ----------
        **kwargs
            Joint state information with the following acceptable keys
            angle : float, optional
                The (angle in radians) of the joint about the joint axis. When
                not defined, does not change from current value.
            omega : float, optional
                The angular velocity (angle in radians / second) of the joint
                about the joint axis.  When not defined, does not change from
                current value.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        targetValue = kwargs.get('angle', self.state.angle)
        targetVelocity = kwargs.get('omega', self.state.omega)
        try:
            targetValue = float(targetValue)
            targetVelocity = float(targetVelocity)
        except (TypeError, ValueError):
            return -1
        self._client.resetJointState(self._body_id,
                                     self._id,
                                     targetValue=targetValue,
                                     targetVelocity=targetVelocity)
        return 0

###############################################################################
#LINK CLASS
###############################################################################
class Link:
    """
    """
    def __init__(self, sim_obj, name, idx):
        self.name = name
        self._client = sim_obj._client
        self._body_id = sim_obj._id
        self._id = idx
        self._set_defaults()

    def _set_defaults(self):
        # Set the default dynamics
        default_dyanamics = {'lateral_contact_friction' : 100.0,
                             'spinning_contact_friction' : 0.0,
                             'rolling_contact_friction' : 0.0,
                             'bounciness' : 0.0,
                             'linear_air_resistance' : 0.0,
                             'angular_air_resistance' : 0.0,
                             'contact_damping' : -1.0,
                             'contact_stiffness' : -1.0,}
        self.set_dynamics(**default_dyanamics)

    def set_dynamics(self, **kwargs):
        args = {}
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
        self._client.changeDynamics(self._body_id, self._id, **args)
        return 0

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
        # Build the object and return it
        body = Body(self._client, path, **kwargs)
        return body

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
