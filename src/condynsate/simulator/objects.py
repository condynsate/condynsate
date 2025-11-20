# -*- coding: utf-8 -*-
"""
This module provides the objects that reside in the simulator class which is
used to run physics simulations using the PyBullet package.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import os
from warnings import warn
import numpy as np
import condynsate.misc.transforms as t
from condynsate.simulator.dataclasses import (BodyState, JointState, LinkState)

###############################################################################
#BODY CLASS
###############################################################################
class Body():
    """
    The class stores information about and allows interaction with a body
    in the simulation. This body is defined from a URDF file and is comprised
    of a set of links and joints. Each Body member has the following attributes
        initial_state : condynsate.core.dataclasses.BodyState
            The initial state of the body. Can be upated with the
            set_initial_state function.
        state : condynsate.core.dataclasses.BodyState
            The current state of the body in simulation. Can be upated either
            by the simulation or with the set_state function.
        center_of_mass : 3 tuple of floats
            The center of mass of the body in world coordinates.
        visual_data : list of dicts
            All data needed to render the body assuming each link is rendered
            individually.
        links : dict of condynsate.core.objects.Link
            A dictionary whose keys are link names (as defined by the .URDF)
            and whose values are the Link objects that facilitate interaction.
        joints : dict of condynsate.core.objects.Joint
            A dictionary whose keys are joints names (as defined by the .URDF)
            and whose values are the Joint objects that facilitate interaction.

    Parameters
    ----------
    client : pybullet_utils.bullet_client.BulletClient
        The PyBullet physics client in which the body lives.
    path : string
        The path pointing to the URDF file that defines the body.
    **kwargs
        Additional arguments for the body. Valid keys are
        fixed : boolean, optional
            A flag that indicates if the body is fixed (has 0 DoF) or free
            (has 6 DoF).

    """
    def __init__(self, client, path, **kwargs):
        self._client = client
        self._id = self._load_urdf(path, **kwargs)
        (self.name, self.links,
         link_ids, self.joints) = self._make_links_joints()
        scales, meshes, poss, oris, colors = self._get_shape_data()
        self._link_data = {'id' : link_ids,
                           'scale' : scales,
                           'mesh' : meshes,
                           'vis_pos' : poss,
                           'vis_ori' : oris,
                           'color' : colors,}

    def _load_urdf(self, urdf_path, **kwargs):
        # Use implicit cylinder for collision and physics calculation
        # Specifies to the engine to use the inertia from the urdf file
        f1 = self._client.URDF_USE_IMPLICIT_CYLINDER
        f2 = self._client.URDF_USE_INERTIA_FROM_FILE
        flags = f1 | f2

        # Get the default initial state
        self._init_state = BodyState()
        basePosition = self._init_state.position
        baseOrientation = self._init_state.orientation
        baseOrientation = t.xyzw_from_wxyz(baseOrientation)
        linearVelocity = self._init_state.velocity
        angularVelocity = self._init_state.omega

        # Load the URDF with default initial conditions
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

    def _make_links_joints(self):
        # Make the base link
        base_name, body_name = self._client.getBodyInfo(self._id)
        base_name = base_name.decode('UTF-8')
        body_name = f"{self._id}_{body_name.decode('UTF-8')}"
        links = {base_name : Link(self, -1)}
        link_ids = {-1 : base_name}
        joints = {}

        # Make each joint and non-base link
        for joint_id in range(self._client.getNumJoints(self._id)):

            # Get the joint's name along with its parent and child links' names
            info = self._client.getJointInfo(self._id, joint_id)
            joint_name = info[1].decode('UTF-8')
            child_name = info[12].decode('UTF-8')
            parent_name = link_ids[info[16]]

            # Build the child link
            links[child_name] = Link(self, joint_id)
            link_ids[joint_id] = child_name

            # Get the parent and children links and make the joint
            parent = links[parent_name]
            child = links[child_name]
            joints[joint_name] = Joint(self, joint_id, parent, child)
        return body_name, links, link_ids, joints

    def _get_shape_data(self):
        data = self._client.getVisualShapeData(self._id)
        scales = [d[3] for d in data]
        meshes = [os.path.realpath(d[4].decode('UTF-8')) for d in data]
        vis_pos = [d[5] for d in data]
        vis_ori = [t.wxyz_from_xyzw(d[6]) for d in data]
        colors = [d[7] for d in data]
        return scales, meshes, vis_pos, vis_ori, colors

    @property
    def initial_state(self):
        """ The initial state of the body. """
        return self._init_state

    @property
    def state(self):
        """ The current state of the body. """
        # Get the base states
        pos, ornObj = self._client.getBasePositionAndOrientation(self._id)
        ori = t.wxyz_from_xyzw(ornObj)
        vel, omg = self._client.getBaseVelocity(self._id)

        # Compile and return
        state = BodyState(position=pos,
                          orientation=ori,
                          velocity=vel,
                          omega=omg)
        return state

    @property
    def center_of_mass(self):
        """ The position of the center of mass of the object. """
        masses = []
        coms = []
        for link in self.links.values():
            masses.append(link.mass)
            coms.append(link.center_of_mass)
        return tuple(np.average(coms, weights=masses, axis=0).tolist())

    @property
    def visual_data(self):
        """ Data needed to render the body. """
        # Get the base state
        base_state = self._client.getBasePositionAndOrientation(self._id)

        # Get all other link states simultaneously
        link_ids = list(range(len(self.links)-1))
        link_states = self._client.getLinkStates(self._id,
                                                 link_ids,
                                                 computeLinkVelocity=0)

        # Compile all positions and orientations
        poss = [s[4] for s in link_states]
        poss.insert(0, base_state[0])
        oris = [t.wxyz_from_xyzw(s[5]) for s in link_states]
        oris.insert(0, t.wxyz_from_xyzw(base_state[1]))

        # Each position and orientation is poss and oris is the position and
        # orientation of the link frame origin (defined by the stl).
        # We must now convert each link frame to its visual frame.
        zipped = zip(self._link_data['vis_pos'], self._link_data['vis_ori'])
        for i, (vis_pos, vis_ori) in enumerate(zipped):
            poss[i] = t.pa_to_pb(t.Rbw_from_wxyz(oris[i]), poss[i], vis_pos)
            oris[i] = t.wxyz_mult(vis_ori, oris[i])

        # Get the name of each link in order
        link_ids.insert(0, -1)
        names = [(self.name, self._link_data['id'][i]) for i in link_ids]

        # Assemble all visual data
        # (name, position, orientation, scale, mesh path, and color)
        zipped = zip(names, poss, oris,
                     self._link_data['scale'],
                     self._link_data['mesh'],
                     self._link_data['color'])
        data = []
        for name, pos, ori, scale, mesh, color in zipped:
            data.append({'name' : name,
                         'path' : mesh,
                         'position' : pos,
                         'wxyz_quat' : ori,
                         'scale' : scale,
                         'color' : color[:-1],
                         'opacity' : color[-1]})
        return data

    def _state_kwargs_ok(self, **kwargs):
        try:
            _ = tuple(float(x) for x in
                      kwargs.get('position', (0.0, 0.0, 0.0)))
            _ = float(kwargs.get('yaw', 0.0))
            _ = float(kwargs.get('pitch', 0.0))
            _ = float(kwargs.get('roll', 0.0))
            _ = tuple(float(x) for x in
                      kwargs.get('velocity', (0.0, 0.0, 0.0)))
            _ = tuple(float(x) for x in
                      kwargs.get('omega', (0.0, 0.0, 0.0)))
            _ = bool(kwargs.get('body', False))
            return True
        except (TypeError, ValueError):
            return False

    def set_initial_state(self, **kwargs):
        """
        Sets the initial state of the body. When the simulation is
        reset, this object will be reset to this state.

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
        if not self._state_kwargs_ok(**kwargs):
            warn('Unable to set state, erroneous kwargs.')
            return -1

        # ypr to orientation
        yaw = float(kwargs.get('yaw', 0.0))
        pitch = float(kwargs.get('pitch', 0.0))
        roll = float(kwargs.get('roll', 0.0))
        kwargs['orientation'] = t.wxyz_from_euler(yaw, pitch, roll)

        # Set the initial base state
        self._init_state = BodyState(**kwargs)
        return self.set_state(**kwargs)

    def set_state(self, **kwargs):
        """
        Sets the state of the body.

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
        if not self._state_kwargs_ok(**kwargs):
            warn('Unable to set state, erroneous kwargs.')
            return -1

        # Get the current state of the body
        state = self.state
        ypr0 = state.ypr

        # Get the new position, if not defined, default to current position
        posObj = kwargs.get('position', state.position)

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
                linearVelocity = state.velocity
            else:
                linearVelocity = tuple(t.va_to_vb(Rbw, vel).tolist())

            # Angular velocity
            omg = kwargs.get('omega', None)
            if omg is None:
                angularVelocity = state.omega
            else:
                angularVelocity = tuple(t.va_to_vb(Rbw, omg).tolist())

        # Velocities in world coords
        else:
            linearVelocity = kwargs.get('velocity', state.velocity)
            angularVelocity = kwargs.get('omega', state.omega)

        # Send the updated state to the physics client
        self._client.resetBasePositionAndOrientation(bodyUniqueId=self._id,
                                                     posObj=posObj,
                                                     ornObj=ornObj)
        self._client.resetBaseVelocity(objectUniqueId=self._id,
                                       linearVelocity=linearVelocity,
                                       angularVelocity=angularVelocity)
        return 0

    def apply_force(self, force, body=False):
        """
        Applies force to the center of mass of the body.

        Parameters
        ----------
        force : 3 tuple of floats
            The force being applied to the center of mass.
        body : bool, optional
            A Boolean flag that indicates if the force argument is in
            body coordinates (True), or in world coordinates (False).
            The default is False.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        try:
            force = (float(force[0]), float(force[1]), float(force[2]))
        except (TypeError, ValueError, IndexError):
            warn('Cannot apply force, invalid force value.')
            return -1

        if body:
            Rbw = t.Rbw_from_wxyz(self.state.orientation)
            force = t.va_to_vb(Rbw, force)

        # Get the required counter torque
        com = self.center_of_mass
        base = self.links[self._link_data['id'][-1]]
        r = np.subtract(base.center_of_mass, com)
        torque = tuple(np.cross(r, force).tolist())

        # Apply force and counter torque
        flag = self._client.WORLD_FRAME
        self._client.applyExternalForce(self._id, -1, force, com, flags=flag)
        self._client.applyExternalTorque(self._id, -1, torque, flags=flag)
        return 0

    def apply_torque(self, torque, body=False):
        """
        Applies external torque to the body.

        Parameters
        ----------
        torque : 3 tuple of floats
            The torque being applied.
        body : bool, optional
            A Boolean flag that indicates if the torque argument is in
            body coordinates (True), or in world coordinates (False).
            The default is False.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        try:
            torque = (float(torque[0]), float(torque[1]), float(torque[2]))
        except (TypeError, ValueError, IndexError):
            warn('Cannot apply torque, invalid torque value.')
            return -1

        if body:
            Rbw = t.Rbw_from_wxyz(self.state.orientation)
            torque = t.va_to_vb(Rbw, torque)

        flag = self._client.WORLD_FRAME
        self._client.applyExternalTorque(self._id, -1, torque, flags=flag)
        return 0

    def reset(self):
        """
        Resets body and each of its joints to their initial conditions.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        kwargs = {}
        kwargs['position'] = self._init_state.position
        ypr = self._init_state.ypr
        kwargs['yaw'] = ypr[0]
        kwargs['pitch'] = ypr[1]
        kwargs['roll'] = ypr[2]
        kwargs['velocity'] = self._init_state.velocity
        kwargs['omega'] = self._init_state.omega
        kwargs['body'] = False
        self.set_state(**kwargs)

        for joint in self.joints.values():
            joint.reset()

        return 0

###############################################################################
#JOINT CLASS
###############################################################################
class Joint:
    """
    The class stores information about and allows interaction with a joint
    on a body in the simulation. Each Joint member has the following attributes
        initial_state : condynsate.core.objects.JointState
            The initial state of the joint. Can be upated with the
            set_initial_state function.
        state : condynsate.core.objects.JointState
            The current state of the joint in simulation. Read only.
        axis : 3 tuple of floats
            The axis, in world coordinates, about which the joint operates.

    Parameters
    ----------
    sim_obj : condynsate.core.objects.Body
        The member of the Body class to which the joint belongs
    idx : int
        The unique number that identifies the joint in the PyBullet client.
    parent : condynsate.core.objects.Link
        The parent link of the joint.
    child : condynsate.core.objects.Link
        The child link of the joint.

    """
    def __init__(self, sim_obj, idx, parent, child):
        self._client = sim_obj._client
        self._body_id = sim_obj._id
        self._id = idx
        self._parent = parent
        self._child = child
        self._init_state = JointState()
        self._set_defaults()

    def _set_defaults(self):
        # Set the default dynamics
        default_dyanamics = {'damping' : 0.005,
                             'max_omega' : 1000.0}
        self.set_dynamics(**default_dyanamics)

        # Set the joint's control forces to 0.0
        mode = self._client.POSITION_CONTROL
        self._client.setJointMotorControlArray(self._body_id,
                                               [self._id, ],
                                               mode,
                                               forces=[0.0, ])

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

    @property
    def axis(self):
        """ The axis about which the joint operates """
        info = self._client.getJointInfo(self._body_id, self._id)
        axis_j = info[13]
        Rjp = t.Rbw_from_wxyz(t.wxyz_from_xyzw(info[15]))
        axis_p = t.va_to_vb(Rjp, axis_j)
        Rpw = t.Rbw_from_wxyz(self._parent.state.orientation)
        axis_w = t.va_to_vb(Rpw, axis_p)
        return axis_w

    def set_dynamics(self, **kwargs):
        """
        Set the joint damping and the maximum joint angular
        velocity.

        Parameters
        ----------
        **kwargs
            Dynamics information with the following acceptable keys
            damping : float, optional
                The damping of the joint about the joint axis.
                The default value is 0.001.
            max_omega : float, optional
                The maximum allowed angular velocity of the joint about the
                joint axis. The default value is 1000.0

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        args = {}
        try:
            if 'damping' in kwargs:
                args['jointDamping'] = float(kwargs['damping'])
            if 'max_omega' in kwargs:
                args['maxJointVelocity'] = float(kwargs['max_omega'])
        except (TypeError, ValueError):
            warn('Unable to set dynamics, erroneous kwargs.')
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
        try:
            angle = float(kwargs.get('angle', 0.0))
            omega = float(kwargs.get('omega', 0.0))
        except (TypeError, ValueError):
            warn('Unable to set state, erroneous kwargs.')
            return -1
        self._init_state = JointState(angle=angle, omega=omega)
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
            warn('Unable to set state, erroneous kwargs.')
            return -1
        self._client.resetJointState(self._body_id,
                                     self._id,
                                     targetValue=targetValue,
                                     targetVelocity=targetVelocity)
        return 0

    def apply_torque(self, torque):
        """
        Applies torque to a joint for a single simulation step.

        Parameters
        ----------
        torque : float
            The torque being applied about the joint's axis..

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        try:
            torque = float(torque)
        except (TypeError, ValueError):
            warn('Cannot apply torque, invalid torque value.')
            return -1
        mode = self._client.TORQUE_CONTROL
        self._client.setJointMotorControlArray(self._body_id,
                                               [self._id, ],
                                               mode,
                                               forces=[torque, ])
        return 0

    def reset(self):
        """
        Resets the joint to its initial conditions.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        kwargs = {}
        kwargs['angle'] = self._init_state.angle
        kwargs['omega'] = self._init_state.omega
        self.set_state(**kwargs)
        return 0

###############################################################################
#LINK CLASS
###############################################################################
class Link:
    """
    The class stores information about and allows interaction with a link
    on a body in the simulation. Each Link member has the following attributes
        mass : float
            The mass of the link. Can be set with the set_dynamics function.
        center_of_mass : 3 tuple of floats
            The current center of mass of the link in world coordinates.

    Parameters
    ----------
    sim_obj : condynsate.core.objects.Body
        The member of the Body class to which the link belongs
    idx : int
        The unique number that identifies the link in the PyBullet client.

    """
    def __init__(self, sim_obj, idx):
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
                             'linear_air_resistance' : 0.005,
                             'angular_air_resistance' : 0.005,}
        self.set_dynamics(**default_dyanamics)

    @property
    def state(self):
        """ The current state of the Link. """
        # Base link case, return base state
        if self._id == -1:
            pos, ori=self._client.getBasePositionAndOrientation(self._body_id)
            ori = t.wxyz_from_xyzw(ori)
            vel, omg = self._client.getBaseVelocity(self._body_id)
            state = LinkState(position=pos,
                              orientation=ori,
                              velocity=vel,
                              omega=omg)
            return state

        # Otherwise return link state
        state = self._client.getLinkState(self._body_id,
                                          self._id,
                                          computeLinkVelocity=1)
        pos = state[0]
        ori = t.wxyz_from_xyzw(state[1])
        vel = state[6]
        omg = state[7]
        state = LinkState(position = pos,
                          orientation = ori,
                          velocity = vel,
                          omega = omg,)
        return state

    @property
    def mass(self):
        """ The mass of the link. """
        info = self._client.getDynamicsInfo(self._body_id, self._id,)
        return info[0]

    @property
    def center_of_mass(self):
        """ The center of mass of the link in world coordinates. """
        info = self._client.getDynamicsInfo(self._body_id, self._id,)
        com_b = info[3]
        state = self.state
        Obw = state.position
        Rbw = t.Rbw_from_wxyz(state.orientation)
        com_w = tuple(t.pa_to_pb(Rbw, Obw, com_b).tolist())
        return com_w

    def set_dynamics(self, **kwargs):
        """
        Sets the dynamics properties of a single link. Allows user to change
        the mass, contact friction, the bounciness, and the air resistance.

        Parameters
        ----------
        **kwargs
            Dynamics values with the following acceptable keys
            mass : float, optional
                The mass of the link. The default is defined by the .URDF file
            lateral_contact_friction : float, optional
                The lateral (linear) contact friction of the link. 0.0 for
                no friction, increasing friction with increasing value.
                The default is 100.0.
            spinning_contact_friction : float, optional
                The torsional contact friction of the link about
                contact normals. 0.0 for no friction, increasing friction
                with increasing value. The default is 0.0.
            rolling_contact_friction : float, optional
                The torsional contact friction of the link orthogonal to
                contact normals. 0.0 for no friction, increasing friction
                with increasing value. Keep this value either 0.0 or very close
                to 0.0, otherwise the simulations can become unstable.
                The default is 0.0.
            bounciness : float, optional
                How bouncy this link is. 0.0 for inelastic collisions, 0.95 for
                mostly elastic collisions. Setting above 0.95 can result in
                unstable simulations. The default is 0.0.
            linear_air_resistance : float, optional
                The air resistance opposing linear movement applied to the
                center of mass of the link. Usually set to either 0.0 or a
                low value less than 0.1. The default is 0.005.
            angular_air_resistance : float, optional
                The air resistance opposing rotational movement applied about
                the center of rotation of the link. Usually set to either 0.0
                or a low value less than 0.1. The default is 0.005.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
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

        # Ensure all args are floats
        try:
            for i in args.items():
                args[i[0]] = float(i[1])
        except (TypeError, ValueError):
            warn('Unable to set dynamics, erroneous kwargs.')
            return -1

        self._client.changeDynamics(self._body_id, self._id, **args)
        return 0

    def apply_force(self, force, body=False):
        """
        Applies force to the center of mass of a link.

        Parameters
        ----------
        force : 3 tuple of floats
            The force being applied to the center of mass.
        body : bool, optional
            A Boolean flag that indicates if the force argument is in
            the body coordinates of the link (True), or in world coordinates
            (False). The default is False.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        try:
            force = (float(force[0]), float(force[1]), float(force[2]))
        except (TypeError, ValueError, IndexError):
            warn('Cannot apply force, invalid force value.')
            return -1

        if body:
            Rbw = t.Rbw_from_wxyz(self.state.orientation)
            force = t.va_to_vb(Rbw, force)

        flag = self._client.WORLD_FRAME
        self._client.applyExternalForce(self._body_id,
                                        self._id,
                                        force,
                                        self.center_of_mass,
                                        flags=flag)
        return 0
