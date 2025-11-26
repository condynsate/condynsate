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
        initial_state : condynsate.simulator.dataclasses.BodyState
            The initial state of the body. Can be upated with the
            set_initial_state function.
        state : condynsate.simulator.dataclasses.BodyState
            The current state of the body in simulation. Can be upated either
            by the simulation or with the set_state function.
        center_of_mass : 3 tuple of floats
            The center of mass of the body in world coordinates.
        visual_data : list of dicts
            All data needed to render the body assuming each link is rendered
            individually.
        links : dict of condynsate.simulator.objects.Link
            A dictionary whose keys are link names (as defined by the .URDF)
            and whose values are the Link objects that facilitate interaction.
        joints : dict of condynsate.simulator.objects.Joint
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
        (self.name, self.links, self.joints) = self._make_links_joints()
        self._arrows = {'com_force' : [],
                        'base_torque' : []}

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
        joints = {}

        # Make each joint and non-base link
        for joint_id in range(self._client.getNumJoints(self._id)):

            # Get the joint's name along with its parent and child's names
            info = self._client.getJointInfo(self._id, joint_id)
            joint_name = info[1].decode('UTF-8')
            child_name = info[12].decode('UTF-8')
            for link in links.values():
                if link.visual_data['id'] == info[16]:
                    parent_link = link
                    break

            # Get the parent and children links and make the joint
            links[child_name] = Link(self, joint_id)
            joints[joint_name] = Joint(self, joint_id, parent_link)
        return body_name, links, joints

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
        link_ids, Obws, oris = self._get_all_link_pos_ori()
        Rbws = [t.Rbw_from_wxyz(ori) for ori in oris]
        masses = []
        coms = []
        for link_id, Obw, Rbw in zip(link_ids, Obws, Rbws):
            info = self._client.getDynamicsInfo(self._id, link_id,)
            masses.append(info[0])
            coms.append(tuple(t.pa_to_pb(Rbw, Obw, info[3]).tolist()))
        return tuple(np.average(coms, weights=masses, axis=0).tolist())

    @property
    def visual_data(self):
        """ Data needed to render the body. """
        # Get the ordered positions and orientations of all links in body
        link_ids, poss, oris = self._get_all_link_pos_ori()

        # Extract visual data from each link
        names = [None for p in poss]
        paths = [None for p in poss]
        scales = [None for p in poss]
        colors = [None for p in poss]
        opacities = [None for p in poss]
        tex_paths = [None for p in poss]
        for link_name, link in self.links.items():
            i = link_ids.index(link.visual_data['id'])

            # Each position and orientation is poss and oris is the position and
            # orientation of the link frame origin (defined by the stl).
            # We must now convert each link frame to its visual frame.
            poss[i] = t.pa_to_pb(t.Rbw_from_wxyz(oris[i]), poss[i],
                                 link.visual_data['vis_pos'])
            oris[i] = t.wxyz_mult(link.visual_data['vis_ori'], oris[i])

            # Get the name, mesh, scale, color, and opacity
            names[i] = (self.name, link_name)
            paths[i] = link.visual_data['mesh']
            scales[i] = link.visual_data['scale']
            colors[i] = link.visual_data['color']
            opacities[i] = link.visual_data['opacity']
            tex_paths[i] = link.visual_data['tex_path']

        # Assemble visual data
        keys = ('name','path','position','wxyz_quat','scale',
                'color','opacity','tex_path')
        data = [dict(zip(keys, vals)) for vals in
                     zip(names, paths, poss, oris, scales,
                         colors, opacities, tex_paths)]

        # Append the arrow data
        for arrow in self._get_arr_vis_dat():
            data.append(dict(zip(keys, arrow)))
        return data

    def clear_visual_buffer(self):
        """
        Clears the body's visual buffer. If visual_data is not collected each
        time step, then clear_visual_buffer must be called to prevent the
        visual_data buffer from growing indeterminately.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        for i in range(len(self._arrows['com_force'])):
            self._arrows['com_force'][i] = None
        for i in range(len(self._arrows['base_torque'])):
            self._arrows['base_torque'][i] = None
        for joint_name, joint in self.joints.items():
            for i in range(len(joint.arrows['torque'])):
                joint.arrows['torque'][i] = None
        for link_name, link in self.links.items():
            for i in range(len(link.arrows['force'])):
                link.arrows['force'][i] = None
        return 0

    def _get_all_link_pos_ori(self):
        # Get the base state
        base_state = self._client.getBasePositionAndOrientation(self._id)

        # Get all other link states simultaneously
        link_ids = sorted(link.visual_data['id']
                          for link in self.links.values())
        link_states = self._client.getLinkStates(self._id, link_ids[1:])

        # Compile all positions and orientations
        poss = [s[4] for s in link_states]
        poss.insert(0, base_state[0])
        oris = [t.wxyz_from_xyzw(s[5]) for s in link_states]
        oris.insert(0, t.wxyz_from_xyzw(base_state[1]))
        return link_ids, poss, oris

    def __get_arr_vis_dat(self, name, vis_file, arrow_dat):
        name = (self.name, name)
        condynsate_src = os.path.dirname(os.path.dirname(__file__))
        assets_dirpath = os.path.join(condynsate_src, "__assets__")
        path = os.path.join(assets_dirpath, vis_file)
        default = (name, path, (0.,)*3, (1.,)+(0.,)*3, (0.01,)*3, (0.,)*3, 0.)
        if arrow_dat is None:
            return default
        magnitude = float(np.linalg.norm(arrow_dat['value']))
        if magnitude == 0.0:
            return default
        dirn = np.divide(arrow_dat['value'], magnitude)
        position = arrow_dat['position']
        orientation = t.wxyz_from_vecs((0., 0., 1.), dirn)
        scale = tuple(magnitude*s for s in arrow_dat['scale'])
        color = (0.,)*3
        opacity = 1.
        tex_path = None
        return name,path,position,orientation,scale,color,opacity,tex_path

    def _get_arr_vis_dat(self):
        # Center of mass force arrow
        arrows = tuple()
        for i, force in enumerate(self._arrows['com_force']):
            args = (f'com_force_{i+1}', 'arrow_lin.stl', force)
            arrows += (self.__get_arr_vis_dat(*args), )
            self._arrows['com_force'][i] = None

        # Base link torque arrow
        for i, torque in enumerate(self._arrows['base_torque']):
            args = (f'base_torque_{i+1}', 'arrow_ccw.stl', torque)
            arrows += (self.__get_arr_vis_dat(*args), )
            self._arrows['base_torque'][i] = None

        # Get each joint's torque arrow
        for joint_name, joint in self.joints.items():
            for i, torque in enumerate(joint.arrows['torque']):
                args = (f'{joint_name}_torque_{i+1}', 'arrow_ccw.stl', torque)
                arrows += (self.__get_arr_vis_dat(*args), )
                joint.arrows['torque'][i] = None

        # Get each link's force arrow
        for link_name, link in self.links.items():
            for i, force in enumerate(link.arrows['force']):
                args = (f'{link_name}_force_{i+1}', 'arrow_lin.stl', force)
                arrows += (self.__get_arr_vis_dat(*args), )
                link.arrows['force'][i] = None
        return arrows

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

        # Get the new position, if not defined, default to current position
        posObj = kwargs.get('position', state.position)

        # Get the new orientation, if not defined, default to current
        # orientation (Tait-Bryan angle-wise)
        ypr0 = state.ypr
        yaw = kwargs.get('yaw', ypr0[0])
        pitch = kwargs.get('pitch', ypr0[1])
        roll = kwargs.get('roll', ypr0[2])
        ornObj = t.xyzw_from_wxyz(t.wxyz_from_euler(yaw, pitch, roll))

        # Convert velocities in body coords to world coords
        if kwargs.get('body', False):
            Rbw = t.Rbw_from_euler(yaw, pitch, roll)
            vel = kwargs.get('velocity', None)
            if vel is None:
                linearVelocity = state.velocity
            else:
                linearVelocity = tuple(t.va_to_vb(Rbw, vel).tolist())
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

    def _get_Obw_Rbw(self):
        Obw, ornObj = self._client.getBasePositionAndOrientation(self._id)
        Rbw = t.Rbw_from_wxyz(t.wxyz_from_xyzw(ornObj))
        return Obw, Rbw

    def _add_arrow(self, position, value, name, sf, **kwargs):
        if kwargs.get('draw_arrow', False):
            scale = kwargs.get('arrow_scale', 1.0)
            scale = tuple(scale if s is None else s for s in sf)
            arrow_dat = {'position' : position,
                         'value' : value,
                         'scale' : scale}
            if len(self._arrows[name]) == 0:
                self._arrows[name].append(arrow_dat)
            else:
                for i, val in enumerate(self._arrows[name]):
                    if val is None:
                        self._arrows[name][i] = arrow_dat
                        break
                    if i == len(self._arrows[name]) - 1:
                        self._arrows[name].append(arrow_dat)
                        break
        return 0


    def apply_force(self, force, **kwargs):
        """
        Applies force to the center of mass of the body.

        Parameters
        ----------
        force : 3 tuple of floats
            The force being applied to the center of mass.
        **kwargs
            Extra arguments. Valid keys include
            body : bool, optional
                A Boolean flag that indicates if the force argument is in
                body coordinates (True), or in world coordinates (False).
                The default is False.
            draw_arrow : bool, optional
                A Boolean flag that indicates if an arrow should be drawn
                to represent the applied force. The default is False.
            arrow_scale : float, optional
                The scaling factor, relative to the size of the applied force,
                that is used to size the force arrow. The default is 1.0.

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

        if kwargs.get('body', False):
            force = t.va_to_vb(self._get_Obw_Rbw()[1], force)

        # Calculate required centers of mass
        # Explicit calc like this requires one less center_of_mass call and
        # also allows use of getLinkStates instead of getLinkState which
        # reduces overhead
        link_ids, Obws, oris = self._get_all_link_pos_ori()
        Rbws = [t.Rbw_from_wxyz(ori) for ori in oris]
        mass = []
        com = []
        base_com = None
        for link_id, Obw, Rbw in zip(link_ids, Obws, Rbws):
            info = self._client.getDynamicsInfo(self._id, link_id,)
            mass.append(info[0])
            com.append(tuple(t.pa_to_pb(Rbw, Obw, info[3]).tolist()))
            if link_id == -1:
                base_com = com[-1]
        com = tuple(np.average(com, weights=mass, axis=0).tolist())

        # Get the required counter torque
        torque = tuple(np.cross(np.subtract(base_com, com), force).tolist())

        # Apply force and counter torque
        self._client.applyExternalForce(self._id, -1, force, com,
                                        flags=self._client.WORLD_FRAME)
        self._client.applyExternalTorque(self._id, -1, torque,
                                         flags=self._client.WORLD_FRAME)

        # Add arrow information for rendering
        self._add_arrow(com, force, 'com_force', (None, None, None), **kwargs)
        return 0

    def apply_torque(self, torque, **kwargs):
        """
        Applies external torque to the body.

        Parameters
        ----------
        torque : 3 tuple of floats
            The torque being applied.
        **kwargs
            Extra arguments. Valid keys include
            body : bool, optional
                A Boolean flag that indicates if the torque argument is in
                body coordinates (True), or in world coordinates (False).
                The default is False.
            draw_arrow : bool, optional
                A Boolean flag that indicates if an arrow should be drawn
                to represent the applied torque. The default is False.
            arrow_scale : float, optional
                The scaling factor, relative to the size of the applied torque,
                that is used to size the torque arrow. The default is 1.0.

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

        Obw, Rbw = self._get_Obw_Rbw()
        if kwargs.get('body', False):
            torque = t.va_to_vb(Rbw, torque)

        flag = self._client.WORLD_FRAME
        self._client.applyExternalTorque(self._id, -1, torque, flags=flag)

        # Add arrow information for rendering
        self._add_arrow(Obw,torque,'base_torque',(None, None, 0.01),**kwargs)
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
        initial_state : condynsate.simulator.dataclasses.JointState
            The initial state of the joint. Can be upated with the
            set_initial_state function.
        state : condynsate.simulator.dataclasses.JointState
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

    """
    def __init__(self, sim_obj, idx, parent):
        self._client = sim_obj._client
        self._body_id = sim_obj._id
        self._id = idx
        self._parent = parent
        self._init_state = JointState()
        self._set_defaults()
        self._type = self._client.getJointInfo(self._body_id, self._id)[2]
        if self._type in (self._client.JOINT_PRISMATIC,
                          self._client.JOINT_PLANAR):
            msg = "Prismatic and Planar joints are not currently supported"
            raise TypeError(msg)
        self.arrows = {'torque' : [],}

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
        _, Rpw = self._parent.Obw_Rbw
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
        angle0,omega0,_,_ = self._client.getJointState(self._body_id, self._id)
        targetValue = kwargs.get('angle', angle0)
        targetVelocity = kwargs.get('omega', omega0)
        try:
            targetValue = float(targetValue)
            targetVelocity = float(targetVelocity)
        except (TypeError, ValueError):
            warn('Unable to set state, erroneous kwargs.')
            return -1
        self._client.resetJointState(self._body_id, self._id,
                                     targetValue=targetValue,
                                     targetVelocity=targetVelocity)
        return 0

    def apply_torque(self, torque, **kwargs):
        """
        Applies torque to a joint for a single simulation step.

        Parameters
        ----------
        torque : float
            The torque being applied about the joint's axis..
        **kwargs
            Extra arguments. Valid keys include
            draw_arrow : bool, optional
                A Boolean flag that indicates if an arrow should be drawn
                to represent the applied torque. The default is False.
            arrow_scale : float, optional
                The scaling factor, relative to the size of the applied torque,
                that is used to size the torque arrow. The default is 1.0.
            arrow_offset : float, optional
                The amount by which the drawn is offset from the center of the
                joint's child link along the joint axis. The default is 0.0.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if self._type == self._client.JOINT_FIXED:
            warn("Cannot apply torque to a fixed joint.")
            return -1
        try:
            torque = float(torque)
        except (TypeError, ValueError):
            warn('Cannot apply torque, invalid torque value.')
            return -1
        self._client.setJointMotorControlArray(self._body_id,
                                               [self._id, ],
                                               self._client.TORQUE_CONTROL,
                                               forces=[torque, ])

        # Add arrow information for rendering
        if kwargs.get('draw_arrow', False):
            info = self._client.getJointInfo(self._body_id, self._id)
            axisp = t.va_to_vb(t.Rbw_from_wxyz(t.wxyz_from_xyzw(info[15])),
                               info[13])
            Opw, Rpw = self._parent.Obw_Rbw
            axisw = t.va_to_vb(Rpw, axisp)
            Ojw = t.pa_to_pb(Rpw, Opw, info[14])
            offset = kwargs.get('arrow_offset', 0.0)
            position = tuple(o+offset*a for o, a in zip(Ojw, axisw))
            scale = kwargs.get('arrow_scale', 1.0)
            arrow_dat = {'position' : position,
                         'value' : tuple(float(torque*x) for x in axisw),
                         'scale' : (scale, scale, 0.01)}
            if len(self.arrows['torque']) == 0:
                self.arrows['torque'].append(arrow_dat)
            else:
                for i, val in enumerate(self.arrows['torque']):
                    if val is None:
                        self.arrows['torque'][i] = arrow_dat
                        break
                    if i == len(self.arrows['torque']) - 1:
                        self.arrows['torque'].append(arrow_dat)
                        break
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
        state : condynsate.simulator.dataclasses.LinkState
            The current state of the link (position, orientation, velocity,
                                           and angular velocity)
        mass : float
            The mass of the link. Can be set with the set_dynamics function.
        center_of_mass : 3 tuple of floats
            The current center of mass of the link in world coordinates.
        visual_data : dict
            A dictionary containing all required data to render the link.
        Obw_Rbw : 2tuple of 3tuple and 3x3 array-like
            The position of the link in world coordinates and the rotation
            matrix of the link relative to the world.

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
        self._visual_data = self._get_visual_data()
        self.arrows = {'force' : [],}

    def _set_defaults(self):
        # Set the default dynamics
        default_dyanamics = {'lateral_contact_friction' : 100.0,
                             'spinning_contact_friction' : 0.0,
                             'rolling_contact_friction' : 0.0,
                             'bounciness' : 0.0,
                             'linear_air_resistance' : 0.005,
                             'angular_air_resistance' : 0.005,}
        self.set_dynamics(**default_dyanamics)

    def _get_visual_data(self):
        data = self._client.getVisualShapeData(self._body_id)
        data = [d for d in data if d[1]==self._id][0]
        mesh = os.path.realpath(data[4].decode('UTF-8'))
        vis_ori = t.wxyz_from_xyzw(data[6])
        keys = ('id', 'scale', 'mesh', 'vis_pos', 'vis_ori',
                'color', 'opacity', 'tex_path')
        dat = (self._id, data[3], mesh, data[5], vis_ori,
               data[7][:-1], data[7][-1], None)
        return dict(zip(keys, dat))

    @property
    def state(self):
        """ The current state of the Link. """
        # Base link case, return base state
        if self._id == -1:
            pos, ori=self._client.getBasePositionAndOrientation(self._body_id)
            ori = t.wxyz_from_xyzw(ori)
            vel, omg = self._client.getBaseVelocity(self._body_id)
            st = LinkState(position=pos,orientation=ori,velocity=vel,omega=omg)
            return st

        # Otherwise return link state
        state = self._client.getLinkState(self._body_id, self._id,
                                          computeLinkVelocity=1)
        pos = state[0]
        ori = t.wxyz_from_xyzw(state[1])
        vel = state[6]
        omg = state[7]
        st = LinkState(position=pos,orientation=ori,velocity=vel,omega=omg)
        return st

    @property
    def mass(self):
        """ The mass of the link. """
        return self._client.getDynamicsInfo(self._body_id, self._id,)[0]

    @property
    def center_of_mass(self):
        """ The center of mass of the link in world coordinates. """
        com_b = self._client.getDynamicsInfo(self._body_id, self._id,)[3]
        Obw, Rbw = self.Obw_Rbw
        com_w = tuple(t.pa_to_pb(Rbw, Obw, com_b).tolist())
        return com_w

    @property
    def visual_data(self):
        """ All visual data required to render the link. """
        return self._visual_data

    @property
    def Obw_Rbw(self):
        """ The position and rotation of the link in the world """
        # Get the position and orientation of the link
        if self._id == -1:
            Obw,ori = self._client.getBasePositionAndOrientation(self._body_id)
            Rbw = t.Rbw_from_wxyz(t.wxyz_from_xyzw(ori))
        else:
            state = self._client.getLinkState(self._body_id, self._id)
            Obw = state[0]
            Rbw = t.Rbw_from_wxyz(t.wxyz_from_xyzw(state[1]))
        return Obw, Rbw

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

    def set_color(self, color):
        """
        Changes the color of the link.

        Parameters
        ----------
        color : 3 tuple of floats
            The color to set the link. In the form (R, G, B) where R is the
            red channel, G is the green channel, and B is the blue channel.
            Each channel has value between 0.0 and 1.0.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        try:
            color = (float(min(max(color[0], 0.0), 1.0)),
                     float(min(max(color[1], 0.0), 1.0)),
                     float(min(max(color[2], 0.0), 1.0)))
        except (TypeError, ValueError, IndexError):
            warn('Cannot set color, invalid color value.')
            return -1
        self._visual_data['color'] = color
        return 0

    def set_texture(self, texture):
        self._visual_data['tex_path'] = texture

    def apply_force(self, force, **kwargs):
        """
        Applies force to the center of mass of a link.

        Parameters
        ----------
        force : 3 tuple of floats
            The force being applied to the center of mass.
        **kwargs
            Extra arguments. Valid keys include
            body : bool, optional
                A Boolean flag that indicates if the force argument is in
                body coordinates (True), or in world coordinates (False).
                The default is False.
            draw_arrow : bool, optional
                A Boolean flag that indicates if an arrow should be drawn
                to represent the applied force. The default is False.
            arrow_scale : float, optional
                The scaling factor, relative to the size of the applied force,
                that is used to size the force arrow. The default is 1.0.

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

        # Convert force from body to world coords
        Obw, Rbw = self.Obw_Rbw
        if kwargs.get('body', False):
            force = t.va_to_vb(Rbw, force)

        # Get the center of mass in world coordinates
        com_b = self._client.getDynamicsInfo(self._body_id, self._id,)[3]
        com_w = tuple(t.pa_to_pb(Rbw, Obw, com_b).tolist())

        # Apply force
        flag = self._client.WORLD_FRAME
        self._client.applyExternalForce(self._body_id, self._id, force, com_w,
                                        flags=flag)

        # Add arrow information for rendering
        if kwargs.get('draw_arrow', False):
            scale = kwargs.get('arrow_scale', 1.0)
            arrow_dat = {'position' : com_w,
                         'value' : force,
                         'scale' : (scale, scale, scale)}
            if len(self.arrows['force']) == 0:
                self.arrows['force'].append(arrow_dat)
            else:
                for i, val in enumerate(self.arrows['force']):
                    if val is None:
                        self.arrows['force'][i] = arrow_dat
                        break
                    if i == len(self.arrows['force']) - 1:
                        self.arrows['force'].append(arrow_dat)
                        break
        return 0
