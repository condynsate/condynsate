# -*- coding: utf-8 -*-
"""
This module provides the Visualizer class.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import zlib
import time
from warnings import warn
from threading import (Thread, Lock)
import numpy as np
import meshcat
import meshcat.geometry as geo
import umsgpack
import cv2
from condynsate.misc import save_recording
from condynsate.visualizer.utilities import (is_instance, is_num, is_nvector,
                                             path_valid, name_valid)
from condynsate.visualizer.utilities import homogeneous_transform
from condynsate.visualizer.utilities import get_scene_path

###############################################################################
#VISUALIZER CLASS
###############################################################################
class Visualizer():
    """
    Visualizer manages the meshcat based visulation.

    Parameters
    ----------
    frame_rate : bool, optional
        The frame rate of the visualizer. When None, attempts to run at
        unlimited. This is not reccomended because it can cause communication
        bottlenecks that cause slow downs. The default value is 45.
    record : bool, optional
        A boolean flag that indicates if the visualizer will record

    """
    def __init__(self, frame_rate=45.0, record=False):
        """
        Constructor method.

        """
        # Calculate time between frames
        if not frame_rate is None:
            self.frame_delta = 1.0 / frame_rate
        else:
            self.frame_delta = 0.0

        # Open a new instance of a meshcat visualizer
        self._scene = meshcat.Visualizer().open()
        self._socket = self._scene.window.zmq_socket

        # Delete all instances from the visualizer
        self._scene.delete()

        # Track each object's geometry, material, and transform. This way
        # we can ignore requests that would not change these.
        self._objects = {}

        # Recording support
        self.record = record
        self._frames = []
        self._frame_ticks = []

        # Start the main thread
        self._actions_buf = {}
        self._done = False
        self._last_refresh = cv2.getTickCount()
        self._LOCK = Lock()
        self._start()

        # Set the default scene settings
        self._set_defaults()

    def __del__(self):
        """
        Deconstructor method.

        """
        self.terminate()

    def _start(self):
        """
        Starts the drawing thread.

        Returns
        -------
        None.

        """
        # Start the main thread
        self._thread = Thread(target=self._main_loop)
        self._thread.daemon = True
        self._thread.start()

    def _main_loop(self):
        """
        Runs a loop that continuously calls sends at the proper frame rate
        until the done flag is set to True.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Continuously redraw
        while True:
            # Time since last frame was rendered
            dt = (cv2.getTickCount()-self._last_refresh)/cv2.getTickFrequency()
            if dt < self.frame_delta:
                time.sleep(0.0005) # Remove CPU stress
                continue

            # Aquire mutex lock to read flags and shared buffer
            actions = []
            priorities = []
            with self._LOCK:
                # If visualizer is closed unexpectedly, end main loop then
                # return failure
                if self._socket.closed:
                    msg = ("Cannot flush actions because visualizer closed"
                           " unexpectedly")
                    warn(msg)
                    self._done = True
                    return -1

                # Extract all of the actions from the shared actions buffer
                for fnc in list(self._actions_buf.keys()):
                    for args, kwargs in self._actions_buf.pop(fnc).values():
                        actions.append((fnc, args, kwargs))
                        priorities.append(self._fnc_priority(fnc))

                # Sort the actions based on their priorities
                actions = [x for _, x in sorted(zip(priorities, actions),
                                                key=lambda pair: pair[0])]

                # If done, do all the last actions under lock and then end
                # the thread loop
                if self._done:
                    for (fnc, args, kwargs) in actions:
                        fnc(*args, **kwargs)
                    return 0

            # Do all the actions
            for (fnc, args, kwargs) in actions:
                fnc(*args, **kwargs)
            self._last_refresh = cv2.getTickCount()

            # If recording, save the current image
            if self.record:
                image = self._scene.get_image(w=800, h=600)
                image = np.asarray(image, dtype=np.uint8)[:,:,:-1]
                image = image.copy()
                self._frames.append((zlib.compress(image), image.shape))
                self._frame_ticks.append(self._last_refresh)

    def _fnc_priority(self, fnc):
        """
        Assigns a priority (with low values being the highest priority) to
        function execution order during main loop.

        Parameters
        ----------
        fnc : function
            The function whose priority is being measured.

        Returns
        -------
        int
            The execution order priority of fnc.

        """
        if fnc in (self._add_object, ):
            return 1
        if fnc in (self._set_transform, self._set_material):
            return 2
        if fnc in (self._set_cam_position, self._set_cam_target,
                   self._set_cam_zoom, self._set_cam_frustum):
            return 3
        if fnc in (self._set_grid, self._set_axes,
                   self._set_background, self._set_light):
            return 4
        return 5

    def _queue_action(self, fnc, scene_path, args, kwargs={}):
        """
        Queues a function argument pair to the scene object at position
        scene_path for execution on the next frame time.

        Parameters
        ----------
        fnc : function
            A function pointer.
        scene_path : String
            The scene path to the object on which the function is operated.
        args : tuple
            The star args applied to function fnc.
        kwargs : dict, optional
            The star star kwargs applied to function fnc. The default is {}.
            When {}, no kwargs are sent to the function.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Aquire mutex lock to interact with actions buffer
        with self._LOCK:
            if self._socket.closed or self._done:
                msg = "Cannot complete action because visualizer is stopped."
                warn(msg)
                return -1

            # If the function is not in the actions buffer, add it.
            if not fnc in self._actions_buf:
                self._actions_buf[fnc] = {scene_path : (args, kwargs)}
                return 0

            # If the function is in the buffer, overwrite the args applied
            # to the object at scene_path
            self._actions_buf[fnc][scene_path] = (args, kwargs)
            return 0

    def _set_defaults(self):
        """
        Sets the default scene settings

        Returns
        -------
        None.

        """
        # Set the default visibility of grid and axes
        self.set_grid(visible=True)
        self.set_axes(visible=True)

        # Set the default background color
        self.set_background(top=(0.44, 0.62, 0.82), bottom=(0.82, 0.62, 0.44))

        # Set the default lights
        self.set_spotlight(on=True, intensity=0.02, distance=0, shadow=True)
        self.set_ptlight_1(on=True, intensity=0.15, distance=0, shadow=True)
        self.set_ptlight_2(on=True, intensity=0.45, distance=0, shadow=True)
        self.set_amblight(on=True, intensity=0.55, shadow=True)
        self.set_dirnlight(on=True, intensity=0.05, shadow=True)

        # Set the default camera properties
        self.set_cam_position((3.0, -6.0, 4))
        self.set_cam_target((0.0, 0.0, 0.0))
        self.set_cam_zoom(1.0)
        self.set_cam_frustum(near=0.01, far=1000.0)

        # Wait just long enough to ensure all default commands go through
        time.sleep(0.25)

    def _set_grid(self, visible):
        """
        Sets the visibility of the grid.

        Parameters
        ----------
        visible : bool
            The boolean value to which the visibility of the grid is set.

        Returns
        -------
        None.

        """
        self._scene["/Grid"].set_property("visible", visible)
        self._scene["/Grid/<object>"].set_property("visible", visible)

    def set_grid(self, visible):
        """
        Controls the visibility state of the XY grid in the visualizer.

        Parameters
        ----------
        visible : bool
            The boolean value to which the visibility of the XY grid is set.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if not is_instance(visible, bool, arg_name='visible'):
            return -1
        scene_path = '/Grid'
        args = (visible,)
        return self._queue_action(self._set_grid, scene_path, args)

    def _set_axes(self, visible):
        """
        Sets the visibility of the axes.

        Parameters
        ----------
        visible : bool
            The boolean value to which the visibility of the axes is set.

        Returns
        -------
        None.

        """
        self._scene["/Axes"].set_property("visible", visible)
        self._scene["/Axes/<object>"].set_property("visible", visible)

    def set_axes(self, visible):
        """
        Controls the visibility state of the axes in the visualizer.

        Parameters
        ----------
        visible : bool
            The boolean value to which the visibility of the axes is set.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if not is_instance(visible, bool, arg_name='visible'):
            return -1
        scene_path = '/Axes'
        args = (visible,)
        return self._queue_action(self._set_axes, scene_path, args)

    def _set_background(self, top, bottom):
        """
        Sets the background color.

        Parameters
        ----------
        top : 3 tuple of floats between 0.0 and 1.0, optional
            The RGB color to apply to the top of the background.
        bottom : 3 tuple of floats between 0.0 and 1.0, optional
            The RGB color to apply to the bottom of the background.

        Returns
        -------
        None.

        """
        # Set the top color
        if not top is None:
            top = [float(np.clip(float(t), 0.0, 1.0)) for t in top]
            self._scene["/Background"].set_property('top_color', top)

        # Set the bottom color
        if not bottom is None:
            bottom = [float(np.clip(float(b), 0.0, 1.0)) for b in bottom]
            self._scene["/Background"].set_property('bottom_color', bottom)

    def set_background(self, top=None, bottom=None):
        """
        Set the top and bottom colors of the background of the scene.

        Parameters
        ----------
        top : 3 tuple of floats between 0.0 and 1.0, optional
            The RGB color to apply to the top of the background.
            If None, is not altered. The default is None
        bottom : 3 tuple of floats between 0.0 and 1.0, optional
            The RGB color to apply to the bottom of the background.
            If None, is not altered. The default is None

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Ensure top and bottom are of the correct format
        if not top is None and not is_nvector(top, 3, 'top'):
            return -1
        if not bottom is None and not is_nvector(bottom, 3, 'bottom'):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Background'
        args = (top, bottom,)
        return self._queue_action(self._set_background, scene_path, args)

    def _set_light(self, light, **kwargs):
        """
        Sets the properties of a light. Pass None to any argument to not
        set that property of the light.

        Parameters
        ----------
        light : String
            The case sensitive name of the light in the scene tree. Choose
            from "SpotLight", "PointLightNegativeX", "PointLightPositiveX",
            "AmbientLight", or "FillLight".
        **kwargs :
            on : bool
                Boolean flag that indicates if the light is on.
            position : 3tuple of floats
                The position of the light source in (x,y,z) world coordinates.
            intensity : float
                Numeric value of the light's strength/intensity.
            distance : float
                Maximum range of the light. Default is 0 (no limit).
            decay : float
                The amount a ptlight or spotlight type light dims along the
                distance of the light.
            angle : float between 0.0 and 1.5707
                The beam angle of a spotlight type light in radians.
            shadow : bool
                Boolean flag that indicates if the light casts a shadow.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Get the scene tree paths
        scene_path_1 = '/Lights/'+light
        scene_path_2 = scene_path_1+'/<object>'

        # Set the properties
        if 'on' in kwargs:
            self._scene[scene_path_1].set_property('visible', kwargs['on'])
            self._scene[scene_path_2].set_property('visible', kwargs['on'])

        if 'position' in kwargs:
            position = tuple(float(p) for p in kwargs['position'])
            self._scene[scene_path_2].set_property('position', position)

        if 'intensity' in kwargs:
            intensity = np.clip(kwargs['intensity'], 0.0, 10.0)
            self._scene[scene_path_2].set_property('intensity', intensity)

        if 'distance' in kwargs:
            distance = np.clip(kwargs['distance'], 0.0, 100.0)
            self._scene[scene_path_2].set_property('distance', distance)

        if 'decay' in kwargs:
            decay = np.clip(kwargs['decay'], 0.0, 10.0)
            self._scene[scene_path_2].set_property('decay', decay)

        if 'angle' in kwargs:
            angle = np.clip(kwargs['angle'], 0.0, 1.5707)
            self._scene[scene_path_2].set_property('angle', angle)

        # Because of a typo in the meshcat repo, setting castShadow is a little
        # harder and requires us to directly send the ZQM message
        if 'shadow' in kwargs:
            cmd_data = {'type': 'set_property',
                        'path': scene_path_2,
                        'property': 'castShadow',
                        'value': kwargs['shadow']}
            self._socket.send_multipart([cmd_data["type"].encode("utf-8"),
                                         cmd_data["path"].encode("utf-8"),
                                         umsgpack.packb(cmd_data)])
            self._socket.recv()

    def _light_args_ok(self, light, **kwargs):
        """
        Ensures that a set of arguments to be passed to _set_light are all
        valid. Returns True if all valid, returns False if at least one
        invalid. Raises exception if the visualizer is not open or if the
        light argument is not a string.

        Parameters
        ----------
        light : String
            The case sensitive name of the light in the scene tree. Choose
            from "SpotLight", "PointLightNegativeX", "PointLightPositiveX",
            "AmbientLight", or "FillLight".
        **kwargs :
            on : bool
                Boolean flag that indicates if the light is on.
            position : 3tuple of floats
                The position of the light source in (x,y,z) world coordinates.
                Does not apply to amblight type sources.
            intensity : float
                Numeric value of the light's strength/intensity.
            distance : float
                Maximum range of the light. Default is 0 (no limit).
            decay : float
                The amount a ptlight or spotlight type light dims along the
                distance of the light.
            angle : float between 0.0 and 1.5707
                The beam angle of a spotlight type light in radians.
            shadow : bool
                Boolean flag that indicates if the light casts a shadow.

        Returns
        -------
        is_okay : bool
            All inputs are valid.

        """
        ok = is_instance(light, str, arg_name='light')
        if 'on' in kwargs:
            ok = ok and is_instance(kwargs['on'], bool, arg_name='on')
        if 'position' in kwargs:
            ok = ok and is_nvector(kwargs['position'], 3, arg_name='position')
        if 'intensity' in kwargs:
            ok = ok and is_num(kwargs['intensity'], arg_name='intensity')
        if 'distance' in kwargs:
            ok = ok and is_num(kwargs['distance'], arg_name='distance')
        if 'decay' in kwargs:
            ok = ok and is_num(kwargs['decay'], arg_name='decay')
        if 'angle' in kwargs:
            ok = ok and is_num(kwargs['angle'], arg_name='angle')
        if 'shadow' in kwargs:
            ok = ok and is_instance(kwargs['shadow'], bool, arg_name='shadow')
        return ok


    def set_spotlight(self, **kwargs):
        """
        Sets the properties of the spotlight in the scene.

        Parameters
        ----------
        **kwargs :
            on : bool
                Boolean flag that indicates if the light is on.
            position : 3tuple of floats
                The position of the light source in (x,y,z) world coordinates.
                Does not apply to amblight type sources.
            intensity : float
                Numeric value of the light's strength/intensity.
            distance : float
                Maximum range of the light. Default is 0 (no limit).
            decay : float
                The amount a ptlight or spotlight type light dims along the
                distance of the light.
            angle : float between 0.0 and 1.5707
                The beam angle of a spotlight type light in radians.
            shadow : bool
                Boolean flag that indicates if the light casts a shadow.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'SpotLight'
        if not self._light_args_ok(name, **kwargs):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Lights/SpotLight'
        args = (name,)
        return self._queue_action(self._set_light, scene_path, args, kwargs)

    def set_ptlight_1(self, **kwargs):
        """
        Sets the properties of the point light along the positive x axis
        in the scene.

        Parameters
        ----------
        **kwargs :
            on : bool
                Boolean flag that indicates if the light is on.
            position : 3tuple of floats
                The position of the light source in (x,y,z) world coordinates.
                Does not apply to amblight type sources.
            intensity : float
                Numeric value of the light's strength/intensity.
            distance : float
                Maximum range of the light. Default is 0 (no limit).
            shadow : bool
                Boolean flag that indicates if the light casts a shadow.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'PointLightPositiveX'
        if not self._light_args_ok(name, **kwargs):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Lights/PointLightPositiveX'
        args = (name,)
        return self._queue_action(self._set_light, scene_path, args, kwargs)

    def set_ptlight_2(self, **kwargs):
        """
        Sets the properties of the point light along the negative x axis
        in the scene.

        Parameters
        ----------
        **kwargs :
            on : bool
                Boolean flag that indicates if the light is on.
            position : 3tuple of floats
                The position of the light source in (x,y,z) world coordinates.
                Does not apply to amblight type sources.
            intensity : float
                Numeric value of the light's strength/intensity.
            distance : float
                Maximum range of the light. Default is 0 (no limit).
            shadow : bool
                Boolean flag that indicates if the light casts a shadow.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'PointLightNegativeX'
        if not self._light_args_ok(name, **kwargs):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Lights/PointLightNegativeX'
        args = (name,)
        return self._queue_action(self._set_light, scene_path, args, kwargs)

    def set_amblight(self, **kwargs):
        """
        Sets the properties ambient light of the scene.

        Parameters
        ----------
        **kwargs :
            on : bool
                Boolean flag that indicates if the light is on.
            intensity : float
                Numeric value of the light's strength/intensity.
            shadow : bool
                Boolean flag that indicates if the light casts a shadow.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'AmbientLight'
        if not self._light_args_ok(name, **kwargs):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Lights/AmbientLight'
        args = (name, )
        return self._queue_action(self._set_light, scene_path, args, kwargs)

    def set_dirnlight(self, **kwargs):
        """
        Sets the properties fill light of the scene.

        Parameters
        ----------
        **kwargs :
            on : bool
                Boolean flag that indicates if the light is on.
            position : 3tuple of floats
                The position of the light source in (x,y,z) world coordinates.
                Does not apply to amblight type sources.
            intensity : float
                Numeric value of the light's strength/intensity.
            shadow : bool
                Boolean flag that indicates if the light casts a shadow.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'FillLight'
        if not self._light_args_ok(name, **kwargs):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Lights/FillLight'
        args = (name, )
        return self._queue_action(self._set_light, scene_path, args, kwargs)

    def _set_cam_position(self, p):
        """
        Set the XYZ position of camera.

        Parameters
        ----------
        p : 3Vec of floats
            The XYZ position to which the camera will be moved.

        Returns
        -------
        None.

        """
        scene_path = "/Cameras/default/rotated/<object>"
        p = (float(p[0]), float(p[2]), -float(p[1])) #Camera is rotated
        self._scene[scene_path].set_property('position', p)

    def set_cam_position(self, p):
        """
        Set the XYZ position of camera.

        Parameters
        ----------
        p : 3Vec of floats
            The XYZ position to which the camera will be moved. After moving,
            the camera will automatically adjust its orientation to continue
            looking directly at its target.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Ensure argument is correct type
        if not is_nvector(p, 3, arg_name='p'):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Cameras/default/rotated'
        args = (p, )
        return self._queue_action(self._set_cam_position, scene_path, args)

    def _set_cam_target(self, t):
        """
        Set the XYZ position of the point the camera is looking at.

        Parameters
        ----------
        t : 3Vec of floats
            The XYZ position for the camera to look at.

        Returns
        -------
        None.

        """
        p = (float(t[0]), float(t[1]), float(t[2]))
        self._scene.set_cam_target(p)

    def set_cam_target(self, t):
        """
        Set the XYZ position of the point the camera is looking at.

        Parameters
        ----------
        t : 3Vec of floats
            The XYZ position for the camera to look at. Regardless of updates
            to the camera position, the camera will always look directly at
            this point.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Ensure argument is correct type
        if not is_nvector(t, 3, arg_name='t'):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Cameras/default/rotated'
        args = (t, )
        return self._queue_action(self._set_cam_target, scene_path, args)

    def _set_cam_zoom(self, zoom):
        """
        Sets the zoom value of the camera

        Parameters
        ----------
        zoom : float greater than 0 and less than or equal to 100.
            The zoom value .

        Returns
        -------
        None.

        """
        # Ensure zoom in (0, 100]
        zoom = np.clip(zoom, 0.0001, 100.0)
        scene_path = "/Cameras/default/rotated/<object>"
        self._scene[scene_path].set_property('zoom', zoom)

    def set_cam_zoom(self, zoom):
        """
        Sets the zoom value of the camera

        Parameters
        ----------
        zoom : float greater than 0 and less than or equal to 100.
            The zoom value .

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Ensure correct type
        if not is_num(zoom, arg_name='zoom'):
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Cameras/default/rotated'
        args = (zoom, )
        return self._queue_action(self._set_cam_zoom, scene_path, args)

    def _set_cam_frustum(self, aspect, fov, near, far):
        """
        Sets the size and shape of the camera's frustum.

        Parameters
        ----------
        aspect : float
            The aspect ratio of the near and far planes of the frustum.
        fov : float
            The vertical field of view of the frustum in degrees.
        near : float less than far
            The distance to the near plane of the frustum.
        far : float greater than near
            The distance to the far plane of the frustum.

        Returns
        -------
        None.

        """
        scene_path = "/Cameras/default/rotated/<object>"
        if not aspect is None:
            self._scene[scene_path].set_property('aspect', aspect)
        if not fov is None:
            self._scene[scene_path].set_property('fov', fov)
        if not near is None:
            self._scene[scene_path].set_property('near', near)
        if not far is None:
            self._scene[scene_path].set_property('far', far)

    def set_cam_frustum(self, aspect=None, fov=None, near=None, far=None):
        """
        Sets the size and shape of the camera's frustum.

        Parameters
        ----------
        aspect : float, optional
            The aspect ratio of the near and far planes of the frustum. When
            None, maintains the current value. The default is None.
        fov : float, optional
            The vertical field of view of the frustum in degrees.
            None, maintains the current value. The default is None.
        near : float less than far, optional
            The distance to the near plane of the frustum.
            None, maintains the current value. The default is None.
        far : float greater than near, optional
            The distance to the far plane of the frustum.
            None, maintains the current value. The default is None.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Check the inputs are in the correct type
        is_okay = True
        if not aspect is None:
            is_okay = is_okay and is_num(aspect, arg_name='aspect')
        if not fov is None:
            is_okay = is_okay and is_num(fov, arg_name='fov')
        if not near is None:
            is_okay = is_okay and is_num(near, arg_name='near')
        if not far is None:
            is_okay = is_okay and is_num(far, arg_name='far')
        if not is_okay:
            return -1

        # Queue the action in thread safe manner
        scene_path = '/Cameras/default/rotated'
        args = (aspect, fov, near, far, )
        return self._queue_action(self._set_cam_frustum, scene_path, args)

    def _get_geometry(self, path):
        """
        Loads an object's mesh.

        Parameters
        ----------
        path : string
            Path pointing to the file that describes the object's
            geometry. The file may be of type .obj, .stl, or .dae.

        Returns
        -------
        geometry : meshcat.geometry.ObjMeshGeometry
            The object's mesh.

        """
        geometry = None
        if path.endswith('.obj'):
            geometry = geo.ObjMeshGeometry.from_file(path)
        elif path.endswith('.stl'):
            geometry = geo.StlMeshGeometry.from_file(path)
        elif path.endswith('.dae'):
            geometry = geo.DaeMeshGeometry.from_file(path)
        return geometry

    def _get_material(self, tex_path, color, shininess, opacity):
        """
        Makes a Phong material.

        Parameters
        ----------
        tex_path : string
            The path pointing to a .png file that defines the texture of
            the object being added. Is only applied correctly if object is
            of type .obj or .dae. .stl files do not support proper
            texturing and attempting to apply texture to .stl may result in
            unexpected viual results. If None, no texture is applied.
        color : 3vec of floats
            The color to apply to the object being added. In the form of
            (R, G, B) where all elements range from 0.0 to 1.0.
        shininess : float
            The shininess of the object being added. Ranges from 0.0 to 1.0.
        opacity : float
            The opacity of the object being added. Ranges from 0.0 to 1.0.

        Returns
        -------
        Material : meshcat.geometry.MeshPhongMaterial
            The object's material.

        """
        if not tex_path is None:
            texture = geo.PngImage.from_file(tex_path)
            texture = geo.ImageTexture(texture, wrap=[1, 1], repeat=[1, 1])
        else:
            texture = None
        color = tuple(int(255*c) for c in color)
        color = int("0x{:02x}{:02x}{:02x}".format(*color), 16)
        mat_kwargs = {'color' : color,
                      'map' : texture,
                      'opacity' : opacity,
                      'shininess' : shininess}
        return geo.MeshPhongMaterial(**mat_kwargs)

    def _add_object(self, name, path, material_kwargs):
        """
        Adds an object to the scene.

        Parameters
        ----------
        name : string or tuple of strings
            A list of strings defining the name of the object as well
            as its position in the scene heirarchy. For example,
            ('foo', 'bar') would insert a new object to the scene at location
            /Scene/foo/bar while 'baz' would insert the object at
            /Scene/baz
        path : string
            Path pointing to the file that describes the object's
            geometry. The file may be of type .obj, .stl, or .dae.
        material_kwargs : dict
            The material kwargs of the object being added. Includes arguments
            tex_path, color, shininess, and opacity.

        Returns
        -------
        None.

        """
        # Add the .obj to the scene
        scene_path = get_scene_path(name)
        geometry = self._get_geometry(path)
        self._objects[scene_path] = {'geometry':geometry,
                                     'tex_path':material_kwargs['tex_path'],
                                     'color':material_kwargs['color'],
                                     'shininess':material_kwargs['shininess'],
                                     'opacity':material_kwargs['opacity'],
                                     'trans_matrix':np.eye(4)}
        material = self._get_material(**material_kwargs)
        self._scene[scene_path].set_object(geometry, material)


    def add_object(self, name, path, **kwargs):
        """
        Adds an object to the visualizer scene.

        Parameters
        ----------
        name : string or tuple of strings
            A list of strings defining the name of the object as well
            as its position in the scene heirarchy. For example,
            ('foo', 'bar') would insert a new object to the scene at location
            /Scene/foo/bar while 'baz' would insert the object at
            /Scene/baz
        path : string
            Path pointing to the file that describes the object's
            geometry. The file may be of type .obj, .stl, or .dae.
        **kwargs
            tex_path : string, optional
                The path pointing to a .png file that defines the texture of
                the object being added. Is only applied correctly if object is
                of type .obj or .dae. .stl files do not support proper
                texturing and attempting to apply texture to .stl may result in
                unexpected viual results. If None, no texture is applied. The
                default is None
            color : 3vec of floats, optional
                The color to apply to the object being added. In the form of
                (R, G, B) where all elements range from 0.0 to 1.0. The default
                is (1.0, 1.0, 1.0).
            shininess : float, optional
                The shininess of the object being added. Ranges from 0.0 to 1.0
                The default value of 0.01.
            opacity : float, optional
                The opacity of the object being added. Ranges from 0.0 to 1.0.
                The default value is 1.0.
            position : 3vec of floats, optional
                The extrinsic position to set.
                The default value is (0., 0., 0.)
            wxyz_quat : 4vec of floats, optional
                The extrinsic rotation to set as defined by a quaternion.
                The default value is (1., 0., 0., 0.)
            yaw : float, optional
                The intrinsic yaw angle to set in degrees. Defined about the
                object's Z axis. The default value is 0.0.
            pitch : float, optional
                The intrinsic pitch angle to set in degrees. Defined about the
                object's Y axis. The default value is 0.0.
            roll : float, optional
                The intrinsic roll angle to set in degrees. Defined about the
                object's X axis. The default value is 0.0.
            scale : 3vec of floats, optional
                The intrinsic scale of the object. When not set, defaults to
                (1., 1., 1.).

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Check the args
        if not name_valid(name, arg_name='name'):
            return -1
        if not path_valid(path, ftype=('.obj', '.dae', '.stl'), arg_name=path):
            return -1

        # If the object is already there, do not add it again
        scene_path = get_scene_path(name)
        if scene_path in self._objects:
            return 0

        # Sanitize the kwargs
        material_kwargs = self._read_material_kwargs(kwargs)
        transform_kwargs = self._read_transform_kwargs(kwargs)

        # Queue loading the object into the scene
        args = (name, path, material_kwargs, )
        ret_code = self._queue_action(self._add_object, scene_path, args)
        if ret_code < 0:
            return ret_code

        # Queue transforming the object
        args = (name, transform_kwargs, )
        return self._queue_action(self._set_transform, scene_path, args)

    def _set_transform(self, name, transform):
        """
        Applies transform to a scene object relative to it's original origin.

        Parameters
        ----------
        name : string or tuple of strings
            A list of strings defining the name of the object as well
            as its position in the scene heirarchy. For example,
            ('foo', 'bar') would insert a new object to the scene at location
            /Scene/foo/bar while 'baz' would insert the object at
            /Scene/baz
        transform : dict
            Defines the transform. Has keys 'position', 'wxyz_quat',
            'yaw', 'pitch', 'roll', and 'scale'. yaw, pitch, and roll are in
            radians

        Returns
        -------
        None.

        """
        # Get the scene path
        scene_path = get_scene_path(name)
        if not scene_path in self._objects:
            msg = f'{name} is not object in scene, cannot set transform.'
            warn(msg)
            return

        # Get the transformation matrix
        position = transform['position']
        wxyz_quat = transform['wxyz_quat']
        yaw = transform['yaw']
        pitch = transform['pitch']
        roll = transform['roll']
        scale = transform['scale']
        args = (position, wxyz_quat, yaw, pitch, roll, scale, )
        trans_matrix = homogeneous_transform(*args)

        # Check if a transform is needed
        cur_trans_matrix = self._objects[scene_path]['trans_matrix']
        if np.allclose(trans_matrix, cur_trans_matrix):
            return

        # Apply the transform
        self._scene[scene_path].set_transform(trans_matrix)

        # Update the store state
        self._objects[scene_path]['trans_matrix'] = trans_matrix

    def _read_transform_kwargs(self, kwargs):
        """
        Reads and sanitizes the kwargs for function self.set_transform.

        Parameters
        ----------
        kwargs : dict
            The kwargs being sanitized.

        Returns
        -------
        sanitized : dict
            The sanitized kwargs with default values set for non user defined
            keys.

        """
        # Set the default transform kwargs
        sanitized = {'position' : (0.0, 0.0, 0.0),
                     'wxyz_quat' : (1.0, 0.0, 0.0, 0.0),
                     'yaw' : 0.0,
                     'pitch' : 0.0,
                     'roll' : 0.0,
                     'scale' : (1.0, 1.0, 1.0)}

        # Validate each key given by user
        for key, val in kwargs.items():
            k = key.lower()

            # Validate the position
            if k == 'position':
                if not is_nvector(val, 3, arg_name=key):
                    continue
                sanitized[key] = val

            # Validate the wxyz_quat
            elif k == 'wxyz_quat':
                if not is_nvector(val, 4, arg_name=key):
                    continue
                sanitized[key] = tuple(float(max(-1, min(x, 1))) for x in val)

            # Validate the roll, pitch, and yaw
            elif k in ('yaw', 'pitch', 'roll'):
                if not is_num(val, arg_name=key):
                    continue
                sanitized[key] = float(max(-180, min(val, 180)))

            # Validate the scale
            elif k == 'scale':
                if not is_nvector(val, 3, arg_name=key):
                    continue
                sanitized[key] = tuple(float(max(0, x)) for x in val)
        return sanitized

    def set_transform(self, name, **kwargs):
        """
        Sets the position, orientation, and scale of a scene object relative to
        its original origin and size.

        Parameters
        ----------
        name : string or tuple of strings
            A list of strings defining the name of the object as well
            as its position in the scene heirarchy. For example,
            ('foo', 'bar') would refers to the object at the scene location
            /Scene/foo/bar while 'baz' refers to the object at scene location
            /Scene/baz
        **kwargs
            position : 3vec of floats, optional
                The extrinsic position to set.
                The default value is (0., 0., 0.)
            wxyz_quat : 4vec of floats, optional
                The extrinsic rotation to set as defined by a quaternion.
                The default value is (1., 0., 0., 0.)
            yaw : float, optional
                The intrinsic yaw angle to set in radians. Defined about the
                object's Z axis. The default value is 0.0.
            pitch : float, optional
                The intrinsic pitch angle to set in radians. Defined about the
                object's Y axis. The default value is 0.0.
            roll : float, optional
                The intrinsic roll angle to set in radians. Defined about the
                object's X axis. The default value is 0.0.
            scale : 3vec of floats, optional
                The intrinsic scale of the object. When not set, defaults to
                (1., 1., 1.).

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Check the args
        if not name_valid(name, arg_name='name'):
            return -1

        # Queue transforming the object
        transform_kwargs = self._read_transform_kwargs(kwargs)
        scene_path = get_scene_path(name)
        args = (name, transform_kwargs, )
        return self._queue_action(self._set_transform, scene_path, args)

    def _read_material_kwargs(self, kwargs):
        """
        Reads and sanitizes the kwargs for function self.set_material.

        Parameters
        ----------
        kwargs : dict
            The kwargs being sanitized.

        Returns
        -------
        sanitized : dict
            The sanitized kwargs with default values set for non user defined
            keys.

        """
        # Set default values
        sanitized = {'tex_path' : None,
                     'color' : (1.0, 1.0, 1.0),
                     'shininess' : 0.01,
                     'opacity' : 1.0,}

        # Validate each key given by user
        for key, val in kwargs.items():
            k = key.lower()

            # Validate the texture path
            if k == 'tex_path':
                if val is None or not path_valid(val, ".png", val):
                    continue
                sanitized[key] = val

            # Validate the color
            elif k == 'color':
                if not is_nvector(val, 3, arg_name=key):
                    continue
                sanitized[key] = tuple(float(max(0, min(c, 1))) for c in val)

            # Validate the and opacity
            elif k == 'opacity':
                if not is_num(val, arg_name=key):
                    continue
                sanitized[key] = float(max(0, min(val, 1)))

            # Validate the shininess
            elif k == 'shininess':
                if not is_num(val, arg_name=key):
                    continue
                sanitized[key] = float(max(0, min(500*val, 500)))
        return sanitized

    def _set_material(self, name, material_kwargs):
        """
        Sets the material of a scene object.

        Parameters
        ----------
        name : string or tuple of strings
            A list of strings defining the name of the object as well
            as its position in the scene heirarchy. For example,
            ('foo', 'bar') would refers to the object at the scene location
            /Scene/foo/bar while 'baz' refers to the object at scene location
            /Scene/baz
        material_kwargs : dict
            Defines the material. Has keys 'tex_path', 'color',
            'shininess', and 'opacity'.

        Returns
        -------
        None.

        """
        scene_path = get_scene_path(name)
        if not scene_path in self._objects:
            msg = f'{name} is not object in scene, cannot set material.'
            warn(msg)
            return

        # Check if a change is needed
        cur_tex_path = self._objects[scene_path]['tex_path']
        cur_color = self._objects[scene_path]['color']
        cur_shininess = self._objects[scene_path]['shininess']
        cur_opacity = self._objects[scene_path]['opacity']
        if (cur_tex_path == material_kwargs['tex_path'] and
            cur_color == material_kwargs['color'] and
            cur_shininess == material_kwargs['shininess'] and
            cur_opacity == material_kwargs['opacity']):
            return

        # Apply the new materi
        geometry = self._objects[scene_path]['geometry']
        material = self._get_material(**material_kwargs)
        self._scene[scene_path].set_object(geometry, material)

        # Update the store material
        self._objects[scene_path]['tex_path'] = material_kwargs['tex_path']
        self._objects[scene_path]['color'] = material_kwargs['color']
        self._objects[scene_path]['shininess'] = material_kwargs['shininess']
        self._objects[scene_path]['opacity'] = material_kwargs['opacity']

    def set_material(self, name, **kwargs):
        """
        Sets an objects material.

        Parameters
        ----------
        name : string or tuple of strings
            A list of strings defining the name of the object as well
            as its position in the scene heirarchy. For example,
            ('foo', 'bar') refers to the object at the scene location
            /Scene/foo/bar while 'baz' refers to the object at scene location
            /Scene/baz
        **kwargs
            tex_path : string, optional
                The path pointing to a .png file that defines the texture of
                the object being added. Is only applied correctly if object is
                of type .obj or .dae. .stl files do not support proper
                texturing and attempting to apply texture to .stl may result in
                unexpected viual results. If None, no texture is applied. The
                default is None
            color : 3vec of floats, optional
                The color to apply to the object. In the form of
                (R, G, B) where all elements range from 0.0 to 1.0. The default
                is (1.0, 1.0, 1.0).
            shininess : float, optional
                The shininess of the object. Ranges from 0.0 to 1.0.
                The default value is 0.01
            opacity : float, optional
                The opacity of the object. Ranges from 0.0 to 1.0.
                The default value is 1.0.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Check the args
        if not name_valid(name, arg_name='name'):
            return -1

        # Queue transforming the object
        material_kwargs = self._read_material_kwargs(kwargs)
        scene_path = get_scene_path(name)
        args = (name, material_kwargs, )
        return self._queue_action(self._set_material, scene_path, args)

    def terminate(self):
        """
        Terminates the visualizer's communication with the web browser.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        with self._LOCK:
            self._done = True
        self._thread.join()

        self._actions_buf = {}
        self._objects = {}

        if self.record and len(self._frames) > 1:
            # Convert frame ticks to frame times
            frame_times = np.array(self._frame_ticks, dtype=float)
            frame_times /= cv2.getTickFrequency()
            frame_times -= frame_times[0]
            save_recording(self._frames, frame_times, 'visualizer')
        self._frames = []
        self._frame_ticks = []

        if not self._socket.closed:
            self._scene.delete()
            self._socket.close()
            return 0
        return -1
