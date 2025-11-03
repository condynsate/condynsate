"""
This module provides the Visualizer class.
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import time
from warnings import warn
import signal
from threading import (Thread, Lock)
import numpy as np
import meshcat
import meshcat.geometry as geo
import umsgpack
import cv2
from condynsate.visualizer.utilities import (is_num, is_nvector,
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

    grid_vis : bool, optional
        The boolean value to which the visibility of the XY grid is set.
        The default is True.
    axes_vis : bool, optional
        The boolean value to which the visibility of the axes is set.
        The default is True.

    """
    def __init__(self, frame_rate=None):
        """
        Constructor method.

        """
        # Asynch listen for script exit
        signal.signal(signal.SIGTERM, self._sig_handler)
        signal.signal(signal.SIGINT, self._sig_handler)

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

        # Start the main thread
        self._actions_buf = {}
        self._done = False
        self._last_refresh = cv2.getTickCount()
        self._LOCK = Lock()
        self._start()

        # Track each object's geometry file in case the user wants to change
        # their textures.
        self._objects = {}

        # Set the default scene settings
        self._set_defaults()

    def __del__(self):
        """
        Deconstructor method.

        """
        self.terminate()

    def _sig_handler(self, sig, frame):
        """
        Handles script termination events so the keyboard listener exits
        gracefully.

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
        m = "Interrupt or termination signal detected. Terminating visualizer."
        warn(m)
        return self.terminate()

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
            # Check if it is a frame time
            dt = (cv2.getTickCount()-self._last_refresh)/cv2.getTickFrequency()
            if dt < self.frame_delta:
                time.sleep(0.005)
                continue

            # Create a dict to hold all the queued actions in actions buffer
            actions = {}

            # Aquire mutex lock to read flags and shared buffer
            with self._LOCK:

                # If visualizer is closed unexpectedly, end main loop then
                # return failure
                if self._socket.closed:
                    msg = ("Cannot flush actions because visualizer closed"
                           " unexpectedly")
                    warn(msg)
                    self._done = True
                    return -1

                # If done, do the last actions then return success
                if self._done:
                    for fnc, args in self._actions_buf.items():
                        fnc(*args)
                    return 0

                # Extract all of the actions from the shared actions buffer
                for fnc in list(self._actions_buf.keys()):
                    actions[fnc] = self._actions_buf.pop(fnc)

            # If visualizer is open and not done, release the mutex lock and do
            # the actions read from the buffer this loop
            for fnc, args in actions.items():
                fnc(*args)

            # Set the current time as the last refresh time
            self._last_refresh = cv2.getTickCount()

            # Remove CPU strain by sleeping for a little bit
            time.sleep(0.01)

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
        self.set_posx_light(on=True, intensity=0.15, distance=0, shadow=True)
        self.set_negx_light(on=True, intensity=0.45, distance=0, shadow=True)
        self.set_ambient_light(on=True, intensity=0.55, shadow=True)
        self.set_fill_light(on=True, intensity=0.05, shadow=True)

        # Set the default camera properties
        self.set_cam_position((3, 0.5, 2))
        self.set_cam_target((0.0, 0.0, 0.0))
        self.set_cam_zoom(1.0)
        self.set_cam_frustum(near=0.01, far=1000.0)

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
        # Input sanitize
        if not isinstance(visible, bool):
            msg='When set_grid, visible must be boolean.'
            warn(msg)
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_grid
            args = (visible,)
            self._actions_buf[fnc] = args
        return 0

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
        # Input sanitize
        if not isinstance(visible, bool):
            msg='When set_axes, visible must be boolean.'
            warn(msg)
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_axes
            args = (visible,)
            self._actions_buf[fnc] = args
        return 0

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
        # Ensure top is of the correct format
        if not top is None:
            if not is_nvector(top, 3):
                m='When set_background, top must be 3 tuple of floats.'
                warn(m)
                return -1

        # Ensure bottom is of the correct format
        if not bottom is None:
            if not is_nvector(bottom, 3):
                m='When set_background, bottom must be 3 tuple of floats.'
                warn(m)
                return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_background
            args = (top, bottom,)
            self._actions_buf[fnc] = args
        return 0

    def _set_light(self, light, on, intensity, distance, shadow):
        """
        Sets the properties of a light. Pass None to any argument to not
        set that property of the light.

        Parameters
        ----------
        light : String
            The case sensitive name of the light in the scene tree. Choose
            from "SpotLight", "PointLightNegativeX", "PointLightPositiveX",
            "AmbientLight", or "FillLight".
        on : bool
            Boolean flag that indicates if the light is on.
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
        # Get the scene tree paths
        p1 = '/Lights/'+light
        p2 = '/Lights/'+light+'/<object>'

        # Set the properties
        if not on is None:
            self._scene[p1].set_property('visible', on)
            self._scene[p2].set_property('visible', on)
        if not intensity is None:
            intensity = np.clip(intensity, 0.0, 20.0)
            self._scene[p2].set_property('intensity', intensity)
        if not distance is None:
            distance = np.clip(distance, 0.0, 100.0)
            self._scene[p2].set_property('distance', distance)

        # Because of a typo in the meshcat repo, setting castShadow is a little
        # harder and requires us to directly send the ZQM message
        if not shadow is None:
            cmd_data = {'type': 'set_property',
                        'path': p2,
                        'property': 'castShadow',
                        'value': shadow}
            self._socket.send_multipart([cmd_data["type"].encode("utf-8"),
                                         cmd_data["path"].encode("utf-8"),
                                         umsgpack.packb(cmd_data)])
            self._socket.recv()

    def _set_lights(self, *args):
        """
        Sets all light parameters given list of args.

        Parameters
        ----------
        *args
            List of self._set_light args.

        Returns
        -------
        None.

        """
        for arg in args:
            self._set_light(*arg)

    def _light_args_ok(self, light, on, intensity, distance, shadow):
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
        on : bool
            Boolean flag that indicates if the light is on.
        intensity : float
            Numeric value of the light's strength/intensity.
        distance : float
            Maximum range of the light. Default is 0 (no limit).
        shadow : bool
            Boolean flag that indicates if the light casts a shadow.

        Returns
        -------
        is_okay : bool
            All inputs are valid.

        """
        # Check light argument
        if not isinstance(light, str):
            warn("When _set_light, argument light must be string.")
            return False

        # Check on argument
        if not on is None:
            if not isinstance(on, bool):
                warn("When setting light, argument on must be a boolean.")
                return False

        # Check intensity argument
        if not intensity is None:
            try:
                intensity = float(intensity)
            except (TypeError, ValueError):
                warn("When setting light, argument intensity must be a float.")
                return False

        # Check distance argument
        if not distance is None:
            try:
                distance = float(distance)
            except (TypeError, ValueError):
                warn("When setting light, argument distance must be a float.")
                return False

        # Check shadow argument
        if not shadow is None:
            if not isinstance(shadow, bool):
                warn("When setting light, argument shadow must be a boolean.")
                return False

        # IF all okay, return True
        return True

    def set_spotlight(self,on=None,intensity=None,distance=None,shadow=None):
        """
        Sets the properties of the spotlight in the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. When None, 
            is not changed from current value. The default is None.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. When None, 
            is not changed from current value. The default is None.
        distance : float [0. to 100.], optional
            Maximum range of the light. When 0, the range in infinite.
            When None, is not changed from current value. The default is None.
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. When None, 
            is not changed from current value. The default is None.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'SpotLight'
        if not self._light_args_ok(name, on, intensity, distance, shadow):
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_lights
            args = (name, on, intensity, distance, shadow, )
            if fnc in self._actions_buf:
                new_name = True
                for i, a in enumerate(self._actions_buf[fnc]):
                    if a[0] == name:
                        self._actions_buf[fnc][i] = args
                        new_name = False
                        break
                if new_name:
                    self._actions_buf[fnc].append(args)
            else:
                self._actions_buf[fnc] = [args, ]
        return 0

    def set_posx_light(self,on=None,intensity=None,distance=None,shadow=None):
        """
        Sets the properties of the point light along the positive x axis
        in the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. When None, 
            is not changed from current value. The default is None.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. When None, 
            is not changed from current value. The default is None.
        distance : float [0. to 100.], optional
            Maximum range of the light. When 0, the range in infinite.
            When None, is not changed from current value. The default is None.
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. When None, 
            is not changed from current value. The default is None.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'PointLightPositiveX'
        if not self._light_args_ok(name, on, intensity, distance, shadow):
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_lights
            args = (name, on, intensity, distance, shadow, )
            if fnc in self._actions_buf:
                new_name = True
                for i, a in enumerate(self._actions_buf[fnc]):
                    if a[0] == name:
                        self._actions_buf[fnc][i] = args
                        new_name = False
                        break
                if new_name:
                    self._actions_buf[fnc].append(args)
            else:
                self._actions_buf[fnc] = [args, ]
        return 0

    def set_negx_light(self,on=None,intensity=None,distance=None,shadow=None):
        """
        Sets the properties of the point light along the negative x axis
        in the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. When None, 
            is not changed from current value. The default is None.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. When None, 
            is not changed from current value. The default is None.
        distance : float [0. to 100.], optional
            Maximum range of the light. When 0, the range in infinite.
            When None, is not changed from current value. The default is None.
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. When None, 
            is not changed from current value. The default is None.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'PointLightNegativeX'
        if not self._light_args_ok(name, on, intensity, distance, shadow):
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_lights
            args = (name, on, intensity, distance, shadow, )
            if fnc in self._actions_buf:
                new_name = True
                for i, a in enumerate(self._actions_buf[fnc]):
                    if a[0] == name:
                        self._actions_buf[fnc][i] = args
                        new_name = False
                        break
                if new_name:
                    self._actions_buf[fnc].append(args)
            else:
                self._actions_buf[fnc] = [args, ]
        return 0

    def set_ambient_light(self, on=None, intensity=None, shadow=None):
        """
        Sets the properties ambient light of the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. When None, 
            is not changed from current value. The default is None.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. When None, 
            is not changed from current value. The default is None.
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. When None, 
            is not changed from current value. The default is None.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'AmbientLight'
        distance = None
        if not self._light_args_ok(name, on, intensity, distance, shadow):
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_lights
            args = (name, on, intensity, distance, shadow, )
            if fnc in self._actions_buf:
                new_name = True
                for i, a in enumerate(self._actions_buf[fnc]):
                    if a[0] == name:
                        self._actions_buf[fnc][i] = args
                        new_name = False
                        break
                if new_name:
                    self._actions_buf[fnc].append(args)
            else:
                self._actions_buf[fnc] = [args, ]
        return 0

    def set_fill_light(self, on=None, intensity=None, shadow=None):
        """
        Sets the properties fill light of the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. When None, is not
            changed from current value. The default is None.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. When None, is not
            changed from current value. The default is None.
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. When None, 
            is not changed from current value. The default is None.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Make sure inputs are valid
        name = 'FillLight'
        distance = None
        if not self._light_args_ok(name, on, intensity, distance, shadow):
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_lights
            args = (name, on, intensity, distance, shadow, )
            if fnc in self._actions_buf:
                new_name = True
                for i, a in enumerate(self._actions_buf[fnc]):
                    if a[0] == name:
                        self._actions_buf[fnc][i] = args
                        new_name = False
                        break
                if new_name:
                    self._actions_buf[fnc].append(args)
            else:
                self._actions_buf[fnc] = [args, ]
        return 0

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
        path = "/Cameras/default/rotated/<object>"
        p = (float(p[0]), float(p[2]), -float(p[1])) #Camera is rotated
        self._scene[path].set_property('position', p)

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
        if not is_nvector(p, 3):
            msg = "When set_cam_position, p must be 3 tuple of floats."
            warn(msg)
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_cam_position
            args = (p, )
            self._actions_buf[fnc] = args
        return 0

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
        if not is_nvector(t, 3):
            msg = "When set_cam_target, t must be 3 tuple of floats."
            warn(msg)
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_cam_target
            args = (t, )
            self._actions_buf[fnc] = args
        return 0

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
        path = "/Cameras/default/rotated/<object>"
        self._scene[path].set_property('zoom', zoom)

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
        # Ensure proper format
        if not is_num(zoom):
            warn("When set_cam_zoom, zoom must be float")
            return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_cam_zoom
            args = (zoom, )
            self._actions_buf[fnc] = args
        return 0

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
        path = "/Cameras/default/rotated/<object>"
        if not aspect is None:
            self._scene[path].set_property('aspect', aspect)
        if not fov is None:
            self._scene[path].set_property('fov', fov)
        if not near is None:
            self._scene[path].set_property('near', near)
        if not far is None:
            self._scene[path].set_property('far', far)

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
        # Check the aspect ratio
        if not aspect is None:
            if not is_num(aspect):
                warn("When set_cam_frustum, aspect must be float")
                return -1

        # Check the vertical field of view
        if not fov is None:
            if not is_num(fov):
                warn("When set_cam_frustum, fov must be float")
                return -1

        # Check the distance to the near plane
        if not near is None:
            if not is_num(near):
                warn("When set_cam_frustum, near must be float")
                return -1

        # Check the distance to the far plane
        if not far is None:
            if not is_num(far):
                warn("When set_cam_frustum, far must be float")
                return -1

        # Queue the action in thread safe manner
        with self._LOCK:
            fnc = self._set_cam_frustum
            args = (aspect, fov, near, far, )
            self._actions_buf[fnc] = args
        return 0

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

    def _get_material(self, tex_path, color, reflectivity, shininess, opacity):
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
        reflectivity : float
            The reflectivity of the object being added. Ranges from 0.0 to
            1.0.
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
            texture = geo.ImageTexture(texture, wrap=[1, 1], repeat=[2, 2])
        else:
            texture = None
        color = tuple(int(255*c) for c in color)
        color = int("0x{:02x}{:02x}{:02x}".format(*color), 16)
        mat_kwargs = {'color' : color,
                      'reflectivity' : reflectivity,
                      'map' : texture,
                      'opacity' : opacity,
                      'shininess' : shininess}
        return geo.MeshPhongMaterial(**mat_kwargs)

    def _add_object(self, name, geometry, material_kwargs):
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
        geometry : meshcat.geometry.ObjMeshGeometry
            The object's mesh.
        material_kwargs : dict
            The material kwargs of the object being added. Includes arguments
            tex_path, color, reflectivity, shininess, and opacity.
        
        Returns
        -------
        None.

        """
        # Add the .obj to the scene
        scene_path = get_scene_path(name)
        material = self._get_material(**material_kwargs)
        self._scene[scene_path].set_object(geometry, material)

    def _add_objects(self, *args):
        """
        Adds multiple objects to the scene.

        Parameters
        ----------
        *args
            List of set_object args. Each element of list has form
            (name, geometry, material_kwargs).

        Returns
        -------
        None.

        """
        for arg in args:
            self._add_object(*arg)

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
            reflectivity : float, optional
                The reflectivity of the object being added. Ranges from 0.0 to
                1.0. The default value is 0.2
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
        # Sanitize the name
        if not name_valid(name):
            msg = "When add_obj, name is not valid."
            warn(msg)
            return -1

        # Check path integrity
        if not path_valid(path):
            msg = "When add_obj, path is not valid."
            warn(msg)
            return -1
        if not path.endswith(('.obj', '.dae', '.stl')):
            msg = "When add_obj, path must point to .obj, .stl, or .dae file."
            warn(msg)
            return -1

        # Sanitize the kwargs
        keys = ('tex_path', 'color', 'reflectivity', 'shininess', 'opacity')
        material_kwargs = {k : v for k,v in kwargs.items() if k in keys}
        material_kwargs = self._read_material_kwargs(material_kwargs)
        keys = ('position', 'wxyz_quat', 'yaw', 'pitch', 'roll', 'scale')
        transform_kwargs = {k: v for k,v in kwargs.items() if k in keys}
        transform_kwargs = self._read_transform_kwargs(transform_kwargs)

        # Keep track of the object's geometry
        geometry = self._get_geometry(path)
        self._objects[get_scene_path(name)] = geometry

        # Queue the actions in thread safe manner
        with self._LOCK:
            # Queue the add action
            fnc = self._add_objects
            args = (name, geometry, material_kwargs, )
            if not fnc in self._actions_buf:
                self._actions_buf[fnc] = [args, ]
            else:
                self._actions_buf[fnc].append(args)

            # Queue the transform action
            fnc = self._set_transforms
            args = (name, transform_kwargs, )
            if fnc in self._actions_buf:
                new_name = True
                for i, a in enumerate(self._actions_buf[fnc]):
                    if a[0] == name:
                        self._actions_buf[fnc][i] = args
                        new_name = False
                        break
                if new_name:
                    self._actions_buf[fnc].append(args)
            else:
                self._actions_buf[fnc] = [args, ]
        return 0

    def _set_transforms(self, *args):
        """
        Sets transform of multiple objects.

        Parameters
        ----------
        *args
            List of self._set_transform arguments.

        Returns
        -------
        None.

        """
        for arg in args:
            self._set_transform(*arg)

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
            degrees

        Returns
        -------
        None.

        """
        # Get the scene path
        scene_path = get_scene_path(name)

        # Get the transformation matrix
        position = transform['position']
        wxyz_quat = transform['wxyz_quat']
        yaw = transform['yaw']
        pitch = transform['pitch']
        roll = transform['roll']
        scale = transform['scale']
        args = (position, wxyz_quat, yaw, pitch, roll, scale, )
        trans_matrix = homogeneous_transform(*args)

        # Apply the transform
        self._scene[scene_path].set_transform(trans_matrix)

    def _read_transform_kwargs(self, kwargs):
        """
        Reads and sanitizes the kwargs for function self.set_transform.

        Parameters
        ----------
        kwargs : dict
            The kwargs being sanitized.

        Returns
        -------
        trans : dict
            The sanitized kwargs with default values set for non user defined
            keys.

        """
        # Set the default transform kwargs
        trans = {'position' : (0.0, 0.0, 0.0),
                 'wxyz_quat' : (1.0, 0.0, 0.0, 0.0),
                 'yaw' : 0.0,
                 'pitch' : 0.0,
                 'roll' : 0.0,
                 'scale' : (1.0, 1.0, 1.0)}

        # Validate each key given by user
        for key, val in kwargs.items():
            k = key.lower()

            # Ensure is valid kwarg
            if not k in trans:
                msg = f"{key} is not recognized kwarg."
                warn(msg)
                continue

            # Validate the position
            if k == 'position':
                if not is_nvector(val, 3):
                    msg = f"{key} must be 3vec of floats"
                    warn(msg)
                    continue
                trans[key] = val

            # Validate the wxyz_quat
            elif k == 'wxyz_quat':
                if not is_nvector(val, 4):
                    msg = f"{key} must be 4vec of floats"
                    warn(msg)
                    continue
                trans[key] = tuple(float(np.clip(c, -1.0, 1.0)) for c in val)

            # Validate the roll, pitch, and yaw
            elif k in ('yaw', 'pitch', 'roll'):
                if not is_num(val):
                    msg = f"{key} must be float"
                    warn(msg)
                    continue
                trans[key] = float(np.clip(val, -180.0, 180.0))

            # Validate the scale
            elif k in 'scale':
                if not is_nvector(val, 3):
                    msg = f"{key} must be 3vec of floats"
                    warn(msg)
                    continue
                trans[key] = tuple(float(np.clip(s, 0.0, np.inf)) for s in val)
        return trans

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
        # Sanitize the name
        if not name_valid(name):
            msg = "When add_obj, name is not valid."
            warn(msg)
            return -1

        # Ensure the object whose transform is being set is alread a
        # member of the scene
        if not get_scene_path(name) in self._objects:
            msg = f"While set_transform, {name} is unset object."
            warn(msg)
            return -1

        # Sanitize the kwargs
        transform = self._read_transform_kwargs(kwargs)

        # Queue the transform action in thread safe manner
        with self._LOCK:
            fnc = self._set_transforms
            args = (name, transform, )
            if fnc in self._actions_buf:
                new_name = True
                for i, a in enumerate(self._actions_buf[fnc]):
                    if a[0] == name:
                        self._actions_buf[fnc][i] = args
                        new_name = False
                        break
                if new_name:
                    self._actions_buf[fnc].append(args)
            else:
                self._actions_buf[fnc] = [args, ]
        return 0

    def _read_material_kwargs(self, kwargs):
        """
        Reads and sanitizes the kwargs for function self.set_material.

        Parameters
        ----------
        kwargs : dict
            The kwargs being sanitized.

        Returns
        -------
        san : dict
            The sanitized kwargs with default values set for non user defined
            keys.

        """
        # Set default values
        san = {'tex_path' : None,
               'color' : (1.0, 1.0, 1.0),
               'reflectivity' : 0.2,
               'shininess' : 0.01,
               'opacity' : 1.0,}

        # Validate each key given by user
        for key, val in kwargs.items():
            k = key.lower()

            # Ensure is valid kwarg
            if not k in san:
                msg = f"{key} is not recognized kwarg."
                warn(msg)
                continue

            # Validate the texture path
            if k == 'tex_path':
                if val is None:
                    san[key] = val
                    continue
                if not path_valid(val):
                    msg = f"{key} is invalid texture path."
                    warn(msg)
                    continue
                if not val.endswith('.png'):
                    msg = f"{key} must point to .png."
                    warn(msg)
                    continue
                san[key] = val

            # Validate the color
            elif k == 'color':
                if not is_nvector(val, 3):
                    msg = f"{key} must be 3vector of floats"
                    warn(msg)
                    continue
                san[key] = tuple(float(np.clip(c, 0.0, 1.0)) for c in val)

            # Validate the reflectivity and opacity
            elif k in ('reflectivity', 'opacity'):
                if not is_num(val):
                    msg = f"{key} must be float"
                    warn(msg)
                    continue
                san[key] = float(np.clip(val, 0.0, 1.0))

            # Validate the shininess
            elif k == 'shininess':
                if not is_num(val):
                    msg = f"{key} must be float"
                    warn(msg)
                    continue
                san[key] = float(np.clip(val*500.0, 0.0, 500.0))

        return san

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
            Defines the material. Has keys 'tex_path', 'color', 'reflectivity', 
            'shininess', and 'opacity'.

        Returns
        -------
        None.

        """
        scene_path = get_scene_path(name)
        geometry = self._objects[scene_path]
        material = self._get_material(**material_kwargs)
        self._scene[scene_path].set_object(geometry, material)

    def _set_materials(self, *args):
        """
        Sets the materials of multiple objects

        Parameters
        ----------
        *args 
            List of args to self._set_material

        Returns
        -------
        None.

        """
        for arg in args:
            self._set_material(*arg)

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
            reflectivity : float, optional
                The reflectivity of the object. Ranges from 0.0 to
                1.0. The default value is 0.2
            shininess : float, optional
                The shininess of the object. Ranges from 0.0 to 1.0.
                The default value is 0.01
            opacity : float, optional
                The opacity of the object. Ranges from 0.0 to 1.0.
                The default value is 1.0.
        Returns
        -------
        None.

        """
        # Sanitize the name
        if not name_valid(name):
            msg = "When add_obj, name is not valid."
            warn(msg)
            return -1

        # Ensure the object whose material is being set is alread a
        # member of the scene
        if not get_scene_path(name) in self._objects:
            msg = f"While set_material, {name} is unset object."
            warn(msg)
            return -1

        # Sanitize the kwargs
        material_kwargs = self._read_material_kwargs(kwargs)

        # Queue the transform action in thread safe manner
        with self._LOCK:
            fnc = self._set_materials
            args = (name, material_kwargs, )
            if fnc in self._actions_buf:
                new_name = True
                for i, a in enumerate(self._actions_buf[fnc]):
                    if a[0] == name:
                        self._actions_buf[fnc][i] = args
                        new_name = False
                        break
                if new_name:
                    self._actions_buf[fnc].append(args)
            else:
                self._actions_buf[fnc] = [args, ]
        return 0

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

        if not self._socket.closed:
            self._scene.delete()
            self._socket.close()
            return 0
        return -1
