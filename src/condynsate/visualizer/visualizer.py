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
from condynsate.misc import (format_path,  wxyz_from_euler)
from condynsate.exceptions import VisualizerClosedError


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
    def __init__(self, frame_rate=None, record=False):
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

        # Start the main thread
        self._LOCK = Lock()
        self._done = False
        self._last_refresh = cv2.getTickCount()
        self._start()

        # Set the default visibility of grid and axes
        self.set_grid()
        self.set_axes()

        # Set the default background color
        self.set_background(top=(0.44, 0.62, 0.82), bottom=(0.82, 0.62, 0.44))

        # Set the default lights
        self.set_spotlight()
        self.set_negx_light()
        self.set_posx_light()
        self.set_ambient_light()
        self.set_fill_light()

        # Set the default camera properties
        self.set_cam_position((3, 0.5, 2))
        self.set_cam_frustum(near=0.01, far=1000.0)

        # Wait for scene to fully load
        time.sleep(0.5)


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
        None.

        """
        # Continuously redraw
        while True:
            self.send()

            # Aquire mutex lock to read flag
            with self._LOCK:

                # If done flag is set, end main loop
                if self._done:
                    break

            # Remove CPU strain by sleeping for a little bit
            time.sleep(0.01)


    def send(self):
        pass


    def _assert_open(self):
        """
        Asserts that the visulizer socket is open and valid.

        Raises
        ------
        VisualizerClosedError
            If the socket is closed

        Returns
        -------
        None.

        """
        if self._socket.closed:
            message = "Cannot complete action because visualizer is closed."
            payload = self
            raise VisualizerClosedError(message, payload)


    def _is_num(self, arg):
        """
        Ensures that an argument is a number.

        Parameters
        ----------
        arg : TYPE
            The argument being tested.

        Returns
        -------
        is_num : bool
            A Boolean flag that indicates if arg is valid.

        """
        # If float castable, not inf, and not nan, is a number
        try:
            f = float(arg)
            return (not np.isinf(f)) and (not np.isnan(f))

        # If something went wrong, is not a number
        except Exception:
            return False


    def _is_3vector(self, arg):
        """
        Ensures that an argument is a 3vector of numbers.

        Parameters
        ----------
        arg : TYPE
            The argument being tested.

        Returns
        -------
        is_3vec : bool
            A Boolean flag that indicates if arg is valid.

        """
        try:
            iter(arg) # Ensure iterable
            if len(arg) != 3: # Ensure of length 3
                raise TypeError('Arg of wrong length')

            # Ensure each arg is number
            return all([self._is_num(a) for a in arg])

        # If something went wrong, arg is not a 3vector
        except Exception:
            return False


    def set_grid(self, visible=True):
        """
        Controls the visibility state of the XY grid in the visualizer.

        Parameters
        ----------
        visible : bool, optional
            The boolean value to which the visibility of the XY grid is set.
            The default is False.

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Raise an error if  the visualizer is closed
        self._assert_open()

        # Input sanitize
        if not isinstance(visible, bool):
            msg='When set_grid, visible must be boolean.'
            warn(msg)
            return -1

        # Set visibility
        self._scene["/Grid"].set_property("visible", visible)
        self._scene["/Grid/<object>"].set_property("visible", visible)


    def set_axes(self, visible=True):
        """
        Controls the visibility state of the axes in the visualizer.

        Parameters
        ----------
        visible : bool, optional
            The boolean value to which the visibility of the axes is set.
            The default is False.

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Raise an error if  the visualizer is closed
        self._assert_open()

        # Input sanitize
        if not isinstance(visible, bool):
            msg='When set_axes, visible must be boolean.'
            warn(msg)
            return -1

        # Set visibility
        self._scene["/Axes"].set_property("visible", visible)
        self._scene["/Axes/<object>"].set_property("visible", visible)
        return 0


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

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        Returns
        -------
        ret_code : int
            0 if successful, less than 0 if something went wrong.

        """
        # Raise an error if  the visualizer is closed
        self._assert_open()

        # Keep track of how well this goes.
        ret_code = 0

        # Ensure top is of the correct format
        if not self._is_3vector(top):
                msg='When set_background, top must be 3 tuple of floats.'
                warn(msg)
                ret_code -= 1

        # If top is correct format, set the top value
        else:
            top = [float(np.clip(float(t), 0.0, 1.0)) for t in top]
            self._scene["/Background"].set_property('top_color', top)


        # Ensure bottom is of the correct format
        if not self._is_3vector(bottom):
                msg='When set_background, bottom must be 3 tuple of floats.'
                warn(msg)
                ret_code -= 1

        # If bottom is correct format, set the bottom value
        else:
            bottom = [float(np.clip(float(b), 0.0, 1.0)) for b in bottom]
            self._scene["/Background"].set_property('bottom_color', bottom)

        return ret_code


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

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        TypeError
            If light is not a string.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Raise an error if  the visualizer is closed
        self._assert_open()

        # Check light argument
        if not isinstance(light, str):
            raise TypeError("When _set_light, argument light must be string.")

        # Check on argument
        if not on is None:
            if not isinstance(on, bool):
                warn("When setting light, argument on must be a boolean.")
                return -1

        # Check intensity argument
        if not intensity is None:
            try:
                intensity = float(intensity)
            except Exception:
                warn("When setting light, argument intensity must be a float.")
                return -1

        # Check distance argument
        if not distance is None:
            try:
                distance = float(distance)
            except Exception:
                warn("When setting light, argument distance must be a float.")
                return -1

        # Check shadow argument
        if not shadow is None:
            if not isinstance(shadow, bool):
                warn("When setting light, argument shadow must be a boolean.")
                return -1

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
            self._socket.send_multipart([
                cmd_data["type"].encode("utf-8"),
                cmd_data["path"].encode("utf-8"),
                umsgpack.packb(cmd_data)
            ])
            self._socket.recv()

        return 0


    def set_spotlight(self, on=False, intensity=0.8, distance=0, shadow=True):
        """
        Sets the properties of the spotlight in the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. The default is
            False.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. The default is 0.8
        distance : float [0. to 100.], optional
            Maximum range of the light. Default is 0 (no limit).
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. The
            default is True

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        name = 'SpotLight'
        return self._set_light(name, on, intensity, distance, shadow)


    def set_posx_light(self, on=True, intensity=0.4, distance=0, shadow=True):
        """
        Sets the properties of the point light along the positive x axis
        in the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. The default is
            True.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. The default is 0.4
        distance : float [0. to 100.], optional
            Maximum range of the light. Default is 0 (no limit).
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. The
            default is True

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        name = 'PointLightPositiveX'
        return self._set_light(name, on, intensity, distance, shadow)


    def set_negx_light(self, on=True, intensity=0.4, distance=0, shadow=True):
        """
        Sets the properties of the point light along the negative x axis
        in the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. The default is
            True.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. The default is 0.4
        distance : float [0. to 100.], optional
            Maximum range of the light. Default is 0 (no limit).
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. The
            default is True

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        name = 'PointLightNegativeX'
        self._set_light(name, on, intensity, distance, shadow)


    def set_ambient_light(self, on=True, intensity=0.6, shadow=True):
        """
        Sets the properties ambient light of the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. The default is
            True.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. The default is 0.6
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. The
            default is True

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        name = 'AmbientLight'
        distance = None
        self._set_light(name, on, intensity, distance, shadow)


    def set_fill_light(self, on=True, intensity=0.4, shadow=True):
        """
        Sets the properties fill light of the scene.

        Parameters
        ----------
        on : bool, optional
            Boolean flag that indicates if the light is on. The default is
            True.
        intensity : float [0. to 20.], optional
            Numeric value of the light's strength/intensity. The default is 0.4
        shadow : bool, optional
            Boolean flag that indicates if the light casts a shadow. The
            default is True

        Raises
        ------
        VisualizerClosedError
            If attempting to set property of closed visualizer.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        name = 'FillLight'
        distance = None
        self._set_light(name, on, intensity, distance, shadow)


    def set_cam_position(self, p):
        """
        # Set the XYZ position of camera.

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
        if not self._is_3vector(p):
            msg = "When set_cam_position, p must be 3 tuple of floats."
            warn(msg)
            return -1

        # Set the camera position in correct XZ-Y format
        path = "/Cameras/default/rotated/<object>"
        formatted = (float(p[0]), float(p[2]), -float(p[1])) #Camera is rotated
        self._scene[path].set_property('position', formatted)
        return 0


    def set_cam_target(self, t):
        """
        # Set the XYZ position of the point the camera is looking at.

        Parameters
        ----------
        t : 3Vec of floats
            The XYZ position the camera is looking at. Regardless of updates
            to the camera position, the camera will always look directly at
            this point.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Ensure argument is correct type
        if not self._is_3vector(t):
            msg = "When set_cam_target, t must be 3 tuple of floats."
            warn(msg)
            return -1

        # Set the camera target
        formatted = (float(t[0]), float(t[1]), float(t[2]))
        self._scene.set_cam_target(formatted)
        return 0


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
        if not self._is_num(zoom):
            warn("When set_cam_zoom, zoom must be float")
            return -1

        # Ensure zoom in (0, 100]
        formatted = np.clip(zoom, 0.0001, 100.0)

        # Set the zoom
        path = "/Cameras/default/rotated/<object>"
        self._scene[path].set_property('zoom', formatted)
        return 0


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
            0 if successful, less than 0 if something went wrong.

        """
        path = "/Cameras/default/rotated/<object>"
        ret_code = 0

        # Set the aspect ratio
        if not aspect is None:
            if not self._is_num(aspect):
                warn("When set_cam_frustum, aspect must be float")
                ret_code -= 1
            else:
                self._scene[path].set_property('aspect', aspect)

        # Set the vertical field of view
        if not fov is None:
            if not self._is_num(fov):
                warn("When set_cam_frustum, fov must be float")
                ret_code -= 1
            else:
                self._scene[path].set_property('fov', fov)

        # Set the distance to the near plane
        if not near is None:
            if not self._is_num(near):
                warn("When set_cam_frustum, near must be float")
                ret_code -= 1
            else:
                self._scene[path].set_property('near', near)

        # Set the distance to the far plane
        if not far is None:
            if not self._is_num(far):
                warn("When set_cam_frustum, far must be float")
                ret_code -= 1
            else:
                self._scene[path].set_property('far', far)

        # Set the aspect ratio
        return ret_code


    def terminate(self):
        """
        Terminates the visualizer's communication with the web browser.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        if not self._socket.closed:
            self._scene.delete()
            self._socket.close()
            return 0
        return -1




































    # def add_obj(self,
    #             urdf_name,
    #             link_name,
    #             obj_path,
    #             tex_path,
    #             scale=[1., 1., 1.],
    #             translate=[0., 0., 0.],
    #             wxyz_quaternion=[1., 0., 0., 0.]):
    #     """
    #     Adds a textured .obj to the visualization.

    #     Parameters
    #     ----------
    #     urdf_name : string
    #         The name of the urdf to which the .obj is being added as a link.
    #         URDF objects define robots or assemblies.
    #     link_name : string
    #         The name of the link.
    #     tex_path : string
    #         Relative path pointing to the .png file that provides the
    #         texture.
    #     scale : array-like, size (3,), optional
    #         The initial scaling along the three axes applied to the .obj.
    #         The default is [1., 1., 1.].
    #     translate : array-like, size (3,), optional
    #         The initial translation along the three axes applied to the .obj.
    #         The default is [0., 0., 0.].
    #     wxyz_quaternion : array-like, size (4,), optional
    #         The wxyz quaternion that defines the initial rotation applied to
    #         the .obj. The default is [1., 0., 0., 0.].

    #     Returns
    #     -------
    #     obj_geometry : meshcat.geometry.ObjMeshGeometry
    #         The object mesh.
    #     obj_texture : meshcat.geometry.MeshPhongMaterial
    #         The object texture.

    #     """
    #     # Get geometry of object from the .obj file at obj_path
    #     obj_path = format_path(obj_path)
    #     obj_geometry = geo.ObjMeshGeometry.from_file(obj_path)

    #     # Get the texture of object from the .png file at tex_path
    #     tex_path = format_path(tex_path)
    #     meshcat_png = geo.PngImage.from_file(tex_path)
    #     im_tex = geo.ImageTexture(image=meshcat_png,
    #                               wrap=[1, 1],
    #                               repeat=[1, 1])
    #     obj_texture = geo.MeshPhongMaterial(map = im_tex)

    #     # Calculate the transform
    #     transform = self._get_transform(scale, translate, wxyz_quaternion)

    #     # Add and transform the object to its orientation and position
    #     self.scene[urdf_name][link_name].set_object(obj_geometry, obj_texture)
    #     self.scene[urdf_name][link_name].set_transform(transform)

    #     # Return the geometry and texture
    #     return obj_geometry, obj_texture


    # def add_stl(self,
    #             urdf_name,
    #             link_name,
    #             stl_path,
    #             color = [91, 155, 213],
    #             transparent=False,
    #             opacity = 1.0,
    #             scale=[1., 1., 1.],
    #             translate=[0., 0., 0.],
    #             wxyz_quaternion=[1., 0., 0., 0.]):
    #     """
    #     Adds a colored .stl to the visualization.

    #     Parameters
    #     ----------
    #     urdf_name : string
    #         The name of the urdf to which the .stl is being added as a link.
    #         URDF objects define robots or assemblies.
    #     link_name : string
    #         The name of the link.
    #     stl_path : string
    #         The relative path pointing to the .stl description of the link that
    #         is being added.
    #     color : array-like, size (3,), optional
    #         The 0-255 RGB color of the .stl. The default is [91, 155, 213].
    #     transparent : boolean, optional
    #         A boolean that indicates if the .stl is transparent.
    #         The default is False.
    #     opacity : float, optional
    #         The opacity of the .stl. Can take float values between 0.0 and 1.0.
    #         The default is 1.0.
    #     scale : array-like, size (3,), optional
    #         The initial scaling along the three axes applied to the .stl.
    #         The default is [1., 1., 1.].
    #     translate : array-like, size (3,), optional
    #         The initial translation along the three axes applied to the .stl.
    #         The default is [0., 0., 0.].
    #     wxyz_quaternion : array-like, size (4,), optional
    #         The wxyz quaternion that defines the initial rotation applied to
    #         the .stl. The default is [1., 0., 0., 0.].

    #     Returns
    #     -------
    #     link_geometry : meshcat.geometry.StlMeshGeometry
    #         The link mesh.
    #     link_mat : meshcat.geometry.MeshPhongMaterial
    #         The link material.

    #     """
    #     # Set the parts's geometry
    #     stl_path = format_path(stl_path)
    #     link_geometry = geo.StlMeshGeometry.from_file(stl_path)

    #     # Set the parts's color
    #     color_int = color[0]*256**2 + color[1]*256 + color[2]

    #     # Set the parts's material
    #     link_mat = geo.MeshPhongMaterial(color=color_int,
    #                                      transparent=False,
    #                                      opacity=opacity,
    #                                      reflectivity=0.3)

    #     # Calculate the transform
    #     transform = self._get_transform(scale, translate, wxyz_quaternion)

    #     # Add and transform the link to its orientation and position
    #     self.scene[urdf_name][link_name].set_object(link_geometry, link_mat)
    #     self.scene[urdf_name][link_name].set_transform(transform)

    #     # Return the geometry and material
    #     return link_geometry, link_mat


    # def set_link_color(self,
    #                    urdf_name,
    #                    link_name,
    #                    link_geometry,
    #                    color = [91, 155, 213],
    #                    transparent = False,
    #                    opacity = 1.0):
    #     """
    #     Set a link color by deleting it and then adding another copy of it.

    #     Parameters
    #     ----------
    #     urdf_name : string
    #         The name of the urdf that contains the link being refreshed.
    #         URDF objects define robots or assemblies.
    #     link_name : string
    #         The name of the link.
    #     link_geometry : meshcat.geometry.StlMeshGeometry
    #         The link mesh.
    #     color : array-like, size (3,), optional
    #         The 0-255 RGB color of the link. The default is [91, 155, 213].
    #     transparent : boolean, optional
    #         A boolean that indicates if the link is transparent.
    #         The default is False.
    #     opacity : float, optional
    #         The opacity of the link. Can take float values between 0.0 and 1.0.
    #         The default is 1.0.

    #     Returns
    #     -------
    #     None.

    #     """
    #     # Set the parts's color
    #     color_int = color[0]*256**2 + color[1]*256 + color[2]
    #     link_mat = geo.MeshPhongMaterial(color=color_int,
    #                                      transparent=transparent,
    #                                      opacity=opacity,
    #                                      reflectivity=0.3)

    #     # Update the part's geometry and color
    #     self.scene[urdf_name][link_name].set_object(link_geometry, link_mat)


    # def _get_transform(self,
    #                    scale=[1., 1., 1.],
    #                    translate=[0., 0., 0.],
    #                    wxyz_quaternion=[1., 0., 0., 0.]):
    #     """
    #     Calculates the spatial transformation matrix that defines a 3D affine
    #     transformation inclunding scaling, translating, and rotating.

    #     Parameters
    #     ----------
    #     scale : array-like, size (3,), optional
    #         The scaling along the three axes. The default is [1., 1., 1.].
    #     translate : array-like, size (3,), optional
    #         The translation along the three axes. The default is [0., 0., 0.].
    #     wxyz_quaternion : array-like, size (4,), optional
    #         The wxyz quaternion that defines the rotation.
    #         The default is [1., 0., 0., 0.].

    #     Returns
    #     -------
    #     transform : array-like, size (4,4)
    #         The resultant 4x4 3D affine transformation matrix.

    #     """
    #     # Extract rotation data
    #     w = wxyz_quaternion[0]
    #     x = wxyz_quaternion[1]
    #     y = wxyz_quaternion[2]
    #     z = wxyz_quaternion[3]

    #     # Perform calculations used to transform quaternion to rotation matrix
    #     xx = 2.*x*x
    #     xy = 2.*x*y
    #     xz = 2.*x*z
    #     yy = 2.*y*y
    #     yz = 2.*y*z
    #     zz = 2.*z*z
    #     wx = 2.*w*x
    #     wy = 2.*w*y
    #     wz = 2.*w*z

    #     # Extract scale data
    #     s1 = scale[0]
    #     s2 = scale[1]
    #     s3 = scale[2]

    #     # Extract translate data
    #     t1 = translate[0]
    #     t2 = translate[1]
    #     t3 = translate[2]

    #     # Directly build the transform matrix
    #     H = np.array([[s1*(1.-yy-zz), s2*(xy-wz),    s3*(xz+wy),    t1],
    #                   [s1*(xy+wz),    s2*(1.-xx-zz), s3*(yz-wx),    t2],
    #                   [s1*(xz-wy),    s2*(yz+wx),    s3*(1.-xx-yy), t3],
    #                   [0.,            0.,            0.,            1.]])

    #     # Return the translation matrix
    #     return H


    # def apply_transform(self,
    #                     urdf_name,
    #                     link_name,
    #                     scale=[1., 1., 1.],
    #                     translate=[0., 0., 0.],
    #                     wxyz_quaternion=[1., 0., 0., 0.]):
    #     """
    #     Applies a 3D affine transformation inclunding scaling, translating,
    #     and rotating to a specified link.

    #     Parameters
    #     ----------
    #     urdf_name : string
    #         The name of the urdf being transformed.
    #     link_name : string
    #         The name of the link being transformed.
    #     scale : array-like, size (3,), optional
    #         The scaling along the three axes. The default is [1., 1., 1.].
    #     translate : array-like, size (3,), optional
    #         The translation along the three axes. The default is [0., 0., 0.].
    #     wxyz_quaternion : array-like, size (4,), optional
    #         The wxyz quaternion that defines the rotation.
    #         The default is [1., 0., 0., 0.].

    #     Returns
    #     -------
    #     transform : array-like, size (4,4)
    #         The 4x4 3D affine transformation matrix applied to the link.

    #     """

    #     # Calculate and apply the transform
    #     transform = self._get_transform(scale, translate, wxyz_quaternion)
    #     self.scene[urdf_name][link_name].set_transform(transform)

    #     # Return the transform
    #     return transform


    # def transform_camera(self,
    #                      scale = [1., 1., 1.],
    #                      translate = [0., 0., 0.],
    #                      wxyz_quaternion = [1., 0., 0., 0.],
    #                      roll=None,
    #                      pitch=None,
    #                      yaw=None):
    #     """
    #     Transforms the position, orientation, and scale of the camera
    #     in the scene.

    #     Parameters
    #     ----------
    #     scale : array-like, size (3,), optional
    #         The scaling of the camera view about the camera point along the
    #         three axes. The default is [1., 1., 1.].
    #     translate : array-like, size (3,), optional
    #         The translation of the camera point along the three axes.
    #         The default is [0., 0., 0.].
    #     wxyz_quaternion : array-like, shape (4,) optional
    #         A wxyz quaternion that describes the intial orientation of camera
    #         about the camera point. When roll, pitch, and yaw all have None
    #         type, the quaternion is used. If any roll, pitch, or yaw have non
    #         None type, the quaternion is ignored.
    #         The default is [1., 0., 0., 0.].
    #     roll : float, optional
    #         The roll of the camera about the camera point.
    #         The default is None.
    #     pitch : float, optional
    #         The pitch of the camera about the camera point.
    #         The default is None.
    #     yaw : float, optional
    #         The yaw of the camera about the camera point.
    #         The default is None.

    #     Returns
    #     -------
    #     None.

    #     """
    #     # If any euler angles are specified, use the euler angles to set the
    #     # initial orientation of the urdf object
    #     # Any unspecified euler angles are set to 0.0
    #     if roll!=None or pitch!=None or yaw!=None:
    #         if roll==None:
    #             roll=0.0
    #         if pitch==None:
    #             pitch=0.0
    #         if yaw==None:
    #             yaw=0.0
    #         wxyz_quaternion = wxyz_from_euler(roll, pitch, yaw)

    #     # Calculate and apply the transform
    #     transform = self._get_transform(scale=scale,
    #                                    translate=translate,
    #                                    wxyz_quaternion=wxyz_quaternion)
    #     self.scene["/Cameras/default"].set_transform(transform)


if __name__ == "__main__":
    vis = Visualizer()
    vis.terminate()
