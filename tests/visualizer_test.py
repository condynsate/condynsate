import os
import time
import condynsate
import numpy as np

if __name__ == "__main__":
    PATH = r'C:\Users\Grayson\Docs\Repos\condynsate\src\condynsate\__assets__'
    vis = condynsate.Visualizer(record=True)
    vis.set_axes(False)
    vis.add_object('Ground', os.path.join(PATH, 'plane_med.obj'),
                   tex_path=os.path.join(PATH, 'concrete.png'))
    vis.add_object('Cube', os.path.join(PATH, 'cube.stl'),
                   color=(0.121, 0.403, 0.749),
                   scale=(0.5, 0.5, 0.5),
                   position=(0., 0., 0.25))

    N = 2000
    P0 = np.array([0., 0., 0.25])
    P1 = np.array([-1., 5., 2.25])
    for i in range(N):
        t = i / (N-1)

        p = (1-t)*P0 + t*P1
        roll = 180*np.sin(t)
        pitch = 180*np.cos(3*t)
        yaw = 90*(np.sin(5*t) + np.cos(7*t))
        vis.set_transform('Cube', position=p, scale=(0.5, 0.5, 0.5),
                          roll=roll, pitch=pitch, yaw=yaw)

        color = (0.121+np.sin(7*t),
                 0.403+np.cos(11*t),
                 0.749+(np.sin(13*t)+np.cos(17*t)))
        color = tuple(float(max(0,min(c,1))) for c in color)
        vis.set_material('Cube', color=color)

        vis.set_cam_target(p)
        time.sleep(0.005)
        
    vis.terminate()