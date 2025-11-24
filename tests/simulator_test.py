import time
import numpy as np
from condynsate import Simulator
from condynsate import Visualizer
from condynsate import __assets__ as assets

if __name__ == "__main__":
    sim = Simulator(gravity=(0.0, 0.0, -9.81), dt=0.01)
    cart1 = sim.load_urdf(assets['cart'], fixed=False)
    cart1.set_initial_state(position=(2,4,0), omega=(1,2,3), pitch=0.75)

    vis = Visualizer()
    vis.set_axes(False)
    for body in sim.bodies:
        for d in body.visual_data:
            vis.add_object(**d)

    t = 10.0
    n = int(np.round(t / sim.dt))
    start = time.time()
    for i in range(n):
        cart1.apply_force((0., 0., 40.))
        barycenter = np.mean([b.center_of_mass for b in sim.bodies], axis=0)
        vis.set_cam_target(barycenter)

        if sim.step(real_time=True) != 0:
            break
        for body in sim.bodies:
            for d in body.visual_data:
                vis.set_transform(**d)

    print(f"Simulation duration: {(time.time() - start):.2f} seconds")

    sim.terminate()
    vis.terminate()
