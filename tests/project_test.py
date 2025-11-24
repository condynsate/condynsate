"""
Simulates an autonomously controlled inverted pendulum on a 4 wheeled cart.
Used to test the operation of condynsate.
"""

import time
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

if __name__ == "__main__":
    # Create the project
    proj = Project(visualizer=True, animator=False,
                   visualizer_record=False, animator_record=False)

    # Turn off the axes and grid visualization
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)

    # Load the ground
    ground = proj.load_urdf(assets['plane_medium'], fixed=True)
    ground.links['plane'].set_texture(assets['concrete_img'])

    # Load the left wall
    left_wall = proj.load_urdf(assets['plane_medium'], fixed=True)
    left_wall.links['plane'].set_texture(assets['bricks_img'])
    left_wall.set_initial_state(roll=1.5708, yaw=1.5708, position=(-10,0,10))

    # Load the right wall
    right_wall = proj.load_urdf(assets['plane_medium'], fixed=True)
    right_wall.links['plane'].set_texture(assets['bricks_img'])
    right_wall.set_initial_state(roll=1.5708, yaw=1.5708, position=(10,0,10))

    # Load the back wall
    back_wall = proj.load_urdf(assets['plane_medium'], fixed=True)
    back_wall.links['plane'].set_texture(assets['bricks_img'])
    back_wall.set_initial_state(roll=1.5708, position=(0,10,10))

    # Load the cart
    cart = proj.load_urdf(assets['cart'])
    cart.set_initial_state(position=(0,0,0.251))
    cart.joints['chassis_to_arm'].set_initial_state(angle=0.25)

    # Position the camera and focus on the cart
    proj.visualizer.set_cam_position((0, -6, 2.5))
    proj.visualizer.set_cam_target(cart.state.position)

    # Reset the project to its initial state
    proj.reset()

    # Store the name of each wheel joint for easy iteration
    wheel_joint_names = ('chassis_to_wheel_1', 'chassis_to_wheel_2',
                         'chassis_to_wheel_3', 'chassis_to_wheel_4',)

    # Set the controls constants
    k = np.array([[ -2., -0.1, -10., -0.1]])
    m_e = np.zeros(4)
    n_e = np.zeros(1)

    # Run a 5 second simulation
    start = time.time()
    while proj.time <= 5.0:
        # Read the states of the pendulum and each wheel
        pendulum_state = cart.joints['chassis_to_arm'].state
        wheel_states = tuple(cart.joints[n].state for n in wheel_joint_names)

        # Calculate the torque required to keep pendulum upright
        m = np.array([pendulum_state.omega,
                      np.mean([s.omega for s in wheel_states]),
                      pendulum_state.angle,
                      np.mean([s.angle for s in wheel_states])])
        torque = float(np.clip((-k@(m - m_e) + n_e)[0], -0.75, 0.75))

        # Apply the torque we calculated
        for joint_name in wheel_joint_names:
            cart.joints[joint_name].apply_torque(torque,
                                                 draw_arrow=True,
                                                 arrow_scale=10.0)

        # Take a simulation step
        proj.step()

    # Note how long the simulation took
    print(f"Simuation ran for {time.time()-start:.2f} seconds.")

    # Terminate the project
    proj.terminate()
