"""
Simulates an autonomously controlled inverted pendulum on a 4 wheeled cart.
Used to test the operation of condynsate.
"""
import sys
import time
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

if __name__ == "__main__":
    # Create the project
    proj = Project(visualizer=True, animator=True, keyboard=True,
                   visualizer_record=True, animator_record=False)

    # Turn off the axes and grid visualization
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)

    # Load the ground
    ground = proj.load_urdf(assets['plane_medium'], fixed=True)
    ground.links['plane'].set_texture(assets['carpet_img'])

    # Load the left wall
    left_wall = proj.load_urdf(assets['half_plane_medium'], fixed=True)
    left_wall.links['plane'].set_texture(assets['window_wall_img'])
    left_wall.set_initial_state(roll=1.5708, yaw=1.5708, position=(-5,0,2.5))

    # Load the right wall
    right_wall = proj.load_urdf(assets['half_plane_medium'], fixed=True)
    right_wall.links['plane'].set_texture(assets['door_wall_img'])
    right_wall.set_initial_state(roll=1.5708, yaw=-1.5708, position=(5,0,2.5))

    # Load the back wall
    back_wall = proj.load_urdf(assets['half_plane_medium'], fixed=True)
    back_wall.links['plane'].set_texture(assets['classroom_wall_img'])
    back_wall.set_initial_state(roll=1.5708, position=(0,5,2.5))

    # Load the cart
    cart = proj.load_urdf(assets['cart'])
    cart.set_initial_state(position=(0,0,0.251))
    cart.joints['chassis_to_arm'].set_initial_state(angle=0.355)

    # Focus the camera on the cart
    proj.visualizer.set_cam_target(cart.center_of_mass)

    # Add a line plot to the animator that tracks the pendulum angle
    plot = proj.animator.add_lineplot(1, y_lim=(-90., 90.), title='Pendulum',
                                   x_label='Time [seconds]',
                                   y_label='Angle [degrees]',
                                   h_zero_line=True,
                                   color='b', line_width=2.5)

    # Store the name of each wheel joint for easy iteration
    wheel_joint_names = ('chassis_to_wheel_1', 'chassis_to_wheel_2',
                         'chassis_to_wheel_3', 'chassis_to_wheel_4',)

    # Set the controls constants
    k = np.array([[ -16.6, -0.6, -100.0, -0.16]])
    m_e = np.zeros(4)
    n_e = np.zeros(1)

    # Run simulations until user requests stop
    DONE = False
    while not DONE:
        # Reset the project to its initial state. This is required to
        # reset the simulation, reset the visualizer, and reset/start the
        # animator.
        proj.reset()

        # Ask the user to continue. Upon timeout, terminate and return
        if proj.await_keypress('enter') < 0:
            proj.terminate()
            sys.exit(0)

        # Run a 10 second simulation
        start = time.time()
        while proj.time <= 10.:
            # Read the states of the pendulum and each wheel
            pen_state = cart.joints['chassis_to_arm'].state
            wheel_states=tuple(cart.joints[n].state for n in wheel_joint_names)

            # If the pendulum angle exceeds 90 degrees, a failure condition is
            # met. Terminate the simulation.
            if abs(pen_state.angle) > 1.5708:
                print('The pendulum fell.')
                DONE = True
                break

            # Calculate the torque required to keep pendulum upright
            m = np.array([pen_state.omega,
                          np.mean([s.omega for s in wheel_states]),
                          pen_state.angle,
                          np.mean([s.angle for s in wheel_states])])
            torque = float(np.clip((-k@(m - m_e) + n_e)[0], -0.75, 0.75))

            # Apply the torque we calculated
            for joint_name in wheel_joint_names:
                offset = ('3' in joint_name or '4' in joint_name)*0.05-0.025
                cart.joints[joint_name].apply_torque(torque,
                                                     draw_arrow=True,
                                                     arrow_scale=0.33,
                                                     arrow_offset=offset)

            # Plot the pendulum angle
            proj.animator.lineplot_append_point(plot, proj.time,
                                                pen_state.angle*180./np.pi)

            # Take a simulation step
            proj.step(real_time=True, stable_step=False)

        # Note how long the simulation took
        print(f"Simuation ran for {time.time()-start:.2f} seconds.",flush=True)

        # Ask the user to continue. Upon timeout, terminate and return
        print("Press 'backspace' to repeat. Press 'esc' to quit.",
              flush=True, end='')
        while True:
            if proj.keyboard.is_pressed('backspace'):
                print(' Repeating.', flush=True)
                break
            if proj.keyboard.is_pressed('esc'):
                DONE = True
                print(' Quiting.', flush=True)
                break
            time.sleep(0.01)

    # Terminate the project
    proj.terminate()
