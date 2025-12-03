# -*- coding: utf-8 -*-
"""
This module gives an example use case for the Project class. In this example,
we create a project that uses the visualizer, the animator, and the keyboard
simultaneously to render a cart keeping an inverted pendulum atop it upright.
"""
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

import time
from condynsate import Project
from condynsate import __assets__ as assets
import numpy as np

if __name__ == "__main__":
    # Create the project
    proj = Project(visualizer=True, animator=True, keyboard=True,
                   visualizer_record=False, animator_record=False)

    # Turn off the axes and grid visualization. Turn on the spotlight
    proj.visualizer.set_axes(False)
    proj.visualizer.set_grid(False)
    proj.visualizer.set_spotlight(on=True)

    # Load a plane with a carpet texture for the ground
    ground = proj.load_urdf(assets['plane_medium.urdf'], fixed=True)
    ground.links['plane'].set_texture(assets['carpet.png'])

    # Load and orient a plane with a windowed wall texture for the left wall
    left_wall = proj.load_urdf(assets['half_plane_medium.urdf'], fixed=True)
    left_wall.links['plane'].set_texture(assets['window_wall.png'])
    left_wall.set_initial_state(roll=1.5708, yaw=1.5708, position=(-5,0,2.5))

    # Load and orient a plane with a doored wall texture for the right wall
    right_wall = proj.load_urdf(assets['half_plane_medium.urdf'], fixed=True)
    right_wall.links['plane'].set_texture(assets['door_wall.png'])
    right_wall.set_initial_state(roll=1.5708, yaw=-1.5708, position=(5,0,2.5))

    # Load and orient a plane with a classroom wall texture for the back wall
    back_wall = proj.load_urdf(assets['half_plane_medium.urdf'], fixed=True)
    back_wall.links['plane'].set_texture(assets['classroom_wall.png'])
    back_wall.set_initial_state(roll=1.5708, position=(0,5,2.5))

    # Load and orient a cart carrying an inverted pendulum. Set the initial
    # angle of the pendulum to a non-zero angle.
    cart = proj.load_urdf(assets['cart.urdf'])
    cart.set_initial_state(position=(0,0,0.126))
    cart.joints['chassis_to_arm'].set_initial_state(angle=0.175)
    
    # Focus the camera on the cart
    proj.visualizer.set_cam_target(cart.center_of_mass)

    # Add a line plot to the animator to track the pendulum angle
    plot1 = proj.animator.add_lineplot(1, y_lim=(-30., 30.), title='Pendulum',
                                       x_label='Time [seconds]',
                                       y_label='Angle [degrees]',
                                       h_zero_line=True,
                                       color='b', line_width=2.5)

    # Add another line plot to the animator to track the cart x position
    plot2 = proj.animator.add_lineplot(1, y_lim=(-5., 5.), title='Cart',
                                   x_label='Time [seconds]',
                                   y_label='Position [meters]',
                                   h_zero_line=True,
                                   color='r', line_width=2.5)

    # Wait for just long enough for all the GUIs to update
    time.sleep(0.5)

    # Store the name of each wheel joint for easy iteration
    wheel_joint_names = ('chassis_to_wheel_1', 'chassis_to_wheel_2',
                         'chassis_to_wheel_3', 'chassis_to_wheel_4',)

    # Set control constants to keep the pendulum upright
    k = np.array([[ 1.5, -0.05, 6.0, -0.075]])
    m_e = np.zeros(4)
    n_e = np.zeros(1)

    # Run simulations until user requests to stop
    DONE = False
    while not DONE:
        # Reset the project to its initial state. This is required to
        # reset the simulation, reset the visualizer, and reset/start the
        # animator.
        proj.reset()

        # Ask the user to continue. If await_keypress is called but the project
        # has no animator, it raises and AttributeError. We can ignore this.
        try:
            proj.await_keypress('enter')
        except AttributeError:
            pass

        # Run a 10 second simulation loop
        start = time.time()
        while proj.simtime <= 10.:
            # Read the states of the pendulum and each wheel
            pen_state = cart.joints['chassis_to_arm'].state
            wheel_states=tuple(cart.joints[n].state for n in wheel_joint_names)

            # If the pendulum angle exceeds 90 degrees, a failure condition is
            # met. Terminate the simulation loop.
            if abs(pen_state.angle) > 1.5708:
                print('The pendulum fell.')
                break

            # Do controls calculations to determine what torque, when applied
            # to each wheel, will keep the pendulum upright.
            m = np.array([pen_state.omega,
                          np.mean([s.omega for s in wheel_states]),
                          pen_state.angle,
                          np.mean([s.angle for s in wheel_states])])
            torque = float(np.clip((-k@(m - m_e) + n_e)[0], -0.75, 0.75))

            # Apply the torque we calculated to each wheel
            for joint_name in wheel_joint_names:
                # This will offset a drawn torque arrow out of the center of
                # the wheels so we can see them. It is required to be
                # different between the front wheels (1 and 2) and the rear
                # wheels (3 and 4) because they are oriented 180 degrees apart
                offset = ('3' in joint_name or '4' in joint_name)*0.1-0.05
                cart.joints[joint_name].apply_torque(torque,
                                                     draw_arrow=True,
                                                     arrow_scale=0.67,
                                                     arrow_offset=offset)

            # Plot the pendulum angle against the current simulation time
            angle_deg = pen_state.angle*180./np.pi
            proj.animator.lineplot_append_point(plot1, proj.simtime, angle_deg)

            # Plot the cart's x position against the current simulation time
            cart_xpos = cart.state.position[0]
            proj.animator.lineplot_append_point(plot2, proj.simtime, cart_xpos)

            # Take a simulation step that attempts real time simulation
            proj.step(real_time=True, stable_step=False)

        # Note how long the simulation took
        print(f"Simuation took {time.time()-start:.2f} seconds.")

        # Ask the user to either continue or reset.
        print("Press 'backspace' to repeat. Press 'esc' to quit.")
        while True:
            # await_anykeys raises and AttributeError if there is no project
            # keyboard. If this is the case, catch and end.
            try:
                keys = proj.await_anykeys()
            except AttributeError:
                DONE = True
                break

            # If the user presses backspace, repeat the sim loop
            # else if the user presse esc, end the program
            if 'backspace' in keys:
                break
            if 'esc' in keys:
                DONE = True
                break

    # Terminate the project. This is required to save any videos that are
    # recorded and gracefully exit all the children threads.
    proj.terminate()
