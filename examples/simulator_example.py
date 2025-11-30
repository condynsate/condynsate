# -*- coding: utf-8 -*-
"""
This module provides an example usage case of the Simulator class. Here we
simulate a cart with an inverted pendulum atop it. Because this is just an
example for the Simulator, no visualization is included.
"""
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

import time
from condynsate import Simulator
from condynsate import __assets__ as assets
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create an instance of the simulator.
    sim = Simulator(gravity=(0.0, 0.0, -9.81), dt=0.001)

    # Load a plane for the ground. This this case, fixed means that the
    # base of the ground will not be affected by external forces
    ground = sim.load_urdf(assets['plane_big.urdf'], fixed=True)

    # Add the cart from the default condynsate assets to the simulator.
    # Set its initial position such that the wheels start 0.001 meters above
    # The ground plane
    cart = sim.load_urdf(assets['cart.urdf'])
    cart.set_initial_state(position=(0,0,0.251))

    # Set the pendulum joint to some non-zero initial angle
    cart.joints['chassis_to_arm'].set_initial_state(angle=0.001)

    # Run a 5 second simulation loop
    start = time.time()
    pendulum_angle = []
    cart_x_pos = []
    simtime = []
    while sim.time < 5.0:
        # Note the angle of the pendulum joint at each time step
        # Note the x coordinate of the cart at each time step
        # Note the simulation time at each step
        pendulum_angle.append(cart.joints['chassis_to_arm'].state.angle)
        cart_x_pos.append(cart.state.position[0])
        simtime.append(sim.time)

        # Apply a small force the the center of mass of the cart
        cart.apply_force((-0.0275, 0.0, 0.0))

        # Attempt a simulation step. If something has gone wrong,
        # break the simulation loop
        if sim.step(real_time=False) != 0:
            break

    # Note the terminate angle, position, and time
    pendulum_angle.append(cart.joints['chassis_to_arm'].state.angle)
    cart_x_pos.append(cart.state.position[0])
    simtime.append(sim.time)

    # Print how long the simulation took in real time
    print(f"Simulation took: {(time.time() - start):.2f} seconds")

    # When done, the terminate command will ensure graceful exit of all
    # children threads
    sim.terminate()

    # Plot the results
    fig, axes = plt.subplots(nrows=2,ncols=1)
    axes[0].plot(simtime, cart_x_pos)
    axes[0].set_xlabel('Simulation Time [seconds]')
    axes[0].set_ylabel('Cart x-position [meters]')
    axes[1].plot(simtime, pendulum_angle)
    axes[1].set_xlabel('Simulation Time [seconds]')
    axes[1].set_ylabel('Pendulum Angle [rad]')
    fig.tight_layout()
    plt.show()
