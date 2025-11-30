# -*- coding: utf-8 -*-
"""
This module provides an example usage case for the animator. Here we create
a figure with 2 plots, 1 line plot and 1 bar chart. On the line plot, we plot
2 lines. On the bar chart, we show 4 bars.
"""
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

import time
from condynsate import Animator
import numpy as np

if __name__ == "__main__":
    # Create the animator with a frame rate of 20 fps
    animator = Animator(frame_rate=20.0, record=False)

    # Create the line plot. add_lineplot will return a list of length
    # n_lines. In the case n_lines==1, then it will return a single value.
    # These values are used to index each line on the line plot.
    lines = animator.add_lineplot(
        n_lines=2, #add 2 lines to the plot
        color=['k','g'], #color then black and green
        tail=[33, -1], #give one a tail and the other infinite length
        line_style=['-','--'], #set the line styles to solid and dashed
        label=['Line 1','Line 2'], #name one Line 1 and the other Line 2
        y_lim=[0,1], #set the y limits of the plot to 0 to 1
        x_label='Index', #set the x axis label of the plot
        y_label='Value', #set the y axis label of the plot
        )

    # Create the bar chart. Similarly, add_barchart returns either a list of
    # length n_bars or a single value.
    bars = animator.add_barchart(
        n_bars=4, #add 4 bars to the chart
        color=['k', 'r', 'g', 'b'], #color each bar a unique color
        label=['Bar 1','Bar 2','Bar 3','Bar 4'], #name each bar
        x_lim=[-1.1, 1.1], #set the x limits of the plot to -1.1 to 1.1
        x_label='Number', #set the x axis label of the plot
        v_zero_line=True #add a thin vertical line at x=0
        )

    # Start the animator. This will open the GUI and start the rendering thread
    animator.start()

    N = 200
    for i in range(N):
        percent_done = i/(N-1)

        # Add a point at the current step index to both lines in the line plot
        animator.lineplot_append_point(lines[0], i, np.random.rand())
        animator.lineplot_append_point(lines[1], i, percent_done**2)

        # Set the values of each bar in the bar chart to something
        if i % 4 == 0: #Every fourth step update the bars
            animator.barchart_set_value(bars[0], percent_done**2)
            animator.barchart_set_value(bars[1], -percent_done**3)
            animator.barchart_set_value(bars[2], np.sin(np.pi*percent_done))
            animator.barchart_set_value(bars[3], np.random.rand()*2-1)
        time.sleep(0.05)

    # When done, terminate the animator to gracefully exit all children threads
    animator.terminate()
