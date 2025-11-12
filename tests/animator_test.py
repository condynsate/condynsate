import time
import condynsate
import numpy as np

if __name__ == "__main__":
    PATH = r'C:\Users\Grayson\Docs\Repos\condynsate\src\condynsate\__assets__'
    ani = condynsate.Animator(frame_rate=20, record=True)
    lines1 = ani.add_lineplot(2, y_lim=[-1,1], h_zero_line=True, tail=(0, 10),
                             title='Plot 1',
                             x_label='Step', y_label='Value',
                             label=('Line1', 'Line2'), color=('r', 'b'), 
                             line_width=2.5, line_style=('dotted', 'solid'))
    lines2 = ani.add_lineplot(1, title='Plot 2', 
                              x_label='Step', y_label='Value',
                              color='k', line_width=3.5, 
                              x_lim=[0,1], y_lim=[0,1])
    bars = ani.add_barchart(4, x_lim=[-1, 1], v_zero_line=True, title='Plot 3',
                            x_label='Value', 
                            label=('Bar1', 'Bar2', 'Bar3', 'Bar4'),
                            color=('r', 'g', 'b', 'k'))

    ani.start()
    N = 30
    for i in range(N):
        t = i / (N-1)
        ani.barchart_set_value(t, bars[0])
        ani.barchart_set_value(-t, bars[1])
        ani.barchart_set_value(np.sin(4*2*np.pi*t), bars[2])
        ani.barchart_set_value(np.cos(4*2*np.pi*t), bars[3])
        ani.lineplot_append_point(i, np.cos(4*2*np.pi*t), lines1[0])
        ani.lineplot_append_point(i, np.sin(4*2*np.pi*t), lines1[1])
        ani.lineplot_append_point(t, t**2, lines2[0])
        time.sleep(0.01)
    ani.terminate()