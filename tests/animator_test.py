import time
from condynsate import Animator
import numpy as np

if __name__ == "__main__":
    animator = Animator(frame_rate=None, record=True)
    lines_1 = animator.add_lineplot(n_lines = 2, color = ['k', 'g'],
                                    tail = [20, -1] , y_lim = [0,1],
                                    line_style=['-', '--'],
                                    x_label = 'Index', y_label = 'Value',
                                    label = ['Line 1', 'Line 2'])
    bars_1 = animator.add_barchart(n_bars = 4, color = ['k', 'r', 'g', 'b'],
                                   x_lim = [-1.1, 1.1],
                                   label = ['B1','B2','B3','B4'],
                                   x_label = 'Number', v_zero_line=True)
    animator.start()
    for i in range(30):
        time.sleep(0.001*np.random.rand())
        animator.lineplot_append_point(i, np.random.rand(), lines_1[0])
        time.sleep(0.0005*np.random.rand())
        animator.lineplot_append_point(i, (i/99)**2, lines_1[1])
        time.sleep(0.1*np.random.rand())
        if i % 4 == 0:
            animator.barchart_set_value((i/99)**2, bars_1[0])
            time.sleep(0.0025*np.random.rand())
        if (i+1) % 4 == 0:
            animator.barchart_set_value(-(i/99)**3, bars_1[1])
            time.sleep(0.01*np.random.rand())
        if (i+2) % 4 == 0:
            animator.barchart_set_value(np.sin(np.pi*3*(i/99)), bars_1[2])
            time.sleep(0.00001*np.random.rand())
        if (i+3) % 4 == 0:
            animator.barchart_set_value(np.random.rand()*2-1, bars_1[3])
            time.sleep(0.015*np.random.rand())
    animator.terminate()
