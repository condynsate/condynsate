import time
from condynsate import Animator
import numpy as np

if __name__ == "__main__":
    animator = Animator(frame_rate=None, record=False)
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
    start = time.time()
    for i in range(100):
        animator.lineplot_append_point(lines_1[0], i, np.random.rand())
        animator.lineplot_append_point(lines_1[1], i, (i/99)**2)
        if i % 4 == 0:
            animator.barchart_set_value(bars_1[0], (i/99)**2)
        if (i+1) % 4 == 0:
            animator.barchart_set_value(bars_1[1], -(i/99)**3)
        if (i+2) % 4 == 0:
            animator.barchart_set_value(bars_1[2], np.sin(np.pi*3*(i/99)))
        if (i+3) % 4 == 0:
            animator.barchart_set_value(bars_1[3], np.random.rand()*2-1)
    print(f'Elapsed: {time.time()-start:.2f}')
    animator.terminate()
