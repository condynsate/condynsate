import cv2
import time
import numpy as np
from copy import copy
from threading import (Thread, Lock)


class Test:
    def __init__(self, name='TEST', pix=(0.1, 0.5, 0.9), frame_rate=5.0):
        self.name = name
        self.pix = pix
        self.frame_rate = frame_rate
        
        self._lock = Lock()
        
        self._done = True
        self._prev_draw_time = time.time()
        self._prev_draw_pix = copy(self.pix)
        
        self._img = np.zeros((240, 320, 3), dtype=np.uint8)
        self._draw_image()
        self._update_ready = True
    
    def __del__(self):
        self.stop()
    
    def _draw_image(self):
        self._img[:,:,0] = np.clip(self.pix[0]*255., 0., 255.).astype(np.uint8)
        self._img[:,:,1] = np.clip(self.pix[1]*255., 0., 255.).astype(np.uint8)
        self._img[:,:,2] = np.clip(self.pix[2]*255., 0., 255.).astype(np.uint8)
        self._update_ready = True
    
    def _start_drawer(self):
        while True:
            c1 = time.time() - self._prev_draw_time >= 1.0 / self.frame_rate
            c2 = not self.pix == self._prev_draw_pix
            
            if c1 and c2:
                with self._lock:
                    self._draw_image()
                    self._prev_draw_time = time.time()
                    self._prev_draw_pix = copy(self.pix)
                    
            else:
                time.sleep(0.01 * (1.0 / self.frame_rate))
                
            with self._lock:
                if self._done:
                    break
    
    def _destory_window(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    def start(self):
        cv2.startWindowThread()
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        self._done = False
        time.sleep(0.1)
        self.update_image(self.pix)
        self._thread = Thread(target=self._start_drawer)
        self._thread.daemon = True
        self._thread.start()
    
    def update_image(self, pix):
        with self._lock:
            self.pix = pix
            if not self._done and self._update_ready:
                cv2.imshow(self.name, self._img)
                cv2.waitKey(1)
    
    def stop(self):
        with self._lock:
            if not self._done:
                self._done = True
                join = True
            else:
                join = False
        self._destory_window()
        if join:
            self._thread.join()


if __name__ == "__main__":
    test = Test()
    test.start()
    for i in range(100):
        pix = (np.random.rand(), np.random.rand(), np.random.rand())
        test.update_image(pix)
    test.stop()