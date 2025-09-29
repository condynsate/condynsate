import cv2
import time
import numpy as np
from copy import copy
from threading import (Thread, Lock)


class Test:
    def __init__(self, num=0, frame_rate=5.0):
        self.name = 'Animator'
        self.num = num
        self.frame_rate = frame_rate
        
        self._lock = Lock()
        
        self._done = True
        self._prev_draw_time = time.time()
        self._prev_num = copy(self.num)
        
        self._img = np.zeros((480, 640, 3), dtype=np.uint8)
        self._draw_image()
        self._update_ready = True
    
    def __del__(self):
        self.stop()
    
    def _draw_image(self):
        self._img[:,:,0] = np.uint8(0)
        self._img[:,:,1] = np.uint8(0)
        self._img[:,:,2] = np.uint8(0)
        
        text = str(self.num)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        thickness = 15
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = int(np.round(0.5 * (640 - w)))
        y = int(np.round(0.5 * (480 - h) + h))
        
        self._img = cv2.putText(self._img, text, (x, y), font, font_scale, 
                                color, thickness, cv2.LINE_AA, False)
        self._update_ready = True
    
    def _start_drawer(self):
        while True:
            c1 = time.time() - self._prev_draw_time >= 1.0 / self.frame_rate
            c2 = not self.num == self._prev_num
            
            if c1 and c2:
                with self._lock:
                    self._draw_image()
                    self._prev_draw_time = time.time()
                    self._prev_num = copy(self.num)
                    
            else:
                time.sleep(0.01 * (1.0 / self.frame_rate))
                
            with self._lock:
                if self._done:
                    break
    
    def _destory_window(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    def start(self):
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        self._done = False
        time.sleep(0.1)
        self.update_image(self.num)
        self._thread = Thread(target=self._start_drawer)
        self._thread.daemon = True
        self._thread.start()
    
    def update_image(self, num):
        with self._lock:
            self.num = num
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
    for i in range(300):
        test.update_image(i+1)
    test.stop()