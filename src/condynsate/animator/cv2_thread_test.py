import cv2
import time
import numpy as np
from threading import (Thread, Lock)


class Test:
    def __init__(self, name='TEST'):
        self.lock = Lock()
        self.done = False
        self.img = (np.random.rand(240,320,3)*255).astype(np.uint8)
        self.name = name
    
    def __del__(self):
        self.stop()
            
    def _destory_window(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    def _run_thread(self):
        while True:
            with self.lock:
                if not self.done:
                    cv2.imshow(self.name, self.img)
                    cv2.waitKey(int(1000.*(1.0/10.0)))
                else:
                    break
        self._destory_window()
    
    def start(self):
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        time.sleep(0.1)
        self.thread = Thread(target=self._run_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def update_image(self, img):
        with self.lock:
           self.img = img
    
    def stop(self):
        with self.lock:
            if not self.done:
                self.done = True
            else:
                return
        self.thread.join()


if __name__ == "__main__":
    test = Test()
    test.start()
    for i in range(100):
        img = (np.random.rand(240,320,3)*255).astype(np.uint8)
        test.update_image(img)
        time.sleep(0.03)
    test.stop()
    