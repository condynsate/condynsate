# -*- coding: utf-8 -*-
"""
This module gives example usage of the keyboard.

@author: G. Schaer
"""
import time
import condynsate

if __name__ == "__main__":
    # Create an instance of the keyboard. Instantiation wil automatically
    # start a keyboard listener thread
    keyboard = condynsate.Keyboard()

    print("Press 'esc' to end.")
    while True:
        # If the esc key is ever pressed, end the loop
        if keyboard.is_pressed('esc'):
            break

        # Get every pressed key. If any pressed keys are detected, print them
        pressed = keyboard.get_pressed()
        if len(pressed) > 0:
            print(f"Keys pressed: {pressed}")

        # Just to remove CPU strain
        time.sleep(0.01)

    # When done, terminate ensures graceful exit of the listener thread
    keyboard.terminate()
