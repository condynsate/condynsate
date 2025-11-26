import time
import condynsate

if __name__ == "__main__":
    keyboard = condynsate.Keyboard()
    while True:
        if keyboard.is_pressed('esc'):
            break

        pressed = keyboard.get_pressed()
        if len(pressed) > 0:
            print(pressed)

        time.sleep(0.01)

    keyboard.terminate()