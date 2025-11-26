# -*- coding: utf-8 -*-
"""
This module provides the Keyboard class which monitors the keyboard and tells
users which keys are pressed.

@author: G. Schaer
"""

###############################################################################
#DEPENDENCIES
###############################################################################
import time
from copy import copy
from warnings import warn
from pynput.keyboard import (Listener, Key)

###############################################################################
#KEYS CLASS
###############################################################################
class Keyboard:
    """
    Keyboard asynchronously detects and reports keyboard events. Used to detect
    which keys are pressed and which modifiers are being used. Works best on
    QWERTY keyboards.
    """
    # Converts unicode ords that use shift on a qwerty keyboard to their
    # lowercase counterparts.
    _QWERTY_LOWER = {126: 96,
                     33: 49,
                     64: 50,
                     35: 51,
                     36: 52,
                     37: 53,
                     94: 54,
                     38: 55,
                     42: 56,
                     40: 57,
                     41: 48,
                     95: 45,
                     43: 61,
                     123: 91,
                     125: 93,
                     124: 92,
                     58: 59,
                     34: 39,
                     60: 44,
                     62: 46,
                     63: 47,}


    def __init__(self):
        """
        Constructor method.

        """
        # Create buffer to hold all keys that are currently down
        self._key_buf = []
        self._mod_buf = []


        # Start the keyboard listener
        self._listener = Listener(self._on_press, self._on_release)
        self._listener.start()
        self._listening = True


    def __del__(self):
        """
        Deconstructor func.
        """
        self.terminate()


    def get_pressed(self):
        """
        Returns a list of all keys that are currently pressed.

        Returns
        -------
        keys_pressed : list of strings
            The list of keys to be detected. Each item in list may be a
            lowercase letter, a digit, a special character, or a punctuation
            mark that does not use shift on a QWERTY keyboard.
            The special keys are as follows:
            "esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
            "f11", "f12", "print_screen", "scroll_lock", "pause", "backspace",
            "insert", "home", "page_up", "num_lock", "tab", "delete", "end",
            "page_down", "caps_lock", "enter", "up", "left", "down", "right"
            The punctuation marks that do not use shift on a QWERTY keyboard
            are as follows:
            "`", "-", "=", "[", "]", "\", ";", "'", ",", ".", "/"
            If one of the following modifiers are used:
            "shift", "alt", "ctrl", "cmd",
            each item in the list will prepend them following format:
            "shift+a", "ctrl+a", "alt+a", "cmd+a", "shift+ctrl+a", etc.

        """
        return self._read_bufs()


    def is_pressed(self, key_str):
        """
        Returns a boolean flag to indicate whether a desired key is pressed.
        The key may be alpha numeric or some special keys.

        Parameters
        ----------
        key_str : string
            The key to be detected. May be a lowercase letter, a digit, a
            special character, or a punctuation mark that does not use shift on
            a QWERTY keyboard.
            The special keys are as follows:
            "esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
            "f11", "f12", "print_screen", "scroll_lock", "pause", "backspace",
            "insert", "home", "page_up", "num_lock", "tab", "delete", "end",
            "page_down", "caps_lock", "enter", "up", "left", "down", "right"
            The punctuation marks that do not use shift on a QWERTY keyboard
            are as follows:
            "`", "-", "=", "[", "]", "\", ";", "'", ",", ".", "/"
            The following modifiers can also be used:
            "shift", "alt", "ctrl", "cmd".
            Modifiers are added with the following format:
            "shift+a", "ctrl+a", "alt+a", "shift+ctrl+alt+a", etc.

        Returns
        -------
        bool
            A boolean flag to indicate whether the desired key is pressed.

        """
        keys_pressed = self._read_bufs()
        return key_str in keys_pressed


    def await_press(self, key_str, timeout=None):
        """
        Waits until the user presses a specified key or until the timeout
        condition is met

        Parameters
        ----------
        key_str : string
            The key to await. May be a lowercase letter, a digit, a
            special character, or a punctuation mark that does not use shift on
            a QWERTY keyboard.
            The special keys are as follows:
            "esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
            "f11", "f12", "print_screen", "scroll_lock", "pause", "backspace",
            "insert", "home", "page_up", "num_lock", "tab", "delete", "end",
            "page_down", "caps_lock", "enter", "up", "left", "down", "right"
            The punctuation marks that do not use shift on a QWERTY keyboard
            are as follows:
            "`", "-", "=", "[", "]", "\", ";", "'", ",", ".", "/"
            The following modifiers can also be used:
            "shift", "alt", "ctrl", "cmd".
            Modifiers are added with the following format:
            "shift+a", "ctrl+a", "alt+a", "shift+ctrl+alt+a", etc.
        timeout : float > 0.0, optional
            The timeout value in seconds. The default is None.

        Returns
        -------
        ret_code: int
            0 is the key is pressed, -1 is the timeout value is reached before
            the key is pressed.

        """
        print(f"Awaiting '{key_str}' to be pressed...", flush=True, end='')
        start_time = time.time()
        while not self.is_pressed(key_str):
            if not timeout is None and time.time() - start_time >= timeout:
                print(f' Timed out after {timeout} seconds.', flush=True)
                return -1
            time.sleep(0.01)
        print(f" '{key_str}' pressed. Continuing", flush=True)
        return 0


    def _read_bufs(self):
        """
        Reads the current key buffers and returns all key strings as a list.

        Returns
        -------
        keys_pressed : list of strings
            The list of keys to be detected. Each item in list may be a
            lowercase letter, a digit, a special character, or a punctuation
            mark that does not use shift on a QWERTY keyboard.
            The special keys are as follows:
            "esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
            "f11", "f12", "print_screen", "scroll_lock", "pause", "backspace",
            "insert", "home", "page_up", "num_lock", "tab", "delete", "end",
            "page_down", "caps_lock", "enter", "up", "left", "down", "right"
            The punctuation marks that do not use shift on a QWERTY keyboard
            are as follows:
            "`", "-", "=", "[", "]", "\", ";", "'", ",", ".", "/"
            If one of the following modifiers are used:
            "shift", "alt", "ctrl", "cmd",
            each item in the list will prepend them following format:
            "shift+a", "ctrl+a", "alt+a", "cmd+a", "shift+ctrl+a", etc.

        """
        keys_pressed = []
        for k in self._key_buf:
            key_str = copy(k)
            for m in self._mod_buf:
                key_str = m + key_str
            keys_pressed.append(key_str)
        return keys_pressed


    def terminate(self):
        """
        Terminates the Keyboard asynchronous listener. Should be called
        when done with the keyboard.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Stop the listener
        if self._listening:
            self._listener.stop()
            self._listener.join()
            self._listening = False

        # Empty all buffers
        self._key_buf = []
        self._mod_buf = []
        return 0


    def _get_key_string(self, key):
        """
        Gets the unique key string for each alpha numeric key and some special
        keys.

        Parameters
        ----------
        key : pynput.keyboard.Key object
            The current key whose string is got.

        Returns
        -------
        key_str : string
            The associated key string.

        """
        # Handle non alphnumeric keys
        if isinstance(key, Key):
            # Collect the key's name
            name = key.name

            # Handle modifier keys
            if name in ['alt', 'alt_l', 'alt_r', 'alt_gr']:
                return 'alt+'
            if name in ['cmd', 'cmd_l', 'cmd_r']:
                return 'cmd+'
            if name in ['ctrl', 'ctrl_l', 'ctrl_r']:
                return 'ctrl+'
            if name in ['shift', 'shift_l', 'shift_r']:
                return 'shift+'

            # Handle all other non alphanumeric keys
            return name

        # Return the lower case alphnumeric string if possible
        try:
            # Get the detected char
            char = key.char
            char_ord = ord(char)

            # If the detected char is a C0 control code other than NUL,
            # convert it to the Latin Alphabet Uppercase or ASCII
            # Punctuation & Symbols char that was used to generate the control
            # code
            if char_ord >= 1 and char_ord <= 31:
                char = chr(char_ord + 64).lower()

            # Take the lower case of the char
            char = self._qwerty_lower(char)
            return char

        # If key is not recognized, return empty string
        except Exception:
            return ""


    def _qwerty_lower(self, char):
        """
        Lowers all chars based on QWERTY keyboard layout.

        Parameters
        ----------
        char : string
            Character being lowered.

        Returns
        -------
        lowered : string
            The lowered char.

        """
        char_ord = ord(char)
        if char_ord >= 65 and char_ord <= 90:
            return chr(char_ord + 32)
        elif char_ord in Keyboard._QWERTY_LOWER.keys():
            return chr(Keyboard._QWERTY_LOWER[char_ord])
        else:
            return char


    def _add_to_buffer(self, key):
        """
        Adds a pressed key to the correct key or mod buffer.

        Parameters
        ----------
        key : pynput.keyboard.Key object
            The key pressed.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Get the key string
        key_str = self._get_key_string(key)

        # If the key is not recognized, return err code
        if len(key_str)==0:
            return -1

        # Handle modifiers
        elif (key_str=="shift+" or key_str=="ctrl+" or
              key_str=="alt+" or key_str=="cmd+"):
            if key_str not in self._mod_buf:
                self._mod_buf.append(key_str)
            return 0

        # Handle other keys
        else:
            if key_str not in self._key_buf:
                self._key_buf.append(key_str)
            return 0


    def _on_press(self, key):
        """
        Takes place on a key down event. Stores the pressed key as a unique
        string and marks that a key is down.

        Parameters
        ----------
        key : pynput.keyboard.Key object
            The current key.

        Returns
        -------
        None.

        """
        # Add the key string to the key buffer
        if not self._add_to_buffer(key) == 0:
            m = 'Pressed key is not recognized. Ignoring.'
            warn(m, UserWarning)


    def _remove_from_buffer(self, key):
        """
        Removes a released key from the correct key or mod buffer.

        Parameters
        ----------
        key : pynput.keyboard.Key object
            The key released.

        Returns
        -------
        ret_code : int
            0 if successful, -1 if something went wrong.

        """
        # Get the key string
        key_str = self._get_key_string(key)

        # If the key is not recognized, return err code
        if len(key_str)==0:
            return -1

        # Handle modifiers
        elif (key_str=="shift+" or key_str=="ctrl+" or
              key_str=="alt+" or key_str=="cmd+"):
            if key_str in self._mod_buf:
                idx = self._mod_buf.index(key_str)
                self._mod_buf.pop(idx)
            return 0

        # Handle other keys
        else:
            if key_str in self._key_buf:
                idx = self._key_buf.index(key_str)
                self._key_buf.pop(idx)
            return 0


    def _on_release(self, key):
        """
        Takes place on a key up event. Removes the pressed key string and
        marks that no keys are down. When 'esc' is pressed, automatically
        terminates the keyboard listener.

        Parameters
        ----------
        key : pynput.keyboard.Key object
            The current key.

        Returns
        -------
        bool
            False on termination event ('esc')

        """
        self._remove_from_buffer(key)
        return True
