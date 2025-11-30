# -*- coding: utf-8 -*-
"""
This module contains condynsate unique exception types.
"""
"""
© Copyright, 2025 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

class InvalidNameException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload

    def __str__(self):
        return str(self.message)


class VisualizerClosedError(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload

    def __str__(self):
        return str(self.message)
