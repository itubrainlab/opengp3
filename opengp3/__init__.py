"""OpenGP3: a macOS (user-space, libusb) driver and gaze tracker for the
Gazepoint GP3 eye tracker.

High level::

    from opengp3 import GazeTracker, Screen

    screen = Screen.detect(width_cm=53, distance_cm=65)
    with GazeTracker(screen) as gt:
        gt.calibrate(points=9)              # fullscreen calibration
        print(gt.validate().report())       # accuracy in degrees
        gt.calibration.save("cal.json")
        for s in gt.gaze(duration=10):      # every frame, ~60 Hz
            if s.valid:
                print(s.x, s.y)             # screen points

Low level: ``GP3Camera`` (registers, frames, IR LEDs), ``detect_eyes`` and
``EyeTracker`` (pupil/glint features per frame).
"""
from .calibration import (VALIDATION_POINTS, Calibration, CalibrationError, Mapping,
                          ValidationResult, grid_points)
from .camera import GP3Camera, GP3Error, GP3NotFoundError, LibUSBNotFoundError
from .detection import EyeFeature, EyeTracker, detect_eyes
from .display import Cancelled, StimulusWindow
from .filters import OneEuroFilter
from .screen import Rect, Screen
from .tracker import GazeSample, GazeTracker

__version__ = "0.1.0"

__all__ = [
    "GazeTracker", "GazeSample", "Screen", "Rect", "Calibration", "CalibrationError",
    "ValidationResult", "Mapping", "grid_points", "VALIDATION_POINTS", "OneEuroFilter",
    "GP3Camera", "GP3Error", "GP3NotFoundError", "LibUSBNotFoundError",
    "EyeFeature", "EyeTracker", "detect_eyes", "StimulusWindow", "Cancelled",
]
