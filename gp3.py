#!/usr/bin/env python3
"""Compatibility shim: the code now lives in the ``opengp3`` package.

    python gp3.py view      ==  opengp3 view
    import gp3              ->  re-exports GP3Camera, detect_eyes, EyeTracker, ...
"""
import sys

from opengp3.camera import *  # noqa: F401,F403
from opengp3.camera import _bit  # noqa: F401
from opengp3.detection import EyeFeature, EyeTracker, detect_eyes  # noqa: F401
from opengp3.detection import _cluster, _find_glints, _fit_pupil  # noqa: F401
from opengp3.cli import main

if __name__ == "__main__":
    sys.exit(main())
