"""A simulated GP3: renders synthetic infrared eye images in real time, so the
whole pipeline (detection, calibration, API server, LSL, Control app) runs
without hardware - for demos, development and teaching.

The simulated person looks at the mouse cursor. During calibration or
validation they follow the targets by themselves (with a human-like
reaction delay). Gentle head movement, fixational jitter, blinks and pupil
size changes are included so the event detectors have something to detect.

    from opengp3 import GazeTracker
    from opengp3.simulate import SimulatedCamera
    cam = SimulatedCamera()
    with GazeTracker(camera=cam) as gt:
        cam.attach(gt)            # look at calibration targets / the mouse
        ...

``opengp3 --simulate`` (or OPENGP3_SIMULATE=1) runs the Control app with it.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .platform import mouse_position


def open_simulated() -> "SimulatedCamera":
    """A SimulatedCamera, announced with a RuntimeWarning (as the CLI and the
    Control app open it), so nobody mistakes its data for real recordings."""
    import warnings
    warnings.warn(SIMULATED_WARNING, RuntimeWarning, stacklevel=2)
    return SimulatedCamera()


SIMULATED_WARNING = ("SIMULATED EYE TRACKER - no real eye-tracking data. The images are "
                     "synthetic and the simulated person looks at your mouse cursor (and follows "
                     "calibration targets by itself); do not use this data for research.")


class SimulatedCamera:
    """Drop-in replacement for ``GP3Camera`` (same interface as used by
    ``GazeTracker``). 752x480 MONO8 at 60 fps."""

    model = "Simulated GP3"
    warning = SIMULATED_WARNING
    warning_title = "SIMULATED (no real eye data)"
    product_id = "SIMULATED"
    company_id = "SIMULATED"
    description = "OpenGP3 simulated GP3 (NOT real eye-tracking data)"
    simulated = True
    rate = 60.0
    width, height = 752, 480
    illumination = "dark"
    serial = "SIMULATED"
    dev = None

    def __init__(self, seed: int = 0, blink_interval_s: float = 4.0, head_motion: bool = True,
                 gain=(-8.0, 6.0)):
        self.rng = np.random.default_rng(seed)
        self.blink_interval = blink_interval_s
        self.head_motion = head_motion
        self.gx, self.gy = gain                 # PCR px per screen width/height
        self.look_override: tuple[float, float] | None = None
        self._tracker = None
        self._seq = 0
        self._next = time.time()
        self._t0 = time.time()
        self._eye = np.array([0.5, 0.5])        # where the eye currently points (screen fraction)
        self._target = np.array([0.5, 0.5])
        self._target_since = time.time()
        self._next_blink = self._t0 + self.rng.uniform(1.5, blink_interval_s)
        self._blink_until = 0.0
        self.leds = True
        self.features = {"gain": 32, "shutter": 400}
        self._background = self._render_face()

    # ------------------------------------------------------------- wiring
    def attach(self, tracker):
        """Let the simulated person see the tracker's screen and its active
        calibration/validation targets."""
        self._tracker = tracker
        return self

    def _wanted_look(self) -> tuple[float, float]:
        if self.look_override is not None:
            return self.look_override
        tr = self._tracker
        run = getattr(tr, "active_run", None) if tr is not None else None
        if run is not None and getattr(run, "current", None):
            _, (fx, fy) = run.current
            x, y = run.area.to_screen(fx, fy)
            return x / tr.screen.width, y / tr.screen.height
        sc = getattr(tr, "screen", None)
        mx, my = mouse_position()
        if sc is None or not sc.width:
            return 0.5, 0.5
        return (float(np.clip((mx - sc.x) / sc.width, 0, 1)),
                float(np.clip((my - sc.y) / sc.height, 0, 1)))

    # ------------------------------------------------------------- camera API
    def set_full_sensor(self, *a, **k):
        pass

    def set_video_mode(self, *a, **k):
        pass

    def set_rate(self, *a, **k):
        pass

    def set_leds(self, on=True):
        self.leds = on

    def start_stream(self, *a, **k):
        self._next = time.time()

    def stop_stream(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def feature(self, name):
        return {"present": True, "min": 16, "max": 64, "auto_capable": True, "on": True,
                "auto": True, "value": self.features.get(name, 0)}

    def set_feature(self, name, value=None, auto=False):
        if value is not None:
            self.features[name] = value

    def info(self) -> dict:
        return {"usb": "simulated GP3", "serial": self.serial, "formats": [0, 7]}

    #: False: frames are produced as fast as they are consumed, stamped with a
    #: simulated time advancing exactly 1/rate per frame (``clock()``), so a
    #: calibration takes a fixed number of frames on any machine (tests)
    @property
    def realtime(self) -> bool:
        return getattr(self, "_realtime", True)

    @realtime.setter
    def realtime(self, on: bool):
        if not on and self.realtime:
            self._vt = time.time()
        self._realtime = on

    def clock(self) -> float:
        return time.time() if self.realtime else self._vt

    def grab(self, timeout_s: float = 2.0, latest: bool = False):
        if not self.realtime:
            self._vt += 1.0 / self.rate
            self._seq += 1
            return self._seq, self._vt, self.frame(self._vt)
        now = time.time()
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(self._next + 1.0 / self.rate, time.time() - 0.05)
        self._seq += 1
        t = time.time()
        return self._seq, t, self.frame(t)

    # ------------------------------------------------------------- rendering
    def _render_face(self) -> np.ndarray:
        """Static, softly lit 'face' background (rendered once)."""
        w, h = self.width, self.height
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        face = 150 - 0.00025 * ((xx - w / 2) ** 2 + 1.6 * (yy - h * 0.55) ** 2)
        face = np.clip(face, 25, 170)
        face += self.rng.normal(0, 3, (h, w))
        return cv2.GaussianBlur(face.astype(np.float32), (0, 0), 2)

    def _update_eye(self, t: float):
        tgt = np.array(self._wanted_look())
        if np.hypot(*(tgt - self._target)) > 0.01:      # a new place to look at
            self._target, self._target_since = tgt, t
        if t - self._target_since > 0.18:                 # reaction time, then a saccade
            self._eye += (self._target - self._eye) * 0.6
        self._eye += self.rng.normal(0, 0.0015, 2)        # fixational jitter

    def frame(self, t: float | None = None) -> np.ndarray:
        t = time.time() if t is None else t
        self._update_eye(t)
        img = self._background.copy()
        if not self.leds:
            return np.clip(img * 0.15, 0, 255).astype(np.uint8)
        # head: slow sway around the image centre
        el = t - self._t0
        hx = 376 + (18 * math.sin(el / 7.0) if self.head_motion else 0)
        hy = 238 + (8 * math.sin(el / 11.0 + 1) if self.head_motion else 0)
        # blinks
        if t >= self._next_blink:
            self._blink_until = t + self.rng.uniform(0.08, 0.18)
            self._next_blink = t + self.rng.exponential(self.blink_interval) + 0.5
        blinking = t < self._blink_until
        fx, fy = self._eye
        # pupil - glint (PCR); the GP3 sits below the screen, so at the screen
        # centre the glints appear a little below the pupil centre
        dx, dy = self.gx * (fx - 0.5), self.gy * (fy - 0.5) - 1.5
        # pupil large enough that the glints stay on it across the screen (as
        # with real eyes at this distance); slow dilation changes
        pupil_r = 8.5 + 0.6 * math.sin(el / 9.0)
        glints = np.zeros((self.height, self.width), np.float32)
        for side in (-1, 1):
            ex, ey = hx + side * 133, hy
            if blinking:  # closed lid: skin over the eye
                cv2.ellipse(img, (int(ex), int(ey)), (30, 18), 0, 0, 360, 120, -1, cv2.LINE_AA)
                continue
            cv2.ellipse(img, (int(ex), int(ey)), (34, 20), 0, 0, 360, 190, -1, cv2.LINE_AA)  # sclera
            px, py = ex + dx, ey + dy
            cv2.circle(img, (int(round(px)), int(round(py))), 17, 85, -1, cv2.LINE_AA)       # iris
            big = np.zeros((self.height * 4, self.width * 4), np.uint8)
            cv2.circle(big, (int(px * 4), int(py * 4)), int(pupil_r * 4), 255, -1)
            m = cv2.resize(big, (self.width, self.height), interpolation=cv2.INTER_AREA) / 255.0
            img = img * (1 - m) + 32 * m                                                      # pupil
            for ox in (-3.5, 3.5):                                                            # glints
                cv2.circle(glints, (int(round(ex + ox)), int(round(ey))), 2, 1.0, -1)
        img = img + self.rng.normal(0, 2.0, img.shape)
        halo = cv2.GaussianBlur(glints, (7, 7), 1.5)
        img = np.maximum(img, 255 * np.minimum(1.0, 2.2 * halo))
        img[glints > 0] = 255
        return np.clip(img, 0, 255).astype(np.uint8)
