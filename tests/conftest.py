"""Shared test helpers: synthetic GP3 images and a fake camera, so the whole
pipeline (detection -> calibration -> API server) runs without hardware."""
import threading
import time

import cv2
import numpy as np
import pytest


def synth_eye_image(eyes, size=(752, 480), seed=0):
    """Grey face, darker iris, dark pupil, and a pair of saturated glints
    (the GP3's two LED banks) per eye. eyes: [(pupil_x, pupil_y, glint_dx, glint_dy)]"""
    w, h = size
    img = np.full((h, w), 150, np.uint8)
    glints = np.zeros((h, w), np.float32)
    for px, py, gdx, gdy in eyes:
        cv2.circle(img, (int(px), int(py)), 28, 80, -1, cv2.LINE_AA)
        # sub-pixel pupil via supersampling
        big = np.zeros((h * 4, w * 4), np.uint8)
        cv2.circle(big, (int(px * 4), int(py * 4)), 36, 255, -1)
        mask = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA) / 255.0
        img = (img * (1 - mask) + 35 * mask).astype(np.uint8)
        gx, gy = px - gdx, py - gdy
        for ox in (-3.5, 3.5):
            cv2.circle(glints, (int(round(gx + ox)), int(round(gy))), 2, 1.0, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    img = img + rng.normal(0, 2, img.shape)
    # real GP3 glints: saturated core (5-20 px) with a ~2 px bloom halo
    halo = cv2.GaussianBlur(glints, (7, 7), 1.5)
    img = np.maximum(img, 255 * np.minimum(1.0, 2.2 * halo))
    img[glints > 0] = 255
    return np.clip(img, 0, 255).astype(np.uint8)




class FakeCamera:
    """Stands in for GP3Camera. Renders a face with two eyes whose
    pupil-glint offset follows ``self.look`` (screen fraction the simulated
    subject looks at), at ~60 fps."""

    width, height = 752, 480

    def __init__(self, fps=60.0, gain_x=-9.0, gain_y=7.0):
        self.look = (0.5, 0.5)
        self.eyes_open = True
        self.fps, self.gx, self.gy = fps, gain_x, gain_y
        self._seq = 0
        self._next = time.time()
        self.leds = False
        self.features = {"gain": 32}
        self.dev = None

    # GP3Camera API used by GazeTracker
    def set_full_sensor(self):
        pass

    def set_video_mode(self, *a):
        pass

    def set_leds(self, on=True):
        self.leds = on

    def start_stream(self):
        pass

    def close(self):
        pass

    def feature(self, name):
        return {"present": True, "min": 16, "max": 64, "value": self.features.get(name, 0)}

    def set_feature(self, name, value=None, auto=False):
        self.features[name] = value

    def frame(self):
        if not self.eyes_open:
            return np.full((self.height, self.width), 150, np.uint8)
        fx, fy = self.look
        dx, dy = self.gx * (fx - 0.5), self.gy * (fy - 0.5)   # PCR (pupil - glint)
        return synth_eye_image([(250 + dx, 240 + dy, dx, dy), (517 + dx, 238 + dy, dx, dy)],
                               seed=self._seq % 7)

    def grab(self, timeout_s=2.0, latest=False):
        now = time.time()
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(self._next + 1 / self.fps, time.time() - 0.05)
        self._seq += 1
        return self._seq, time.time(), self.frame()


@pytest.fixture
def fake_camera():
    return FakeCamera()
