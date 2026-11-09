import cv2
import numpy as np
import pytest

from opengp3.detection import EyeFeature
from opengp3.eyemodel import EyeModel, MarkerTracker


def eye(x, y, r=6.0):
    return EyeFeature(x, y, r, x, y, 2, 1.0)


def test_depth_position_and_pupil_mm():
    m = EyeModel(focal_px=2750, ipd_mm=63)
    out = m.update({"R": eye(242.5, 240), "L": eye(509.5, 240)})   # 267 px apart
    z = 2750 * 0.063 / 267
    assert m.depth_m == pytest.approx(z)
    assert out["R"].z_m == pytest.approx(z)
    assert out["R"].x_m > 0 > out["L"].x_m          # subject's right eye: +x
    assert out["L"].x_m - out["R"].x_m == pytest.approx(-0.063, abs=1e-6)
    assert out["R"].pupil_mm == pytest.approx(12 * z / 2750 * 1000)
    assert out["R"].cx == pytest.approx(242.5 / 752)
    m.set_reference()
    out = m.update({"R": eye(242.5, 240), "L": eye(509.5, 240)})
    assert out["R"].scale == pytest.approx(1.0)


def test_one_eye_keeps_depth():
    m = EyeModel()
    m.update({"R": eye(250, 240), "L": eye(517, 240)})
    out = m.update({"L": eye(517, 240)})
    assert out["L"].valid and out["L"].z_m > 0 and not out["R"].valid


def test_marker_scale():
    img = MarkerTracker.image(size_mm=11.0, dpi=254)        # 110 px black square (+ border)
    canvas = np.full((480, 752), 255, np.uint8)
    canvas[100:100 + img.shape[0], 300:300 + img.shape[1]] = img
    r = MarkerTracker(size_mm=11.0).detect(canvas)
    assert r.valid
    assert r.mm_per_px == pytest.approx(0.1, rel=0.03)
