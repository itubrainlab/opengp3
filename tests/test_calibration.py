import numpy as np
import pytest

from opengp3 import Calibration, CalibrationError, Screen, grid_points
from opengp3.calibration import CalibrationSample, evaluate, presentation_order
from opengp3.detection import EyeFeature

SCREEN = Screen(1920, 1080, width_cm=53.0, distance_cm=65.0)


def true_map(px, py, eye):
    """A smooth, slightly nonlinear PCR -> screen relation."""
    off = 0.6 if eye == "R" else -0.6
    x = 960 - 110 * (px - off) - 3.0 * (px - off) ** 2
    y = 540 + 95 * py + 2.0 * py ** 2
    return x, y



def pcr_for(x, y, eye):
    # numerically invert true_map
    px, py = 0.0, 0.0
    for _ in range(50):
        fx, fy = true_map(px, py, eye)
        jx = (true_map(px + 1e-4, py, eye)[0] - fx) / 1e-4
        jy = (true_map(px, py + 1e-4, eye)[1] - fy) / 1e-4
        px -= (fx - x) / jx
        py -= (fy - y) / jy
    return px, py


def make_samples(points, area, rng, noise=0.05, eyes=("R", "L"), n=40, drop=()):
    out = []
    for i, (fx, fy) in enumerate(points):
        x, y = area.to_screen(fx, fy)
        for eye in eyes:
            if (i, eye) in drop:
                continue
            px, py = pcr_for(x, y, eye)
            for _ in range(n):
                out.append(CalibrationSample(i, x, y, eye, px + rng.normal(0, noise),
                                             py + rng.normal(0, noise)))
    return out


def feat(px, py):
    return EyeFeature(100 + px, 100 + py, 6, 100, 100, 2, 1.0)


@pytest.mark.parametrize("n", [5, 9, 13, 16, 25])
def test_grids(n):
    pts = grid_points(n)
    assert len(pts) == n and len(set(pts)) == n
    assert all(0 < x < 1 and 0 < y < 1 for x, y in pts)
    assert sorted(presentation_order(pts)) == list(range(n))


def test_fit_and_predict_full_screen():
    rng = np.random.default_rng(1)
    cal = Calibration.fit(make_samples(grid_points(9), SCREEN.rect, rng), SCREEN)
    assert set(cal.eyes) == {"R", "L"}
    # unseen point
    x, y = 700, 400
    pr, pl = pcr_for(x, y, "R"), pcr_for(x, y, "L")
    gx, gy, per = cal.predict({"R": feat(*pr), "L": feat(*pl)})
    assert np.hypot(gx - x, gy - y) < 15
    # one eye only still works
    gx, gy, per = cal.predict({"L": feat(*pl)})
    assert set(per) == {"L"} and np.hypot(gx - x, gy - y) < 20
    assert cal.predict({}) is None


def test_eye_with_missing_points_falls_back_to_simpler_model():
    rng = np.random.default_rng(2)
    drop = {(0, "L"), (3, "L"), (2, "L")}  # left eye lost at 3 of 9 targets
    cal = Calibration.fit(make_samples(grid_points(9), SCREEN.rect, rng, drop=drop), SCREEN)
    assert cal.eyes["L"].n_points == 6
    assert cal.eyes["L"].mapping.model in ("affine", "x2/y2")
    assert cal.eyes["R"].n_points == 9


def test_no_data_raises():
    with pytest.raises(CalibrationError):
        Calibration.fit([], SCREEN)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    area = SCREEN.area(22) if SCREEN.width_cm else SCREEN.rect
    cal = Calibration.fit(make_samples(grid_points(9), area, rng), SCREEN, area)
    p = tmp_path / "cal.json"
    cal.save(str(p))
    cal2 = Calibration.load(str(p))
    pr = pcr_for(900, 500, "R")
    assert cal.predict({"R": feat(*pr)})[:2] == pytest.approx(cal2.predict({"R": feat(*pr)})[:2])
    assert cal2.area == cal.area and cal2.screen == cal.screen


def test_evaluate_accuracy_in_degrees():
    rng = np.random.default_rng(4)
    gaze = []
    for i, (x, y) in enumerate([(500, 300), (1400, 800)]):
        for k in range(60):
            gaze.append((i, x, y, k / 60, x + 20 + rng.normal(0, 3), y + rng.normal(0, 3)))
    res = evaluate(gaze, SCREEN)
    assert len(res.points) == 2
    # 20 points offset on a 53 cm / 1920 pt screen at 65 cm ~= 0.49 deg near centre
    assert 0.35 < res.accuracy_deg < 0.6
    assert res.precision_rms_s2s_deg > 0
    assert "ACCURACY" in res.report()
