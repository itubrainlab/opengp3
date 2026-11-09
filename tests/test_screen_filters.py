import numpy as np
import pytest

from opengp3 import OneEuroFilter, Rect, Screen


def test_area_22_inch_on_ultrawide():
    s = Screen(3440, 1440, width_cm=80.0, distance_cm=65.0)
    a = s.area(22)
    assert a.w * s.cm_per_point == pytest.approx(48.7, abs=0.1)
    assert a.h * s.cm_per_point == pytest.approx(27.4, abs=0.1)
    assert a.x + a.w / 2 == pytest.approx(1720) and a.y + a.h / 2 == pytest.approx(720)


def test_area_needs_physical_size():
    with pytest.raises(ValueError):
        Screen(1920, 1080).area(22)


def test_angle():
    s = Screen(1920, 1080, width_cm=53.0, distance_cm=65.0)
    assert s.angle_deg((960, 540), (960, 540)) == pytest.approx(0)
    # 10 cm to the side of centre at 65 cm
    x = 960 + 10 / s.cm_per_point
    assert s.angle_deg((960, 540), (x, 540)) == pytest.approx(np.degrees(np.arctan(10 / 65)), abs=1e-6)


def test_rect_mapping():
    r = Rect(100, 50, 200, 100)
    assert r.to_screen(0.5, 0.5) == (200, 100)
    assert r.contains(150, 60) and not r.contains(10, 10)


def test_one_euro_reduces_jitter_but_follows_jumps():
    f = OneEuroFilter()
    rng = np.random.default_rng(0)
    t = np.arange(240) / 60
    x = np.where(t < 2, 500.0, 1200.0) + rng.normal(0, 15, t.size)
    out = np.array([f(ti, xi, 300.0)[0] for ti, xi in zip(t, x)])
    still = slice(30, 110)
    assert out[still].std() < 0.5 * x[still].std()
    # within 150 ms after the saccade it should be most of the way there
    assert out[np.searchsorted(t, 2.15)] > 1000


def test_certified_area_limits_large_screens_to_22_inch():
    big = Screen(3440, 1440, width_cm=80.0)                 # 34" ultrawide
    a = big.certified_area()
    assert big.diagonal_in == pytest.approx(34.1, abs=0.2)
    assert a.w * big.cm_per_point == pytest.approx(48.7, abs=0.1)
    assert big.certified_area(None) == big.rect              # explicit override
    small = Screen(1920, 1080, width_cm=47.6)                # 21.5"
    assert small.certified_area() == small.rect
    unknown = Screen(1920, 1080)                             # no physical size known
    assert unknown.certified_area() == unknown.rect
