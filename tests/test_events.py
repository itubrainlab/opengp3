import numpy as np
import pytest

from opengp3.events import BlinkDetector, FixationFilter, MovingAverage


def feed(f, pts, t0=0.0, dt=1 / 60):
    out = []
    for k, p in enumerate(pts):
        out.append(f.update(t0 + k * dt, *(p if p else (None, None))))
    return out


def test_fixation_ids_start_duration_and_saccade():
    f = FixationFilter(radius=0.02, min_duration=0.06)
    rng = np.random.default_rng(0)
    a = [(0.3 + rng.normal(0, 0.003), 0.4 + rng.normal(0, 0.003)) for _ in range(30)]
    b = [(0.7 + rng.normal(0, 0.003), 0.6 + rng.normal(0, 0.003)) for _ in range(30)]
    out = feed(f, a + b)
    assert not out[0].valid                       # too short to be a fixation yet
    assert out[29].valid and out[29].id == 1
    assert out[29].x == pytest.approx(0.3, abs=0.005)
    assert out[29].duration == pytest.approx(29 / 60, abs=1e-6)
    assert out[30].id == 1                        # saccade not yet confirmed (1 sample)
    assert out[32].id == 1 and not out[32].valid  # confirmed after 3: new candidate
    assert out[32].start == pytest.approx(30 / 60)  # back-dated to the saccade
    assert out[59].valid and out[59].id == 2


def test_fixation_survives_short_gap_but_not_long_one():
    f = FixationFilter(radius=0.02, min_duration=0.05, max_gap=0.15)
    pts = [(0.5, 0.5)] * 20 + [None] * 5 + [(0.5, 0.5)] * 5     # 83 ms blink
    out = feed(f, pts)
    assert not out[22].valid                      # invalid during the gap
    assert out[-1].valid and out[-1].id == 1      # same fixation afterwards
    f2 = FixationFilter(radius=0.02, min_duration=0.05, max_gap=0.15)
    out2 = feed(f2, [(0.5, 0.5)] * 20 + [None] * 30 + [(0.5, 0.5)] * 10)
    assert out2[-1].id == 2                       # 0.5 s gap: a new fixation


def test_blinks():
    b = BlinkDetector()
    seq = [True] * 30 + [False] * 6 + [True] * 30 + [False] * 120 + [True] * 10
    infos = [b.update(k / 60, v) for k, v in enumerate(seq)]
    ids = [i.id for i in infos if i.id]
    assert ids == [1]                             # the 2 s loss is not a blink
    assert infos[-1].duration == pytest.approx(0.1, abs=0.02)
    assert infos[-1].per_minute == 1


def test_moving_average():
    m = MovingAverage(3)
    for v in (1, 2, 3, 4):
        r = m.update(v, 0)
    assert r == (3.0, 0.0)
    assert m.update(None, None) == (3.0, 0.0)     # holds during invalid samples
    m.set_length(1)
    assert m.update(10, 1) == (10.0, 1.0)


def test_fixation_ignores_isolated_noise_but_detects_saccade():
    f = FixationFilter(radius=0.02, min_duration=0.05, confirm=3)
    pts = [(0.5, 0.5)] * 20 + [(0.56, 0.5)] + [(0.5, 0.5)] * 10 + [(0.8, 0.2)] * 10
    out = feed(f, pts)
    assert out[21].valid and out[21].id == 1      # single outlier: still the same fixation
    assert out[30].id == 1 and out[30].x == pytest.approx(0.5, abs=0.003)
    assert out[33].start == pytest.approx(31 / 60)  # saccade confirmed, back-dated start
    assert out[-1].valid and out[-1].id == 2
