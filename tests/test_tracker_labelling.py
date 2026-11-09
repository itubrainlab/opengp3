"""Regression test for sample labelling by capture time: samples captured
before a target appeared (or during the saccade) must never be attributed to
it, however late they are processed."""
from opengp3.tracker import GazeSample, GazeTracker


def test_label_by_capture_time():
    shown = [(0, 100, 100, 10.0, 12.0), (1, 900, 100, 12.0, 14.0)]
    samples = [GazeSample(t=10.0 + k / 60, frame=k, eyes={}) for k in range(240)]
    lab = GazeTracker._label(samples, shown, settle=0.6)
    for idx, x, y, s in lab:
        t_on, t_off = shown[idx][3], shown[idx][4]
        assert t_on + 0.6 <= s.t < t_off
    assert {l[0] for l in lab} == {0, 1}
    # each target: (2.0 - 0.6) s * 60 Hz = 84 samples
    assert sum(1 for l in lab if l[0] == 0) == 84
