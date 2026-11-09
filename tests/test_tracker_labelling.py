"""Regression test for sample labelling by capture time: samples captured
before a target's collection window (or during the animation/saccade) must
never be attributed to it, however late they are processed."""
import queue

from opengp3.screen import Rect
from opengp3.tracker import CalibrationRun, GazeSample


class _DummyTracker:
    def subscribe(self):
        return queue.Queue()

    def unsubscribe(self, q):
        pass


def test_label_by_capture_time():
    run = CalibrationRun(_DummyTracker(), [(0.1, 0.1), (0.5, 0.1)], Rect(0, 0, 1000, 1000),
                         delay=0.6, timeout=1.4, retry=False)
    # collection windows (after the 0.6 s animation): 10.6-12.0 and 12.6-14.0
    run.windows = [(0, 100, 100, 10.6, 12.0), (1, 500, 100, 12.6, 14.0)]
    run.samples = [GazeSample(t=10.0 + k / 60, frame=k, eyes={}) for k in range(240)]
    lab = run.labelled()
    for idx, x, y, s in lab:
        t0, t1 = run.windows[idx][3], run.windows[idx][4]
        assert t0 <= s.t < t1
    assert {l[0] for l in lab} == {0, 1}
    assert sum(1 for l in lab if l[0] == 0) == 84   # 1.4 s * 60 Hz


def test_retried_target_uses_last_window_only():
    run = CalibrationRun(_DummyTracker(), [(0.1, 0.1)], Rect(0, 0, 1000, 1000), retry=True)
    run.windows = [(0, 100, 100, 1.0, 2.0), (0, 100, 100, 5.0, 6.0)]
    run.samples = [GazeSample(t=t, frame=0, eyes={}) for t in (1.5, 5.5)]
    assert [s.t for *_, s in run.labelled()] == [5.5]
