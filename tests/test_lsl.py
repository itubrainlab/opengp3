"""LSL output, end to end: fake camera -> tracker -> LSL outlet -> inlet."""
import time

import pytest

pylsl = pytest.importorskip("pylsl")

from opengp3 import GazeTracker, Screen
from opengp3.lsl import CHANNELS, LSLStreamer


def _resolve(name, timeout=5.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        found = [s for s in pylsl.resolve_streams(wait_time=0.5) if s.name() == name]
        if found:
            return found[0]
    pytest.fail(f"LSL stream {name!r} not found")


def test_lsl_gaze_and_markers(fake_camera):
    gt = GazeTracker(Screen(1920, 1080, width_cm=53.0), camera=fake_camera, input_monitor=False)
    with gt:
        lsl = LSLStreamer(gt, name="OpenGP3-test", source_id="opengp3-test")
        info = _resolve("OpenGP3-test Gaze")
        inlet = pylsl.StreamInlet(info, max_buflen=10)
        full = inlet.info(timeout=5)
        assert full.channel_count() == len(CHANNELS) and full.nominal_srate() == 60
        labels = []
        ch = full.desc().child("channels").child("channel")
        while not ch.empty():
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
        assert labels == [c[0] for c in CHANNELS]
        minlet = pylsl.StreamInlet(_resolve("OpenGP3-test Markers"))
        minlet.open_stream(timeout=5)
        inlet.open_stream(timeout=5)
        time.sleep(0.3)
        gt.user_data = "TRIAL_1"
        samples, stamps = [], []
        t_end = time.time() + 6            # generous: CI runners are slow
        while len(samples) < 60 and time.time() < t_end:
            s, ts = inlet.pull_sample(timeout=1.0)
            if s:
                samples.append(s)
                stamps.append(ts)
        marker, mts = minlet.pull_sample(timeout=3.0)
        lsl.close()
    assert len(samples) >= 20
    lpv, rpv = labels.index("LPCX"), labels.index("RPCX")
    assert all(s[lpv] > 0 and s[rpv] > 0 for s in samples[-10:])   # both eyes seen
    assert abs(stamps[-1] - pylsl.local_clock()) < 2.0
    assert all(b > a for a, b in zip(stamps, stamps[1:]))          # monotonic capture times
    assert marker == ["TRIAL_1"]
