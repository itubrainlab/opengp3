"""The simulated GP3: detectable images, and a full calibration in which the
simulated person follows the targets by themselves."""
import pytest
import time

from opengp3 import GazeTracker, Screen, detect_eyes
from opengp3.simulate import SimulatedCamera


def test_frames_are_detectable():
    cam = SimulatedCamera(head_motion=False, blink_interval_s=1e9)
    eyes = detect_eyes(cam.frame())
    assert len(eyes) == 2 and all(e.n_glints == 2 for e in eyes)


@pytest.mark.slow
def test_calibration_follows_targets_then_tracks_gaze():
    cam = SimulatedCamera(head_motion=False, blink_interval_s=1e9)
    cam.realtime = False   # simulated time: fixed frames per point on any machine
    screen = Screen(1920, 1080, width_cm=47.0)            # 21": no 22" restriction
    with GazeTracker(screen, camera=cam, input_monitor=False) as gt:
        cam.attach(gt)
        run = gt.start_calibration(9, delay=0.3, timeout=0.4, min_samples=5)
        while not run.tick(None):
            time.sleep(0.005)
        cal = gt.finish_calibration(run)
        assert set(cal.eyes) == {"R", "L"} and cal.valid_points == 9
        cam.look_override = (0.3, 0.7)
        time.sleep(0.5)                                   # the eye gets there
        q = gt.subscribe()
        gaze = []
        t_end = time.time() + 10
        while len(gaze) < 20 and time.time() < t_end:   # median of 20 samples:
            s = q.get(timeout=2)                          # single frames are noisy
            if s.valid:
                gaze.append((s.raw_x / 1920, s.raw_y / 1080))
        gt.unsubscribe(q)
    import numpy as np
    mx, my = np.median(gaze, axis=0)
    assert len(gaze) == 20 and abs(mx - 0.3) < 0.04 and abs(my - 0.7) < 0.04
    assert gt.active_run is None if hasattr(gt, "active_run") else True


def test_simulated_device_is_flagged_everywhere(monkeypatch):
    """Like the webcam fallback, the simulated GP3 must never pass for real data."""
    import pytest
    from opengp3.server import OpenGazeServer, _Client
    from opengp3.simulate import SIMULATED_WARNING, open_simulated
    with pytest.warns(RuntimeWarning, match="SIMULATED EYE TRACKER"):
        cam = open_simulated()
    with GazeTracker(Screen(1920, 1080), camera=cam, input_monitor=False) as gt:
        cam.attach(gt)
        assert gt.warning == SIMULATED_WARNING and "SIMULATED" in gt.warning_title
        srv = OpenGazeServer(gt, "127.0.0.1", port=0)
        c = _Client(sock=None, addr=None)
        prod = srv.handle(c, "GET", "PRODUCT_ID", {})[0]
        assert 'VALUE="SIMULATED"' in prod and "WARNING=" in prod and "no real eye" in prod
        assert 'VALUE="SIMULATED"' in srv.handle(c, "GET", "COMPANY_ID", {})[0]
        assert "WARNING=" in srv.handle(c, "GET", "API_ID", {})[0]


@pytest.mark.slow
def test_simulated_lsl_metadata_and_marker():
    import pytest
    pylsl = pytest.importorskip("pylsl")
    from opengp3.lsl import LSLStreamer
    cam = SimulatedCamera()
    with GazeTracker(Screen(1920, 1080), camera=cam, input_monitor=False) as gt:
        lsl = LSLStreamer(gt, name="OpenGP3-sim-test", source_id="opengp3-sim-test")
        try:
            info, t_end = None, time.time() + 5
            while info is None and time.time() < t_end:
                found = [s for s in pylsl.resolve_streams(wait_time=0.5)
                         if s.name() == "OpenGP3-sim-test Gaze"]
                info = found[0] if found else None
            full = pylsl.StreamInlet(info).info(timeout=5)
            assert "NOT real eye-tracking data" in full.desc().child_value("manufacturer")
            assert "SIMULATED EYE TRACKER" in full.desc().child_value("warning")
        finally:
            lsl.close()
