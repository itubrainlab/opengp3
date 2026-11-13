"""The simulated GP3: detectable images, and a full calibration in which the
simulated person follows the targets by themselves."""
import time

from opengp3 import GazeTracker, Screen, detect_eyes
from opengp3.simulate import SimulatedCamera


def test_frames_are_detectable():
    cam = SimulatedCamera(head_motion=False, blink_interval_s=1e9)
    eyes = detect_eyes(cam.frame())
    assert len(eyes) == 2 and all(e.n_glints == 2 for e in eyes)


def test_calibration_follows_targets_then_tracks_gaze():
    cam = SimulatedCamera(head_motion=False, blink_interval_s=1e9)
    screen = Screen(1920, 1080, width_cm=47.0)            # 21": no 22" restriction
    with GazeTracker(screen, camera=cam, input_monitor=False) as gt:
        cam.attach(gt)
        run = gt.start_calibration(9, delay=0.5, timeout=0.8, min_samples=5)
        while not run.tick(None):
            time.sleep(0.005)
        cal = gt.finish_calibration(run)
        assert set(cal.eyes) == {"R", "L"} and cal.valid_points == 9
        cam.look_override = (0.3, 0.7)
        time.sleep(0.8)
        s = gt.latest()
    assert s.valid
    assert abs(s.raw_x / 1920 - 0.3) < 0.04 and abs(s.raw_y / 1080 - 0.7) < 0.04
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
