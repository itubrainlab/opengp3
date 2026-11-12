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
