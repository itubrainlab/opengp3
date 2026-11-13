"""Webcam fallback: feature maths, fallback logic and warnings. No camera or
MediaPipe needed (landmarks are synthesised, the detector is stubbed)."""
import math
import time
import warnings

import numpy as np
import pytest

import opengp3.camera as camera
import opengp3.webcam as webcam
from opengp3 import GazeTracker, Screen
from opengp3.server import OpenGazeServer, _Client
from opengp3.webcam import FEATURE_SCALE, WEBCAM_WARNING, eyes_from_landmarks


def face(look=(0.0, 0.0), roll_deg=0.0, scale=1.0, open_=0.3, centre=(640, 360)):
    """478 face-mesh points; iris displaced by look * eye width."""
    pts = np.zeros((478, 2))
    width = 60.0
    for (o, i, up, lo, iris, ring), cx in (((33, 133, 159, 145, 468, range(469, 473)), -70),
                                           ((362, 263, 386, 374, 473, range(474, 478)), 70)):
        mid = np.array([cx, 0.0])
        pts[o] = mid + (-width / 2, 0)
        pts[i] = mid + (width / 2, 0)
        pts[up] = mid + (0, -open_ * width / 2)
        pts[lo] = mid + (0, open_ * width / 2)
        pts[iris] = mid + (look[0] * width, look[1] * width)
        for k, j in enumerate(ring):
            a = k * math.pi / 2
            pts[j] = pts[iris] + 6 * np.array([math.cos(a), math.sin(a)])
    r = math.radians(roll_deg)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return (pts @ R.T) * scale + centre


def test_feature_follows_iris_and_ignores_roll_and_distance():
    base = eyes_from_landmarks(face((0.10, -0.05)))
    assert len(base) == 2
    for e in base:
        assert e.pcr == pytest.approx((0.10 * FEATURE_SCALE, -0.05 * FEATURE_SCALE), abs=1e-6)
        assert e.pupil_r == 0.0                       # pupil size: not measurable
    for variant in (face((0.10, -0.05), roll_deg=12), face((0.10, -0.05), scale=0.6)):
        for e, b in zip(eyes_from_landmarks(variant), base):
            assert e.pcr == pytest.approx(b.pcr, abs=1e-6)


def test_closed_eyes_are_missing():
    assert eyes_from_landmarks(face(open_=0.05)) == []           # blink
    assert eyes_from_landmarks(np.zeros((10, 2))) == []          # not a face mesh


def test_open_camera_falls_back_with_warning(monkeypatch):
    def no_gp3(opener, auto_setup):
        raise camera.GP3NotFoundError("no GP3 or GP3 HD found")
    monkeypatch.setattr(camera, "_open_with_setup", no_gp3)
    monkeypatch.delenv("OPENGP3_NO_WEBCAM_FALLBACK", raising=False)
    sentinel = object()
    monkeypatch.setattr(webcam, "open_webcam", lambda index=0: sentinel)
    with pytest.warns(RuntimeWarning, match="NO GAZEPOINT EYE TRACKER FOUND"):
        assert camera.open_camera() is sentinel
    # disabled: the original error
    monkeypatch.setenv("OPENGP3_NO_WEBCAM_FALLBACK", "1")
    with pytest.raises(camera.GP3NotFoundError, match="no GP3"):
        camera.open_camera()
    monkeypatch.delenv("OPENGP3_NO_WEBCAM_FALLBACK")

    def unavailable(index=0):
        raise webcam.WebcamUnavailable('pip install "opengp3[webcam]"')
    monkeypatch.setattr(webcam, "open_webcam", unavailable)
    with pytest.raises(camera.GP3NotFoundError, match=r"Webcam fallback not available.*opengp3\[webcam\]"):
        camera.open_camera()


class _Detector:
    """Stand-in for WebcamEyeDetector: landmarks of a face looking at cam.look."""
    max_eyes = 2

    def __init__(self, cam):
        self.cam = cam

    def update(self, img):
        fx, fy = self.cam.look
        return eyes_from_landmarks(face(((fx - 0.5) * 0.3, (fy - 0.5) * 0.2)))


@pytest.fixture
def webcam_tracker(fake_camera, monkeypatch):
    fake_camera.realtime = False   # simulated time: fixed frames per point on any machine
    for attr in ("illumination", "model", "warning", "warning_title", "product_id",
                 "company_id", "description"):  # a webcam's self-description
        setattr(fake_camera, attr, getattr(webcam.WebcamCamera, attr))
    monkeypatch.setattr(webcam, "WebcamEyeDetector", lambda: _Detector(fake_camera))
    gt = GazeTracker(Screen(1920, 1080, width_cm=47.0), camera=fake_camera, input_monitor=False)
    with gt:
        yield gt, fake_camera


@pytest.mark.slow
def test_webcam_pipeline_calibrates_and_warns(webcam_tracker):
    gt, cam = webcam_tracker
    assert gt.warning == WEBCAM_WARNING

    def follow(ev, n, fx, fy):
        if ev == "start_pt":
            cam.look = (fx, fy)
    run = gt.start_calibration(9, delay=0.3, timeout=0.4, min_samples=5, listener=follow)
    while not run.tick(None):
        time.sleep(0.005)
    cal = gt.finish_calibration(run)
    assert set(cal.eyes) == {"R", "L"} and cal.valid_points == 9
    cam.look = (0.3, 0.7)
    time.sleep(0.4)
    s = gt.latest()
    assert s.valid and abs(s.raw_x / 1920 - 0.3) < 0.03 and abs(s.raw_y / 1080 - 0.7) < 0.03
    assert not s.metrics["R"].pupil_mm                   # no pupil size from a webcam

    srv = OpenGazeServer(gt, "127.0.0.1", port=0)
    c = _Client(sock=None, addr=None)
    prod = srv.handle(c, "GET", "PRODUCT_ID", {})[0]
    assert 'VALUE="WEBCAM"' in prod and "WARNING=" in prod and "NO GAZEPOINT" in prod
    assert 'VALUE="WEBCAM"' in srv.handle(c, "GET", "COMPANY_ID", {})[0]
    assert "WARNING=" in srv.handle(c, "GET", "API_ID", {})[0]


def test_no_warning_attributes_with_a_gazepoint_tracker(fake_camera):
    with GazeTracker(Screen(1920, 1080), camera=fake_camera, input_monitor=False) as gt:
        srv = OpenGazeServer(gt, "127.0.0.1", port=0)
        c = _Client(sock=None, addr=None)
        assert gt.warning is None
        for ident in ("PRODUCT_ID", "API_ID", "COMPANY_ID"):
            assert "WARNING" not in srv.handle(c, "GET", ident, {})[0]
        assert 'VALUE="GAZEPOINT"' in srv.handle(c, "GET", "COMPANY_ID", {})[0]


@pytest.mark.slow
def test_lsl_metadata_and_marker_warn(webcam_tracker):
    pylsl = pytest.importorskip("pylsl")
    from opengp3.lsl import LSLStreamer
    gt, cam = webcam_tracker
    lsl = LSLStreamer(gt, name="OpenGP3-webcam-test", source_id="opengp3-webcam-test")
    try:
        info = None
        t_end = time.time() + 5
        while info is None and time.time() < t_end:
            found = [s for s in pylsl.resolve_streams(wait_time=0.5)
                     if s.name() == "OpenGP3-webcam-test Gaze"]
            info = found[0] if found else None
        assert info is not None
        full = pylsl.StreamInlet(info).info(timeout=5)
        assert "NOT a Gazepoint" in full.desc().child_value("manufacturer")
        assert "NO GAZEPOINT" in full.desc().child_value("warning")
    finally:
        lsl.close()


@pytest.mark.slow
def test_gui_banner(webcam_tracker, tmp_path):
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from opengp3.gui import _import_qt, main_window_class
    _import_qt()
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    gt, cam = webcam_tracker
    srv = OpenGazeServer(gt, "127.0.0.1", port=0).start()
    try:
        w = main_window_class()(gt, srv, lsl=None, media_dir=None,
                                save_path=tmp_path / "cal.json", mirror=True, show_tracker=True,
                                screen_args={})
        assert not w.banner.isHidden() and "NO GAZEPOINT EYE TRACKER FOUND" in w.banner.text()
        assert "WEBCAM FALLBACK" in w.windowTitle()
        w.close()
    finally:
        srv.stop()


def test_calibration_files_are_separate(monkeypatch, tmp_path):
    from opengp3.control import default_calibration_path
    monkeypatch.setenv("OPENGP3_HOME", str(tmp_path))
    names = {default_calibration_path().name, default_calibration_path(simulated=True).name,
             default_calibration_path(webcam=True).name}
    assert names == {"calibration.json", "calibration-simulated.json", "calibration-webcam.json"}


def test_auto_open_skips_black_virtual_camera(monkeypatch):
    """Index 0 can be a virtual camera (e.g. OBS) that only sends black frames:
    the fallback must move on to the first camera with a real image."""
    import opengp3.sources as sources
    monkeypatch.setattr(webcam, "_require_mediapipe", lambda: None)
    monkeypatch.setattr(webcam, "model_path", lambda download=True: None)
    monkeypatch.setattr(sources, "preferred_webcam_order", lambda: [])   # unknown: 0, 1, 2...
    tried = []

    class FakeWebcam:
        def __init__(self, index, first_frame_timeout=4.0):
            tried.append(index)
            if index == 0:
                raise webcam.BlackCamera("camera 0 delivers only black frames - probably virtual")
            if index >= 3:
                raise webcam.WebcamUnavailable(f"no webcam at index {index}")
            self.index = index
    monkeypatch.setattr(webcam, "WebcamCamera", FakeWebcam)
    assert webcam.open_webcam().index == 1 and tried == [0, 1]
    assert webcam.open_webcam(index=2).index == 2          # explicit choice is respected

    class AllBlack(FakeWebcam):
        def __init__(self, index, first_frame_timeout=4.0):
            tried.append(index)
            if index < 2:
                raise webcam.BlackCamera(f"camera {index} delivers only black frames - x")
            raise webcam.WebcamUnavailable(f"no webcam at index {index}")
    monkeypatch.setattr(webcam, "WebcamCamera", AllBlack)
    with pytest.raises(webcam.WebcamUnavailable, match="camera 0 delivers only black frames"):
        webcam.open_webcam()



def test_auto_open_prefers_real_cameras(monkeypatch):
    """With names known, built-in cameras are tried before phones and virtual
    cameras (whose placeholder image isn't black, so it would be accepted)."""
    import opengp3.sources as sources
    monkeypatch.setattr(webcam, "_require_mediapipe", lambda: None)
    monkeypatch.setattr(webcam, "model_path", lambda download=True: None)
    monkeypatch.setattr(sources, "preferred_webcam_order", lambda: [1, 0, 2])
    tried = []

    class FakeWebcam:
        def __init__(self, index, first_frame_timeout=4.0):
            tried.append(index)
            self.index = index
    monkeypatch.setattr(webcam, "WebcamCamera", FakeWebcam)
    assert webcam.open_webcam().index == 1 and tried == [1]
