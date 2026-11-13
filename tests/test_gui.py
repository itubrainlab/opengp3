"""The Qt control app (``opengp3 server``), headless: Qt's offscreen platform
and the simulated camera, so it runs in CI without a display."""
import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from opengp3 import GazeTracker, Screen          # noqa: E402
from opengp3.gui import calibration_points, main_window_class, point_errors   # noqa: E402
from opengp3.server import OpenGazeServer        # noqa: E402

SCREEN = Screen(1920, 1080, width_cm=53.0, distance_cm=65.0)


pytestmark = pytest.mark.slow   # GUI / LSL: run on release


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, fake_camera, tmp_path):
    gt = GazeTracker(SCREEN, camera=fake_camera, input_monitor=False)
    gt.start()
    srv = OpenGazeServer(gt, "127.0.0.1", port=0).start()
    w = main_window_class()(gt, srv, lsl=None, media_dir=None, save_path=tmp_path / "cal.json",
                            mirror=True, show_tracker=True, screen_args={})
    w.show()
    yield w, gt, srv, fake_camera
    w.close()
    srv.stop()
    gt.stop()


def pump(app, w, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.005)


def test_calibration_point_sets():
    assert calibration_points(5)[0] == (0.5, 0.5) and len(calibration_points(5)) == 5
    assert len(calibration_points(9)) == 9
    assert len(calibration_points(13)) == 13 and len(calibration_points(25)) == 25


def test_window_shows_tracker_and_status(app, window):
    w, gt, srv, cam = window
    pump(app, w, 1.0)
    assert w.view.pixmap() is not None and not w.view.pixmap().isNull()
    assert w.lb_eyes.text() == "both"
    assert "samples/s" in w.lb_fps.text()
    assert w.lb_clients.text() == "0"
    assert w.lb_model.text().startswith("GP3")
    w.cmb_points.setCurrentIndex(w.cmb_points.findData(13))
    assert len(srv.cal_points) == 13
    w.cb_display.setChecked(False)
    assert srv.tracker_display is False and not w.view.isVisible()


def test_calibrate_from_the_window_headless(app, window):
    """Calibration started from the app; the simulated subject follows the
    targets; the results table and summary appear."""
    w, gt, srv, cam = window
    srv.cal_show = False            # headless: the server times the points, no stimulus window
    cam.realtime = False   # simulated time: fixed frames per point on any machine
    srv.cal_delay, srv.cal_timeout = 0.3, 0.4
    w.cmb_points.setCurrentIndex(w.cmb_points.findData(5))
    orig = srv.calibration_listener

    def follow(event, n, fx, fy):
        orig(event, n, fx, fy)
        if event == "start_pt":
            cam.look = (fx, fy)
    srv.calibration_listener = follow
    w.start_calibration()
    assert w.btn_cal.text() == "Stop"
    t_end = time.time() + 40
    while w.run is not None and time.time() < t_end:
        app.processEvents()
        time.sleep(0.005)
    assert w.run is None and gt.calibration is not None
    assert w.tbl.rowCount() == 5
    assert "5/5 points valid" in w.lb_cal.text()
    # every point has a valid eye, and every valid eye's result is accurate (an
    # eye may be invalid at a point if its pupil fits were rejected as misfits)
    errors = point_errors(gt.calibration)
    assert all(any(e is not None for e in errs.values()) for _, _, _, errs in errors)
    assert all(e < 60 for _, _, _, errs in errors for e in errs.values() if e is not None)
    assert w.btn_cal.text() == "Calibrate" and w.btn_val.isEnabled()
    assert (w.save_path).exists()
    w.cb_pointer.setChecked(True)
    assert gt.gaze_pointer
    gt.gaze_pointer = False


def test_screen_selection(app, window, monkeypatch):
    import opengp3.platform as plat
    w, gt, srv, cam = window
    monkeypatch.setattr(plat, "displays", lambda: [
        plat.Display(1, 0, 0, 2560, 1440, True, 597.0),
        plat.Display(2, 217, -1080, 1920, 1080, False, 527.0)])
    gt.set_display(1)
    assert (gt.screen.width, gt.screen.x, gt.screen.y, gt.screen.width_cm) == (1920, 217, -1080, 52.7)
    with pytest.raises(ValueError, match="display 5 not found"):
        gt.set_display(5)
    w.displays = plat.displays()
    w.sp_width.setValue(52.7)
    w.cmb_screen.addItem("x", 1)
    w.cmb_screen.setCurrentIndex(w.cmb_screen.count() - 1)
    assert gt.screen.width == 1920 and gt.screen.width_cm == pytest.approx(52.7)


def test_remote_access_switch(app, window):
    w, gt, srv, cam = window
    assert srv.local_only and not w.cb_remote.isChecked()           # fixture: local
    w.cb_remote.setChecked(True)
    assert not srv.local_only and srv.port > 0
    assert w.lb_address.text() == "\n".join(srv.addresses())
    w.cb_remote.setChecked(False)
    assert srv.local_only and w.lb_address.text() == f"127.0.0.1:{srv.port}"
