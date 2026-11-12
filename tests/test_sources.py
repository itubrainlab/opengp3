"""Camera selector: listing sources, switching cameras while tracking, and
the Control app's drop-down (no hardware: fake and simulated cameras)."""
import time

import pytest

import opengp3.sources as sources
from opengp3 import GazeTracker, Screen
from opengp3.simulate import SimulatedCamera


def test_list_sources(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "linux")   # names map onto indices there
    monkeypatch.setattr(sources, "gazepoint_devices", lambda: ["GP3"])
    monkeypatch.setattr(sources, "webcam_names", lambda: ["FaceTime HD Camera", "StreamCam"])
    monkeypatch.setattr(sources, "webcam_supported", lambda: False)
    src = sources.list_sources()
    assert [s.kind for s in src] == ["gazepoint", "webcam", "webcam", "simulated"]
    assert src[0].label == "Gazepoint GP3"
    assert src[2].index == 1 and "StreamCam" in src[2].label and "low accuracy" in src[2].label
    assert not src[1].available and "opengp3[webcam]" in src[1].note   # extra missing
    assert src[-1].available


def test_macos_labels_are_neutral(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "darwin")
    monkeypatch.setattr(sources, "gazepoint_devices", lambda: [])
    monkeypatch.setattr(sources, "webcam_names", lambda: ["OBS Virtual Camera", "MacBook Pro Camera"])
    monkeypatch.setattr(sources, "webcam_supported", lambda: True)
    labels = [s.label for s in sources.list_sources()]
    assert labels[:2] == ["Webcam 0 (low accuracy)", "Webcam 1 (low accuracy)"]
    tip = sources.webcam_tooltip()
    assert "MacBook Pro Camera" in tip and "not in camera order" in tip


def test_source_of():
    class Gp3: pass
    class Cam:
        webcam, index = True, 2
    assert sources.source_of(Gp3()) == ("gazepoint", 0)
    assert sources.source_of(Cam()) == ("webcam", 2)
    assert sources.source_of(SimulatedCamera()) == ("simulated", 0)


def test_switch_camera_while_tracking(fake_camera):
    gt = GazeTracker(Screen(1920, 1080), camera=fake_camera, input_monitor=False)
    with gt:
        time.sleep(0.3)
        assert gt.latest() is not None and gt.model == "GP3"
        class _Cal:                                    # stands for a GP3 calibration
            meta = {}

            def predict(self, eyes):
                return None
        gt.calibration = _Cal()
        sim = SimulatedCamera()
        gt.switch_camera(sim)
        assert gt.camera is sim and gt.calibration is None   # features differ: cleared
        assert sim._tracker is gt                      # simulated camera attached
        t_end = time.time() + 3
        while (gt.latest() is None or gt.latest().frame < 5) and time.time() < t_end:
            time.sleep(0.05)
        s = gt.latest()
        assert s is not None and len(s.eyes) == 2      # tracking resumed on the new camera


def test_gui_camera_selector(fake_camera, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("OPENGP3_HOME", str(tmp_path))
    from opengp3.gui import _import_qt, main_window_class
    from opengp3.server import OpenGazeServer
    _import_qt()
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(sources, "gazepoint_devices", lambda: ["GP3"])
    monkeypatch.setattr(sources, "webcam_names", lambda: [])
    real_open = sources.open_source
    monkeypatch.setattr(sources, "open_source",
                        lambda src: fake_camera if src.kind == "gazepoint" else real_open(src))
    # a saved simulated calibration must be picked up after switching
    sim_cal = tmp_path / "calibration-simulated.json"
    gt0 = GazeTracker(Screen(1920, 1080), camera=SimulatedCamera(), input_monitor=False)
    gt = GazeTracker(Screen(1920, 1080), camera=fake_camera, input_monitor=False)
    with gt:
        srv = OpenGazeServer(gt, "127.0.0.1", port=0).start()
        try:
            w = main_window_class()(gt, srv, lsl=None, media_dir=None,
                                    save_path=tmp_path / "calibration.json", mirror=True,
                                    show_tracker=True, screen_args={})
            labels = [w.cmb_camera.itemText(i) for i in range(w.cmb_camera.count())]
            assert labels == ["Gazepoint GP3", "Simulated GP3 (no hardware)"]
            assert w.cmb_camera.currentIndex() == 0 and w.banner.isHidden()
            # switch to the simulated camera
            w._camera_selected(1)
            assert gt.camera.simulated and w.save_path.name == "calibration-simulated.json"
            assert "Simulated" in w.statusBar().currentMessage()
            assert w.cmb_rate.itemText(0) == "60 Hz" and not w.cmb_rate.isEnabled()
            # a webcam-like camera shows the warning banner; back to the GP3 hides it
            fake_camera.warning = "LOW ACCURACY TEST"
            w._camera_selected(0)
            assert gt.camera is fake_camera and w.save_path.name == "calibration.json"
            assert not w.banner.isHidden() and "LOW ACCURACY TEST" in w.banner.text()
            del fake_camera.warning
            w._update_warning()
            assert w.banner.isHidden()
            w.close()
        finally:
            srv.stop()
