"""Camera selector: listing sources, switching cameras while tracking, and
the Control app's drop-down (no hardware: fake and simulated cameras)."""
import time

import pytest

import opengp3.sources as sources
from opengp3 import GazeTracker, Screen
from opengp3.simulate import SimulatedCamera


def test_list_sources(monkeypatch):
    monkeypatch.setattr(sources, "gazepoint_devices", lambda: ["GP3"])
    monkeypatch.setattr(sources, "webcams", lambda: [
        sources.Webcam(0, "Neko Camera", "continuity"),
        sources.Webcam(1, "MacBook Pro Camera", "builtin"),
        sources.Webcam(2, "OBS Virtual Camera", "virtual")])
    monkeypatch.setattr(sources, "webcam_supported", lambda: False)
    src = sources.list_sources()
    assert [s.kind for s in src] == ["gazepoint", "webcam", "webcam", "webcam", "simulated"]
    assert src[0].label == "Gazepoint GP3"
    # real cameras first; each entry keeps its own OpenCV index
    assert [(s.index, s.label) for s in src[1:4]] == [
        (1, "Webcam: MacBook Pro Camera (low accuracy)"),
        (0, "Webcam: Neko Camera (low accuracy, iPhone/iPad)"),
        (2, "Webcam: OBS Virtual Camera (low accuracy, virtual)")]
    assert not src[1].available and "opengp3[webcam]" in src[1].note   # extra missing
    assert src[-1].available


def test_macos_names_follow_opencv_order(monkeypatch):
    """OpenCV numbers AVFoundation devices by sorting their unique IDs."""
    import json
    report = {"SPCameraDataType": [
        {"_name": "MacBook Pro Camera", "spcamera_model-id": "MacBook Pro Camera",
         "spcamera_unique-id": "6C707041-05AC-0011-0007-000000000001"},
        {"_name": "OBS Virtual Camera", "spcamera_model-id": "OBS Camera Extension",
         "spcamera_unique-id": "7626645E-4425-469E-9D8B-97E0FA59AC75"},
        {"_name": "Neko Camera", "spcamera_model-id": "iPhone15,4",
         "spcamera_unique-id": "4E166378-3B94-4DA3-B382-87C500000001"}]}

    class R:
        stdout = json.dumps(report)
    monkeypatch.setattr(sources.sys, "platform", "darwin")
    monkeypatch.setattr(sources.subprocess, "run", lambda *a, **k: R())
    cams = sources.webcams()
    assert [(w.index, w.name, w.kind) for w in cams] == [
        (0, "Neko Camera", "continuity"), (1, "MacBook Pro Camera", "builtin"),
        (2, "OBS Virtual Camera", "virtual")]
    assert sources.preferred_webcam_order() == [1, 0, 2]


def test_classify():
    assert sources._classify("FaceTime HD Camera") == "builtin"
    assert sources._classify("Logitech StreamCam") == "external"
    assert sources._classify("OBS Virtual Camera", "OBS Camera Extension") == "virtual"
    assert sources._classify("Dummy video device (0x0000)") == "virtual"
    assert sources._classify("Neko Camera", "iPhone15,4") == "continuity"


def test_source_of():
    class Gp3: pass
    class Cam:
        webcam, index = True, 2
    assert sources.source_of(Gp3()) == ("gazepoint", 0)
    assert sources.source_of(Cam()) == ("webcam", 2)
    assert sources.source_of(SimulatedCamera()) == ("simulated", 0)


@pytest.mark.slow
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


@pytest.mark.slow
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
    monkeypatch.setattr(sources, "webcams", lambda: [])
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
            # switch to the simulated camera (announced with a warning)
            with pytest.warns(RuntimeWarning, match="SIMULATED EYE TRACKER"):
                w._camera_selected(1)
            assert gt.camera.simulated and w.save_path.name == "calibration-simulated.json"
            assert not w.banner.isHidden() and "SIMULATED EYE TRACKER" in w.banner.text()
            assert "SIMULATED (no real eye data)" in w.windowTitle()
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
