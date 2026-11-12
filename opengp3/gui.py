"""OpenGP3 Control: the Qt application of ``opengp3 server`` (the default
command). It is the counterpart of Gazepoint Control: tracker display with
head-positioning guidance, calibration and its results, gaze pointer, screen
selection and the Open Gaze API server, as one window.

Window layout
-------------
* left: the camera image with pupil/glint overlay (mirrored like a mirror),
  head-position box and distance scale;
* right: panels for the eye tracker, calibration (points, speed, sound,
  custom targets, one-point drift correction, undo, validation, results per
  point), gaze output (gaze pointer, AAC filter, single-eye tracking, gain
  sweep, pupil-size marker), screen (display, physical width, distance, 22"
  area) and the API server (address, clients, LSL);
* calibration and validation run in a fullscreen window on the selected
  screen; after a calibration it shows the targets with each eye's
  calibrated gaze and the live gaze point.

Network clients only queue requests (calibrate, show/hide windows); the GUI
thread executes them, as OpenCV/Qt windows must live on the main thread.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from .calibration import (API_DEFAULT_POINTS, NINE_POINTS, Calibration, CalibrationError,
                          grid_points, presentation_order)
from .display import Cancelled
from .screen import Screen
from .server import API_VERSION, OpenGazeServer
from .tracker import GazeTracker, distance_label

POINT_SETS = {5: API_DEFAULT_POINTS, 9: NINE_POINTS}
SPEEDS = {"Slow": 1.5, "Normal": 1.0, "Fast": 0.7}


def calibration_points(n: int) -> list[tuple[float, float]]:
    """Gazepoint's 5- and 9-point layouts, or an n-point grid."""
    if n in POINT_SETS:
        return list(POINT_SETS[n])
    pts = grid_points(n)
    return [pts[i] for i in presentation_order(pts)]


def point_errors(cal: Calibration) -> list[tuple[int, float, float, dict[str, float | None]]]:
    """Per calibration target: (target, x, y, {eye: error in screen points or None})."""
    out = []
    for p in sorted(cal.points, key=lambda p: p.target):
        errs = {}
        for eye in ("R", "L"):
            g = p.gaze.get(eye)
            errs[eye] = (float(np.hypot(g[0] - p.x, g[1] - p.y))
                         if g is not None and p.valid.get(eye, True) else None)
        out.append((p.target, p.x, p.y, errs))
    return out


def _import_qt():
    # opengp3 imports OpenCV, whose Linux wheels point Qt at their own (Qt 5)
    # plugins: undo that before PySide6 (Qt 6) loads its platform plugin
    for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        if "cv2" in os.environ.get(var, ""):
            os.environ.pop(var)
    import PySide6  # noqa: F401  (ImportError -> caller falls back)


def run_app(tracker: GazeTracker, server: OpenGazeServer, *, lsl=None, media_dir=None,
            save_path: Path | None = None, mirror: bool = True, show_tracker: bool = True,
            screen_args: dict | None = None) -> int:
    """Run the Qt control app until its window is closed."""
    _import_qt()
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("OpenGP3")
    from .qtwindow import QtStimulusWindow
    tracker.window_factory = QtStimulusWindow
    w = main_window_class()(tracker, server, lsl=lsl, media_dir=media_dir, save_path=save_path,
                            mirror=mirror, show_tracker=show_tracker,
                            screen_args=screen_args or {})
    w.show()
    return app.exec()


def _make_main_window():
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                                   QFileDialog,
                                   QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                                   QHeaderView, QLabel, QMainWindow, QPlainTextEdit, QPushButton,
                                   QScrollArea, QSizePolicy, QSpinBox, QTableWidget,
                                   QTableWidgetItem, QVBoxLayout, QWidget)

    from .control import default_calibration_path, render_tracker_view

    class HeadBox(QWidget):
        """Head-positioning guide: where the eyes are in the camera's view
        (mirrored, like a mirror) and how far the head is, coloured green /
        orange / red for good / borderline / poor."""

        def __init__(self):
            super().__init__()
            self.setMinimumSize(220, 150)
            self.eyes, self.size, self.z = {}, (1, 1), None

        def set(self, eyes, size, z):
            self.eyes, self.size, self.z = eyes, size, z
            self.update()

        def paintEvent(self, _):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            W, H = self.width(), self.height()
            iw, ih = self.size
            bh = H - 40
            bw = min(W - 20, bh * iw / ih)
            bh = bw * ih / iw
            x0, y0 = (W - bw) / 2, 6
            p.fillRect(int(x0), int(y0), int(bw), int(bh), QColor(30, 30, 30))
            p.setPen(QPen(QColor(90, 90, 90), 1))
            p.drawRect(int(x0), int(y0), int(bw), int(bh))
            m = 0.1
            p.setPen(QPen(QColor(60, 60, 60), 1, Qt.DashLine))
            p.drawRect(int(x0 + m * bw), int(y0 + m * bh), int((1 - 2 * m) * bw), int((1 - 2 * m) * bh))
            for side, e in self.eyes.items():
                fx, fy = 1 - e.pupil_x / iw, e.pupil_y / ih
                inner = m < fx < 1 - m and m < fy < 1 - m
                col = QColor(80, 220, 80) if inner else QColor(255, 160, 0)
                p.setBrush(col)
                p.setPen(Qt.NoPen)
                p.drawEllipse(int(x0 + fx * bw) - 7, int(y0 + fy * bh) - 7, 14, 14)
                p.setPen(col)
                p.drawText(int(x0 + fx * bw) - 4, int(y0 + fy * bh) - 10, side)
            # distance scale: 45 cm (close) .. 85 cm (far), good 55-75 cm
            y = int(y0 + bh + 18)
            p.setPen(QPen(QColor(90, 90, 90), 1))
            p.drawLine(int(x0), y, int(x0 + bw), y)
            p.fillRect(int(x0 + bw * 10 / 40), y - 3, int(bw * 20 / 40), 6, QColor(40, 90, 40))
            if self.z:
                cm = self.z * 100
                frac = min(1.0, max(0.0, (cm - 45) / 40))
                good = 55 <= cm <= 75
                p.setBrush(QColor(80, 220, 80) if good else QColor(255, 160, 0))
                p.setPen(Qt.NoPen)
                p.drawEllipse(int(x0 + frac * bw) - 6, y - 6, 12, 12)
                p.setPen(QColor(200, 200, 200))
                p.drawText(int(x0), y + 16, f"{distance_label(cm)}")
            else:
                p.setPen(QColor(150, 150, 150))
                p.drawText(int(x0), y + 16, "no eyes" if not self.eyes else "distance: both eyes needed")

    class MainWindow(QMainWindow):
        def __init__(self, gt: GazeTracker, srv: OpenGazeServer, *, lsl, media_dir, save_path,
                     mirror, show_tracker, screen_args):
            super().__init__()
            self.gt, self.srv, self.lsl = gt, srv, lsl
            self.media_dir = media_dir
            self.save_path = save_path or default_calibration_path()
            self.mirror = mirror
            self.screen_args = dict(screen_args)
            self.run = None
            self.cal_win = None
            self.sound = False
            self.speed = 1.0
            self.validation = None
            self._busy = False
            self._last_view = 0.0
            self._last_status = 0.0
            self._quit = False
            srv.tracker_display = show_tracker
            self.setWindowTitle(f"OpenGP3 Control - {gt.model}")
            self._build()
            self._refresh_calibration()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(5)

        # ---------------------------------------------------------- layout
        def _build(self):
            root = QWidget()
            lay = QHBoxLayout(root)
            left = QVBoxLayout()
            # degraded-data warning (webcam fallback): impossible to miss
            self.banner = QLabel()
            self.banner.setWordWrap(True)
            self.banner.setStyleSheet("background: #b00020; color: white; font-weight: bold; "
                                      "padding: 8px; border-radius: 4px;")
            left.addWidget(self.banner)
            self._update_warning()
            self.view = QLabel("starting camera...")
            self.view.setAlignment(Qt.AlignCenter)
            self.view.setMinimumSize(560, 448)
            self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.view.setStyleSheet("background: #111; color: #aaa;")
            left.addWidget(self.view, 1)
            row = QHBoxLayout()
            self.cb_display = QCheckBox("Show tracker display")
            self.cb_display.setChecked(self.srv.tracker_display)
            self.cb_display.toggled.connect(self._set_tracker_display)
            self.cb_mirror = QCheckBox("Mirror image")
            self.cb_mirror.setChecked(self.mirror)
            self.cb_mirror.toggled.connect(lambda v: setattr(self, "mirror", v))
            row.addWidget(self.cb_display)
            row.addWidget(self.cb_mirror)
            row.addStretch(1)
            left.addLayout(row)
            lay.addLayout(left, 3)

            panel = QWidget()
            pl = QVBoxLayout(panel)
            pl.addWidget(self._tracker_group())
            pl.addWidget(self._calibration_group())
            pl.addWidget(self._gaze_group())
            pl.addWidget(self._screen_group())
            pl.addWidget(self._server_group())
            quit_btn = QPushButton("Quit")
            quit_btn.clicked.connect(self.close)
            pl.addWidget(quit_btn)
            pl.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidget(panel)
            scroll.setWidgetResizable(True)
            scroll.setMinimumWidth(420)
            lay.addWidget(scroll, 2)
            self.setCentralWidget(root)
            self.statusBar().showMessage("ready")
            self.resize(1280, 820)

        def _tracker_group(self):
            g = QGroupBox("Eye tracker")
            f = QFormLayout(g)
            # camera selector: Gazepoint tracker(s), webcams (fallback), simulated
            self.cmb_camera = QComboBox()
            self.cmb_camera.setToolTip("Webcam eye tracking is far less accurate than a "
                                       "Gazepoint eye tracker")
            self.cmb_camera.activated.connect(self._camera_selected)
            f.addRow("Camera", self.cmb_camera)
            self.lb_model = QLabel()
            f.addRow("Model", self.lb_model)
            self.lb_camera = QLabel()
            f.addRow("Camera image", self.lb_camera)
            self.cmb_rate = QComboBox()
            self.cmb_rate.currentIndexChanged.connect(self._set_rate)
            f.addRow("Frame rate", self.cmb_rate)
            self.lb_illum = QLabel()
            f.addRow("Illumination", self.lb_illum)
            self._fill_cameras()
            self._update_camera_info()
            self.lb_fps = QLabel("-")
            f.addRow("Tracking", self.lb_fps)
            self.lb_eyes = QLabel("-")
            f.addRow("Eyes", self.lb_eyes)
            self.head = HeadBox()
            f.addRow("Head position", self.head)
            return g

        def _calibration_group(self):
            g = QGroupBox("Calibration")
            v = QVBoxLayout(g)
            grid = QGridLayout()
            self.cmb_points = QComboBox()
            for n in (5, 9, 13, 16, 25):
                self.cmb_points.addItem(f"{n} points", n)
            self.cmb_points.setCurrentIndex(1 if len(self.srv.cal_points) == 9 else 0)
            self.cmb_points.currentIndexChanged.connect(self._set_points)
            self.cmb_speed = QComboBox()
            self.cmb_speed.addItems(list(SPEEDS))
            self.cmb_speed.setCurrentText("Normal")
            self.cmb_speed.currentTextChanged.connect(lambda t: setattr(self, "speed", SPEEDS[t]))
            self.cb_sound = QCheckBox("Tone at each point")
            self.cb_sound.toggled.connect(lambda v: setattr(self, "sound", v))
            self.btn_media = QPushButton("Target images...")
            self.btn_media.setToolTip("Folder with CalibPoint<N>.png images and .mp3 sounds")
            self.btn_media.clicked.connect(self._choose_media)
            self.lb_media = QLabel(self._media_label())
            grid.addWidget(QLabel("Points"), 0, 0)
            grid.addWidget(self.cmb_points, 0, 1)
            grid.addWidget(QLabel("Speed"), 0, 2)
            grid.addWidget(self.cmb_speed, 0, 3)
            grid.addWidget(self.cb_sound, 1, 0, 1, 2)
            grid.addWidget(self.btn_media, 1, 2)
            grid.addWidget(self.lb_media, 1, 3)
            v.addLayout(grid)
            row = QHBoxLayout()
            self.btn_cal = QPushButton("Calibrate")
            self.btn_cal.setStyleSheet("font-weight: bold;")
            self.btn_cal.clicked.connect(self._calibrate_clicked)
            self.btn_one = QPushButton("1-point")
            self.btn_one.setToolTip("One-point drift correction of the current calibration")
            self.btn_one.clicked.connect(self._one_point)
            self.btn_undo = QPushButton("Undo")
            self.btn_undo.setToolTip("Switch back to the previous calibration (again to toggle)")
            self.btn_undo.clicked.connect(self._undo)
            self.btn_val = QPushButton("Validate")
            self.btn_val.setToolTip("Accuracy and precision on points the calibration didn't use")
            self.btn_val.clicked.connect(self._validate)
            self.btn_results = QPushButton("Show results")
            self.btn_results.clicked.connect(self._show_results)
            for b in (self.btn_cal, self.btn_one, self.btn_undo, self.btn_val, self.btn_results):
                row.addWidget(b)
            v.addLayout(row)
            self.lb_cal = QLabel("no calibration")
            self.lb_cal.setWordWrap(True)
            v.addWidget(self.lb_cal)
            self.tbl = QTableWidget(0, 5)
            self.tbl.setHorizontalHeaderLabels(["#", "x", "y", "right eye", "left eye"])
            self.tbl.verticalHeader().setVisible(False)
            self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            self.tbl.setMaximumHeight(200)
            v.addWidget(self.tbl)
            return g

        def _gaze_group(self):
            g = QGroupBox("Gaze")
            f = QFormLayout(g)
            self.cb_pointer = QCheckBox("Mouse follows gaze (gaze pointer)")
            self.cb_pointer.toggled.connect(self._set_pointer)
            f.addRow(self.cb_pointer)
            self.sp_aac = QSpinBox()
            self.sp_aac.setRange(1, 300)
            self.sp_aac.setValue(self.gt.aac.length)
            self.sp_aac.setSuffix(" samples")
            self.sp_aac.valueChanged.connect(self.gt.set_aac_length)
            f.addRow("AAC filter", self.sp_aac)
            self.cmb_mono = QComboBox()
            self.cmb_mono.addItems(["Both eyes", "Right eye only", "Left eye only"])
            self.cmb_mono.setCurrentIndex({None: 0, "R": 1, "L": 2}[self.gt.monocular])
            self.cmb_mono.currentIndexChanged.connect(
                lambda i: setattr(self.gt, "monocular", [None, "R", "L"][i]))
            f.addRow("Track", self.cmb_mono)
            self.cb_gain = QCheckBox("Gain sweep while no eyes are found")
            self.cb_gain.setChecked(self.gt.auto_gain)
            self.cb_gain.toggled.connect(lambda v: setattr(self.gt, "auto_gain", v))
            f.addRow(self.cb_gain)
            row = QHBoxLayout()
            self.cb_marker = QCheckBox("Printed marker")
            self.sp_marker = QDoubleSpinBox()
            self.sp_marker.setRange(3, 50)
            self.sp_marker.setValue(self.srv.marker_mm)
            self.sp_marker.setSuffix(" mm")
            self.cb_marker.toggled.connect(self._set_marker)
            self.sp_marker.valueChanged.connect(lambda _: self._set_marker(self.cb_marker.isChecked()))
            row.addWidget(self.cb_marker)
            row.addWidget(self.sp_marker)
            f.addRow("Pupil size", row)
            return g

        def _screen_group(self):
            from .platform import displays
            g = QGroupBox("Screen")
            f = QFormLayout(g)
            self.cmb_screen = QComboBox()
            self.displays = displays()
            for i, d in enumerate(self.displays):
                size = f", {d.width_mm / 10:.1f} cm" if d.width_mm else ""
                self.cmb_screen.addItem(f"{i}: {int(d.width)} x {int(d.height)}{size}"
                                        + (" (main)" if d.main else ""), i)
            self.cmb_screen.setCurrentIndex(self.screen_args.get("display", 0))
            self.cmb_screen.currentIndexChanged.connect(self._screen_changed)
            f.addRow("Display", self.cmb_screen)
            self.sp_width = QDoubleSpinBox()
            self.sp_width.setRange(0, 500)
            self.sp_width.setSuffix(" cm")
            self.sp_width.setSpecialValueText("unknown")
            self.sp_width.setValue(self.gt.screen.width_cm or 0)
            self.sp_width.editingFinished.connect(self._apply_screen)
            f.addRow("Physical width", self.sp_width)
            self.sp_dist = QDoubleSpinBox()
            self.sp_dist.setRange(30, 150)
            self.sp_dist.setSuffix(" cm")
            self.sp_dist.setValue(self.gt.screen.distance_cm)
            self.sp_dist.editingFinished.connect(self._apply_screen)
            f.addRow("Viewing distance", self.sp_dist)
            self.cb_area = QCheckBox("Calibrate within a 22\" area on larger screens")
            self.cb_area.setChecked(self.gt.max_diagonal_in is not None)
            self.cb_area.toggled.connect(self._apply_screen)
            f.addRow(self.cb_area)
            self.lb_area = QLabel()
            f.addRow("Calibration area", self.lb_area)
            self._update_area_label()
            return g

        # ----------------------------------------------------- camera
        def _fill_cameras(self):
            from .sources import list_sources, source_of, webcam_tooltip
            self.sources = list_sources()
            self.cmb_camera.setToolTip(webcam_tooltip())
            current = source_of(self.gt.camera)
            self.cmb_camera.blockSignals(True)
            self.cmb_camera.clear()
            for i, src in enumerate(self.sources):
                self.cmb_camera.addItem(src.label + ("" if src.available else f"  [{src.note}]"))
                item = self.cmb_camera.model().item(i)
                if item is not None and not src.available:
                    item.setEnabled(False)
                if src.key == current:
                    self.cmb_camera.setCurrentIndex(i)
            self.cmb_camera.blockSignals(False)

        def _update_camera_info(self):
            cam = self.gt.camera
            serial = getattr(cam, "serial", None) or self.srv._serial()
            self.lb_model.setText(f"{self.gt.model}" + (f"  (serial {serial})" if serial else ""))
            self.lb_camera.setText(f"{self.gt.image_size[0]} x {self.gt.image_size[1]}")
            self.cmb_rate.blockSignals(True)
            self.cmb_rate.clear()
            if getattr(cam, "model", "") == "GP3 HD" and hasattr(cam, "set_rate"):
                self.cmb_rate.addItems(["60 Hz", "150 Hz"])
                self.cmb_rate.setCurrentIndex(1 if self.gt.rate > 100 else 0)
                self.cmb_rate.setEnabled(True)
            else:
                self.cmb_rate.addItem(f"{self.gt.rate:.0f} Hz")
                self.cmb_rate.setEnabled(False)
            self.cmb_rate.blockSignals(False)
            illum = {"bright-dark": "bright/dark pupil (Gazepoint)", "dark": "dark pupil",
                     "webcam": "none (webcam, visible light)"}
            self.lb_illum.setText(illum.get(self.gt.illumination, self.gt.illumination))
            self._update_warning()

        def _update_warning(self):
            warning = self.gt.warning
            if not hasattr(self, "banner"):
                return
            self.banner.setText(f"⚠ {warning}" if warning else "")
            self.banner.setVisible(bool(warning))
            title = f"OpenGP3 Control - {self.gt.model}"
            self.setWindowTitle(title + (" - WEBCAM FALLBACK (low accuracy)" if warning else ""))

        def _camera_selected(self, i: int):
            from .calibration import Calibration
            from .control import default_calibration_path
            from .sources import open_source, source_of
            if i >= len(self.sources):
                return
            src = self.sources[i]
            if src.key == source_of(self.gt.camera):
                return
            if self.run is not None or self._busy:
                self._message("finish the calibration first")
                self._fill_cameras()
                return
            self._message(f"opening {src.label}...")
            QApplication.processEvents()
            try:
                cam = open_source(src)
            except Exception as e:
                self._message(f"could not open {src.label}: {e}")
                self._fill_cameras()
                return
            self.gt.switch_camera(cam)
            # this camera's own calibration (features differ between cameras)
            self.save_path = default_calibration_path(src.kind == "simulated", src.kind == "webcam")
            if self.save_path.exists():
                try:
                    cal = Calibration.load(str(self.save_path))
                    self.gt.calibration = cal
                    self.gt._apply_calibration_meta(cal)
                except Exception:
                    pass
            if self.lsl is not None:  # LSL metadata (rate, warning) must match the camera
                from .lsl import LSLStreamer
                self.lsl.close()
                self.lsl = LSLStreamer(self.gt)
            self._update_camera_info()
            self._refresh_calibration()
            self._message(f"camera: {src.label}"
                          + ("" if self.gt.calibration else " - not calibrated yet"))

        def _server_group(self):
            g = QGroupBox("Open Gaze API server")
            f = QFormLayout(g)
            self.lb_address = QLabel()
            self.lb_address.setTextInteractionFlags(Qt.TextSelectableByMouse)
            f.addRow("Connect to", self.lb_address)
            self.cb_remote = QCheckBox("Allow connections from other computers")
            self.cb_remote.setToolTip("The API has no password: anyone on these networks can "
                                      "read gaze data and send commands")
            self.cb_remote.setChecked(not self.srv.local_only)
            self.cb_remote.toggled.connect(self._set_remote)
            f.addRow("", self.cb_remote)
            self._update_address()
            f.addRow("API version", QLabel(API_VERSION))
            self.lb_clients = QLabel("0")
            f.addRow("Clients", self.lb_clients)
            self.lb_lsl = QLabel("off" if self.lsl is None else "streaming")
            f.addRow("Lab Streaming Layer", self.lb_lsl)
            return g

        def _update_address(self):
            self.lb_address.setText("\n".join(self.srv.addresses()))

        def _set_remote(self, remote: bool):
            try:
                self.srv.set_local_only(not remote)
            except OSError as e:
                self._message(f"could not change the API address: {e}")
                self.cb_remote.blockSignals(True)
                self.cb_remote.setChecked(not self.srv.local_only)
                self.cb_remote.blockSignals(False)
                return
            self._update_address()
            self._message("API reachable from other computers" if remote
                          else "API restricted to this computer")

        # --------------------------------------------------------- actions
        def _message(self, text: str):
            self.statusBar().showMessage(text)

        def _set_tracker_display(self, on: bool):
            self.srv.tracker_display = on
            self.view.setVisible(on)
            self.cb_mirror.setEnabled(on)

        def _set_rate(self, i: int):
            hz = (60, 150)[i]
            try:
                self.gt.camera.set_rate(hz)
                self._message(f"frame rate {self.gt.rate:.0f} Hz")
            except Exception as e:
                self._message(f"could not change the frame rate: {e}")

        def _set_points(self, _):
            self.srv.cal_points = calibration_points(self.cmb_points.currentData())

        def _media_label(self):
            return Path(self.media_dir).name if self.media_dir else "default dots"

        def _choose_media(self):
            d = QFileDialog.getExistingDirectory(self, "Folder with calibration target images",
                                                 self.media_dir or str(Path.home()))
            self.media_dir = d or None
            self.lb_media.setText(self._media_label())

        def _set_pointer(self, on: bool):
            if on and self.gt.calibration is None:
                self.cb_pointer.setChecked(False)
                self._message("calibrate first: the gaze pointer needs a calibration")
                return
            self.gt.gaze_pointer = on

        def _set_marker(self, on: bool):
            self.srv.marker_mm = float(self.sp_marker.value())
            self.srv.marker_on = on
            self.gt.enable_marker(self.srv.marker_mm if on else None)

        def _screen_changed(self, i: int):
            # the item's data is the display number; the row index need not be
            n = self.cmb_screen.itemData(i)
            n = i if n is None else n
            d = self.displays[n] if 0 <= n < len(self.displays) else None
            self.sp_width.setValue(d.width_mm / 10 if d is not None and d.width_mm else 0)
            self._apply_screen()

        def _apply_screen(self, *_):
            if self.run is not None or self._busy:
                return
            i = self.cmb_screen.currentData() or 0
            width = self.sp_width.value() or None
            try:
                scr = Screen.detect(width_cm=width, distance_cm=self.sp_dist.value(), display=i)
            except (ValueError, IndexError) as e:
                self._message(str(e))
                return
            self.gt.max_diagonal_in = 22.0 if self.cb_area.isChecked() else None
            self.cal_win = None
            self.gt.set_screen(scr)
            self._update_area_label()
            self._message(f"screen {scr.width} x {scr.height}"
                          + (f", {scr.width_cm:.1f} cm" if scr.width_cm else ""))

        def _update_area_label(self):
            a, s = self.gt.default_area(), self.gt.screen
            if a.w >= s.width and a.h >= s.height:
                self.lb_area.setText("whole screen")
            else:
                self.lb_area.setText(f"centred {a.w:.0f} x {a.h:.0f} (22\")")

        # ----------------------------------------------------- calibration
        def _open_cal_window(self):
            if self.cal_win is None:
                self.cal_win = self.gt.window()

        def _close_cal_window(self):
            if self.cal_win is not None:
                self.gt.close_window()
                self.cal_win = None

        def _calibrate_clicked(self):
            if self.run is not None:
                self._stop_calibration()
                return
            self.srv.cal_show = True
            self._open_cal_window()
            self.start_calibration()

        def start_calibration(self, points=None):
            if points is not None:
                self.srv.cal_points = list(points)
            if not self.srv.cal_points:
                self._message("no calibration points (CALIBRATE_RESET restores the default)")
                return
            self.srv.cal_running = True
            self.run = self.gt.start_calibration(
                self.srv.cal_points, delay=self.srv.cal_delay * self.speed,
                timeout=self.srv.cal_timeout * self.speed, retry=False,
                listener=self._calibration_event, sound=self.sound,
                media_dir=self.media_dir, keep_order=True)
            self.btn_cal.setText("Stop")
            self._message(f"calibrating: {len(self.srv.cal_points)} points")

        def _stop_calibration(self):
            if self.run is not None:
                self.run.close()
                self.run = None
            self.srv.cal_running = False
            self.btn_cal.setText("Calibrate")
            self._message("calibration stopped")

        def _calibration_event(self, event, n, fx, fy):
            self.srv.calibration_listener(event, n, fx, fy)
            if self.lsl is not None:
                self.lsl.marker(f"calibration_{event} {n} {fx:.4f} {fy:.4f}")

        def _finish_calibration(self):
            run, self.run = self.run, None
            self.btn_cal.setText("Calibrate")
            try:
                cal = self.gt.finish_calibration(run)
                self._save(cal)
                self._message(f"calibrated: {cal.valid_points}/{len(run.points)} points, "
                              f"average error {cal.average_error:.0f} px")
            except CalibrationError as e:
                run.close()
                self._message(f"calibration failed: {e}")
            self.srv.send_calibration_result()
            self._refresh_calibration()

        def _save(self, cal: Calibration):
            try:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                cal.save(str(self.save_path))
            except OSError:
                pass

        def _blocking(self, fn):
            """Run a calibration-window task that has its own loop (one-point,
            validation); the window's show() keeps Qt responsive meanwhile."""
            if self._busy or self.run is not None:
                return None
            self._busy = True
            try:
                self.srv.cal_show = True
                self._open_cal_window()
                return fn()
            except Cancelled:
                self._message("cancelled")
            except (CalibrationError, RuntimeError) as e:
                self._message(str(e))
            finally:
                self._busy = False
            return None

        def _one_point(self):
            if self.gt.calibration is None:
                self._message("calibrate first: one-point correction adjusts a calibration")
                return
            cal = self._blocking(lambda: self.gt.calibrate_one_point(window=self.cal_win,
                                                                     wait_for_eyes=False))
            if cal is not None:
                self._save(cal)
                self._message("one-point drift correction applied")
                self._refresh_calibration()

        def _undo(self):
            cal = self.gt.undo_calibration()
            self._message("switched to the other calibration (Undo again to toggle back)"
                          if cal is not None else "no previous calibration")
            self._refresh_calibration()

        def _validate(self):
            if self.gt.calibration is None:
                self._message("calibrate first")
                return
            res = self._blocking(lambda: self.gt.validate(window=self.cal_win))
            if res is None:
                return
            self.validation = res
            acc = res.accuracy_deg
            self._message(f"validation: accuracy {acc:.2f} deg, precision "
                          f"{res.precision_rms_s2s_deg:.2f} deg" if acc is not None else
                          f"validation: mean offset {res.offset_points:.0f} px "
                          "(set the screen's physical width for degrees)")
            self._show_text("Validation", res.report())
            self._refresh_calibration()

        def _show_results(self):
            if self.gt.calibration is None:
                self._message("no calibration yet")
                return
            self.srv.cal_show = True
            self._open_cal_window()

        def _show_text(self, title, text):
            d = QDialog(self)
            d.setWindowTitle(title)
            lay = QVBoxLayout(d)
            t = QPlainTextEdit(text)
            t.setReadOnly(True)
            t.setFont(QFont("Courier New", 10))
            lay.addWidget(t)
            d.resize(760, 360)
            d.show()

        def _refresh_calibration(self):
            cal = self.gt.calibration
            self.btn_one.setEnabled(cal is not None)
            self.btn_val.setEnabled(cal is not None)
            self.btn_results.setEnabled(cal is not None)
            self.btn_undo.setEnabled(self.gt.previous_calibration is not None)
            if cal is None:
                self.lb_cal.setText("no calibration")
                self.tbl.setRowCount(0)
                return
            k = self.gt.screen.cm_per_point
            parts = []
            for eye, name in (("R", "right"), ("L", "left")):
                f = cal.eyes.get(eye)
                if f is None:
                    parts.append(f"{name} eye: not calibrated")
                    continue
                model = getattr(f.mapping, "model", getattr(f.mapping, "name", ""))
                parts.append(f"{name} eye: {f.n_points} points, {model} model, "
                             f"error {f.residual:.0f} px (leave-one-out {f.loo_error:.0f} px)")
            avg = cal.average_error
            line = f"{cal.valid_points}/{len(cal.points)} points valid, average error {avg:.0f} px"
            if k:
                deg = np.degrees(np.arctan(avg * k / self.gt.screen.distance_cm))
                line += f" ({deg:.2f} deg)"
            if self.validation is not None and self.validation.accuracy_deg is not None:
                v = self.validation
                line += (f"\nlast validation: accuracy {v.accuracy_deg:.2f} deg, "
                         f"precision {v.precision_rms_s2s_deg:.2f} deg RMS-S2S")
            self.lb_cal.setText(line + "\n" + "\n".join(parts))
            rows = point_errors(cal)
            self.tbl.setRowCount(len(rows))
            for r, (t, x, y, errs) in enumerate(rows):
                vals = [str(t + 1), f"{x:.0f}", f"{y:.0f}"] + [
                    "-" if errs[e] is None else f"{errs[e]:.0f} px" for e in ("R", "L")]
                for c, val in enumerate(vals):
                    it = QTableWidgetItem(val)
                    it.setTextAlignment(Qt.AlignCenter)
                    if c >= 3:
                        e = errs["RL"[c - 3]]
                        col = (QColor(200, 60, 60) if e is None else
                               QColor(40, 150, 40) if e < 40 else QColor(220, 140, 0))
                        it.setForeground(col)
                    self.tbl.setItem(r, c, it)

        def _draw_result(self, win):
            """Calibration window after a calibration: targets, each eye's
            calibrated gaze and the live gaze point (Esc closes)."""
            gt = self.gt
            win.clear((25, 25, 25))
            cal = gt.calibration
            if cal is not None:
                if cal.area.w < win.screen.width or cal.area.h < win.screen.height:
                    win.rect(cal.area)
                for p in cal.points:
                    win.dot(p.x, p.y, 14, (200, 200, 200), 2)
                    for eye, col in (("L", (255, 160, 60)), ("R", (60, 160, 255))):
                        g = p.gaze.get(eye)
                        if g:
                            win.dot(g[0], g[1], 5, col)
            s = gt.latest()
            if s is not None and s.valid:
                win.dot(s.x, s.y, 16, (60, 220, 60), 3)
            win.text(self.statusBar().currentMessage() or "no calibration", 0.06, 0.7,
                     (220, 220, 220))
            win.text("blue: right eye   orange: left eye   green: live gaze      "
                     "c: recalibrate   1: one-point   u: undo   Esc: close",
                     0.95, 0.55, (150, 150, 150))
            key = win.show()
            ch = chr(key).lower() if 0 <= key < 256 else ""
            if ch == "c":
                self.start_calibration()
            elif ch == "1":
                self._one_point()
            elif ch == "u":
                self._undo()

        # ------------------------------------------------------------ loop
        def _requests(self):
            while not self.srv.requests.empty():
                what, arg = self.srv.requests.get_nowait()
                if what == "calibrate":
                    if self.srv.cal_show:
                        self._open_cal_window()
                    self.start_calibration()
                elif what == "calibrate_stop" and self.run is not None:
                    self._stop_calibration()
                elif what == "calibrate_show":
                    if arg:
                        self._open_cal_window()
                    elif self.run is None:
                        self._close_cal_window()
                elif what == "tracker_display":
                    self.cb_display.setChecked(bool(arg))

        def _tick(self):
            if self._busy:  # a blocking task is running its own loop
                return
            self._requests()
            try:
                if self.run is not None:
                    win = self.cal_win if self.srv.cal_show else None
                    try:
                        if self.run.tick(win):
                            self._finish_calibration()
                    except Cancelled:
                        self._stop_calibration()
                        self.srv.cal_show = False
                        self._close_cal_window()
                elif self.cal_win is not None:
                    try:
                        self._draw_result(self.cal_win)
                    except Cancelled:
                        self.srv.cal_show = False
                        self._close_cal_window()
            except RuntimeError as e:  # e.g. the window could not go fullscreen
                self._message(str(e))
                self._stop_calibration()
                self._close_cal_window()
            now = time.time()
            if now - self._last_view >= 1 / 30:
                self._last_view = now
                self._update_view()
            if now - self._last_status >= 0.25:
                self._last_status = now
                self._update_status()

        def _update_view(self):
            img, s = self.gt.latest_frame()
            if s is not None:
                zs = [m.z_m for m in s.metrics.values() if m.valid and m.z_m]
                self.head.set(s.eyes, self.gt.image_size, zs[0] if zs else None)
            if img is None or not self.srv.tracker_display:
                return
            vis = render_tracker_view(img, s, self.gt.fps, (), self.mirror)
            rgb = np.ascontiguousarray(vis[:, :, ::-1])
            h, w = rgb.shape[:2]
            qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.view.setPixmap(QPixmap.fromImage(qi).scaled(
                self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def _update_status(self):
            gt = self.gt
            self.lb_fps.setText(f"{gt.fps:.1f} samples/s   dropped frames {gt.dropped_frames}")
            s = gt.latest()
            eyes = s.eyes if s else {}
            self.lb_eyes.setText("both" if len(eyes) == 2 else
                                 f"{'right' if 'R' in eyes else 'left'} only" if eyes else "none")
            self.lb_clients.setText(str(len(self.srv.clients)))
            if self.lsl is not None:
                self.lb_lsl.setText(f"streaming ({self.lsl.samples} samples)")
            self.cb_pointer.blockSignals(True)
            self.cb_pointer.setChecked(gt.gaze_pointer)
            self.cb_pointer.blockSignals(False)
            if gt.screen.width_cm and abs(self.sp_width.value() - gt.screen.width_cm) > 0.05 \
                    and not self.sp_width.hasFocus():
                self.sp_width.setValue(gt.screen.width_cm)  # e.g. SCREEN_SIZE from a client

        def closeEvent(self, e):
            self.timer.stop()
            if self.run is not None:
                self.run.close()
                self.run = None
            self._close_cal_window()
            super().closeEvent(e)

    return MainWindow


_main_window = None


def main_window_class():
    """The MainWindow class, built on first use so that importing
    opengp3.gui needs no Qt."""
    global _main_window
    if _main_window is None:
        _import_qt()
        _main_window = _make_main_window()
    return _main_window
