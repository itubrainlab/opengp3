"""The ``opengp3 server`` application: an Open Gaze API server with the
tracker display, calibration window and keyboard controls of a typical
eye-tracker control program.

Runs the GUI on the main thread (required on macOS); network clients only
queue requests that this loop executes.

Keys - tracker display window:
    c  calibrate            p  gaze pointer on/off (mouse follows gaze)
    g  gain sweep on/off    q  quit
Keys - calibration window (after a calibration):
    c  recalibrate          5 / 9  recalibrate with 5 / 9 points
    1  one-point drift correction
    u  undo (toggle previous calibration)
    s  tone at each point   + / -  slower / faster   d  default speed
    Esc  close the window
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import API_DEFAULT_POINTS, NINE_POINTS, Calibration, CalibrationError
from .display import Cancelled, StimulusWindow
from .server import OpenGazeServer
from .tracker import GazeTracker, distance_label



def default_calibration_path(simulated: bool = False) -> Path:
    """Where the last calibration is kept. Simulated runs use their own file
    so they never overwrite the real tracker's calibration."""
    name = "calibration-simulated.json" if simulated else "calibration.json"
    return Path(os.environ.get("OPENGP3_HOME", Path.home() / ".opengp3")) / name


def render_tracker_view(img: np.ndarray, sample, fps: float, lines=(), mirror: bool = False) -> np.ndarray:
    """Camera image with eye overlay, head-position and status text.

    mirror=True flips the image horizontally so it behaves like a mirror for
    the person in front of the tracker (only the display; data is unchanged).
    Shapes are drawn before flipping, text after, so labels stay readable.
    """
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    eyes = sample.eyes if sample else {}
    for side, e in eyes.items():
        cv2.circle(vis, (int(e.pupil_x), int(e.pupil_y)), int(e.pupil_r) + 1, (0, 255, 0), 1)
        cv2.drawMarker(vis, (int(e.glint_x), int(e.glint_y)), (0, 0, 255), cv2.MARKER_CROSS, 8)
    if mirror:
        vis = cv2.flip(vis, 1)
    for side, e in eyes.items():
        x = w - 1 - e.pupil_x if mirror else e.pupil_x
        cv2.putText(vis, side, (int(x) - 5, int(e.pupil_y) - int(e.pupil_r) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if len(eyes) == 2:
        status, col = "both eyes", (0, 255, 0)
    elif eyes:
        status, col = f"only {'right' if 'R' in eyes else 'left'} eye", (0, 200, 255)
    else:
        status, col = "no eyes", (0, 0, 255)
    if eyes:
        # in camera coordinates the subject's right is the image left
        xs = [e.pupil_x for e in eyes.values()]
        ys = [e.pupil_y for e in eyes.values()]
        m, hint = 0.08, []
        if min(xs) < m * w: hint.append("move to your left")
        if max(xs) > (1 - m) * w: hint.append("move to your right")
        if min(ys) < m * h: hint.append("lower your head / tilt GP3 up")
        if max(ys) > (1 - m) * h: hint.append("raise your head / tilt GP3 down")
        if hint:
            status += " - " + ", ".join(hint)
    cv2.putText(vis, f"{fps:.1f} fps  {status}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    # head depth bar: close (left) .. far (right), ideal in the middle
    z = None
    if sample is not None:
        zs = [m.z_m for m in sample.metrics.values() if m.valid and m.z_m]
        z = zs[0] if zs else None
    if z:
        x0, x1, y = 200, w - 200, h - 22
        cv2.rectangle(vis, (x0, y - 6), (x1, y + 6), (90, 90, 90), 1)
        frac = min(1.0, max(0.0, (z * 100 - 45) / 40))  # 45..85 cm
        good = 55 <= z * 100 <= 75
        cv2.circle(vis, (int(x0 + frac * (x1 - x0)), y), 7, (0, 220, 0) if good else (0, 160, 255), -1)
        cv2.putText(vis, "close", (x0 - 50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, "far", (x1 + 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, distance_label(z * 100), (x0, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    for k, line in enumerate(lines):
        cv2.putText(vis, line, (8, 42 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return vis


class ControlApp:
    TRACKER_WIN = "OpenGP3 tracker (c: calibrate, p: pointer, g: gain sweep, q: quit)"

    def __init__(self, tracker: GazeTracker, server: OpenGazeServer, show_tracker: bool = True,
                 media_dir: str | None = None, save_path: Path | None = None,
                 mirror: bool = True, lsl=None):
        self.gt, self.srv = tracker, server
        self.mirror = mirror
        self.lsl = lsl  # optional LSLStreamer: gets calibration event markers
        self.media_dir = media_dir
        self.save_path = save_path or default_calibration_path()
        self.show_tracker = show_tracker
        server.tracker_display = show_tracker
        self.run = None
        self.cal_win: StimulusWindow | None = None
        self.sound = False
        self.speed = 1.0
        self.message = ""
        self._quit = False

    # ------------------------------------------------------------- actions
    def start_calibration(self, points=None):
        if points is not None:
            self.srv.cal_points = list(points)
        if not self.srv.cal_points:
            self.message = "no calibration points (CALIBRATE_RESET restores the default 5)"
            return
        self.srv.cal_running = True
        self.run = self.gt.start_calibration(
            self.srv.cal_points, delay=self.srv.cal_delay * self.speed,
            timeout=self.srv.cal_timeout * self.speed, retry=False,
            listener=self._calibration_event, sound=self.sound,
            media_dir=self.media_dir, keep_order=True)

    def _calibration_event(self, event, n, fx, fy):
        self.srv.calibration_listener(event, n, fx, fy)
        if self.lsl is not None:
            self.lsl.marker(f"calibration_{event} {n} {fx:.4f} {fy:.4f}")

    def _finish_calibration(self):
        run, self.run = self.run, None
        try:
            cal = self.gt.finish_calibration(run)
            self._save(cal)
            self.message = (f"calibrated: {cal.valid_points}/{len(run.points)} points, "
                            f"average error {cal.average_error:.0f} px")
        except CalibrationError as e:
            run.close()
            self.message = f"calibration failed: {e}"
        self.srv.send_calibration_result()

    def _save(self, cal: Calibration):
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            cal.save(str(self.save_path))
        except OSError:
            pass

    def _open_cal_window(self):
        if self.cal_win is None:
            self.cal_win = self.gt.window()

    def _close_cal_window(self):
        if self.cal_win is not None:
            self.gt.close_window()
            self.cal_win = None

    # ------------------------------------------------------------- loop
    def _requests(self):
        while not self.srv.requests.empty():
            what, arg = self.srv.requests.get_nowait()
            if what == "calibrate":
                if self.srv.cal_show:
                    self._open_cal_window()
                self.start_calibration()
            elif what == "calibrate_stop" and self.run is not None:
                self.run.close()
                self.run = None
                self.srv.cal_running = False
            elif what == "calibrate_show":
                if arg:
                    self._open_cal_window()
                elif self.run is None:
                    self._close_cal_window()
            elif what == "tracker_display":
                self.show_tracker = bool(arg)
                if not arg:
                    try:
                        cv2.destroyWindow(self.TRACKER_WIN)
                    except cv2.error:
                        pass

    def _draw_result(self, win: StimulusWindow):
        """After calibration: targets, per-eye calibration results and live gaze."""
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
        win.text(self.message or "no calibration", 0.06, 0.7, (220, 220, 220))
        win.text("c: recalibrate   5/9: points   1: one-point   u: undo   s: sound "
                 f"({'on' if self.sound else 'off'})   +/-/d: speed x{self.speed:.2f}   Esc: close",
                 0.95, 0.55, (150, 150, 150))
        key = win.show()
        self._result_key(key)

    def _result_key(self, key: int):
        if key < 0:
            return
        ch = chr(key).lower() if key < 256 else ""
        if ch == "c":
            self.start_calibration()
        elif ch == "5":
            self.start_calibration(API_DEFAULT_POINTS)
        elif ch == "9":
            self.start_calibration(NINE_POINTS)
        elif ch == "1":
            try:
                cal = self.gt.calibrate_one_point(window=self.cal_win, wait_for_eyes=False)
                self._save(cal)
                self.message = "one-point correction applied"
            except CalibrationError as e:
                self.message = str(e)
        elif ch == "u":
            self.gt.undo_calibration()
            self.message = "switched to the other calibration (u again to toggle back)"
        elif ch == "s":
            self.sound = not self.sound
        elif ch == "+":
            self.speed *= 1.25
        elif ch == "-":
            self.speed /= 1.25
        elif ch == "d":
            self.speed = 1.0

    def _draw_tracker(self):
        img = self.gt.latest_image()
        if img is None:
            return
        s = self.gt.latest()
        cal = self.gt.calibration
        clients = len(self.srv.clients)
        lines = [f"server {self.srv.host}:{self.srv.port} - {clients} client(s)",
                 f"calibration: {'yes' if cal else 'none'}   IPD model {self.gt.eye_model.ipd_mm:.1f} mm"
                 + ("   pointer ON" if self.gt.gaze_pointer else "")
                 + ("   gain sweep ON" if self.gt.auto_gain else "")]
        if self.message:
            lines.append(self.message)
        if self.lsl is not None:
            lines.append(f"LSL streaming: {self.lsl.samples} samples")
        cv2.imshow(self.TRACKER_WIN, render_tracker_view(img, s, self.gt.fps, lines, self.mirror))

    def _tracker_key(self, key: int):
        if key < 0 or key == 255:
            return
        ch = chr(key).lower()
        if ch == "q":
            self._quit = True
        elif ch == "c":
            self.srv.cal_show = True
            self._open_cal_window()
            self.start_calibration()
        elif ch == "p":
            self.gt.gaze_pointer = not self.gt.gaze_pointer and self.gt.calibration is not None
        elif ch == "g":
            self.gt.auto_gain = not self.gt.auto_gain

    def loop(self):
        while not self._quit:
            self._requests()
            if self.run is not None:
                win = self.cal_win if self.srv.cal_show else None
                try:
                    if self.run.tick(win):
                        self._finish_calibration()
                except Cancelled:
                    self.run.close()
                    self.run = None
                    self.srv.cal_running = False
                    self.srv.cal_show = False
                    self._close_cal_window()
                if win is None:
                    time.sleep(0.005)
            elif self.cal_win is not None:
                try:
                    self._draw_result(self.cal_win)
                except Cancelled:
                    self.srv.cal_show = False
                    self._close_cal_window()
            if self.show_tracker and (self.run is None or not self.srv.cal_show):
                self._draw_tracker()
            if self.cal_win is None or self.run is None:
                key = cv2.waitKey(10) & 0xFF
                self._tracker_key(-1 if key == 255 else key)
        self._close_cal_window()
        cv2.destroyAllWindows()
