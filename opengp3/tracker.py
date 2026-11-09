"""High-level API: calibrate on the screen and stream gaze in screen points.

    from opengp3 import GazeTracker, Screen

    with GazeTracker(Screen.detect(width_cm=53, distance_cm=65)) as gt:
        gt.calibrate()                      # fullscreen 9-point calibration
        print(gt.validate().report())       # optional accuracy check
        gt.calibration.save("cal.json")
        for s in gt.gaze(duration=10):      # every frame, 60 Hz
            if s.valid:
                print(s.t, s.x, s.y)
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .calibration import (VALIDATION_POINTS, Calibration, CalibrationError,
                          CalibrationSample, ValidationResult, evaluate, grid_points,
                          presentation_order)
from .camera import GP3Camera
from .detection import EyeFeature, EyeTracker, detect_eyes
from .display import Cancelled, StimulusWindow
from .filters import OneEuroFilter
from .screen import Rect, Screen


@dataclass
class GazeSample:
    """One camera frame's worth of tracking output."""
    t: float                         # capture time (time.time() clock)
    frame: int                       # camera frame sequence number
    eyes: dict[str, EyeFeature]      # detected eyes, "R"/"L" = subject's right/left
    x: float | None = None           # gaze on screen, points (smoothed if enabled)
    y: float | None = None
    raw_x: float | None = None       # unsmoothed gaze
    raw_y: float | None = None
    per_eye: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """True when a calibrated eye was visible (x/y available)."""
        return self.x is not None


class GazeTracker:
    """Owns the camera, runs detection on every frame in a background thread,
    and maps eyes to screen coordinates once calibrated.

    screen:      the display gaze is reported on (default: main display).
                 Set ``width_cm``/``distance_cm`` for results in degrees.
    calibration: a saved Calibration (``Calibration.load(path)``), optional.
    smoothing:   One-Euro filter on the gaze point (``x``/``y``); ``raw_x``/
                 ``raw_y`` are always unfiltered. Pass a OneEuroFilter to tune.
    """

    def __init__(self, screen: Screen | None = None, calibration: Calibration | None = None,
                 smoothing: bool | OneEuroFilter = True, camera: GP3Camera | None = None,
                 full_sensor: bool = True):
        self.screen = screen or Screen.detect()
        self.calibration = calibration
        self.filter = (smoothing if isinstance(smoothing, OneEuroFilter)
                       else OneEuroFilter() if smoothing else None)
        self._cam = camera
        self._own_cam = camera is None
        self._full_sensor = full_sensor
        self._thread = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []
        self._callbacks: list[Callable[[GazeSample], None]] = []
        self._latest: GazeSample | None = None
        self.image_size = (752, 480)
        self.dropped_frames = 0
        self.error: Exception | None = None
        self._win: StimulusWindow | None = None
        #: labelled samples of the last calibrate()/validate() run, for diagnostics:
        #: list of (target, target_x, target_y, GazeSample)
        self.last_calibration_samples: list = []
        self.last_validation_samples: list = []

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "GazeTracker":
        if self._thread:
            return self
        if self._cam is None:
            self._cam = GP3Camera.open()
        if self._full_sensor:
            self._cam.set_full_sensor()
        else:
            self._cam.set_video_mode(5, 5)
        self.image_size = (self._cam.width, self._cam.height)
        self._cam.set_leds(True)
        self._cam.start_stream()
        self._running.set()
        self._thread = threading.Thread(target=self._process, name="opengp3", daemon=True)
        self._thread.start()
        return self

    def window(self) -> StimulusWindow:
        """The session's fullscreen stimulus window (created on first use and
        reused, so calibration and validation share identical geometry)."""
        if self._win is None:
            self._win = StimulusWindow(self.screen, "OpenGP3")
        return self._win

    def close_window(self):
        if self._win is not None:
            self._win.close()
            self._win = None

    def stop(self):
        self.close_window()
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cam and self._own_cam:
            self._cam.close()
            self._cam = None

    close = stop

    def __enter__(self) -> "GazeTracker":
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @property
    def camera(self) -> GP3Camera | None:
        return self._cam

    # ------------------------------------------------------------- pipeline
    def _process(self):
        eye_tracker = EyeTracker()
        last_seq = 0
        while self._running.is_set():
            try:
                seq, t, img = self._cam.grab(timeout_s=1.0)
            except TimeoutError as e:
                self.error = e
                continue
            if time.time() - t > 0.15:  # fell behind: skip to the newest frame
                seq, t, img = self._cam.grab(latest=True)
            if last_seq and seq > last_seq + 1:
                self.dropped_frames += seq - last_seq - 1
            last_seq = seq
            eyes = eye_tracker.update(detect_eyes(img))
            s = GazeSample(t, seq, eyes)
            cal = self.calibration
            if cal is not None:
                g = cal.predict(eyes)
                if g is not None:
                    s.raw_x, s.raw_y, s.per_eye = g
                    s.x, s.y = (self.filter(t, s.raw_x, s.raw_y) if self.filter
                                else (s.raw_x, s.raw_y))
            with self._lock:
                self._latest = s
                subs, cbs = list(self._subs), list(self._callbacks)
            for q in subs:
                try:
                    q.put_nowait(s)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(s)
                    except queue.Empty:
                        pass
            for cb in cbs:
                try:
                    cb(s)
                except Exception as e:  # a faulty callback must not kill tracking
                    self.error = e

    # ------------------------------------------------------------- output
    def latest(self) -> GazeSample | None:
        """Most recent sample (non-blocking), or None before the first frame."""
        with self._lock:
            return self._latest

    def subscribe(self, maxsize: int = 6000) -> queue.Queue:
        """Queue receiving every GazeSample (oldest dropped if not consumed)."""
        q = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def add_callback(self, fn: Callable[[GazeSample], None]):
        """Call ``fn(sample)`` for every frame, from the tracking thread
        (keep it fast; do GUI work elsewhere)."""
        with self._lock:
            self._callbacks.append(fn)

    def remove_callback(self, fn):
        with self._lock:
            if fn in self._callbacks:
                self._callbacks.remove(fn)

    def gaze(self, duration: float | None = None, timeout: float = 2.0) -> Iterator[GazeSample]:
        """Iterate over every sample (about 60 Hz) for ``duration`` seconds
        (forever if None). Includes invalid samples (eyes lost/blinks)."""
        self._require_started()
        q = self.subscribe()
        t_end = time.time() + duration if duration else None
        try:
            while t_end is None or time.time() < t_end:
                try:
                    yield q.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError(f"no frames from the GP3 ({self.error})")
        finally:
            self.unsubscribe(q)

    def _require_started(self):
        if not self._thread:
            raise RuntimeError("call start() first (or use 'with GazeTracker() as gt:')")

    # ------------------------------------------------------------- stimuli
    def _wait_for_eyes(self, win: StimulusWindow, title: str, area: Rect,
                       stable_s: float = 1.2, timeout: float = 60.0):
        """Positioning screen: shows the eyes in the camera view and starts
        once they have been stable for ``stable_s`` (Space starts at once)."""
        t0 = time.time()
        since = None
        while True:
            s = self.latest()
            eyes = s.eyes if s else {}
            now = time.time()
            since = (since or now) if eyes else None
            win.clear()
            if area.w < self.screen.width or area.h < self.screen.height:
                win.rect(area)
            win.text(title, 0.32, 1.4)
            win.text("Keep your head still and look at the centre of each dot.", 0.42, 0.8)
            status = {2: "both eyes detected", 1: "one eye detected", 0: "no eyes detected"}[len(eyes)]
            col = (80, 220, 80) if len(eyes) == 2 else (0, 200, 255) if eyes else (60, 60, 255)
            win.text(status + (" - starting..." if since and now - since > 0.3 else ""),
                     0.52, 0.8, col)
            win.eye_box(eyes, self.image_size)
            win.text("Space: start now    Esc: cancel", 0.92, 0.6, (150, 150, 150))
            if win.show() == ord(" ") or (since and now - since >= stable_s):
                return
            if now - t0 > timeout:
                raise CalibrationError("eyes not detected - check the GP3 position (opengp3 view)")

    def _present(self, win: StimulusWindow, targets: list[tuple[int, float, float]],
                 duration: float, area: Rect) -> list[tuple[int, float, float, float, float]]:
        """Show each target; return (idx, x, y, t_on, t_off) with capture-time
        windows, so samples are labelled by when the frame was captured."""
        shown = []
        for idx, x, y in targets:
            t_on = time.time()
            while (el := time.time() - t_on) < duration:
                win.clear()
                if area.w < self.screen.width or area.h < self.screen.height:
                    win.rect(area)
                win.target(x, y, el / duration)
                win.show()
            shown.append((idx, x, y, t_on, time.time()))
        return shown

    @staticmethod
    def _label(samples: list[GazeSample], shown, settle: float):
        """Assign samples to targets by capture time (skip the saccade)."""
        out = []
        for idx, x, y, t_on, t_off in shown:
            for s in samples:
                if t_on + settle <= s.t < t_off:
                    out.append((idx, x, y, s))
        return out

    # ------------------------------------------------------------- calibration
    def calibrate(self, points: int = 9, area: Rect | None = None, target_duration: float = 1.8,
                  settle: float = 0.6, min_samples: int = 10, retry: bool = True,
                  window: StimulusWindow | None = None) -> Calibration:
        """Fullscreen calibration. Call from the main thread.

        points: 5, 9, 13, 16 or 25 targets (more = better edges on big screens)
        area:   Rect to calibrate (default: whole screen; e.g. ``screen.area(22)``)
        The result is stored in ``self.calibration`` and returned. Raises
        CalibrationError if no eye could be calibrated, Cancelled on Esc.
        """
        self._require_started()
        area = area or self.screen.rect
        grid = grid_points(points)
        targets = [(i, *area.to_screen(*grid[i])) for i in presentation_order(grid)]
        win = window or self.window()
        q = self.subscribe()
        try:
            self._wait_for_eyes(win, f"Calibration ({points} points)", area)
            shown = self._present(win, targets, target_duration, area)
            samples = self._drain(q)
            labelled = self._label(samples, shown, settle)
            if retry:  # redo targets where neither eye got enough data once
                counts = {i: max(sum(1 for (j, _, _, s) in labelled if j == i and side in s.eyes)
                                 for side in ("R", "L")) for i, _, _ in targets}
                redo = [t for t in targets if counts[t[0]] < min_samples]
                if redo:
                    shown2 = self._present(win, redo, target_duration, area)
                    samples += self._drain(q)
                    labelled = [l for l in labelled if l[0] not in {t[0] for t in redo}]
                    labelled += self._label(samples, shown2, settle)
        finally:
            self.unsubscribe(q)
        self.last_calibration_samples = labelled
        cal_samples = [CalibrationSample(i, x, y, side, *e.pcr, s.t)
                       for i, x, y, s in labelled for side, e in s.eyes.items()]
        cal = Calibration.fit(cal_samples, self.screen, area, min_samples=min_samples)
        self.calibration = cal
        if self.filter:
            self.filter.reset()
        return cal

    def validate(self, points=VALIDATION_POINTS, area: Rect | None = None,
                 target_duration: float = 1.6, settle: float = 0.6, shuffle: bool = True,
                 window: StimulusWindow | None = None) -> ValidationResult:
        """Measure accuracy on targets the calibration did not use (by
        default 8 points between the 3x3 calibration grid, in random order).
        Uses unsmoothed gaze. Call from the main thread."""
        import random
        self._require_started()
        if self.calibration is None:
            raise RuntimeError("calibrate() or load a calibration first")
        area = area or self.calibration.area
        targets = [(i, *area.to_screen(fx, fy)) for i, (fx, fy) in enumerate(points)]
        if shuffle:
            random.shuffle(targets)
        win = window or self.window()
        q = self.subscribe()
        try:
            self._wait_for_eyes(win, f"Validation ({len(targets)} points)", area)
            shown = self._present(win, targets, target_duration, area)
            labelled = self._label(self._drain(q), shown, settle)
        finally:
            self.unsubscribe(q)
        self.last_validation_samples = labelled
        gaze = [(i, x, y, s.t, s.raw_x, s.raw_y) for i, x, y, s in labelled if s.valid]
        return evaluate(gaze, self.screen)

    @staticmethod
    def _drain(q: queue.Queue) -> list[GazeSample]:
        out = []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------------------------------------- demo
    def show_gaze(self, duration: float | None = None, trail: int = 20,
                  window: StimulusWindow | None = None):
        """Fullscreen gaze cursor (Esc to quit). Call from the main thread."""
        self._require_started()
        if self.calibration is None:
            raise RuntimeError("calibrate() or load a calibration first")
        win = window or self.window()
        hist: list[tuple[float, float]] = []
        t_end = time.time() + duration if duration else None
        try:
            while t_end is None or time.time() < t_end:
                s = self.latest()
                win.clear((20, 20, 20))
                win.rect(self.calibration.area, (60, 60, 60))
                for fx in (0.1, 0.5, 0.9):  # reference marks
                    for fy in (0.1, 0.5, 0.9):
                        win.dot(*self.calibration.area.to_screen(fx, fy), 4, (90, 90, 90))
                if s and s.valid:
                    hist = (hist + [(s.x, s.y)])[-trail:]
                    for k, (hx, hy) in enumerate(hist[:-1]):
                        c = int(60 + 120 * k / max(1, len(hist)))
                        win.dot(hx, hy, 6, (c // 2, c, c))
                    win.dot(s.x, s.y, 18, (80, 220, 255), 3)
                    status = f"eyes: {''.join(sorted(s.eyes)) or '-'}"
                else:
                    status = "eyes lost" if s else "waiting for camera"
                win.text(f"{status}   (Esc to quit)", 0.97, 0.6, (150, 150, 150))
                win.show()
        except Cancelled:
            pass
