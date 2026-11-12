"""High-level API: calibrate on the screen and stream gaze in screen points.

    from opengp3 import GazeTracker, Screen

    with GazeTracker(Screen.detect(width_cm=53, distance_cm=65)) as gt:
        gt.calibrate()                      # fullscreen 9-point calibration
        print(gt.validate().report())       # optional accuracy check
        gt.calibration.save("cal.json")
        for s in gt.gaze(duration=10):      # every frame, 60 Hz (GP3 HD: 150 Hz)
            if s.valid:
                print(s.t, s.x, s.y, s.fixation.valid)
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

from .calibration import (VALIDATION_POINTS, Calibration, CalibrationError,
                          CalibrationSample, ValidationResult, evaluate, grid_points,
                          presentation_order)
from .camera import GP3Camera, open_camera
from .brightdark import BrightDarkDetector, PupilGainControl
from .detection import EyeFeature, EyeTracker, detect_eyes
from .display import Cancelled, StimulusWindow
from .events import BlinkDetector, BlinkInfo, Fixation, FixationFilter, MovingAverage
from .eyemodel import EyeMetrics, EyeModel, MarkerReading, MarkerTracker
from .filters import OneEuroFilter
from .platform import InputMonitor, move_mouse
from .screen import Rect, Screen


@dataclass
class GazeSample:
    """One camera frame's worth of tracking output.

    Gaze coordinates (``x``, ``raw_x``, ``fixation.x``, ``aac`` ...) are in
    screen points relative to the tracked screen's top-left corner.
    """
    t: float                         # capture time (time.time() clock)
    frame: int                       # camera frame sequence number
    eyes: dict[str, EyeFeature]      # detected eyes, "R"/"L" = subject's right/left
    x: float | None = None           # gaze, smoothed (One-Euro) if enabled
    y: float | None = None
    raw_x: float | None = None       # unsmoothed gaze, mean of visible calibrated eyes
    raw_y: float | None = None
    per_eye: dict[str, tuple[float, float]] = field(default_factory=dict)
    time: float = 0.0                # seconds since the tracker started
    tick: int = 0                    # cv2.getTickCount() when the frame was processed
    fixation: Fixation = field(default_factory=Fixation)   # fixation filter output
    aac: tuple[float, float] | None = None                 # moving-average gaze (pointer)
    blink: BlinkInfo = field(default_factory=BlinkInfo)
    metrics: dict[str, EyeMetrics] = field(default_factory=dict)  # pupil / 3-D per eye
    cursor: tuple[float, float, int] | None = None  # global x, y, button state (API CS)
    key: tuple[str, int] = (" ", 0)                 # key name, state (API KB/KBS)
    marker: MarkerReading | None = None
    user: str = ""                                  # user marker active for this frame

    @property
    def valid(self) -> bool:
        """True when a calibrated eye was visible (x/y available)."""
        return self.x is not None


# ===================================================================== calibration
class CalibrationRun:
    """One calibration (or validation) sequence, driven by time.

    Each target has an animation phase (``delay`` s: the dot moves there and
    shrinks; no data used) and a collection phase (``timeout`` s). Samples
    are labelled by *capture* time, so slow rendering can't mislabel data.
    Call ``tick()`` regularly (with a window to draw it, or ``None`` for
    headless runs where an API client draws the targets); it returns True
    when finished. ``listener(event, n, fx, fy)`` receives "start_pt" /
    "result_pt" with the 1-based point number and the point's position as
    fractions of the whole screen.
    """

    def __init__(self, tracker: "GazeTracker", points: Sequence[tuple[float, float]], area: Rect,
                 delay: float = 0.6, timeout: float = 1.2, retry: bool = True,
                 min_samples: int = 10, listener: Callable | None = None,
                 sound: bool = False, images: dict | None = None, title: str = "Calibration",
                 keep_order: bool = False):
        self.tracker, self.area = tracker, area
        self.points = list(points)
        self.delay, self.timeout = delay, timeout
        self.retry, self.min_samples = retry, min_samples
        self.listener, self.sound, self.images = listener, sound, images or {}
        self.title = title
        self.queue = tracker.subscribe()
        tracker.active_run = self  # lets e.g. the simulated camera see the current target
        self.order = (list(range(len(self.points))) if keep_order or len(self.points) <= 2
                      else presentation_order(self.points))
        self.windows: list[tuple[int, float, float, float, float]] = []  # idx, x, y, t0, t1
        self.samples: list[GazeSample] = []
        self._i = -1
        self._t_point = None
        self._phase = "wait"
        self._retried = False
        self._prev_xy = area.to_screen(0.5, 0.5)
        self.done = False

    # -- state
    @property
    def current(self):
        if 0 <= self._i < len(self.order):
            idx = self.order[self._i]
            return idx, self.points[idx]
        return None

    def _advance(self, now):
        if self._i >= 0 and self.current:
            idx, (fx, fy) = self.current
            x, y = self.area.to_screen(fx, fy)
            self.windows.append((idx, x, y, self._t_point + self.delay, now))
            self._emit("result_pt", idx, fx, fy)
            self._prev_xy = (x, y)
        self._i += 1
        if self._i >= len(self.order) and self.retry and not self._retried:
            self._retried = True
            redo = self._missing()
            if redo:
                self.order = self.order + redo
        if self._i >= len(self.order):
            self.done = True
            return
        self._t_point = now
        self._phase = "animate"
        idx, (fx, fy) = self.current
        self._emit("start_pt", idx, fx, fy)
        if self.sound:
            _play(self.images.get(("sound", idx + 1)))

    def _missing(self) -> list[int]:
        self._collect()
        lab = self.labelled()
        counts = {i: max((sum(1 for (j, _, _, s) in lab if j == i and side in s.eyes)
                          for side in ("R", "L")), default=0) for i in range(len(self.points))}
        return [i for i in range(len(self.points)) if counts[i] < self.min_samples]

    def _emit(self, ev, idx, fx, fy):
        """Notify the listener with the point's position as a fraction of the
        whole screen (what Open Gaze API clients expect)."""
        if self.listener:
            x, y = self.area.to_screen(fx, fy)
            sc = self.tracker.screen
            try:
                self.listener(ev, idx + 1, x / sc.width, y / sc.height)
            except Exception:
                pass

    def _collect(self):
        while True:
            try:
                self.samples.append(self.queue.get_nowait())
            except queue.Empty:
                return

    def start(self):
        self._advance(time.time())

    def tick(self, win: StimulusWindow | None = None) -> bool:
        if self._phase == "wait":
            self.start()
        now = time.time()
        if not self.done:
            el = now - self._t_point
            if el >= self.delay + self.timeout:
                self._advance(now)
            elif el >= self.delay:
                self._phase = "collect"
        self._collect()
        if win is not None and not self.done:
            self.draw(win, now)
        return self.done

    def draw(self, win: StimulusWindow, now: float):
        idx, (fx, fy) = self.current
        tx, ty = self.area.to_screen(fx, fy)
        el = now - self._t_point
        win.clear()
        if self.area.w < win.screen.width or self.area.h < win.screen.height:
            win.rect(self.area)
        if el < self.delay:  # animation: travel from the previous point, shrink
            a = el / self.delay if self.delay else 1.0
            a = a * a * (3 - 2 * a)  # smoothstep
            x = self._prev_xy[0] + (tx - self._prev_xy[0]) * a
            y = self._prev_xy[1] + (ty - self._prev_xy[1]) * a
            prog = 0.0
        else:
            x, y = tx, ty
            prog = (el - self.delay) / self.timeout if self.timeout else 1.0
        img = self.images.get(idx + 1)
        if img is not None:
            win.image(img, x, y, 80 - 40 * min(1.0, prog))
        else:
            win.target(x, y, prog)
        win.show()

    def labelled(self):
        out = []
        for idx, x, y, t0, t1 in self.windows:
            for s in self.samples:
                if t0 <= s.t < t1:
                    out.append((idx, x, y, s))
        # retried targets: keep only the last window per target
        last = {}
        for w in self.windows:
            last[w[0]] = w
        return [l for l in out if last[l[0]][3] <= l[3].t < last[l[0]][4]]

    def close(self):
        self.tracker.unsubscribe(self.queue)
        if getattr(self.tracker, "active_run", None) is self:
            self.tracker.active_run = None


def _play(path: str | None = None):
    from .platform import play_sound
    play_sound(path)


def load_calibration_media(folder: str) -> dict:
    """Custom calibration targets: ``CalibPoint<N>.png`` images and
    ``CalibPoint<N>.mp3``/``.aiff``/``.wav`` sounds in ``folder``."""
    media = {}
    if not folder or not os.path.isdir(folder):
        return media
    for name in os.listdir(folder):
        base, ext = os.path.splitext(name)
        if not base.startswith("CalibPoint") or not base[10:].isdigit():
            continue
        n = int(base[10:])
        path = os.path.join(folder, name)
        if ext.lower() == ".png":
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                media[n] = img
        elif ext.lower() in (".mp3", ".aiff", ".wav", ".m4a"):
            media[("sound", n)] = path
    return media


# ===================================================================== tracker
class GazeTracker:
    """Owns the camera, runs detection on every frame in a background thread,
    and maps eyes to screen coordinates once calibrated.

    screen:      the display gaze is reported on (default: main display).
    display:     alternatively, the display's number in ``opengp3.platform.displays()``
                 (0 = main); ``set_display()`` changes it later.
                 Set ``width_cm``/``distance_cm`` for results in degrees.
    calibration: a saved Calibration (``Calibration.load(path)``), optional.
    smoothing:   One-Euro filter on ``x``/``y``; ``raw_x``/``raw_y`` stay raw.
    monocular:   None (both eyes), "R" or "L": track one eye only and label the
                 visible eye as that side (e.g. subjects with one usable eye).
    auto_gain:   sweep the camera gain while no eyes are found (very bright or
                 dim pupils: children, elderly subjects).
    max_diagonal_in: on screens larger than this (inches), calibrate and
                 validate within a centred area of this size (default 22, the
                 GP3's specified maximum). None = always the whole screen.
    input_monitor: record mouse position/buttons and keystrokes with each sample.
    aac_length:  moving-average window (samples) of the AAC / pointer gaze.
    rate_hz:     GP3 HD frame rate, 60 or 150 (default 150). Ignored by the GP3 (60 Hz).
    """

    def __init__(self, screen: Screen | None = None, calibration: Calibration | None = None,
                 smoothing: bool | OneEuroFilter = True, camera: GP3Camera | None = None,
                 full_sensor: bool = True, monocular: str | None = None,
                 auto_gain: bool = False, input_monitor: bool = True, aac_length: int = 15,
                 fixation_radius_deg: float = 1.0, max_diagonal_in: float | None = 22.0,
                 rate_hz: float | None = None, display: int | None = None):
        if screen is None:
            screen = Screen.detect(display=display or 0)
        elif display is not None:
            raise ValueError("pass either screen or display, not both")
        self.screen = screen
        #: calibration/validation targets stay within a centred area of this
        #: diagonal (GP3 specification: screens up to 22"); None = whole screen
        self.max_diagonal_in = max_diagonal_in
        self.calibration = calibration
        self.previous_calibration: Calibration | None = None
        self.filter = (smoothing if isinstance(smoothing, OneEuroFilter)
                       else OneEuroFilter() if smoothing else None)
        if monocular not in (None, "R", "L"):
            raise ValueError("monocular must be None, 'R' or 'L'")
        self.monocular = monocular
        self.auto_gain = auto_gain
        self.input = InputMonitor() if input_monitor else None
        self.aac = MovingAverage(aac_length)
        self.fixation_filter = FixationFilter(self._fix_radius(fixation_radius_deg),
                                              aspect=self.screen.width / self.screen.height)
        self.blinks = BlinkDetector()
        self.eye_model = EyeModel()
        self.marker: MarkerTracker | None = None
        self.gaze_pointer = False
        self.user_data = ""
        self._cam = camera
        self._own_cam = camera is None
        self._full_sensor = full_sensor
        #: GP3 HD frame rate (60 or 150 Hz; None = the camera default, 150).
        #: The GP3 always runs at 60 Hz.
        self.rate_hz = rate_hz
        self._thread = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []
        self._callbacks: list[Callable[[GazeSample], None]] = []
        self._latest: GazeSample | None = None
        self._latest_image: np.ndarray | None = None
        self._image_sample: GazeSample | None = None
        self._t0 = time.time()
        self.image_size = (752, 480)
        self.dropped_frames = 0
        self.fps = 0.0
        self.error: Exception | None = None
        self._win: StimulusWindow | None = None
        if calibration is not None:
            self._apply_calibration_meta(calibration)
        #: labelled samples of the last calibrate()/validate() run, for diagnostics:
        #: list of (target, target_x, target_y, GazeSample)
        self.last_calibration_samples: list = []
        self.last_validation_samples: list = []

    def _fix_radius(self, deg: float) -> float:
        """Fixation radius as a fraction of screen width."""
        k = self.screen.cm_per_point
        if not k:
            return 0.025 * deg
        return float(np.tan(np.radians(deg)) * self.screen.distance_cm / k / self.screen.width)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "GazeTracker":
        if self._thread:
            return self
        if self._cam is None:
            self._cam = open_camera()
        if self._full_sensor:
            self._cam.set_full_sensor()
        else:
            self._cam.set_video_mode(5, 5)
        if self.rate_hz and hasattr(self._cam, "set_rate"):
            self._cam.set_rate(self.rate_hz)
        self.image_size = (self._cam.width, self._cam.height)
        self.eye_model.w, self.eye_model.h = self.image_size
        # each tracker model has its own lens/sensor: focal length in pixels
        self.eye_model.focal_px = getattr(self._cam, "focal_px", self.eye_model.focal_px)
        self._cam.set_leds(True)
        self._cam.start_stream()
        self._t0 = time.time()
        self._running.set()
        self._thread = threading.Thread(target=self._process, name="opengp3", daemon=True)
        self._thread.start()
        return self

    def window(self) -> StimulusWindow:
        """The session's fullscreen stimulus window (created on first use and
        reused, so calibration and validation share identical geometry)."""
        if self._win is None:
            self._win = self.window_factory(self.screen, "OpenGP3")
        return self._win

    #: creates the stimulus window: (screen, title) -> StimulusWindow-like.
    #: The Qt control app replaces it with a Qt window (opengp3.qtwindow).
    window_factory = StimulusWindow

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

    @property
    def model(self) -> str:
        """Eye tracker model: "GP3" or "GP3 HD"."""
        return getattr(self._cam, "model", "GP3")

    @property
    def rate(self) -> float:
        """Nominal camera frame rate (Hz)."""
        return float(getattr(self._cam, "rate", 60.0))

    def latest_image(self) -> np.ndarray | None:
        """Most recent camera frame (for display)."""
        with self._lock:
            return self._latest_image

    def latest_frame(self) -> tuple[np.ndarray | None, GazeSample | None]:
        """Most recent camera frame for display, together with its own tracking
        result. With the GP3 HD's alternating illumination these are the
        dark-pupil frames only, so a display doesn't flicker."""
        with self._lock:
            return self._latest_image, self._image_sample

    # ------------------------------------------------------------- pipeline
    def _detect(self, img, eye_tracker, bright_dark=None):
        max_eyes = 1 if self.monocular else 2
        if bright_dark is not None:
            bright_dark.max_eyes = max_eyes
            found = bright_dark.update(img)
        else:
            found = detect_eyes(img, max_eyes=max_eyes)
        if self.monocular:
            return {self.monocular: found[0]} if found else {}
        return eye_tracker.update(found)

    @property
    def warning(self) -> str | None:
        """Why the data may be unreliable (e.g. the webcam fallback), else None."""
        return getattr(self._cam, "warning", None)

    @property
    def illumination(self) -> str:
        """"bright-dark" (GP3 HD, Gazepoint's alternating LED banks: detected
        with ``BrightDarkDetector``) or "dark" (dark pupil, ``detect_eyes``)."""
        return getattr(self._cam, "illumination", "dark")

    def _process(self):
        # pupil-size consistency check for dark-pupil (GP3) images only: the
        # GP3 HD's bright/dark frames measure the pupil differently
        eye_tracker = EyeTracker(size_check=self.illumination == "dark")
        if self.illumination == "webcam":       # fallback: landmarks in webcam images
            from .webcam import WebcamEyeDetector
            bright_dark = WebcamEyeDetector()
        else:
            bright_dark = BrightDarkDetector() if self.illumination == "bright-dark" else None
        pupil_gain = PupilGainControl(self._cam) if self.illumination == "bright-dark" else None
        last_seq = 0
        gain = _GainSweep(self)
        fps_t, fps_n = time.time(), 0
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
            fps_n += 1
            if t - fps_t >= 1.0:
                self.fps, fps_t, fps_n = fps_n / (t - fps_t), t, 0
            eyes = self._detect(img, eye_tracker, bright_dark)
            if pupil_gain is not None and not self.auto_gain:
                pupil_gain.update(bright_dark, t)
            s = self._enrich(GazeSample(t, seq, eyes, time=t - self._t0, tick=cv2.getTickCount()),
                             img)
            if self.auto_gain:
                gain.update(t, bool(eyes))
            with self._lock:
                self._latest = s
                # display frames: with the GP3 HD's alternating illumination only the
                # dark-pupil frames (50 Hz), else the view flickers between frame types
                # (detectors without frame types, e.g. the webcam's, show every frame)
                if getattr(bright_dark, "last_kind", "D") == "D":
                    self._latest_image, self._image_sample = img, s  # image + its own result
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

    def _enrich(self, s: GazeSample, img) -> GazeSample:
        """Calibrated gaze, fixation/blink/AAC filters, eye metrics, inputs."""
        t, W, H = s.t, self.screen.width, self.screen.height
        s.user = self.user_data
        cal = self.calibration
        if cal is not None:
            g = cal.predict(s.eyes)
            if g is not None:
                s.raw_x, s.raw_y, s.per_eye = g
                s.x, s.y = (self.filter(t, s.raw_x, s.raw_y) if self.filter
                            else (s.raw_x, s.raw_y))
        fx = s.raw_x / W if s.raw_x is not None else None
        fy = s.raw_y / H if s.raw_y is not None else None
        f = self.fixation_filter.update(t - self._t0, fx, fy)
        s.fixation = Fixation(f.x * W, f.y * H, f.start, f.duration, f.id, f.valid)
        # AAC: moving average of fixation data; valid while a fixation is valid
        a = self.aac.update(s.fixation.x if f.valid else None, s.fixation.y if f.valid else None)
        s.aac = a if f.valid else None
        s.blink = self.blinks.update(t, bool(s.eyes))
        if self.marker is not None:
            s.marker = self.marker.detect(img)
        s.metrics = self.eye_model.update(s.eyes)
        if self.input is not None:
            cx, cy, cs, kb, kbs = self.input.poll()
            s.cursor, s.key = (cx, cy, cs), (kb, kbs)
        if self.gaze_pointer and a is not None and f.valid:
            move_mouse(*self.screen.to_global(*a))
        return s

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

    def default_area(self) -> Rect:
        """Where calibration targets go by default (see ``max_diagonal_in``)."""
        return self.screen.certified_area(self.max_diagonal_in)

    def _require_started(self):
        if not self._thread:
            raise RuntimeError("call start() first (or use 'with GazeTracker() as gt:')")

    # ------------------------------------------------------------- settings
    def set_aac_length(self, n: int):
        self.aac.set_length(n)

    def set_screen(self, screen: Screen):
        """Change the tracked screen (needs recalibration): gaze is reported in
        its coordinates, and calibration, validation and the gaze demo run on
        it."""
        self.screen = screen
        self.close_window()
        self.fixation_filter.aspect = screen.width / screen.height

    def set_display(self, display: int, width_cm: float | None = None,
                    distance_cm: float | None = None) -> Screen:
        """Track and calibrate on display number ``display`` (an index into
        ``opengp3.platform.displays()``, 0 = main display), with its physical
        width from the OS unless ``width_cm`` is given. Returns the new Screen."""
        scr = Screen.detect(width_cm=width_cm, display=display,
                            distance_cm=distance_cm or self.screen.distance_cm)
        self.set_screen(scr)
        return scr

    def enable_marker(self, size_mm: float | None):
        """Track a printed ArUco marker of ``size_mm`` (None = off)."""
        self.marker = MarkerTracker(size_mm) if size_mm else None

    def undo_calibration(self) -> Calibration | None:
        """Swap back to the previous calibration (calling again re-applies
        the newer one)."""
        if self.previous_calibration is None:
            return self.calibration
        self.calibration, self.previous_calibration = self.previous_calibration, self.calibration
        self._apply_calibration_meta(self.calibration)
        return self.calibration

    def _apply_calibration_meta(self, cal: Calibration):
        m = cal.meta or {}
        if m.get("ipd_mm"):
            self.eye_model.ipd_mm = m["ipd_mm"]
        if m.get("reference_depth_m"):
            self.eye_model.reference_depth = m["reference_depth_m"]

    # ------------------------------------------------------------- stimuli
    def _wait_for_eyes(self, win: StimulusWindow, title: str, area: Rect,
                       stable_s: float = 1.2, timeout: float = 60.0):
        """Positioning screen: shows the eyes in the camera view and the head
        distance, and starts once the eyes have been stable for ``stable_s``
        (Space starts at once)."""
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
            win.text(title, 0.30, 1.4)
            win.text("Keep your head still and look at the centre of each dot.", 0.40, 0.8)
            status = {2: "both eyes detected", 1: "one eye detected", 0: "no eyes detected"}[len(eyes)]
            col = (80, 220, 80) if len(eyes) == 2 else (0, 200, 255) if eyes else (60, 60, 255)
            win.text(status + (" - starting..." if since and now - since > 0.3 else ""),
                     0.50, 0.8, col)
            win.eye_box(eyes, self.image_size)
            z = self.eye_model.depth_m
            if z:
                win.text(distance_label(z * 100), 0.86, 0.7, (200, 200, 200))
            win.text("Space: start now    Esc: cancel", 0.94, 0.6, (150, 150, 150))
            if win.show() == ord(" ") or (since and now - since >= stable_s):
                return
            if now - t0 > timeout:
                raise CalibrationError("eyes not detected - check the GP3 position (opengp3 view)")

    # ------------------------------------------------------------- calibration
    def start_calibration(self, points: int | Sequence[tuple[float, float]] = 9,
                          area: Rect | None = None, delay: float = 0.6, timeout: float = 1.2,
                          retry: bool = True, listener: Callable | None = None,
                          sound: bool = False, media_dir: str | None = None,
                          min_samples: int = 10, keep_order: bool = False) -> CalibrationRun:
        """Non-blocking calibration: returns a CalibrationRun to ``tick()``
        (used by the API server). Most users want ``calibrate()``."""
        self._require_started()
        pts = grid_points(points) if isinstance(points, int) else [tuple(p) for p in points]
        return CalibrationRun(self, pts, area or self.default_area(), delay, timeout, retry,
                              min_samples, listener, sound, load_calibration_media(media_dir),
                              keep_order=keep_order)

    def finish_calibration(self, run: CalibrationRun) -> Calibration:
        """Fit and activate the calibration collected by ``run``."""
        run._collect()
        run.close()
        labelled = run.labelled()
        self.last_calibration_samples = labelled
        cal_samples = [CalibrationSample(i, x, y, side, *e.pcr, s.t)
                       for i, x, y, s in labelled for side, e in s.eyes.items()]
        cal = Calibration.fit(cal_samples, self.screen, run.area, min_samples=run.min_samples)
        # eye model: calibration depth (scale = 1 there) and marker-measured IPD
        depths = [s.metrics["R"].z_m for _, _, _, s in labelled
                  if s.metrics.get("R") and s.metrics["R"].valid and s.metrics["R"].z_m]
        ipds = [s.marker.mm_per_px * np.hypot(s.eyes["L"].pupil_x - s.eyes["R"].pupil_x,
                                              s.eyes["L"].pupil_y - s.eyes["R"].pupil_y)
                for _, _, _, s in labelled
                if s.marker is not None and s.marker.valid and "R" in s.eyes and "L" in s.eyes]
        if ipds:
            self.eye_model.ipd_mm = float(np.median(ipds))
            cal.meta["ipd_mm"] = self.eye_model.ipd_mm
            cal.meta["ipd_source"] = "marker"
        if depths:
            ref = float(np.median(depths))
            if ipds:  # depth was computed with the old IPD; rescale
                ref *= self.eye_model.ipd_mm / (cal.meta.get("ipd_mm") or self.eye_model.ipd_mm)
            self.eye_model.reference_depth = ref
            cal.meta["reference_depth_m"] = ref
        self.previous_calibration, self.calibration = self.calibration, cal
        if self.filter:
            self.filter.reset()
        self.fixation_filter.reset()
        return cal

    def calibrate(self, points: int | Sequence[tuple[float, float]] = 9, area: Rect | None = None,
                  delay: float = 0.6, timeout: float = 1.2, min_samples: int = 10,
                  retry: bool = True, sound: bool = False, media_dir: str | None = None,
                  wait_for_eyes: bool = True, window: StimulusWindow | None = None,
                  listener: Callable | None = None) -> Calibration:
        """Fullscreen calibration. Call from the main thread.

        points:  5, 9, 13, 16 or 25 (grid), a list of (fx, fy) area fractions,
                 or 1 for a one-point drift correction of the current calibration
        area:    Rect to calibrate (default: ``default_area()``: the whole screen,
                 or a centred 22" area on larger screens)
        delay:   seconds the dot travels to the next point (not recorded)
        timeout: seconds of data collected per point
        sound:   short tone at each point; media_dir: CalibPoint<N>.png/.mp3 targets
        The result is stored in ``self.calibration`` (the previous one is kept
        for ``undo_calibration()``) and returned. Raises CalibrationError if no
        eye could be calibrated, Cancelled on Esc.
        """
        if points == 1:
            return self.calibrate_one_point(area=area, delay=delay, timeout=timeout,
                                            window=window, wait_for_eyes=wait_for_eyes)
        run = self.start_calibration(points, area, delay, timeout, retry, listener, sound,
                                     media_dir, min_samples)
        win = window or self.window()
        try:
            if wait_for_eyes:
                self._wait_for_eyes(win, f"Calibration ({len(run.points)} points)", run.area)
            while not run.tick(win):
                pass
        finally:
            run.close()
        return self.finish_calibration(run)

    def calibrate_one_point(self, base: Calibration | None = None, area: Rect | None = None,
                            delay: float = 0.6, timeout: float = 1.5,
                            window: StimulusWindow | None = None,
                            wait_for_eyes: bool = True) -> Calibration:
        """Correct the current (or ``base``) calibration with one central
        point: fast, but only fixes a constant offset (e.g. after the subject
        moved). Without any previous calibration this cannot work."""
        base = base or self.calibration or self.previous_calibration
        if base is None:
            raise CalibrationError("one-point calibration needs an existing calibration to correct")
        area = area or base.area
        run = CalibrationRun(self, [(0.5, 0.5)], area, delay, timeout, retry=False)
        win = window or self.window()
        try:
            if wait_for_eyes:
                self._wait_for_eyes(win, "Calibration (1 point)", area)
            while not run.tick(win):
                pass
            run._collect()
        finally:
            run.close()
        tx, ty = area.to_screen(0.5, 0.5)
        offsets = {}
        for side, fit in base.eyes.items():
            g = [fit.mapping(s.eyes[side].pcr) for _, _, _, s in run.labelled() if side in s.eyes]
            if len(g) >= 5:
                mx, my = np.median(g, axis=0)
                offsets[side] = (tx - mx, ty - my)
        if not offsets:
            raise CalibrationError("no eye data at the calibration point")
        cal = base.with_offset(offsets)
        self.previous_calibration, self.calibration = self.calibration, cal
        if self.filter:
            self.filter.reset()
        return cal

    def validate(self, points=VALIDATION_POINTS, area: Rect | None = None,
                 delay: float = 0.6, timeout: float = 1.0, shuffle: bool = True,
                 window: StimulusWindow | None = None) -> ValidationResult:
        """Measure accuracy on targets the calibration did not use (by
        default 8 points between the 3x3 calibration grid, in random order).
        Uses unsmoothed gaze. Call from the main thread."""
        import random
        self._require_started()
        if self.calibration is None:
            raise RuntimeError("calibrate() or load a calibration first")
        area = area or self.calibration.area
        pts = list(points)
        run = CalibrationRun(self, pts, area, delay, timeout, retry=False)
        if shuffle:
            random.shuffle(run.order)
        win = window or self.window()
        try:
            self._wait_for_eyes(win, f"Validation ({len(pts)} points)", area)
            while not run.tick(win):
                pass
            run._collect()
        finally:
            run.close()
        labelled = run.labelled()
        self.last_validation_samples = labelled
        gaze = [(i, x, y, s.t, s.raw_x, s.raw_y) for i, x, y, s in labelled if s.valid]
        return evaluate(gaze, self.screen)

    # ------------------------------------------------------------- demo
    def show_gaze(self, duration: float | None = None, trail: int = 20,
                  window: StimulusWindow | None = None, use: str = "x"):
        """Fullscreen gaze cursor (Esc to quit). Call from the main thread.
        use: "x" (smoothed gaze), "raw", "fixation" or "aac"."""
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
                p = _pick(s, use) if s else None
                if p is not None:
                    hist = (hist + [p])[-trail:]
                    for k, (hx, hy) in enumerate(hist[:-1]):
                        c = int(60 + 120 * k / max(1, len(hist)))
                        win.dot(hx, hy, 6, (c // 2, c, c))
                    win.dot(p[0], p[1], 18, (80, 220, 255), 3)
                    status = f"eyes: {''.join(sorted(s.eyes)) or '-'}"
                else:
                    status = "eyes lost" if s else "waiting for camera"
                win.text(f"{status}   (Esc to quit)", 0.97, 0.6, (150, 150, 150))
                win.show()
        except Cancelled:
            pass


def _pick(s: GazeSample, use: str):
    if use == "raw":
        return (s.raw_x, s.raw_y) if s.raw_x is not None else None
    if use == "fixation":
        return (s.fixation.x, s.fixation.y) if s.fixation.valid else None
    if use == "aac":
        return s.aac
    return (s.x, s.y) if s.valid else None


def distance_label(cm: float, ideal=(55, 75)) -> str:
    if cm < ideal[0]:
        return f"distance {cm:.0f} cm - move back"
    if cm > ideal[1]:
        return f"distance {cm:.0f} cm - move closer"
    return f"distance {cm:.0f} cm - good"


class _GainSweep:
    """While no eyes are found for > 1 s, step the camera gain through its
    range (manual gain); once eyes are found, keep that gain."""

    def __init__(self, tracker: GazeTracker, step_s: float = 0.3, steps: int = 8):
        self.tr, self.step_s, self.steps = tracker, step_s, steps
        self._lost = None
        self._next = 0.0
        self._k = 0
        self._range = None

    def update(self, t: float, eyes: bool):
        cam = self.tr.camera
        if cam is None:
            return
        if eyes:
            self._lost = None
            return
        self._lost = self._lost or t
        if t - self._lost < 1.0 or t < self._next:
            return
        try:
            if self._range is None:
                f = cam.feature("gain")
                self._range = (f["min"], f["max"])
            lo, hi = self._range
            value = lo + (hi - lo) * (self._k % self.steps) / (self.steps - 1)
            cam.set_feature("gain", int(value))
            self._k += 1
        except Exception as e:
            self.tr.error = e
        self._next = t + self.step_s
