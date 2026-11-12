"""Webcam-based eye tracking: a low-accuracy FALLBACK for when no Gazepoint
eye tracker (GP3 or GP3 HD) is connected.

Needs ``pip install "opengp3[webcam]"`` (Google MediaPipe). Face and iris
landmarks are found in ordinary webcam images; for each eye, the iris centre
relative to the eye corners (normalised by the eye's width) is the gaze
feature, used by the same calibration, filters, API server and LSL streams as
the pupil-glint vector of the GP3.

Expect far lower accuracy than with an infrared eye tracker (typically
2-5 degrees instead of ~0.5), strong sensitivity to head movement (recalibrate
after moving) and lighting, and no pupil size. OpenGP3 reports this
prominently: console, Python warning, Control app banner, Open Gaze API
(PRODUCT_ID "WEBCAM" with a WARNING attribute) and LSL metadata.
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import GP3Error
from .detection import EyeFeature

WEBCAM_WARNING = ("NO GAZEPOINT EYE TRACKER FOUND - using webcam-based eye tracking. Accuracy is "
                  "much lower (typically 2-5 degrees instead of about 0.5), it is very sensitive "
                  "to head movement and lighting (keep your head still; recalibrate after "
                  "moving), and pupil size is not available.")

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")

# MediaPipe face mesh indices: eye corners, upper/lower lid, iris centre and ring
_EYES = [dict(corners=(33, 133), lids=(159, 145)), dict(corners=(362, 263), lids=(386, 374))]
_IRIS = [(468, (469, 470, 471, 472)), (473, (474, 475, 476, 477))]
FEATURE_SCALE = 100.0   # gaze feature units: hundredths of an eye width


class WebcamUnavailable(GP3Error):
    """The webcam fallback can't be used (extra not installed, no camera,
    no permission)."""


@dataclass
class WebcamEyeFeature(EyeFeature):
    """An eye found in a webcam image. ``pupil_x/y`` is the iris centre,
    ``glint_x/y`` the eye-corner midpoint (the reference point); ``pcr`` is
    the iris position within the eye, in hundredths of the eye's width."""
    feature_x: float = 0.0
    feature_y: float = 0.0
    openness: float = 0.0

    @property
    def pcr(self) -> tuple[float, float]:
        return self.feature_x, self.feature_y


def eyes_from_landmarks(pts: np.ndarray, closed_below: float = 0.12) -> list[WebcamEyeFeature]:
    """Eye features from face landmarks in image pixels (N x 2, MediaPipe
    face-mesh order with the 10 iris points). Closed eyes are left out (so
    blinks are detected as missing eyes)."""
    if pts.shape[0] < 478:
        return []
    eyes = []
    centres = []
    for e in _EYES:
        a, b = pts[e["corners"][0]], pts[e["corners"][1]]
        left, right = (a, b) if a[0] <= b[0] else (b, a)        # image left -> right
        centres.append(((left + right) / 2, left, right, e))
    for ci, ring in _IRIS:
        iris = pts[ci]
        mid, left, right, e = min(centres, key=lambda c: float(np.hypot(*(c[0] - iris))))
        width = float(np.hypot(*(right - left)))
        if width < 4:
            continue
        ax = (right - left) / width                              # along the eye, image-rightwards
        nx = np.array([-ax[1], ax[0]])                           # across the eye, image-downwards
        lid = float(np.hypot(*(pts[e["lids"][0]] - pts[e["lids"][1]]))) / width
        if lid < closed_below:
            continue
        d = iris - mid
        fx = float(d @ ax) / width * FEATURE_SCALE
        fy = float(d @ nx) / width * FEATURE_SCALE
        eyes.append(WebcamEyeFeature(
            pupil_x=float(iris[0]), pupil_y=float(iris[1]),
            pupil_r=0.0,              # pupil size can't be measured in webcam images
            glint_x=float(mid[0]), glint_y=float(mid[1]), n_glints=0, confidence=1.0,
            feature_x=fx, feature_y=fy, openness=lid))
    eyes.sort(key=lambda e: e.pupil_x)
    return eyes


def model_path(download: bool = True) -> Path:
    """The face landmark model, downloaded once from Google's MediaPipe model
    storage into ~/.opengp3/models (override with OPENGP3_HOME)."""
    home = Path(os.environ.get("OPENGP3_HOME", Path.home() / ".opengp3"))
    p = home / "models" / "face_landmarker.task"
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    if not download:
        raise WebcamUnavailable(f"face landmark model missing: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    print(f"downloading the face landmark model for webcam eye tracking (~4 MB) to {p}",
          flush=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except OSError as e:
        raise WebcamUnavailable(f"could not download the face landmark model ({e})") from e
    if tmp.stat().st_size < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise WebcamUnavailable("face landmark model download was incomplete")
    tmp.replace(p)
    return p


def _require_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        return mp, BaseOptions, vision
    except ImportError as e:
        raise WebcamUnavailable('webcam eye tracking needs: pip install "opengp3[webcam]"') from e


class WebcamEyeDetector:
    """Face/iris landmarks per frame (MediaPipe, video mode) -> eye features."""

    def __init__(self, model: Path | None = None):
        mp, BaseOptions, vision = _require_mediapipe()
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model or model_path())),
            running_mode=vision.RunningMode.VIDEO, num_faces=1)
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = 0
        self.max_eyes = 2

    def update(self, img: np.ndarray) -> list[WebcamEyeFeature]:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB)
        ms = max(self._last_ms + 1, int(time.monotonic() * 1000))   # strictly increasing
        self._last_ms = ms
        res = self._lm.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ms)
        if not res.face_landmarks:
            return []
        h, w = img.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], np.float64)
        return eyes_from_landmarks(pts)[: self.max_eyes]


class WebcamCamera:
    """A webcam with the camera interface GazeTracker uses. Frames are
    delivered in grayscale (display and API image size)."""

    model = "WEBCAM"
    bus = "USB"
    illumination = "webcam"
    simulated = False
    webcam = True
    dev = None
    warning = WEBCAM_WARNING

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: float = 30.0):
        self.index = index
        backend = cv2.CAP_AVFOUNDATION if os.uname().sysname == "Darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise WebcamUnavailable(_permission_hint(f"no webcam at index {index}"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        frame = self._first_frame()
        self.height, self.width = frame.shape[:2]
        r = self.cap.get(cv2.CAP_PROP_FPS)
        self.rate = float(r) if r and 1 <= r <= 240 else 30.0
        # pinhole focal length for a typical ~65 degree horizontal field of view
        self.focal_px = self.width / 2 / math.tan(math.radians(65 / 2))
        self.serial = f"webcam-{index}"
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread = None
        self._seq = 0

    def _first_frame(self, timeout: float = 4.0) -> np.ndarray:
        t_end = time.time() + timeout
        while time.time() < t_end:
            ok, f = self.cap.read()
            if ok and f is not None and f.size and f.mean() > 2:
                return f
            time.sleep(0.05)
        self.cap.release()
        raise WebcamUnavailable(_permission_hint("the webcam delivered no image"))

    # camera interface ----------------------------------------------------
    def set_full_sensor(self, *a, **k):
        pass

    def set_video_mode(self, *a, **k):
        pass

    def set_rate(self, *a, **k):
        pass

    def set_leds(self, on=True):
        pass

    def feature(self, name):
        return {"present": False, "min": 0, "max": 0, "value": 0}

    def set_feature(self, *a, **k):
        pass

    def info(self) -> dict:
        return {"usb": f"webcam {self.index}", "size": [self.width, self.height],
                "rate": self.rate, "warning": self.warning}

    def start_stream(self, *a, **k):
        if self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="webcam", daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if not ok or f is None:
                time.sleep(0.01)
                continue
            t = time.time()
            self._seq += 1
            item = (self._seq, t, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            try:
                self._q.put_nowait(item)
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put_nowait(item)

    def grab(self, timeout_s: float = 2.0, latest: bool = False):
        try:
            item = self._q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError("no frames from the webcam")
        if latest:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
        return item

    def stop_stream(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def stop(self):
        self.stop_stream()

    def close(self):
        self.stop_stream()
        self.cap.release()


def _permission_hint(msg: str) -> str:
    if os.uname().sysname == "Darwin":
        return (f"{msg}. On macOS, allow camera access for the app running OpenGP3 "
                "(Terminal, your IDE, ...) in System Settings > Privacy & Security > Camera")
    return msg


def open_webcam(index: int = 0) -> WebcamCamera:
    """Open the webcam fallback (checks the extra and the model first)."""
    _require_mediapipe()
    model_path()
    return WebcamCamera(index)
