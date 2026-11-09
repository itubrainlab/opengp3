"""Geometric eye model: 3-D eye positions, pupil size in mm, head depth.

The GP3's lens is fixed, so the camera is modelled as a pinhole with focal
length ``focal_px`` (pixels). Depth comes from the inter-pupillary distance:
both eyes are ``ipd_mm`` apart in reality and ``d`` pixels apart in the image,
so Z = focal_px * ipd / d. With one eye visible the last depth is kept.

Defaults: ``ipd_mm`` = 63 (adult average); ``focal_px`` = 2750, estimated for
this unit's lens from images at a known distance. Absolute values (mm, metres)
are therefore approximate (about +-10 %, mostly from individual IPD); relative
changes (pupil dilation, depth changes) are accurate. Supply the subject's IPD,
or measure it with a printed marker (``MarkerTracker``), for better values.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_IPD_MM = 63.0
DEFAULT_FOCAL_PX = 2750.0


@dataclass
class EyeMetrics:
    """Per-eye image and 3-D data (``valid`` False when the eye was not seen)."""
    valid: bool = False
    cx: float = 0.0          # pupil centre, fraction of camera image width / height
    cy: float = 0.0
    diameter_px: float = 0.0
    scale: float = 1.0       # depth / calibration depth (1 at calibration distance)
    x_m: float = 0.0         # eye position relative to the camera (m): +x = subject's right,
    y_m: float = 0.0         # +y = up, +z = away from the camera
    z_m: float = 0.0
    pupil_mm: float = 0.0


class EyeModel:
    def __init__(self, image_size=(752, 480), focal_px: float = DEFAULT_FOCAL_PX,
                 ipd_mm: float = DEFAULT_IPD_MM):
        self.w, self.h = image_size
        self.focal_px, self.ipd_mm = focal_px, ipd_mm
        self.reference_depth: float | None = None  # depth at calibration (m)
        self._z: float | None = None

    @property
    def depth_m(self) -> float | None:
        """Current eye depth estimate (m), None until both eyes were seen."""
        return self._z

    def update(self, eyes: dict) -> dict[str, EyeMetrics]:
        """eyes: {"R": EyeFeature, "L": ...} -> {"R": EyeMetrics, "L": ...}."""
        if "R" in eyes and "L" in eyes:
            r, l = eyes["R"], eyes["L"]
            d = float(np.hypot(l.pupil_x - r.pupil_x, l.pupil_y - r.pupil_y))
            if d > 1:
                z = self.focal_px * self.ipd_mm / 1000.0 / d
                self._z = z if self._z is None else 0.7 * self._z + 0.3 * z  # light smoothing
        out = {}
        for side in ("R", "L"):
            e = eyes.get(side)
            if e is None:
                out[side] = EyeMetrics()
                continue
            m = EyeMetrics(True, e.pupil_x / self.w, e.pupil_y / self.h, 2 * e.pupil_r)
            if self._z is not None:
                z = self._z
                m.z_m = z
                m.x_m = (self.w / 2 - e.pupil_x) * z / self.focal_px
                m.y_m = (self.h / 2 - e.pupil_y) * z / self.focal_px
                m.pupil_mm = m.diameter_px * z / self.focal_px * 1000.0
                if self.reference_depth:
                    m.scale = z / self.reference_depth
            out[side] = m
        return out

    def set_reference(self):
        """Mark the current depth as the calibration depth (scale = 1 there)."""
        self.reference_depth = self._z

    def calibrate_ipd(self, mm_per_px_at_eyes: float, ipd_px: float):
        """Measure the subject's IPD from a known image scale (e.g. a marker on
        the forehead, at the same depth as the eyes)."""
        self.ipd_mm = mm_per_px_at_eyes * ipd_px


@dataclass
class MarkerReading:
    valid: bool = False
    x: float = 0.0           # marker centre, fraction of the camera image
    y: float = 0.0
    mm_per_px: float = 0.0   # scale factor at the marker's depth


class MarkerTracker:
    """Finds a printed ArUco marker (4x4 dictionary, id 0) of known size in
    the camera image and reports the pixel -> millimetre scale at its depth.
    Worn on the forehead between the eyes it gives real-world pupil sizes and
    the subject's IPD. Print it with ``opengp3 marker`` (laser printers work
    best: some inkjet inks are transparent in infrared)."""

    DICT = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else None

    def __init__(self, size_mm: float = 11.0, marker_id: int = 0):
        self.size_mm, self.marker_id = size_mm, marker_id
        d = cv2.aruco.getPredefinedDictionary(self.DICT)
        self._det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    def detect(self, img: np.ndarray) -> MarkerReading:
        corners, ids, _ = self._det.detectMarkers(img)
        if ids is None:
            return MarkerReading()
        for c, i in zip(corners, ids.ravel()):
            if i != self.marker_id:
                continue
            c = c.reshape(4, 2)
            side_px = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
            if side_px < 4:
                continue
            h, w = img.shape[:2]
            cx, cy = c.mean(axis=0)
            return MarkerReading(True, float(cx / w), float(cy / h), self.size_mm / side_px)
        return MarkerReading()

    @staticmethod
    def image(size_mm: float = 11.0, dpi: int = 600, marker_id: int = 0) -> np.ndarray:
        """Printable marker image (with a white quiet zone) at ``dpi``."""
        px = int(round(size_mm / 25.4 * dpi))
        d = cv2.aruco.getPredefinedDictionary(MarkerTracker.DICT)
        m = cv2.aruco.generateImageMarker(d, marker_id, px)
        pad = px // 2
        return cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
