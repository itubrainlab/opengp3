"""Bright/dark-pupil eye detection for the GP3 HD's alternating illumination.

The GP3 HD has two IR LED banks. The one next to the lens lights the retina
through the pupil, so the pupil glows (bright pupil). The off-axis one leaves
the pupil black (dark pupil) and makes a pair of corneal reflections (glints).
Gazepoint Control alternates them in a 3-frame cycle: one dark-pupil frame,
two bright-pupil frames (``opengp3.u3v``, LED mode "gazepoint"). This module
uses that illumination the way it is designed to be used:

1. frame type: in each cycle the dark-pupil frame is clearly brighter overall
   (the off-axis bank is the stronger one), which classifies every frame
   without relying on the frame counter's phase;
2. eye finding: the latest bright-pupil frame minus the latest dark-pupil
   frame. Only the pupils are much brighter in the first, so the difference
   image leaves them as round blobs and cancels skin, eyelashes, dark irises,
   eyebrows and spectacle frames that fool a single-frame dark-pupil search;
3. pupil fit in the *current* frame (bright disc or dark disc), so a sample
   never mixes two moments in time;
4. glint reference: the on-axis glint (bright-pupil frames). In dark-pupil
   frames the glint-pair midpoint is shifted to the on-axis reference with an
   offset learned per eye from neighbouring frames (relative to the pupil, so
   it also absorbs the bright/dark pupil-fit difference): pupil-glint vectors
   (PCR) from both frame types share one reference and one calibration.

Gazepoint's own image processing is not published; this is the standard
bright/dark-pupil technique applied to Gazepoint's illumination.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .detection import EyeFeature, _fit_pupil

SATURATED = 230


def _glint_blobs(img: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """Saturated spots in a window: [(x, y, area)] in image coordinates."""
    roi = img[y0:y1, x0:x1]
    n, _, st, cen = cv2.connectedComponentsWithStats((roi >= SATURATED).astype(np.uint8))
    return [(cen[i][0] + x0, cen[i][1] + y0, int(st[i, cv2.CC_STAT_AREA]))
            for i in range(1, n) if st[i, cv2.CC_STAT_AREA] <= 150]


class BrightDarkDetector:
    """Feed every frame in order with ``update(img)``; returns the eyes found
    in that frame (like ``detect_eyes``), sorted left to right in the image."""

    def __init__(self, max_eyes: int = 2, min_contrast: float = 12.0,
                 dark_ratio: float = 1.15, min_confidence: float = 0.5, carry_frames: int = 10):
        self.max_eyes = max_eyes
        self.carry_frames = carry_frames
        self.min_contrast = min_contrast
        self.dark_ratio = dark_ratio
        self.min_confidence = min_confidence
        self.bright: np.ndarray | None = None   # latest bright-pupil frame
        self.dark: np.ndarray | None = None     # latest dark-pupil frame
        self._means: deque[float] = deque(maxlen=3)
        self._onaxis: list[tuple[float, float, float, float]] = []  # last bright frame: (px, py, gx, gy)
        self._offsets: list[list[float]] = []   # per eye: [px, py, dx, dy] pair-mid minus on-axis
        self._prev: list[list[float]] = []      # last eyes: [px, py, r, frames since confirmed]
        self._levels: list[float] = []
        self.pupil_level: float | None = None
        self.last_kind = ""

    # ----------------------------------------------------------- frame type
    def frame_kind(self, img: np.ndarray) -> str:
        """"D" (dark-pupil, off-axis bank) or "B" (bright-pupil, on-axis)."""
        m = float(img[::8, ::8].mean())
        self._means.append(m)
        if len(self._means) >= 2 and m > self.dark_ratio * min(self._means):
            return "D"
        return "B"

    # ------------------------------------------------------------- finding
    def _candidates(self):
        """Pupil candidates from bright minus dark: [(x, y, r, contrast)]."""
        b = np.ascontiguousarray(self.bright[::2, ::2])
        d = np.ascontiguousarray(self.dark[::2, ::2])
        # saturating subtraction: only where the bright frame is brighter. The
        # on-axis glint lies inside the bright pupil and the dark frame's glint
        # pair only leaves a small hole, so glints need no masking (masking
        # saturated pixels would erase a saturated bright pupil)
        diff = cv2.subtract(b, d)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        peak = float(diff.max())
        if peak < self.min_contrast:
            return []
        thr = max(self.min_contrast, 0.4 * peak)
        mask = cv2.morphologyEx((diff >= thr).astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            r = np.sqrt(area / np.pi)
            if not 1.5 <= r <= 15:          # half resolution: 3-30 px pupil radius
                continue
            if area < 0.5 * np.pi * (max(w, h) / 2) ** 2 or max(w, h) > 2.2 * min(w, h):
                continue                    # not round: a motion edge, not a pupil
            contrast = float(diff[lab == i].mean())
            out.append((cen[i][0] * 2, cen[i][1] * 2, r * 2, contrast))
        out.sort(key=lambda c: -c[3])
        return out[:2 * self.max_eyes]

    # ------------------------------------------------------------ refining
    def _fit(self, img, kind, cx, cy, r0):
        roi_r = int(max(25, 3 * r0))
        if kind == "D":
            return _fit_pupil(img, cx, cy, roi_r=roi_r)
        return self._fit_bright(img, cx, cy, r0, roi_r)

    def _fit_bright(self, img, cx, cy, r0, roi_r):
        """Bright pupil: remove the glint and its bloom, threshold halfway
        between the pupil's inside and the iris around it, fit an ellipse."""
        h, w = img.shape
        x0, y0 = max(0, int(cx) - roi_r), max(0, int(cy) - roi_r)
        x1, y1 = min(w, int(cx) + roi_r + 1), min(h, int(cy) + roi_r + 1)
        roi = img[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        self._levels.append(float(np.percentile(roi[d <= 0.8 * r0], 75)))  # for gain control
        # remove glints (+ bloom): only *small* bright spots; a large saturated
        # disc is the bright pupil itself
        n, lab, st, _ = cv2.connectedComponentsWithStats((roi >= 200).astype(np.uint8))
        small = np.isin(lab, [i for i in range(1, n)
                              if st[i, cv2.CC_STAT_AREA] < 0.35 * np.pi * r0 * r0])
        gl = cv2.dilate(small.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        if gl.any():
            roi = cv2.inpaint(roi, gl, 4, cv2.INPAINT_TELEA)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        inner, ring = roi[d <= 0.6 * r0], roi[(d >= 1.7 * r0) & (d <= 2.5 * r0)]
        if inner.size < 5 or ring.size < 5:
            return None
        lo, hi = float(np.median(ring)), float(np.median(inner))
        if hi - lo < self.min_contrast:
            return None
        n, lab, st, _ = cv2.connectedComponentsWithStats((roi > (lo + hi) / 2).astype(np.uint8))
        lid = lab[int(round(cy)) - y0, int(round(cx)) - x0]
        if lid == 0:
            return None
        bx, by, bw, bh, area = st[lid]
        if bx == 0 or by == 0 or bx + bw >= roi.shape[1] or by + bh >= roi.shape[0]:
            return None  # leaked into the surroundings
        cnts, _ = cv2.findContours((lab == lid).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 8:
            return None
        (ex, ey), (ma, mi), _ = cv2.fitEllipse(c.astype(np.float32))
        hull = cv2.convexHull(c)
        perim = cv2.arcLength(hull, True)
        q = 4 * np.pi * cv2.contourArea(hull) / (perim * perim + 1e-6)
        if not np.isfinite([ex, ey, ma, mi]).all() or max(ma, mi) > 2 * min(ma, mi):
            return None
        return ex + x0, ey + y0, (ma + mi) / 4, float(min(1.0, q))

    def _offset_for(self, px, py):
        best = None
        for o in self._offsets:
            dist = np.hypot(o[0] - px, o[1] - py)
            if dist < 60 and (best is None or dist < best[0]):
                best = (dist, o)
        return best[1] if best else None

    def _learn_offset(self, px, py, mx, my):
        """How far the dark frame's glint-pair midpoint sits from where the
        on-axis glint would be, measured relative to the pupil in each frame:
        (pair mid - dark pupil) - (on-axis glint - bright pupil) from the latest
        bright frame. Relative to the pupil, it also absorbs the small
        difference between fitting a bright and a dark pupil."""
        near = [g for g in self._onaxis if np.hypot(g[0] - px, g[1] - py) < 20]
        if not near:
            return
        bpx, bpy, gx, gy = near[0]
        dx, dy = (mx - px) - (gx - bpx), (my - py) - (gy - bpy)
        o = self._offset_for(px, py)
        if o is None:
            self._offsets.append([px, py, dx, dy])
            del self._offsets[:-4]
        else:
            o[0], o[1] = px, py
            o[2] += 0.2 * (dx - o[2])
            o[3] += 0.2 * (dy - o[3])

    def update(self, img: np.ndarray) -> list[EyeFeature]:
        kind = self.frame_kind(img)
        self.last_kind = kind
        self._levels = []
        if kind == "B":
            self.bright = img
        else:
            self.dark = img
        if self.bright is None or self.dark is None:
            return []
        h, w = img.shape
        eyes = []
        onaxis = []
        cands = [c + (True,) for c in self._candidates()]
        # during fast eye movements the pupil moves between the bright and the
        # dark frame and the difference blob stops being round: keep following
        # recently confirmed eyes from their last position for a few frames
        for px, py, pr, age in self._prev:
            if age < self.carry_frames and all(np.hypot(px - c[0], py - c[1]) > 2 * pr + 4
                                               for c in cands):
                cands.append((px, py, pr, 2.5 * self.min_contrast, False))
        found = []
        for cx, cy, r0, contrast, confirmed in cands:
            fit = self._fit(img, kind, cx, cy, r0)
            if fit is None:
                continue
            px, py, pr, q = fit
            if np.hypot(px - cx, py - cy) > max(5.0, r0) or not 0.4 * r0 <= pr <= 2.5 * r0 + 2:
                continue  # the fit drifted off the difference blob
            reach = 1.4 * pr + 6
            blobs = [g for g in _glint_blobs(img, max(0, int(px - reach)), max(0, int(py - reach)),
                                             min(w, int(px + reach) + 1), min(h, int(py + reach) + 1))
                     if np.hypot(g[0] - px, g[1] - py) <= reach]
            if not blobs:
                continue
            if kind == "B":
                # the on-axis glint is a small spot; a saturated pupil is not a glint
                blobs = [g for g in blobs if g[2] < 0.35 * np.pi * pr * pr]
                if not blobs:
                    continue
                gx, gy, _ = min(blobs, key=lambda g: np.hypot(g[0] - px, g[1] - py))
                n_glints = 1
                onaxis.append((px, py, gx, gy))
            else:
                top = sorted(blobs, key=lambda g: -g[2])[:2]
                mx = float(np.mean([g[0] for g in top]))
                my = float(np.mean([g[1] for g in top]))
                n_glints = len(top)
                self._learn_offset(px, py, mx, my)
                o = self._offset_for(px, py)
                if o is None:
                    continue  # no on-axis reference for this eye yet
                gx, gy = mx - o[2], my - o[3]
            conf = float(q) * min(1.0, contrast / 30.0)
            if conf >= self.min_confidence:
                eyes.append(EyeFeature(px, py, pr, gx, gy, n_glints, conf))
                found.append((eyes[-1], confirmed, cx, cy))
        if kind == "B":
            self._onaxis = onaxis
            if self._levels:
                #: brightest bright-pupil level in the last bright frame (for gain control)
                self.pupil_level = max(self._levels)
        prev = []
        for e, confirmed, cx, cy in found:
            age = 0 if confirmed else next((p[3] + 1 for p in self._prev
                                            if np.hypot(p[0] - cx, p[1] - cy) < 1), 1)
            prev.append([e.pupil_x, e.pupil_y, e.pupil_r, age])
        self._prev = prev
        eyes.sort(key=lambda e: -e.confidence)
        eyes = eyes[:self.max_eyes]
        eyes.sort(key=lambda e: e.pupil_x)
        return eyes


class PupilGainControl:
    """Keeps the bright pupil just below saturation by adjusting the camera
    gain. A saturated bright pupil hides the on-axis glint inside it; how
    bright it gets depends on the subject, distance and gaze direction, so
    a fixed gain can't suit everyone (Gazepoint Control's gain also varies
    between sessions). Steps down quickly when saturated, up slowly when dim.
    """

    def __init__(self, camera, low: float = 110.0, high: float = 225.0,
                 step_db: float = 0.5, interval_s: float = 0.2):
        self.cam, self.low, self.high = camera, low, high
        self.step_db, self.interval_s = step_db, interval_s
        self._next = 0.0
        self._range = None

    def update(self, detector: BrightDarkDetector, t: float):
        level = detector.pupil_level
        if level is None or t < self._next:
            return
        detector.pupil_level = None  # use each measurement once
        if self.low <= level <= self.high:
            return
        try:
            f = self.cam.feature("gain")
            if self._range is None:
                self._range = (f["min"], f["max"])
            step = -2 * self.step_db if level > self.high else self.step_db
            gain = min(self._range[1], max(self._range[0], float(f["value"]) + step))
            if gain != f["value"]:
                self.cam.set_feature("gain", gain)
        except Exception:
            return
        self._next = t + self.interval_s
