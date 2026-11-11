"""Pupil and corneal-reflection detection for GP3 infrared eye images.

Glint-anchored dark-pupil method:
1. find corneal reflections (glints): small saturated blobs on a dark surround;
2. group them per cornea (the GP3's two LED banks give a pair per eye);
3. inpaint the glints, segment the pupil with an edge-scored threshold sweep,
   and fit an ellipse.
The pupil-minus-glint vector (PCR) is the gaze signal a calibration maps to
screen coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class EyeFeature:
    pupil_x: float          # pupil centre (ellipse fit), image px
    pupil_y: float
    pupil_r: float          # mean ellipse radius, px
    glint_x: float          # centroid of the corneal reflections, px
    glint_y: float
    n_glints: int = 0
    confidence: float = 0.0  # 0..1, pupil ellipse circularity * fit quality

    @property
    def pcr(self) -> tuple[float, float]:
        """Pupil-minus-corneal-reflection vector: the gaze signal that a
        calibration maps to screen coordinates (robust to small head shifts)."""
        return self.pupil_x - self.glint_x, self.pupil_y - self.glint_y


def _find_glints(img: np.ndarray, blur: np.ndarray):
    """Small saturated spots sitting on a dark (pupil) surround."""
    bright = (img >= 230).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(bright)
    h, w = img.shape
    out = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if not 2 <= area <= 120:
            continue
        cx, cy = cen[i]
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        r = int(max(bw, bh)) + 4
        x0, x1 = int(cx) - r, int(cx) + r + 1
        y0, y1 = int(cy) - r, int(cy) + r + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        ring = blur[y0:y1, x0:x1][bright[y0:y1, x0:x1] == 0]
        # surround must be dark (pupil), unlike specular spots on skin
        if ring.size and np.percentile(ring, 30) < 120:
            out.append((cx, cy, area))
    return out


def _cluster(glints, max_dist=25.0):
    """Group glints belonging to the same cornea (GP3 has 2 LED banks)."""
    groups = []
    for g in sorted(glints, key=lambda g: -g[2]):
        for grp in groups:
            if any(np.hypot(g[0] - q[0], g[1] - q[1]) < max_dist for q in grp):
                grp.append(g)
                break
        else:
            groups.append([g])
    return groups


def _fit_pupil(img: np.ndarray, sx: float, sy: float, roi_r: int = 60, glint_level: int = 140):
    """Segment the dark pupil around the glints and fit an ellipse.

    Sweeps a threshold upward from the darkest pixel near the glints and
    keeps the seed-connected region whose boundary has the strongest edge
    (the pupil/iris step) times roundness. The ellipse is then fitted only to
    boundary points outside the inpainted glint area.
    Returns (x, y, radius, quality) in image px, or None.
    """
    h, w = img.shape
    x0, y0 = max(0, int(sx) - roi_r), max(0, int(sy) - roi_r)
    x1, y1 = min(w, int(sx) + roi_r), min(h, int(sy) + roi_r)
    roi = img[y0:y1, x0:x1].copy()
    # Inpaint glints (+ their ~2px bloom halo) from the surrounding pixels.
    # Painting them with a flat dark value instead creates a glint-shaped blob
    # that becomes the "pupil" when the pupil is small, locking pupil to glint.
    gl = cv2.dilate((roi >= glint_level).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    roi = cv2.inpaint(roi, gl, 5, cv2.INPAINT_TELEA)
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    gl_edge = cv2.dilate(gl, np.ones((3, 3), np.uint8)) > 0  # unreliable px

    cx, cy = int(sx) - x0, int(sy) - y0
    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    near = (xx - cx) ** 2 + (yy - cy) ** 2 <= 10 ** 2
    sy_, sx_ = np.unravel_index(np.argmin(np.where(near, roi, 255)), roi.shape)
    p_level = float(roi[sy_, sx_])
    rh, rw = roi.shape

    best = None  # (score, contour_pts, quality)
    for thr in np.arange(p_level + 4, p_level + 70, 3):
        mask = (roi < thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        lid = lab[sy_, sx_]
        if lid == 0:
            continue
        bx, by, bw, bh, area = st[lid]
        if area < 8:
            continue
        if bx == 0 or by == 0 or bx + bw >= rw or by + bh >= rh:
            break  # leaked out of the ROI: larger thresholds only get worse
        if best is not None and area > 4 * best[3]:
            # the region jumped from the pupil to the whole (dark) iris: its
            # boundary with the sclera can out-score the pupil edge (GP3 HD)
            break
        cnts, _ = cv2.findContours((lab == lid).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)[:, 0, :]
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        perim = cv2.arcLength(hull, True)
        circ = 4 * np.pi * ha / (perim * perim + 1e-6)
        solidity = area / (ha + 1e-6)
        if circ < 0.6 or solidity < 0.7:
            continue
        ok = ~gl_edge[c[:, 1], c[:, 0]]
        if ok.sum() < 6:
            continue
        edge = float(np.median(grad[c[ok, 1], c[ok, 0]]))
        score = edge * circ * solidity
        if best is None or score > best[0]:
            best = (score, c, c[ok], area, circ * solidity)
    if best is None:
        return None
    _, c_all, pts, area, q = best
    # region moments: robust centre/radius even when glints hide much of the edge
    M = cv2.moments(c_all.reshape(-1, 1, 2).astype(np.int32))
    if M["m00"] <= 0:
        return None
    mx, my = M["m10"] / M["m00"], M["m01"] / M["m00"]
    r_area = float(np.sqrt(area / np.pi))
    # prefer the sub-pixel ellipse fit on reliable edge points, but only if it
    # agrees with the region (small pupils + glints leave few/one-sided points)
    if len(pts) >= 8:
        (ex, ey), (ma, mi), _ = cv2.fitEllipse(pts.astype(np.float32))
        r_fit = (ma + mi) / 4
        if (np.isfinite([ex, ey, ma, mi]).all() and max(ma, mi) <= 2.5 * min(ma, mi)
                and 0.7 * r_area <= r_fit <= 1.4 * r_area
                and np.hypot(ex - mx, ey - my) <= 0.5 * r_area + 1):
            return ex + x0, ey + y0, r_fit, float(min(1.0, q))
    return mx + x0, my + y0, r_area, float(min(1.0, q)) * 0.9


def detect_eyes(img: np.ndarray, max_eyes: int = 2,
                min_confidence: float = 0.6) -> list[EyeFeature]:
    """Glint-anchored dark-pupil detector for GP3 IR images.

    1. find corneal reflections: small saturated blobs on a dark surround
    2. group them per eye (the GP3's two LED banks give a pair per cornea)
    3. inpaint glints, segment the pupil (edge-scored threshold sweep), fit an ellipse
    Returns up to `max_eyes` eyes sorted left-to-right in the image
    (the subject's right eye is on the image left).
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    eyes = []
    for grp in _cluster(_find_glints(img, blur)):
        # unweighted midpoint of the (up to) two largest glints: an area
        # weighting shifts as one glint shrinks, faking gaze movement
        top = sorted(grp, key=lambda g: -g[2])[:2]
        gx = float(np.mean([g[0] for g in top]))
        gy = float(np.mean([g[1] for g in top]))
        fit = _fit_pupil(img, gx, gy)
        if fit is None:
            continue
        px, py, pr, q = fit
        if not 2 <= pr <= 60 or np.hypot(px - gx, py - gy) > 1.2 * pr + 3:
            continue  # glints must lie on the pupil/cornea
        pair_bonus = 1.0 if len(grp) >= 2 else 0.7
        conf = q * pair_bonus
        if conf >= min_confidence:
            eyes.append(EyeFeature(px, py, pr, gx, gy, len(grp), conf))
    eyes.sort(key=lambda e: -e.confidence)
    eyes = eyes[:max_eyes]
    eyes.sort(key=lambda e: e.pupil_x)
    return eyes


class EyeTracker:
    """Keeps eye identity across frames so one-eye tracking works.

    The camera faces the subject, so the subject's RIGHT eye is on the image
    LEFT. With two eyes visible, identity is by x order. With one, it's
    assigned to the nearest last-known eye position (predicted with the last
    inter-eye offset). This keeps per-eye calibrations valid while one eye is
    out of the GP3's narrow field of view, which roughly doubles the usable
    sideways head movement.
    """

    def __init__(self, max_jump: float = 0.45):
        self.pos = {"R": None, "L": None}  # last pupil positions, image px
        self.offset = None                  # L - R pupil vector
        self.max_jump = max_jump            # fraction of eye spacing

    def update(self, eyes: list[EyeFeature]) -> dict[str, EyeFeature]:
        out: dict[str, EyeFeature] = {}
        if len(eyes) >= 2:
            r, l = sorted(eyes, key=lambda e: e.pupil_x)[:2]
            out = {"R": r, "L": l}
            self.offset = np.array([l.pupil_x - r.pupil_x, l.pupil_y - r.pupil_y])
        elif len(eyes) == 1:
            e = eyes[0]
            p = np.array([e.pupil_x, e.pupil_y])
            pred = {}
            if self.pos["R"] is not None:
                pred["R"] = self.pos["R"]
            if self.pos["L"] is not None:
                pred["L"] = self.pos["L"]
            if self.pos["R"] is not None and self.offset is not None:
                pred.setdefault("L", self.pos["R"] + self.offset)
            if self.pos["L"] is not None and self.offset is not None:
                pred.setdefault("R", self.pos["L"] - self.offset)
            if pred:
                side = min(pred, key=lambda k: np.hypot(*(p - pred[k])))
                spacing = np.hypot(*self.offset) if self.offset is not None else 200.0
                if np.hypot(*(p - pred[side])) <= self.max_jump * spacing:
                    out = {side: e}
            # no history / ambiguous: leave unlabelled rather than guess
        for side in ("R", "L"):
            if side in out:
                self.pos[side] = np.array([out[side].pupil_x, out[side].pupil_y])
            elif side != "R" and "R" in out and self.offset is not None:
                self.pos["L"] = self.pos["R"] + self.offset  # follow the visible eye
            elif side != "L" and "L" in out and self.offset is not None:
                self.pos["R"] = self.pos["L"] - self.offset
        return out
