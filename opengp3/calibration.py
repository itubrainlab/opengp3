"""Mapping eye features to screen coordinates.

A calibration is fitted per eye from fixations on known targets: the median
pupil-glint (PCR) vector at each target is mapped to the target's screen
position. For each eye the model (affine / x2-y2 / quadratic) is chosen by
leave-one-out error over the calibration points, so it adapts to how many
points that eye actually has (an eye can lose its glint at extreme angles).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .screen import Rect, Screen

EYES = ("R", "L")  # subject's right / left eye


# ---------------------------------------------------------------- target grids
def grid_points(n: int = 9, margin: float = 0.1) -> list[tuple[float, float]]:
    """Calibration target positions as fractions of the calibrated area.

    n = 5 (corners + centre), 9 (3x3), 13 (3x3 + 4 in between), 16 (4x4),
    25 (5x5). More points model the screen edges better (useful for full-screen
    calibration on large displays) at the cost of a longer calibration.
    """
    lo, hi = margin, 1 - margin
    lin = lambda k: list(np.linspace(lo, hi, k))
    if n == 5:
        pts = [(lo, lo), (hi, lo), (0.5, 0.5), (lo, hi), (hi, hi)]
    elif n == 9:
        pts = [(x, y) for y in lin(3) for x in lin(3)]
    elif n == 13:
        q1, q3 = lo + (hi - lo) / 4, lo + 3 * (hi - lo) / 4
        pts = [(x, y) for y in lin(3) for x in lin(3)] + [(q1, q1), (q3, q1), (q1, q3), (q3, q3)]
    elif n == 16:
        pts = [(x, y) for y in lin(4) for x in lin(4)]
    elif n == 25:
        pts = [(x, y) for y in lin(5) for x in lin(5)]
    else:
        raise ValueError("n must be one of 5, 9, 13, 16, 25")
    return [(float(x), float(y)) for x, y in pts]


def presentation_order(points) -> list[int]:
    """Centre first, then nearest-neighbour tour (short eye movements)."""
    pts = np.array(points)
    left = list(range(len(pts)))
    cur = int(np.argmin(np.hypot(pts[:, 0] - 0.5, pts[:, 1] - 0.5)))
    order = [cur]
    left.remove(cur)
    while left:
        cur = min(left, key=lambda i: np.hypot(*(pts[i] - pts[cur])))
        order.append(cur)
        left.remove(cur)
    return order


# Validation points sit between the 3x3 calibration points, so validation
# measures interpolation accuracy on positions the calibration never saw.
VALIDATION_POINTS = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7),
                     (0.5, 0.25), (0.5, 0.75), (0.2, 0.5), (0.8, 0.5)]


# ---------------------------------------------------------------- models
def _cols(*fs):
    def terms(v):
        v = np.atleast_2d(np.asarray(v, float))
        return np.column_stack([f(v[:, 0], v[:, 1]) for f in fs])
    return terms


_1 = lambda x, y: np.ones_like(x)
_x = lambda x, y: x
_y = lambda x, y: y
MODELS = {  # name: (terms for screen x, terms for screen y)
    "affine": (_cols(_1, _x, _y), _cols(_1, _x, _y)),
    "x2/y2": (_cols(_1, _x, _y, lambda x, y: x * x), _cols(_1, _x, _y, lambda x, y: y * y)),
    "quadratic": (_cols(_1, _x, _y, lambda x, y: x * y, lambda x, y: x * x, lambda x, y: y * y),) * 2,
}


def _n_params(name: str) -> int:
    return max(MODELS[name][i](np.zeros((1, 2))).shape[1] for i in (0, 1))


class Mapping:
    """PCR vector -> screen point for one eye."""

    def __init__(self, model: str, cx, cy):
        self.model = model
        self.cx, self.cy = np.asarray(cx, float), np.asarray(cy, float)
        self._tx, self._ty = MODELS[model]

    @classmethod
    def fit(cls, model: str, pcr: np.ndarray, screen_pts: np.ndarray) -> "Mapping":
        tx, ty = MODELS[model]
        cx = np.linalg.lstsq(tx(pcr), screen_pts[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(ty(pcr), screen_pts[:, 1], rcond=None)[0]
        return cls(model, cx, cy)

    def __call__(self, pcr) -> tuple[float, float]:
        return float((self._tx(pcr) @ self.cx)[0]), float((self._ty(pcr) @ self.cy)[0])

    def to_dict(self) -> dict:
        return {"model": self.model, "cx": self.cx.tolist(), "cy": self.cy.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Mapping":
        return cls(d["model"], d["cx"], d["cy"])


@dataclass
class EyeFit:
    mapping: Mapping
    n_points: int
    loo_error: float        # median leave-one-out error, screen points
    residual: float         # median fit residual, screen points


@dataclass
class CalibrationSample:
    """One eye observation while a target was shown (after settling)."""
    target: int
    x: float                # target position, screen points
    y: float
    eye: str                # "R" / "L"
    pcr_x: float
    pcr_y: float
    t: float = 0.0


class CalibrationError(RuntimeError):
    pass


@dataclass
class Calibration:
    """Per-eye PCR -> screen mappings plus the context they were made in."""
    eyes: dict[str, EyeFit]
    screen: Screen
    area: Rect
    created: float = field(default_factory=time.time)

    # -------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, samples: list[CalibrationSample], screen: Screen, area: Rect | None = None,
            min_samples: int = 10, models=tuple(MODELS)) -> "Calibration":
        """Fit each eye from samples collected on the calibration targets.

        Per target, the median PCR (robust to blinks/outliers) is used;
        targets with fewer than ``min_samples`` samples for an eye are skipped
        for that eye. Raises CalibrationError if no eye can be calibrated.
        """
        fits: dict[str, EyeFit] = {}
        targets = sorted({s.target for s in samples})
        for eye in EYES:
            P, S = [], []
            for ti in targets:
                ss = [s for s in samples if s.eye == eye and s.target == ti]
                if len(ss) < min_samples:
                    continue
                P.append(np.median([(s.pcr_x, s.pcr_y) for s in ss], axis=0))
                S.append((ss[0].x, ss[0].y))
            if not P:
                continue
            P, S = np.array(P), np.array(S, float)
            best = None
            for name in models:
                if len(P) < _n_params(name) + 2:  # need spare points for LOO
                    continue
                errs = []
                for i in range(len(P)):
                    keep = np.arange(len(P)) != i
                    g = Mapping.fit(name, P[keep], S[keep])(P[i])
                    errs.append(float(np.hypot(g[0] - S[i, 0], g[1] - S[i, 1])))
                loo = float(np.median(errs))
                if best is None or loo < best[1]:
                    best = (name, loo)
            if best is None:
                continue
            m = Mapping.fit(best[0], P, S)
            res = [np.hypot(*(np.array(m(p)) - s)) for p, s in zip(P, S)]
            fits[eye] = EyeFit(m, len(P), best[1], float(np.median(res)))
        if not fits:
            raise CalibrationError(
                "no eye had enough calibration points - check that the eyes are "
                "visible (opengp3 view) and that the targets are within the GP3's range")
        return cls(fits, screen, area or screen.rect)

    # -------------------------------------------------------------- use
    def predict(self, eyes: dict) -> tuple[float, float, dict[str, tuple[float, float]]] | None:
        """Gaze point from ``{"R": EyeFeature, "L": ...}`` (either may be
        missing). Returns (x, y, per_eye) with x, y the mean of the calibrated
        visible eyes, or None if no calibrated eye is visible."""
        per_eye = {}
        for side, e in eyes.items():
            if side in self.eyes:
                per_eye[side] = self.eyes[side].mapping(e.pcr)
        if not per_eye:
            return None
        xs, ys = zip(*per_eye.values())
        return float(np.mean(xs)), float(np.mean(ys)), per_eye

    def summary(self) -> str:
        lines = []
        k = self.screen.cm_per_point
        for side in EYES:
            if side not in self.eyes:
                lines.append(f"eye {side}: not calibrated")
                continue
            f = self.eyes[side]
            deg = ""
            if k:
                deg = f" ({np.degrees(np.arctan(f.loo_error * k / self.screen.distance_cm)):.2f}°)"
            lines.append(f"eye {side}: {f.n_points} points, model '{f.mapping.model}', "
                         f"leave-one-out {f.loo_error:.0f} pt{deg}")
        return "\n".join(lines)

    # -------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "created": self.created,
            "screen": {"width": self.screen.width, "height": self.screen.height,
                       "width_cm": self.screen.width_cm, "distance_cm": self.screen.distance_cm},
            "area": [self.area.x, self.area.y, self.area.w, self.area.h],
            "eyes": {side: {**f.mapping.to_dict(), "n_points": f.n_points,
                            "loo_error": f.loo_error, "residual": f.residual}
                     for side, f in self.eyes.items()},
        }

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        s = d["screen"]
        eyes = {side: EyeFit(Mapping.from_dict(e), e["n_points"], e["loo_error"], e["residual"])
                for side, e in d["eyes"].items()}
        return cls(eyes, Screen(s["width"], s["height"], s.get("width_cm"),
                                s.get("distance_cm", 65.0)),
                   Rect(*d["area"]), d.get("created", 0.0))

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------- validation
@dataclass
class PointResult:
    target: int
    x: float
    y: float
    n: int
    gaze_x: float           # median gaze
    gaze_y: float
    offset: float           # screen points
    accuracy_deg: float | None
    rms_s2s_deg: float | None
    sd_deg: float | None
    samples: np.ndarray = field(repr=False, default=None)


@dataclass
class ValidationResult:
    points: list[PointResult]
    screen: Screen

    def _vals(self, attr):
        v = [getattr(p, attr) for p in self.points if getattr(p, attr) is not None]
        return np.array(v) if v else None

    @property
    def accuracy_deg(self) -> float | None:
        v = self._vals("accuracy_deg")
        return float(v.mean()) if v is not None else None

    @property
    def precision_rms_s2s_deg(self) -> float | None:
        v = self._vals("rms_s2s_deg")
        return float(v.mean()) if v is not None else None

    @property
    def offset_points(self) -> float:
        return float(np.mean([p.offset for p in self.points])) if self.points else float("nan")

    def report(self) -> str:
        k = self.screen.cm_per_point
        out = ["target   screen(x,y)      n   offset pt   offset cm   accuracy°   RMS-S2S°   SD°"]
        f = lambda v, w=8: f"{v:>{w}.2f}" if v is not None else " " * (w - 1) + "-"
        for p in self.points:
            out.append(f"{p.target:>4}   ({p.x:>5.0f},{p.y:>5.0f})  {p.n:>4}   {p.offset:>8.0f}   "
                       f"{f(p.offset * k if k else None, 8)}   {f(p.accuracy_deg)}    "
                       f"{f(p.rms_s2s_deg, 7)}   {f(p.sd_deg, 5)}")
        acc = self._vals("accuracy_deg")
        if acc is not None:
            out.append(f"\nACCURACY  mean {acc.mean():.2f}°  median {np.median(acc):.2f}°  "
                       f"worst {acc.max():.2f}°   (mean {self.offset_points * k:.1f} cm on screen)")
            out.append(f"PRECISION RMS-S2S {self.precision_rms_s2s_deg:.2f}°   "
                       f"SD {self._vals('sd_deg').mean():.2f}°")
            out.append(f"(viewing distance {self.screen.distance_cm:.0f} cm, "
                       f"screen {self.screen.width_cm:.0f} cm wide)")
        else:
            out.append(f"\nmean offset {self.offset_points:.0f} pt "
                       "(set Screen.width_cm for errors in degrees)")
        return "\n".join(out)


def evaluate(gaze: list[tuple[int, float, float, float, float, float]], screen: Screen,
             min_samples: int = 10) -> ValidationResult:
    """Accuracy/precision from validation gaze samples.

    gaze: (target, target_x, target_y, t, gaze_x, gaze_y) per frame.
    accuracy = angle between median gaze and target; precision = RMS of
    sample-to-sample angles (RMS-S2S) and RMS deviation from the median (SD).
    """
    deg = screen.angle_deg if screen.width_cm else None
    points = []
    for ti in sorted({g[0] for g in gaze}):
        gs = sorted((g for g in gaze if g[0] == ti), key=lambda g: g[3])
        if len(gs) < min_samples:
            continue
        tx, ty = gs[0][1], gs[0][2]
        G = np.array([(g[4], g[5]) for g in gs])
        med = np.median(G, axis=0)
        acc = s2s = sd = None
        if deg:
            acc = deg(med, (tx, ty))
            steps = [deg(G[i], G[i + 1]) for i in range(len(G) - 1)]
            s2s = float(np.sqrt(np.mean(np.square(steps))))
            sd = float(np.sqrt(np.mean(np.square([deg(g, med) for g in G]))))
        points.append(PointResult(ti, tx, ty, len(G), float(med[0]), float(med[1]),
                                  float(np.hypot(*(med - (tx, ty)))), acc, s2s, sd, G))
    return ValidationResult(points, screen)
