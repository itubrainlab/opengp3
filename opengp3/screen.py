"""Screen geometry: size, test areas and visual-angle conversion."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rect:
    """Axis-aligned area on the screen, in screen points."""
    x: float
    y: float
    w: float
    h: float

    def to_screen(self, fx: float, fy: float) -> tuple[float, float]:
        """Map fractions (0..1) of this area to screen coordinates."""
        return self.x + fx * self.w, self.y + fy * self.h

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h


@dataclass
class Screen:
    """The display that gaze is reported on.

    Coordinates are screen *points* with the origin at the top-left: the same
    units macOS uses for windows and the mouse (on a Retina display 1 point =
    2 pixels). ``width_cm`` and ``distance_cm`` are only needed to express
    errors in degrees of visual angle.
    """
    width: int
    height: int
    width_cm: float | None = None
    distance_cm: float = 65.0
    x: float = 0.0            # global position of the screen (multi-monitor setups)
    y: float = 0.0

    @classmethod
    def detect(cls, width_cm: float | None = None, distance_cm: float = 65.0,
               display: int = 0) -> "Screen":
        """A display's size and position in points. ``display`` indexes
        ``opengp3.platform.displays()`` (0 = main display)."""
        from .platform import displays
        ds = displays()
        if ds:
            if not 0 <= display < len(ds):
                raise ValueError(f"display {display} not found: {len(ds)} display(s), numbered "
                                 f"0-{len(ds) - 1} (0 = main)")
            d = ds[display]
            if width_cm is None and d.width_mm:  # physical size reported by the OS
                width_cm = round(d.width_mm / 10, 1)
            return cls(int(d.width), int(d.height), width_cm, distance_cm, d.x, d.y)
        if display:
            raise ValueError("display selection is not available on this system")
        w, h = _main_display_size()
        if width_cm is None and sys.platform.startswith("linux"):
            # no X11 display list (e.g. Wayland): the EDID in sysfs, if unambiguous
            from .platform import linux_edid_widths
            widths = set(linux_edid_widths().values())
            if len(widths) == 1:
                width_cm = round(widths.pop() / 10, 1)
        return cls(w, h, width_cm, distance_cm)

    def to_global(self, x: float, y: float) -> tuple[float, float]:
        """Screen-local points -> global desktop points."""
        return self.x + x, self.y + y

    @property
    def rect(self) -> Rect:
        return Rect(0, 0, self.width, self.height)

    @property
    def cm_per_point(self) -> float | None:
        return self.width_cm / self.width if self.width_cm else None

    def area(self, diagonal_in: float, aspect: float = 16 / 9) -> Rect:
        """Centred area the size of a monitor with this diagonal (inches).

        Useful on very large/wide screens whose edges are beyond the GP3's
        gaze range (about +-20 degrees): e.g. ``screen.area(22)`` on a 34"
        ultrawide. Needs ``width_cm``.
        """
        if not self.width_cm:
            raise ValueError("Screen.width_cm is required to size an area in inches")
        d_cm = diagonal_in * 2.54
        w_cm = d_cm * aspect / np.hypot(aspect, 1)
        h_cm = d_cm / np.hypot(aspect, 1)
        w = min(self.width, w_cm / self.cm_per_point)
        h = min(self.height, h_cm / self.cm_per_point)
        return Rect(float((self.width - w) / 2), float((self.height - h) / 2), float(w), float(h))

    @property
    def diagonal_in(self) -> float | None:
        """Physical diagonal in inches (None if the physical width is unknown)."""
        if not self.width_cm:
            return None
        return float(np.hypot(self.width_cm, self.width_cm * self.height / self.width) / 2.54)

    def certified_area(self, max_diagonal_in: float | None = 22.0, aspect: float = 16 / 9) -> Rect:
        """Area to calibrate/test on: the whole screen if it is no larger than
        ``max_diagonal_in`` (the GP3 is specified for screens up to 22"),
        otherwise a centred area of that size. Without a known physical
        width the whole screen is used."""
        d = self.diagonal_in
        if max_diagonal_in is None or d is None or d <= max_diagonal_in:
            return self.rect
        return self.area(max_diagonal_in, aspect)

    def angle_deg(self, p, q) -> float:
        """Visual angle between screen points p and q, for an eye facing the
        screen centre at ``distance_cm``. Needs ``width_cm``."""
        if not self.width_cm:
            raise ValueError("Screen.width_cm is required for visual angles")
        k = self.cm_per_point

        def ray(pt):
            return np.array([(pt[0] - self.width / 2) * k, (pt[1] - self.height / 2) * k,
                             self.distance_cm])
        a, b = ray(p), ray(q)
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    def eccentricity_deg(self, x: float, y: float) -> float:
        """Visual angle of a screen point from the screen centre."""
        return self.angle_deg((x, y), (self.width / 2, self.height / 2))


def _main_display_size() -> tuple[int, int]:
    if sys.platform == "darwin":
        # main display's "UI looks like" size, in points (what OpenCV's
        # fullscreen window covers). Finder's desktop bounds would span all
        # displays, so they are only a fallback.
        try:
            import json
            out = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"],
                                 capture_output=True, text=True, timeout=15).stdout
            for gpu in json.loads(out).get("SPDisplaysDataType", []):
                for d in gpu.get("spdisplays_ndrvs", []):
                    if d.get("spdisplays_main") == "spdisplays_yes":
                        res = d.get("_spdisplays_resolution") or d.get("spdisplays_resolution")
                        w, h = res.split("@")[0].split("x")
                        return int(float(w)), int(float(h))
        except Exception:
            pass
        try:
            out = subprocess.run(
                ["osascript", "-e", 'tell application "Finder" to get bounds of window of desktop'],
                capture_output=True, text=True, timeout=5).stdout
            x0, y0, x1, y1 = (int(v) for v in out.split(","))
            return x1 - x0, y1 - y0
        except Exception:
            pass
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1440, 900
