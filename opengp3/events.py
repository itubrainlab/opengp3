"""Online gaze event detection: fixations, blinks and the AAC moving average.

All filters are causal (one sample at a time, no look-ahead) so they can run
in the 60 Hz tracking loop. Times are in seconds on any monotonic clock.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Fixation:
    x: float = 0.0           # centroid, same units as the input (e.g. screen fraction)
    y: float = 0.0
    start: float = 0.0       # time the fixation began
    duration: float = 0.0
    id: int = 0
    valid: bool = False      # True only while a fixation is in progress and eyes are seen


class FixationFilter:
    """Dispersion-based online fixation detector.

    A sample within ``radius`` of the running fixation centroid extends the
    fixation. A saccade is only accepted once ``confirm`` consecutive samples
    fall outside the radius (isolated noisy samples are ignored instead of
    breaking the fixation); the new fixation then starts at the first of
    them. A candidate becomes a valid fixation after ``min_duration``. Short
    data gaps (blinks, < ``max_gap``) do not end a fixation, but samples
    during the gap are reported invalid. ``radius`` is in input units: with
    screen fractions, 0.025 is ~1 degree on a 50 cm wide screen at 65 cm.
    """

    def __init__(self, radius: float = 0.025, min_duration: float = 0.06, max_gap: float = 0.15,
                 aspect: float = 1.0, confirm: int = 3):
        self.radius, self.min_duration, self.max_gap = radius, min_duration, max_gap
        self.aspect = aspect  # width/height: makes the radius circular on screen
        self.confirm = max(1, confirm)
        self._n = 0
        self._sx = self._sy = 0.0
        self._start = None
        self._last_t = None
        self._id = 0
        self._counted = False
        self._pending: list[tuple[float, float, float]] = []  # samples outside the radius
        self._last = Fixation()

    def reset(self):
        self.__init__(self.radius, self.min_duration, self.max_gap, self.aspect, self.confirm)

    def _dist(self, x, y, cx, cy):
        return float(np.hypot(x - cx, (y - cy) / self.aspect))

    def _begin(self, pts):
        self._n = len(pts)
        self._sx = sum(p[1] for p in pts)
        self._sy = sum(p[2] for p in pts)
        self._start = pts[0][0]
        self._counted = False

    def update(self, t: float, x: float | None, y: float | None) -> Fixation:
        if x is None or y is None:  # no gaze
            if self._start is not None and self._last_t is not None and t - self._last_t > self.max_gap:
                self._n, self._start = 0, None  # gap too long: fixation over
                self._pending = []
            f = self._last
            return Fixation(f.x, f.y, f.start, f.duration, f.id, False)
        if not self._n:
            self._begin([(t, x, y)])
        elif self._dist(x, y, self._sx / self._n, self._sy / self._n) <= self.radius:
            self._n += 1
            self._sx += x
            self._sy += y
            self._pending = []  # the excursion was noise
        else:
            self._pending.append((t, x, y))
            # pending samples must agree with each other to count as a saccade target
            px, py = self._pending[0][1], self._pending[0][2]
            if any(self._dist(q[1], q[2], px, py) > self.radius for q in self._pending[1:]):
                self._pending = self._pending[-1:]
            if len(self._pending) >= self.confirm:
                self._begin(self._pending)
                self._pending = []
        self._last_t = t
        dur = t - self._start
        valid = dur >= self.min_duration
        if valid and not self._counted:
            self._id += 1
            self._counted = True
        self._last = Fixation(self._sx / self._n, self._sy / self._n, self._start, dur,
                              self._id, valid)
        return self._last


@dataclass
class BlinkInfo:
    id: int = 0              # blink number on the sample where a blink ended, else 0
    duration: float = 0.0    # duration of the most recent blink (s)
    per_minute: int = 0      # blinks in the last 60 s


class BlinkDetector:
    """Blink = both eyes lost for ``min_duration``..``max_duration`` and then
    found again (longer losses are looking away / leaving, not blinks)."""

    def __init__(self, min_duration: float = 0.05, max_duration: float = 0.5):
        self.min_duration, self.max_duration = min_duration, max_duration
        self._lost_since = None
        self._seen = False
        self._count = 0
        self._last_dur = 0.0
        self._times: deque[float] = deque()

    def update(self, t: float, eyes_visible: bool) -> BlinkInfo:
        bid = 0
        if not eyes_visible:
            if self._seen and self._lost_since is None:
                self._lost_since = t
        else:
            if self._lost_since is not None:
                dur = t - self._lost_since
                if self.min_duration <= dur <= self.max_duration:
                    self._count += 1
                    bid = self._count
                    self._last_dur = dur
                    self._times.append(t)
                self._lost_since = None
            self._seen = True
        while self._times and t - self._times[0] > 60.0:
            self._times.popleft()
        return BlinkInfo(bid, self._last_dur, len(self._times))


class MovingAverage:
    """Moving-window average of the last ``length`` valid points (the AAC POG:
    steadier targeting for assistive / gaze-pointer use)."""

    def __init__(self, length: int = 15):
        self.length = max(1, int(length))
        self._buf: deque[tuple[float, float]] = deque(maxlen=self.length)

    def set_length(self, length: int):
        self.length = max(1, int(length))
        self._buf = deque(self._buf, maxlen=self.length)

    def update(self, x: float | None, y: float | None) -> tuple[float, float] | None:
        if x is None or y is None:
            return (tuple(np.mean(self._buf, axis=0)) if self._buf else None)
        self._buf.append((x, y))
        m = np.mean(self._buf, axis=0)
        return float(m[0]), float(m[1])
