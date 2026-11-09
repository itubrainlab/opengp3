"""Gaze smoothing."""
from __future__ import annotations

import math


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012) for a 2-D gaze point.

    Smooths heavily while the gaze is still (removes jitter during fixations)
    and lightly during fast movements (keeps saccades responsive).

    min_cutoff: Hz, lower = smoother fixations (more lag at low speeds)
    beta:       speed coefficient, higher = less lag during saccades
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.005, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._t = None
        self._x = None
        self._dx = (0.0, 0.0)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]:
        if self._t is None or t <= self._t or t - self._t > 0.5:  # first / gap
            self._t, self._x, self._dx = t, (x, y), (0.0, 0.0)
            return x, y
        dt = t - self._t
        self._t = t
        a_d = self._alpha(self.d_cutoff, dt)
        dx = tuple(a_d * (v - p) / dt + (1 - a_d) * pd
                   for v, p, pd in zip((x, y), self._x, self._dx))
        speed = math.hypot(*dx)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self._x = tuple(a * v + (1 - a) * p for v, p in zip((x, y), self._x))
        self._dx = dx
        return self._x
