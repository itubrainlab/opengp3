"""Fullscreen stimulus window (OpenCV) used for calibration, validation and
the gaze demo. OpenCV GUI calls must run on the main thread on macOS."""
from __future__ import annotations

import cv2
import numpy as np

from .screen import Rect, Screen


class Cancelled(Exception):
    """The user pressed Esc / q or closed the window."""


class StimulusWindow:
    """Black fullscreen canvas in screen-point coordinates.

    The canvas is rendered at a reduced resolution (OpenCV scales it up to
    the screen) so drawing stays fast on large displays.
    """

    def __init__(self, screen: Screen, title: str = "OpenGP3", max_canvas_width: int = 1720):
        self.screen = screen
        self.s = min(1.0, max_canvas_width / screen.width)
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        if screen.x or screen.y:  # place on the selected display before going fullscreen
            cv2.moveWindow(title, int(screen.x) + 50, int(screen.y) + 50)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.canvas = None
        self.clear()
        self.show()
        self._ensure_fullscreen()

    def _ensure_fullscreen(self, timeout: float = 4.0):
        """Wait until the window really covers the screen.

        On macOS a fullscreen window created while another one is leaving
        fullscreen can end up offset/not fullscreen; targets would then be
        drawn somewhere else than their screen coordinates say, silently
        corrupting calibration/validation. Re-request fullscreen until
        OpenCV reports the image at (0, 0) at full canvas size.
        """
        import time
        want = (self.canvas.shape[1], self.canvas.shape[0])
        t0, t_retry = time.time(), time.time() + 0.8
        stable_since = None
        while time.time() - t0 < timeout:
            self.clear()
            self.show()
            x, y, w, h = cv2.getWindowImageRect(self.title)
            sw, sh = self.screen.width, self.screen.height
            ok = (((x, y) == (0, 0) and abs(w - want[0]) <= 2 and abs(h - want[1]) <= 2)
                  or (abs(w - sw) <= 2 and abs(h - sh) <= 2 and
                      (x, y) in ((0, 0), (int(self.screen.x), int(self.screen.y)))))
            if ok:
                stable_since = stable_since or time.time()
                if time.time() - stable_since > 0.3:
                    self.geometry = (x, y, w, h)
                    return
            else:
                stable_since = None
                if time.time() > t_retry:  # toggle fullscreen again
                    cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.waitKey(50)
                    cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    t_retry = time.time() + 0.8
        # Window-rect semantics differ between OpenCV GUI backends (Cocoa reports
        # canvas units at (0, 0); Win32/GTK/Qt report screen pixels at the
        # monitor origin). Accept a full-screen rect in either form, else fail.
        sw, sh = self.screen.width, self.screen.height
        if abs(w * want[1] - h * want[0]) <= 0.02 * want[0] * h and (
                abs(w - sw) <= 0.02 * sw or abs(w - want[0]) <= 2 or w >= sw):
            self.geometry = (x, y, w, h)
            return
        raise RuntimeError(
            f"stimulus window did not become fullscreen (image rect {x, y, w, h}, expected "
            f"the full {sw}x{sh} screen); targets would be misplaced")

    # ------------------------------------------------------------- drawing
    def _p(self, x, y):
        return int(round(x * self.s)), int(round(y * self.s))

    def clear(self, color=(0, 0, 0)):
        self.canvas = np.zeros((int(self.screen.height * self.s), int(self.screen.width * self.s), 3),
                               np.uint8)
        if color != (0, 0, 0):
            self.canvas[:] = color

    def dot(self, x, y, r, color=(255, 255, 255), thickness=-1):
        cv2.circle(self.canvas, self._p(x, y), max(1, int(r * self.s)), color, thickness,
                   cv2.LINE_AA)

    def target(self, x, y, progress: float = 0.0):
        """Fixation target that shrinks as ``progress`` goes 0 -> 1."""
        self.dot(x, y, 26 - 16 * min(1.0, max(0.0, progress)))
        self.dot(x, y, 4, (0, 0, 255))

    def image(self, img: np.ndarray, x, y, size: float):
        """Draw a BGR/BGRA image centred at (x, y), ``size`` points wide."""
        px = max(2, int(size * self.s))
        im = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
        cx, cy = self._p(x, y)
        x0, y0 = cx - px // 2, cy - px // 2
        H, W = self.canvas.shape[:2]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(W, x0 + px), min(H, y0 + px)
        if x1c <= x0c or y1c <= y0c:
            return
        crop = im[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
        if crop.ndim == 3 and crop.shape[2] == 4:  # alpha blend
            a = crop[:, :, 3:4] / 255.0
            roi = self.canvas[y0c:y1c, x0c:x1c]
            roi[:] = (a * crop[:, :, :3] + (1 - a) * roi).astype(np.uint8)
        else:
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            self.canvas[y0c:y1c, x0c:x1c] = crop[:, :, :3]

    def rect(self, r: Rect, color=(50, 50, 50)):
        cv2.rectangle(self.canvas, self._p(r.x, r.y), self._p(r.x + r.w, r.y + r.h), color, 1)

    def text(self, msg: str, y_frac: float = 0.5, size: float = 1.0, color=(255, 255, 255),
             x: float | None = None):
        fs = size * 0.9
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        px = int(x * self.s) if x is not None else (self.canvas.shape[1] - tw) // 2
        cv2.putText(self.canvas, msg, (px, int(self.canvas.shape[0] * y_frac) + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)

    def eye_box(self, eyes: dict, image_size: tuple[int, int], y_frac: float = 0.72):
        """Small 'mirror' showing where the eyes are in the camera image, to
        help position the head. Mirrored so it moves like a mirror image."""
        iw, ih = image_size
        bw = 320
        bh = int(bw * ih / iw)
        cx = self.canvas.shape[1] // 2
        top = int(self.canvas.shape[0] * y_frac) - bh // 2
        x0 = cx - bw // 2
        cv2.rectangle(self.canvas, (x0, top), (x0 + bw, top + bh), (90, 90, 90), 1)
        m = 0.1  # margin zone where tracking gets unreliable
        cv2.rectangle(self.canvas, (x0 + int(m * bw), top + int(m * bh)),
                      (x0 + int((1 - m) * bw), top + int((1 - m) * bh)), (45, 45, 45), 1)
        for side, e in eyes.items():
            ex = x0 + int((1 - e.pupil_x / iw) * bw)
            ey = top + int(e.pupil_y / ih * bh)
            cv2.circle(self.canvas, (ex, ey), 7, (80, 220, 80), -1, cv2.LINE_AA)
            cv2.putText(self.canvas, side, (ex - 5, ey - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (80, 220, 80), 1, cv2.LINE_AA)

    # ------------------------------------------------------------- events
    def show(self) -> int:
        """Present the canvas; returns the key code (-1 if none).
        Raises Cancelled on Esc / q / window closed."""
        cv2.imshow(self.title, self.canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            raise Cancelled()
        try:
            if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                raise Cancelled()
        except cv2.error:
            raise Cancelled()
        return -1 if k == 255 else k

    def close(self):
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass
        for _ in range(3):
            cv2.waitKey(1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
