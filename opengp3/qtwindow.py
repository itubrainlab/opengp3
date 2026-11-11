"""Fullscreen stimulus window in Qt, for the Qt control app.

Same interface and drawing as ``display.StimulusWindow`` (targets, text and
images are drawn on a numpy canvas with OpenCV's drawing functions, which
need no OpenCV GUI); only the window is Qt. The Qt app can't use OpenCV
windows: on Linux, opencv-python ships its own Qt 5, which conflicts with
PySide6's Qt 6 in one process.

``show()`` presents the canvas and processes Qt events, so the calibration and
validation loops (which call it repeatedly) keep the whole app responsive.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from .display import Cancelled, StimulusWindow
from .screen import Screen


def qscreen_for(screen: Screen):
    """The QScreen showing ``screen``. OpenGP3 screens are in OS units (macOS
    points, Windows/X11 pixels); Qt uses device-independent pixels, so both
    the logical and the physical geometry are compared."""
    best, best_score = None, None
    for qs in QGuiApplication.screens():
        g, r = qs.geometry(), qs.devicePixelRatio()
        score = min(abs(g.x() - screen.x) + abs(g.y() - screen.y)
                    + abs(g.width() - screen.width) + abs(g.height() - screen.height),
                    abs(g.x() * r - screen.x) + abs(g.y() * r - screen.y)
                    + abs(g.width() * r - screen.width) + abs(g.height() * r - screen.height))
        if best_score is None or score < best_score:
            best, best_score = qs, score
    return best or QGuiApplication.primaryScreen()


class _Canvas(QWidget):
    KEYS = {Qt.Key_Escape: 27, Qt.Key_Space: 32, Qt.Key_Return: 13, Qt.Key_Enter: 13}

    def __init__(self, title: str):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setCursor(Qt.BlankCursor)
        self.image: QImage | None = None
        self.keys: deque[int] = deque()
        self.closed = False

    def paintEvent(self, _):
        p = QPainter(self)
        if self.image is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(self.rect(), self.image)
        else:
            p.fillRect(self.rect(), Qt.black)

    def keyPressEvent(self, e):
        k = self.KEYS.get(e.key())
        if k is None and e.text():
            k = ord(e.text()[0])
        if k is not None:
            self.keys.append(k)

    def closeEvent(self, e):
        self.closed = True
        super().closeEvent(e)


class QtStimulusWindow(StimulusWindow):
    """``StimulusWindow`` shown as a frameless fullscreen Qt window on the
    QScreen that matches ``screen``."""

    def __init__(self, screen: Screen, title: str = "OpenGP3", max_canvas_width: int = 1720):
        if QApplication.instance() is None:
            raise RuntimeError("QtStimulusWindow needs a running QApplication")
        self.screen = screen
        self.s = min(1.0, max_canvas_width / screen.width)
        self.title = title
        self.canvas = None
        self.clear()
        self.qscreen = qscreen_for(screen)
        self.w = _Canvas(title)
        self.w.setGeometry(self.qscreen.geometry())
        self.w.create()
        self.w.windowHandle().setScreen(self.qscreen)
        self.w.showFullScreen()
        self.w.raise_()
        self.w.activateWindow()
        self._ensure_fullscreen()

    def _ensure_fullscreen(self, timeout: float = 4.0):
        """Wait until the window covers its whole screen: targets drawn in a
        window that doesn't would land somewhere else than their coordinates
        say and silently corrupt calibration."""
        t_end = time.time() + timeout
        g = self.qscreen.geometry()
        while time.time() < t_end:
            self.show()
            wg = self.w.geometry()
            if (self.w.isFullScreen() and abs(wg.width() - g.width()) <= 2
                    and abs(wg.height() - g.height()) <= 2):
                self.geometry = (wg.x(), wg.y(), wg.width(), wg.height())
                return
            time.sleep(0.02)
        raise RuntimeError(f"stimulus window did not become fullscreen on {self.qscreen.name()} "
                           f"(window {wg.width()}x{wg.height()}, screen {g.width()}x{g.height()})")

    def show(self) -> int:
        """Present the canvas; returns the key pressed (-1 if none). Raises
        Cancelled on Esc / q / window closed."""
        rgb = np.ascontiguousarray(self.canvas[:, :, ::-1])
        h, w = rgb.shape[:2]
        self.w.image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.w.update()
        QApplication.processEvents()
        if self.w.closed:
            raise Cancelled()
        k = self.w.keys.popleft() if self.w.keys else -1
        if k in (27, ord("q")):
            raise Cancelled()
        return k

    def close(self):
        if not self.w.closed:
            self.w.closed = True
            self.w.close()
        QApplication.processEvents()
