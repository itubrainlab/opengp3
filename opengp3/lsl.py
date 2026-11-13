"""Lab Streaming Layer (LSL) output for a running GazeTracker.

Two outlets (``opengp3 server`` publishes them by default):

* **OpenGP3 Gaze** (type ``Gaze``, 60 Hz; GP3 HD 150 Hz, float32): one sample per camera
  frame; channel names follow the Open Gaze API fields (see ``CHANNELS``).
  Gaze positions are fractions of the tracked screen (0..1, top-left origin).
* **OpenGP3 Markers** (type ``Markers``, irregular, string): user markers
  (``USER_DATA`` from API clients or ``GazeTracker.user_data``) and calibration
  events.

Timestamps are the frames' capture times converted to the LSL clock, so
data from other LSL sources (EEG, stimulus markers, ...) aligns with the
moment the eye image was taken, not when it was processed.

    from opengp3 import GazeTracker
    from opengp3.lsl import LSLStreamer
    with GazeTracker(calibration=cal) as gt:
        lsl = LSLStreamer(gt)        # streams until gt stops (or lsl.close())
"""
from __future__ import annotations

import time

from .tracker import GazeSample, GazeTracker

# (label, unit, description)
CHANNELS = [
    ("BPOGX", "normalized", "gaze x, mean of visible eyes"),
    ("BPOGY", "normalized", "gaze y, mean of visible eyes"),
    ("BPOGV", "bool", "gaze valid"),
    ("FPOGX", "normalized", "fixation x"),
    ("FPOGY", "normalized", "fixation y"),
    ("FPOGD", "s", "fixation duration so far"),
    ("FPOGID", "count", "fixation number"),
    ("FPOGV", "bool", "fixation valid"),
    ("LPOGX", "normalized", "left eye gaze x"),
    ("LPOGY", "normalized", "left eye gaze y"),
    ("LPOGV", "bool", "left eye gaze valid"),
    ("RPOGX", "normalized", "right eye gaze x"),
    ("RPOGY", "normalized", "right eye gaze y"),
    ("RPOGV", "bool", "right eye gaze valid"),
    ("LPMM", "mm", "left pupil diameter"),
    ("LPMMV", "bool", "left pupil valid"),
    ("RPMM", "mm", "right pupil diameter"),
    ("RPMMV", "bool", "right pupil valid"),
    ("LPCX", "normalized", "left pupil x in camera image"),
    ("LPCY", "normalized", "left pupil y in camera image"),
    ("RPCX", "normalized", "right pupil x in camera image"),
    ("RPCY", "normalized", "right pupil y in camera image"),
    ("EYEZ", "m", "eye distance from the tracker"),
    ("BKID", "count", "blink number (on the sample where a blink ended, else 0)"),
    ("CX", "normalized", "mouse cursor x"),
    ("CY", "normalized", "mouse cursor y"),
]


def _require_pylsl():
    try:
        import pylsl
        return pylsl
    except ImportError as e:
        raise RuntimeError("LSL streaming needs pylsl (pip install pylsl)") from e


class LSLStreamer:
    """Publishes a GazeTracker's samples and markers on LSL."""

    def __init__(self, tracker: GazeTracker, name: str = "OpenGP3", source_id: str | None = None):
        lsl = _require_pylsl()
        self.lsl, self.tracker = lsl, tracker
        if source_id is None:
            source_id = f"opengp3-{_serial(tracker) or 'gp3'}"
        cam = getattr(tracker, "camera", None)
        rate = float(getattr(cam, "rate", 60.0))
        info = lsl.StreamInfo(f"{name} Gaze", "Gaze", len(CHANNELS), rate, "float32", source_id)
        desc = info.desc()
        warning = getattr(tracker, "warning", None)
        if warning:  # webcam / simulated: make it unmistakable in recordings (XDF)
            desc.append_child_value("manufacturer",
                                    getattr(cam, "description", "OpenGP3 (degraded data)"))
            desc.append_child_value("warning", warning)
        else:
            desc.append_child_value("manufacturer",
                                    f"OpenGP3 (Gazepoint {getattr(cam, 'model', 'GP3')} hardware)")
        ch = desc.append_child("channels")
        for label, unit, d in CHANNELS:
            c = ch.append_child("channel")
            c.append_child_value("label", label)
            c.append_child_value("unit", unit)
            c.append_child_value("type", "Gaze")
            c.append_child_value("description", d)
        sc = tracker.screen
        scr = desc.append_child("screen")
        scr.append_child_value("width_px", str(sc.width))
        scr.append_child_value("height_px", str(sc.height))
        if sc.width_cm:
            scr.append_child_value("width_cm", f"{sc.width_cm:g}")
        scr.append_child_value("distance_cm", f"{sc.distance_cm:g}")
        self.gaze = lsl.StreamOutlet(info, chunk_size=1)
        minfo = lsl.StreamInfo(f"{name} Markers", "Markers", 1, lsl.IRREGULAR_RATE, "string",
                               source_id + "-markers")
        self.markers = lsl.StreamOutlet(minfo)
        self._last_user = ""
        self.samples = 0
        if warning:
            self.marker(f"WARNING: {warning}")
        tracker.add_callback(self._on_sample)

    def close(self):
        self.tracker.remove_callback(self._on_sample)

    def _lsl_time(self, t_capture: float) -> float:
        """Capture time (time.time clock) -> LSL clock."""
        return self.lsl.local_clock() - (time.time() - t_capture)

    def marker(self, text: str, t: float | None = None):
        """Push a marker string (timestamp: now, or capture time ``t``)."""
        ts = self._lsl_time(t) if t is not None else self.lsl.local_clock()
        self.markers.push_sample([text], ts)

    def _on_sample(self, s: GazeSample):
        self.gaze.push_sample(sample_values(s, self.tracker), self._lsl_time(s.t))
        self.samples += 1
        if s.user and s.user != self._last_user:
            self.marker(s.user, s.t)
        self._last_user = s.user


def sample_values(s: GazeSample, tracker: GazeTracker) -> list[float]:
    """One LSL gaze sample, in CHANNELS order (invalid values are 0)."""
    W, H = tracker.screen.width, tracker.screen.height
    v = s.raw_x is not None
    f = s.fixation
    out = [s.raw_x / W if v else 0.0, s.raw_y / H if v else 0.0, float(v),
           f.x / W, f.y / H, f.duration, float(f.id), float(f.valid)]
    for side in ("L", "R"):
        g = s.per_eye.get(side)
        out += [g[0] / W if g else 0.0, g[1] / H if g else 0.0, float(g is not None)]
    for side in ("L", "R"):
        m = s.metrics.get(side)
        ok = bool(m and m.valid and m.pupil_mm)
        out += [m.pupil_mm if ok else 0.0, float(ok)]
    for side in ("L", "R"):
        m = s.metrics.get(side)
        ok = bool(m and m.valid)
        out += [m.cx if ok else 0.0, m.cy if ok else 0.0]
    zs = [m.z_m for m in s.metrics.values() if m.valid and m.z_m]
    out.append(zs[0] if zs else 0.0)
    out.append(float(s.blink.id))
    if s.cursor:
        sc = tracker.screen
        out += [(s.cursor[0] - sc.x) / W, (s.cursor[1] - sc.y) / H]
    else:
        out += [0.0, 0.0]
    return out


def _serial(tracker: GazeTracker) -> str:
    cam = tracker.camera
    if isinstance(getattr(cam, "serial", None), str):  # GP3 HD: from the camera's bootstrap registers
        return cam.serial
    try:
        import usb.util
        return usb.util.get_string(cam.dev, cam.dev.iSerialNumber) if cam is not None and cam.dev else ""
    except Exception:
        return ""
