"""Camera sources the tracker can use: a connected Gazepoint eye tracker
(GP3 / GP3 HD), webcams (low-accuracy fallback, needs opengp3[webcam]) and a
simulated GP3. Used by the Control app's camera selector."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSource:
    kind: str                 # "gazepoint", "webcam" or "simulated"
    label: str
    index: int = 0            # webcam index (OpenCV numbering)
    available: bool = True
    note: str = ""            # why it is unavailable, or a caveat

    @property
    def key(self) -> tuple[str, int]:
        return self.kind, self.index


def gazepoint_devices() -> list[str]:
    """Models of the Gazepoint trackers on USB ("GP3", "GP3 HD")."""
    try:
        import usb.core
        from .camera import PIDS, VID, _libusb_backend, is_u3v
        devs = list(usb.core.find(find_all=True, idVendor=VID, backend=_libusb_backend()) or ())
    except Exception:
        return []
    out = []
    for d in devs:
        try:
            if is_u3v(d):
                out.append("GP3 HD")
            elif d.idProduct in PIDS:
                out.append("GP3")
        except Exception:
            continue
    return out


def webcam_names() -> list[str]:
    """Names of the webcams the OS reports. On Linux they are in OpenCV's
    index order (/dev/videoN); on macOS the order can differ from OpenCV's
    numbering, so there they are only a list of what is connected."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                                 capture_output=True, text=True, timeout=10).stdout
            cams = json.loads(out).get("SPCameraDataType", [])
            return [c.get("_name", f"Camera {i}") for i, c in enumerate(cams)]
        if sys.platform.startswith("linux"):
            names = []
            for dev in sorted(glob.glob("/sys/class/video4linux/video*"),
                              key=lambda p: int(p.rsplit("video", 1)[1] or 0)):
                try:  # capture nodes only (each camera also has metadata nodes)
                    if open(os.path.join(dev, "index")).read().strip() != "0":
                        continue
                    names.append(open(os.path.join(dev, "name")).read().strip())
                except OSError:
                    continue
            return names
        if sys.platform == "win32":  # no names without extra libraries: probe a few indices
            import cv2
            names = []
            for i in range(4):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                ok = cap.isOpened()
                cap.release()
                if not ok:
                    break
                names.append(f"Camera {i}")
            return names
    except Exception:
        pass
    return []


def webcam_supported() -> bool:
    try:
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False


def list_sources() -> list[CameraSource]:
    """Everything the selector can offer, best first."""
    out = [CameraSource("gazepoint", f"Gazepoint {m}", i) for i, m in enumerate(gazepoint_devices())]
    ok = webcam_supported()
    note = "" if ok else 'needs: pip install "opengp3[webcam]"'
    names = webcam_names()
    named = not sys.platform == "darwin"   # macOS: names don't map onto OpenCV indices
    for i, name in enumerate(names):
        label = f"Webcam {i}: {name}" if named else f"Webcam {i}"
        out.append(CameraSource("webcam", label + " (low accuracy)", i, ok, note))
    out.append(CameraSource("simulated", "Simulated GP3 (no hardware)"))
    return out


def webcam_tooltip() -> str:
    """Connected webcams by name, for the selector's tooltip."""
    names = webcam_names()
    if not names:
        return "No webcams found."
    text = "Webcams on this computer: " + ", ".join(names) + "."
    if sys.platform == "darwin":
        text += (" macOS may number them differently, so the list is not in camera order: "
                 "the tracker display shows which camera is active.")
    return text + " Webcam eye tracking is far less accurate than a Gazepoint eye tracker."


def source_of(camera) -> tuple[str, int]:
    """The (kind, index) key of an open camera object."""
    if getattr(camera, "simulated", False):
        return "simulated", 0
    if getattr(camera, "webcam", False):
        return "webcam", getattr(camera, "index", 0)
    return "gazepoint", 0


def open_source(src: CameraSource):
    """Open the camera for a source (raises GP3Error subclasses on failure)."""
    if src.kind == "simulated":
        from .simulate import SimulatedCamera
        return SimulatedCamera()
    if src.kind == "webcam":
        from .webcam import WebcamCamera, _require_mediapipe, model_path
        _require_mediapipe()
        model_path()
        return WebcamCamera(src.index)
    from .camera import open_camera
    return open_camera(webcam_fallback=False)
