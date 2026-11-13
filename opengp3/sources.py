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


@dataclass(frozen=True)
class Webcam:
    index: int        # OpenCV camera index
    name: str
    kind: str         # "builtin", "external", "continuity" (iPhone/iPad) or "virtual"


# order in which the automatic webcam fallback tries camera kinds
KIND_PREFERENCE = ("builtin", "external", "continuity", "virtual")


def _classify(name: str, model: str = "") -> str:
    text = f"{name} {model}".lower()
    if any(w in text for w in ("virtual", "obs ", "obs-", "camera extension", "v4l2loopback",
                               "dummy", "snap camera", "mmhmm", "manycam", "droidcam")):
        return "virtual"
    if model.startswith(("iPhone", "iPad")) or "continuity" in text or "desk view" in text:
        return "continuity"
    if any(w in text for w in ("macbook", "facetime", "imac", "built-in", "integrated")):
        return "builtin"
    return "external"


def webcams() -> list[Webcam]:
    """The webcams, in OpenCV's index order, with their real names.

    macOS: OpenCV's AVFoundation backend numbers the cameras by sorting the
    capture devices by their unique ID, so the same sort applied to the IDs
    system_profiler reports gives each index its name. Linux: index N is
    /dev/videoN. Windows: names aren't available without extra libraries."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                                 capture_output=True, text=True, timeout=10).stdout
            cams = json.loads(out).get("SPCameraDataType", [])
            if any("spcamera_unique-id" not in c for c in cams):
                return [Webcam(i, c.get("_name", f"Camera {i}"), "external")
                        for i, c in enumerate(cams)]  # can't reproduce the order
            cams = sorted(cams, key=lambda c: c["spcamera_unique-id"])
            return [Webcam(i, c.get("_name", f"Camera {i}"),
                           _classify(c.get("_name", ""), c.get("spcamera_model-id", "")))
                    for i, c in enumerate(cams)]
        if sys.platform.startswith("linux"):
            out = []
            for dev in sorted(glob.glob("/sys/class/video4linux/video*"),
                              key=lambda p: int(p.rsplit("video", 1)[1] or 0)):
                try:  # capture nodes only (each camera also has metadata nodes)
                    if open(os.path.join(dev, "index")).read().strip() != "0":
                        continue
                    name = open(os.path.join(dev, "name")).read().strip()
                    index = int(dev.rsplit("video", 1)[1])
                    out.append(Webcam(index, name, _classify(name)))
                except (OSError, ValueError):
                    continue
            return out
        if sys.platform == "win32":  # no names without extra libraries: probe a few indices
            import cv2
            out = []
            for i in range(4):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                ok = cap.isOpened()
                cap.release()
                if not ok:
                    break
                out.append(Webcam(i, f"Camera {i}", "external"))
            return out
    except Exception:
        pass
    return []


def webcam_names() -> list[str]:
    """Webcam names in OpenCV index order."""
    return [w.name for w in webcams()]


def preferred_webcam_order() -> list[int]:
    """OpenCV indices in the order the automatic fallback should try them:
    built-in cameras first, virtual cameras (e.g. OBS's placeholder image)
    last. Empty if the cameras can't be listed (then 0, 1, 2... is tried)."""
    cams = webcams()
    return [w.index for w in sorted(cams, key=lambda w: (KIND_PREFERENCE.index(w.kind), w.index))]


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
    tags = {"virtual": ", virtual", "continuity": ", iPhone/iPad"}
    for w in sorted(webcams(), key=lambda w: (KIND_PREFERENCE.index(w.kind), w.index)):
        out.append(CameraSource("webcam", f"Webcam: {w.name} (low accuracy{tags.get(w.kind, '')})",
                                w.index, ok, note))
    out.append(CameraSource("simulated", "Simulated GP3 (no hardware)"))
    return out


def webcam_tooltip() -> str:
    """Help text for the camera selector."""
    if not webcams():
        return "No webcams found."
    return ("Webcam eye tracking is far less accurate than a Gazepoint eye tracker. Virtual "
            "cameras (e.g. OBS) and phones (Continuity Camera) only work while they are "
            "streaming.")


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
        from .simulate import open_simulated
        return open_simulated()
    if src.kind == "webcam":
        from .webcam import WebcamCamera, _require_mediapipe, model_path
        _require_mediapipe()
        model_path()
        return WebcamCamera(src.index)
    from .camera import open_camera
    return open_camera(webcam_fallback=False)
