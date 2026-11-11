"""Operating-system access: displays, mouse, keyboard, sounds.

Backends (all via ctypes, no extra dependencies):

* macOS   - CoreGraphics
* Windows - user32 (the process is made DPI-aware so coordinates are pixels)
* Linux   - X11 (+ XRandR for multiple monitors). Under Wayland, global
            pointer/keyboard access is not permitted; the functions then
            return defaults (no displays, cursor (0, 0), no keys) instead of
            failing.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class Display:
    id: int
    x: float          # global position (points/pixels), primary display at (0, 0)
    y: float
    width: float
    height: float
    main: bool
    width_mm: float | None = None   # physical width if the OS reports it


# Open Gaze API key names, shared by all backends
API_KEYS = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "SPACE", "RETURN", "TAB", "BACKSPACE",
            "ESCAPE", "LEFT", "RIGHT", "UP", "DOWN", "COMMA", "PERIOD"]


class _Backend:
    available = False

    def displays(self) -> list[Display]:
        return []

    def mouse_position(self) -> tuple[float, float]:
        return 0.0, 0.0

    def mouse_buttons(self) -> tuple[bool, bool]:
        return False, False

    def move_mouse(self, x: float, y: float):
        pass

    def pressed_keys(self) -> set[str]:
        return set()


# ===================================================================== macOS
class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class _CGRect(ctypes.Structure):
    _fields_ = [("origin", _CGPoint), ("size", _CGSize)]


class _MacBackend(_Backend):
    # kVK_* virtual key codes
    KEYCODES = {
        0x00: "A", 0x0B: "B", 0x08: "C", 0x02: "D", 0x0E: "E", 0x03: "F", 0x05: "G", 0x04: "H",
        0x22: "I", 0x26: "J", 0x28: "K", 0x25: "L", 0x2E: "M", 0x2D: "N", 0x1F: "O", 0x23: "P",
        0x0C: "Q", 0x0F: "R", 0x01: "S", 0x11: "T", 0x20: "U", 0x09: "V", 0x0D: "W", 0x07: "X",
        0x10: "Y", 0x06: "Z",
        0x1D: "0", 0x12: "1", 0x13: "2", 0x14: "3", 0x15: "4", 0x17: "5", 0x16: "6", 0x1A: "7",
        0x1C: "8", 0x19: "9",
        0x31: "SPACE", 0x24: "RETURN", 0x30: "TAB", 0x33: "BACKSPACE", 0x35: "ESCAPE",
        0x7B: "LEFT", 0x7C: "RIGHT", 0x7D: "DOWN", 0x7E: "UP", 0x2B: "COMMA", 0x2F: "PERIOD",
    }
    MODIFIERS = {0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F}
    STATE = 0  # kCGEventSourceStateCombinedSessionState

    def __init__(self):
        cg = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("CoreGraphics")
            or "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cf = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("CoreFoundation")
            or "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        cg.CGEventCreate.restype = ctypes.c_void_p
        cg.CGEventCreate.argtypes = [ctypes.c_void_p]
        cg.CGEventGetLocation.restype = _CGPoint
        cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        cg.CGEventSourceButtonState.restype = ctypes.c_bool
        cg.CGEventSourceButtonState.argtypes = [ctypes.c_int32, ctypes.c_uint32]
        cg.CGEventSourceKeyState.restype = ctypes.c_bool
        cg.CGEventSourceKeyState.argtypes = [ctypes.c_int32, ctypes.c_uint16]
        cg.CGGetActiveDisplayList.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                              ctypes.POINTER(ctypes.c_uint32)]
        cg.CGDisplayBounds.restype = _CGRect
        cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
        cg.CGDisplayScreenSize.restype = _CGSize
        cg.CGDisplayScreenSize.argtypes = [ctypes.c_uint32]
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayPixelsWide.restype = cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
        cg.CGDisplayPixelsWide.argtypes = cg.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
        cg.CGWarpMouseCursorPosition.argtypes = [_CGPoint]
        cg.CGAssociateMouseAndMouseCursorPosition.argtypes = [ctypes.c_bool]
        self.cg, self.cf = cg, cf
        self.available = True

    def displays(self):
        ids = (ctypes.c_uint32 * 32)()
        n = ctypes.c_uint32()
        self.cg.CGGetActiveDisplayList(32, ids, ctypes.byref(n))
        main = self.cg.CGMainDisplayID()
        out = []
        for i in range(n.value):
            r = self.cg.CGDisplayBounds(ids[i])
            size = self.cg.CGDisplayScreenSize(ids[i])  # mm, from EDID (72 dpi guess without)
            px_w = self.cg.CGDisplayPixelsWide(ids[i]) or r.size.width
            px_h = self.cg.CGDisplayPixelsHigh(ids[i]) or r.size.height
            mm = plausible_width_mm(size.width, px_w, px_h, size.height, dpi_guess=72)
            out.append(Display(int(ids[i]), r.origin.x, r.origin.y, r.size.width, r.size.height,
                               ids[i] == main, mm))
        return sorted(out, key=lambda d: not d.main)

    def mouse_position(self):
        ev = self.cg.CGEventCreate(None)
        p = self.cg.CGEventGetLocation(ev)
        self.cf.CFRelease(ev)
        return p.x, p.y

    def mouse_buttons(self):
        return (self.cg.CGEventSourceButtonState(self.STATE, 0),
                self.cg.CGEventSourceButtonState(self.STATE, 1))

    def move_mouse(self, x, y):
        self.cg.CGWarpMouseCursorPosition(_CGPoint(x, y))
        self.cg.CGAssociateMouseAndMouseCursorPosition(True)

    def pressed_keys(self):
        out = set()
        for code in range(128):
            if code not in self.MODIFIERS and self.cg.CGEventSourceKeyState(self.STATE, code):
                out.add(self.KEYCODES.get(code, "OTHER"))
        return out


# ===================================================================== Windows
def plausible_width_mm(width_mm, width_px=None, height_px=None, height_mm=None,
                       dpi_guess: float | None = None) -> float | None:
    """``width_mm`` if it looks like a real monitor size, else None.

    Rejects sizes OSes report when they don't know: 0, the EDID "aspect ratio
    instead of size" placeholders (16x9 / 16x10 cm), a size computed from an
    assumed DPI (macOS uses 72 dpi without EDID), widths outside 10 cm-5 m,
    and physical aspect ratios that don't match the pixels."""
    if not width_mm:
        return None
    w = float(width_mm)
    if not 100 <= w <= 5000:
        return None
    if height_mm and (round(w), round(float(height_mm))) in ((160, 90), (160, 100), (1600, 900)):
        return None
    if dpi_guess and width_px and abs(w - width_px / dpi_guess * 25.4) < 1.0:
        return None
    if height_mm and width_px and height_px:
        r = (w / float(height_mm)) / (width_px / height_px)
        if not 0.8 <= r <= 1.25:
            return None
    return w


def linux_edid_widths(drm_dir: str = "/sys/class/drm") -> dict[str, float]:
    """Physical widths (mm) of connected monitors from their EDID in sysfs,
    by connector name as X11 names outputs ("DP-1", "HDMI-1", "eDP-1"). Works
    without X11 (e.g. under Wayland)."""
    out: dict[str, float] = {}
    try:
        entries = sorted(os.listdir(drm_dir))
    except OSError:
        return out
    for name in entries:
        if "-" not in name:
            continue
        base = os.path.join(drm_dir, name)
        try:
            with open(os.path.join(base, "status")) as fh:
                if fh.read().strip() != "connected":
                    continue
            with open(os.path.join(base, "edid"), "rb") as fh:
                w = edid_width_mm(fh.read())
        except OSError:
            continue
        if w:
            conn = name.split("-", 1)[1]                     # card0-HDMI-A-1 -> HDMI-A-1
            out[conn] = w
            out.setdefault(conn.replace("-A-", "-"), w)     # X11: HDMI-1
    return out


def edid_width_mm(edid: bytes) -> float | None:
    """Image width (mm) from an EDID block: the first detailed timing
    descriptor's size in mm, else the basic display parameters' size in cm
    (some monitors leave the descriptor's size at 0)."""
    if len(edid) < 128 or edid[:8] != b"\x00\xff\xff\xff\xff\xff\xff\x00":
        return None
    if edid[54] or edid[55]:  # first 18-byte descriptor is a detailed timing
        mm = edid[66] | ((edid[68] & 0xF0) << 4)
        if mm:
            return float(mm)
    return float(edid[21] * 10) if edid[21] else None


class _WinBackend(_Backend):
    VK = {**{0x41 + i: c for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
          **{0x30 + i: str(i) for i in range(10)},
          0x20: "SPACE", 0x0D: "RETURN", 0x09: "TAB", 0x08: "BACKSPACE", 0x1B: "ESCAPE",
          0x25: "LEFT", 0x26: "UP", 0x27: "RIGHT", 0x28: "DOWN", 0xBC: "COMMA", 0xBE: "PERIOD"}
    # mouse buttons and modifier / lock keys are not keystrokes
    SKIP = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x10, 0x11, 0x12, 0x14, 0x5B, 0x5C, 0x90, 0x91,
            0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5}

    def __init__(self):
        from ctypes import wintypes
        self.wt = wintypes
        self.u32 = ctypes.windll.user32
        try:  # per-monitor DPI awareness: cursor/monitor coordinates in real pixels
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                self.u32.SetProcessDPIAware()
            except Exception:
                pass
        self.u32.GetAsyncKeyState.restype = ctypes.c_short
        self.available = True

    def displays(self):
        wt = self.wt

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [("cbSize", wt.DWORD), ("rcMonitor", wt.RECT), ("rcWork", wt.RECT),
                        ("dwFlags", wt.DWORD), ("szDevice", wt.WCHAR * 32)]

        out = []
        proc_t = ctypes.WINFUNCTYPE(wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT),
                                    wt.LPARAM)

        def cb(hmon, hdc, rect, data):
            mi = MONITORINFOEXW()
            mi.cbSize = ctypes.sizeof(MONITORINFOEXW)
            self.u32.GetMonitorInfoW(hmon, ctypes.byref(mi))
            r = mi.rcMonitor
            out.append(Display(int(hmon or 0), r.left, r.top, r.right - r.left, r.bottom - r.top,
                               bool(mi.dwFlags & 1), self._width_mm(mi.szDevice)))
            return True

        self.u32.EnumDisplayMonitors(None, None, proc_t(cb), 0)
        return sorted(out, key=lambda d: not d.main)

    def _width_mm(self, adapter: str) -> float | None:
        """Physical width of the monitor on display adapter output ``adapter``
        (e.g. \\\\.\\DISPLAY1), from its EDID in the registry. Windows' own
        GetDeviceCaps(HORZSIZE) is unreliable with DPI scaling."""
        wt = self.wt

        class DISPLAY_DEVICEW(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("DeviceName", wt.WCHAR * 32),
                        ("DeviceString", wt.WCHAR * 128), ("StateFlags", wt.DWORD),
                        ("DeviceID", wt.WCHAR * 128), ("DeviceKey", wt.WCHAR * 128)]

        try:
            import winreg
            i = 0
            while True:
                dd = DISPLAY_DEVICEW()
                dd.cb = ctypes.sizeof(DISPLAY_DEVICEW)
                # EDD_GET_DEVICE_INTERFACE_NAME: DeviceID = \\?\DISPLAY#<model>#<instance>#{guid}
                if not self.u32.EnumDisplayDevicesW(adapter, i, ctypes.byref(dd), 1):
                    return None
                i += 1
                if not dd.StateFlags & 1:  # DISPLAY_DEVICE_ACTIVE
                    continue
                parts = dd.DeviceID.split("#")
                if len(parts) < 3:
                    continue
                key = rf"SYSTEM\CurrentControlSet\Enum\DISPLAY\{parts[1]}\{parts[2]}\Device Parameters"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                    edid, _ = winreg.QueryValueEx(k, "EDID")
                w = plausible_width_mm(edid_width_mm(bytes(edid)))
                if w:
                    return w
        except Exception:
            return None

    def mouse_position(self):
        p = self.wt.POINT()
        self.u32.GetCursorPos(ctypes.byref(p))
        return float(p.x), float(p.y)

    def _down(self, vk):
        return bool(self.u32.GetAsyncKeyState(vk) & 0x8000)

    def mouse_buttons(self):
        return self._down(0x01), self._down(0x02)

    def move_mouse(self, x, y):
        self.u32.SetCursorPos(int(round(x)), int(round(y)))

    def pressed_keys(self):
        return {self.VK.get(vk, "OTHER") for vk in range(0x08, 0xFF)
                if vk not in self.SKIP and self._down(vk)}


# ===================================================================== Linux / X11
class _XRRMonitorInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_ulong), ("primary", ctypes.c_int), ("automatic", ctypes.c_int),
                ("noutput", ctypes.c_int), ("x", ctypes.c_int), ("y", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int), ("mwidth", ctypes.c_int),
                ("mheight", ctypes.c_int), ("outputs", ctypes.c_void_p)]


class _X11Backend(_Backend):
    KEYSYMS = {**{c.lower(): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
               **{str(i): str(i) for i in range(10)},
               "space": "SPACE", "Return": "RETURN", "KP_Enter": "RETURN", "Tab": "TAB",
               "BackSpace": "BACKSPACE", "Escape": "ESCAPE", "Left": "LEFT", "Right": "RIGHT",
               "Up": "UP", "Down": "DOWN", "comma": "COMMA", "period": "PERIOD"}
    MODIFIER_PREFIXES = ("Shift", "Control", "Alt", "Super", "Meta", "Hyper", "Caps_Lock",
                         "Num_Lock", "ISO_Level", "Mode_switch")

    def __init__(self):
        if not os.environ.get("DISPLAY"):
            raise OSError("no X display")
        x = ctypes.cdll.LoadLibrary(ctypes.util.find_library("X11") or "libX11.so.6")
        x.XInitThreads()
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        d = x.XOpenDisplay(None)
        if not d:
            raise OSError("cannot open X display")
        x.XDefaultRootWindow.restype = ctypes.c_ulong
        x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x.XQueryPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong] + [ctypes.c_void_p] * 7
        x.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        x.XkbKeycodeToKeysym.restype = ctypes.c_ulong
        x.XkbKeycodeToKeysym.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_int, ctypes.c_int]
        x.XKeysymToString.restype = ctypes.c_char_p
        x.XKeysymToString.argtypes = [ctypes.c_ulong]
        x.XWarpPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong] + \
            [ctypes.c_int] * 2 + [ctypes.c_uint] * 2 + [ctypes.c_int] * 2
        x.XFlush.argtypes = [ctypes.c_void_p]
        x.XDisplayWidth.argtypes = x.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XDisplayWidthMM.argtypes = x.XDisplayHeightMM.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XGetAtomName.restype = ctypes.c_char_p
        x.XGetAtomName.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.x, self.d = x, d
        self.root = x.XDefaultRootWindow(d)
        self.lock = threading.Lock()
        try:
            xr = ctypes.cdll.LoadLibrary(ctypes.util.find_library("Xrandr") or "libXrandr.so.2")
            xr.XRRGetMonitors.restype = ctypes.POINTER(_XRRMonitorInfo)
            xr.XRRGetMonitors.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_int)]
            xr.XRRFreeMonitors.argtypes = [ctypes.c_void_p]
            self.xr = xr
        except OSError:
            self.xr = None
        self._names: dict[int, str | None] = {}
        self.available = True

    def displays(self):
        edid = linux_edid_widths()  # fallback when X reports no/placeholder sizes
        with self.lock:
            if self.xr is not None:
                n = ctypes.c_int()
                mons = self.xr.XRRGetMonitors(self.d, self.root, 1, ctypes.byref(n))
                if mons and n.value:
                    out = []
                    for i in range(n.value):
                        m = mons[i]
                        mm = plausible_width_mm(m.mwidth, m.width, m.height, m.mheight)
                        if mm is None:
                            name = self.x.XGetAtomName(self.d, m.name) if m.name else None
                            mm = edid.get(name.decode()) if name else None
                            if mm is None and n.value == 1 and len(set(edid.values())) == 1:
                                mm = next(iter(edid.values()))
                        out.append(Display(i, m.x, m.y, m.width, m.height, bool(m.primary), mm))
                    self.xr.XRRFreeMonitors(mons)
                    if not any(d.main for d in out):
                        out[0] = Display(*[*out[0].__dict__.values()][:5], True, out[0].width_mm)
                    return sorted(out, key=lambda d: not d.main)
            w, h = self.x.XDisplayWidth(self.d, 0), self.x.XDisplayHeight(self.d, 0)
            mm = plausible_width_mm(self.x.XDisplayWidthMM(self.d, 0), w, h,
                                    self.x.XDisplayHeightMM(self.d, 0))
            if mm is None and len(set(edid.values())) == 1:
                mm = next(iter(edid.values()))
            return [Display(0, 0, 0, w, h, True, mm)]

    def _pointer(self):
        rr, cr = ctypes.c_ulong(), ctypes.c_ulong()
        rx, ry, wx, wy = (ctypes.c_int() for _ in range(4))
        mask = ctypes.c_uint()
        with self.lock:
            self.x.XQueryPointer(self.d, self.root, ctypes.byref(rr), ctypes.byref(cr),
                                 ctypes.byref(rx), ctypes.byref(ry), ctypes.byref(wx),
                                 ctypes.byref(wy), ctypes.byref(mask))
        return rx.value, ry.value, mask.value

    def mouse_position(self):
        x, y, _ = self._pointer()
        return float(x), float(y)

    def mouse_buttons(self):
        _, _, m = self._pointer()
        return bool(m & (1 << 8)), bool(m & (1 << 10))  # Button1Mask, Button3Mask

    def move_mouse(self, x, y):
        with self.lock:
            self.x.XWarpPointer(self.d, 0, self.root, 0, 0, 0, 0, int(round(x)), int(round(y)))
            self.x.XFlush(self.d)

    def _name(self, keycode: int) -> str | None:
        if keycode not in self._names:
            ks = self.x.XkbKeycodeToKeysym(self.d, keycode, 0, 0)
            s = self.x.XKeysymToString(ks) if ks else None
            s = s.decode() if s else None
            if s is None or s.startswith(self.MODIFIER_PREFIXES):
                self._names[keycode] = None
            else:
                self._names[keycode] = self.KEYSYMS.get(s, "OTHER")
        return self._names[keycode]

    def pressed_keys(self):
        buf = ctypes.create_string_buffer(32)
        with self.lock:
            self.x.XQueryKeymap(self.d, buf)
            raw = buf.raw
            out = set()
            for byte_i, byte in enumerate(raw):
                if not byte:
                    continue
                for bit in range(8):
                    if byte & (1 << bit):
                        name = self._name(byte_i * 8 + bit)
                        if name:
                            out.add(name)
        return out


def _make_backend() -> _Backend:
    try:
        if sys.platform == "darwin":
            return _MacBackend()
        if sys.platform == "win32":
            return _WinBackend()
        if sys.platform.startswith("linux"):
            return _X11Backend()
    except Exception:
        pass
    return _Backend()


_backend: _Backend | None = None


def backend() -> _Backend:
    """The OS backend (created on first use)."""
    global _backend
    if _backend is None:
        _backend = _make_backend()
    return _backend


def displays() -> list[Display]:
    """Active displays, primary/main display first ([] if unavailable)."""
    return backend().displays()


def mouse_position() -> tuple[float, float]:
    """Global cursor position."""
    return backend().mouse_position()


def mouse_buttons() -> tuple[bool, bool]:
    """(left, right) button pressed."""
    return backend().mouse_buttons()


def move_mouse(x: float, y: float):
    """Move the cursor to a global position."""
    backend().move_mouse(x, y)


def pressed_keys() -> set[str]:
    """API names of the pressed keys; unnamed keys are reported as "OTHER"."""
    return backend().pressed_keys()


def play_sound(path: str | None = None):
    """Short non-blocking tone (or the given sound file)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["afplay", path or "/System/Library/Sounds/Tink.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            import winsound
            if path and path.lower().endswith(".wav"):
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.MessageBeep()
        else:
            for player in (["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
                if path and _which(player[0]):
                    subprocess.Popen(player + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def _which(cmd):
    from shutil import which
    return which(cmd)


class InputMonitor:
    """Polls mouse and keyboard once per frame and reports transitions in
    the Open Gaze API's terms.

    cursor state: 0 idle, 1 left down, 2 right down, 3 left up, 4 right up
    key state:    0 idle (key " "), 1 down, 2 up
    One transition is reported per poll; simultaneous ones are queued.
    """

    def __init__(self, keyboard: bool = True):
        self.keyboard = keyboard
        self._buttons = (False, False)
        self._keys: set[str] = set()
        self._cursor_events: list[int] = []
        self._key_events: list[tuple[str, int]] = []

    @property
    def available(self) -> bool:
        return backend().available

    def poll(self) -> tuple[float, float, int, str, int]:
        """-> (cursor_x, cursor_y, cursor_state, key, key_state); global coordinates."""
        x, y = mouse_position()
        left, right = mouse_buttons()
        pl, pr = self._buttons
        if left and not pl:
            self._cursor_events.append(1)
        if right and not pr:
            self._cursor_events.append(2)
        if pl and not left:
            self._cursor_events.append(3)
        if pr and not right:
            self._cursor_events.append(4)
        self._buttons = (left, right)
        if self.keyboard:
            keys = pressed_keys()
            for k in sorted(keys - self._keys):
                self._key_events.append((k, 1))
            for k in sorted(self._keys - keys):
                self._key_events.append((k, 2))
            self._keys = keys
        cs = self._cursor_events.pop(0) if self._cursor_events else 0
        kb, kbs = self._key_events.pop(0) if self._key_events else (" ", 0)
        return x, y, cs, kb, kbs
