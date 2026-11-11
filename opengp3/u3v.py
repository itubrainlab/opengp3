"""User-space USB3 Vision driver for the Gazepoint GP3 HD.

The GP3 HD replaced the GP3's Firefly MV with a FLIR (Point Grey) Chameleon3
USB 3 camera running at up to 150 Hz. That camera speaks the standard USB3
Vision protocol instead of the Firefly's IIDC tunnel:

* control: GenCP ReadMem/WriteMem commands on the control interface's bulk
  OUT/IN endpoints (class 0xEF, subclass 0x05, protocol 0). Bootstrap
  registers (ABRM at address 0, SBRM and SIRM) locate everything else;
* features: a GenICam XML file stored on the camera (usually zipped) maps
  feature names (Width, ExposureTime, AcquisitionStart ...) to registers,
  see ``genicam.NodeMap``;
* images: the streaming interface's bulk IN endpoint (protocol 2) sends each
  frame as a leader, the payload in transfers of a size the host configures
  in the SIRM, and a trailer.

This is the same protocol libraries such as Aravis use, implemented on top of
libusb/pyusb so no vendor SDK or kernel driver is needed.

The IR illuminators are assumed to hang off a camera GPIO line driven by the
exposure strobe, like the GP3's (strobe 2, active-low). Which line, and its
polarity, has not been verified on hardware: ``opengp3 probe-leds`` finds it
empirically and saves it to ``~/.opengp3/gp3hd.json``; the environment
variables OPENGP3_HD_IR_LINE (e.g. ``Line2``) and OPENGP3_HD_IR_INVERT (0/1)
override it.
"""
from __future__ import annotations

import ctypes
import io
import json
import os
import struct
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import usb.core
import usb.util

from .camera import FrameStream, GP3Error, _libusb_backend, prepare_device, VID, is_u3v, \
    GP3NotFoundError, _open_with_setup
from .genicam import GenICamError, NodeMap

# --- GenCP -------------------------------------------------------------------
U3V_PREFIX = 0x43563355            # "U3VC"
LEADER_MAGIC = 0x4C563355          # "U3VL"
TRAILER_MAGIC = 0x54563355         # "U3VT"
READMEM_CMD, READMEM_ACK = 0x0800, 0x0801
WRITEMEM_CMD, WRITEMEM_ACK = 0x0802, 0x0803
PENDING_ACK = 0x0805
FLAG_REQUEST_ACK = 0x4000
GENCP_STATUS = {
    0x8001: "not implemented", 0x8002: "invalid parameter", 0x8003: "invalid address",
    0x8004: "write protect", 0x8005: "bad alignment", 0x8006: "access denied", 0x8007: "busy",
    0x800B: "message timeout", 0x800E: "invalid header", 0x800F: "wrong config",
    0x8FFF: "error",
}

# --- bootstrap registers -----------------------------------------------------
ABRM_MANUFACTURER = 0x0004
ABRM_MODEL = 0x0044
ABRM_FAMILY = 0x0084
ABRM_DEVICE_VERSION = 0x00C4
ABRM_SERIAL = 0x0144
ABRM_MAX_RESPONSE_MS = 0x01CC
ABRM_MANIFEST_TABLE = 0x01D0
ABRM_SBRM = 0x01D8

SBRM_MAX_CMD_LEN = 0x14
SBRM_MAX_ACK_LEN = 0x18
SBRM_N_STREAMS = 0x1C
SBRM_SIRM = 0x20

SIRM_INFO = 0x00
SIRM_CONTROL = 0x04
SIRM_REQ_PAYLOAD = 0x08
SIRM_REQ_LEADER = 0x10
SIRM_REQ_TRAILER = 0x14
SIRM_MAX_LEADER = 0x18
SIRM_PAYLOAD_SIZE = 0x1C
SIRM_PAYLOAD_COUNT = 0x20
SIRM_FINAL1 = 0x24
SIRM_FINAL2 = 0x28
SIRM_MAX_TRAILER = 0x2C

MAX_TRANSFER = 1 << 20             # payload transfer size (as Aravis)
PIXEL_MONO8 = 0x01080001
CONFIG_PATH = Path.home() / ".opengp3" / "gp3hd.json"
# GP3 HD LEDs, measured: two banks on GPIO Line2 and Line3, driven by the
# exposure strobe, not inverted. Line2 is the bank next to the lens: it lights
# the retina (bright pupil). Line3 is off axis: dark pupil and a glint pair.
#
# LED modes:
#   "gazepoint" (default): Gazepoint Control's illumination, read from the
#       camera after a Gazepoint Control session and reproduced register for
#       register: a 3-frame cycle of one dark-pupil frame (Line3) and two
#       bright-pupil frames (Line2), 6.6 ms pulses. Detection uses the
#       bright/dark-pupil method (opengp3.brightdark).
#   "dark": Line3 only, every frame, for the plain dark-pupil detector.
LED_MODES = ("gazepoint", "dark")
DEFAULT_IR_MODE = "gazepoint"
DEFAULT_IR_LINES, DEFAULT_IR_INVERT = ("Line3",), False   # "dark" mode
BRIGHT_PUPIL_LINE, DARK_PUPIL_LINE = "Line2", "Line3"
# Point Grey strobe pattern (IIDC registers, not in the GenICam file): a
# pattern of N frames (GPIO_STRPAT_CTRL) and per-pin GPIO control words and
# frame masks. Gazepoint Control's values: 3-frame pattern, all masks 0x9F00
# (pattern frame 0 only); pin 3's control word makes Line3 follow its mask,
# pin 2's makes Line2 fire on the other two frames.
PGR_SHUTTER, PGR_GAIN = 0x81C, 0x820   # IIDC feature control registers
PGR_STRPAT_CTRL = 0x110C
PGR_GPIO_CTRL = 0x1110        # + 0x10 * pin
PGR_STRPAT_MASK = 0x1118      # + 0x10 * pin
PGR_STRPAT_ALL = 0x8000FFFF   # presence + fire on every frame of the pattern
GAZEPOINT_STROBE_REGS = {
    PGR_STRPAT_CTRL: 0x80000300,
    PGR_GPIO_CTRL + 0x20: 0x80080000, PGR_STRPAT_MASK + 0x20: 0x80009F00,   # pin 2: Line2
    PGR_GPIO_CTRL + 0x30: 0x80080001, PGR_STRPAT_MASK + 0x30: 0x80009F00,   # pin 3: Line3
}
GAZEPOINT_STROBE_US = {"Line2": 6695, "Line3": 6492}   # Gazepoint's strobe durations
#: exposure (= LED pulse, us) and gain (dB) per LED mode.
#: "gazepoint": Gazepoint Control's exposure, so the IR output is the same as
#: with Gazepoint Control; its gain varied (11-14 dB) between sessions.
#: "dark": Line3 on every frame at 3 ms is a 45 % duty cycle, below the 66 %
#: Gazepoint runs Line2 at; 2.2 ms was too dark, 18 dB (maximum) detected best.
MODE_SETTINGS = {"gazepoint": (6600.0, 12.0), "dark": (3000.0, 18.0)}
DEFAULT_EXPOSURE_US, DEFAULT_GAIN_DB = MODE_SETTINGS["dark"]


class U3VError(GP3Error):
    """USB3 Vision protocol error."""


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a if a > 1 else n


# ----------------------------------------------------------------- transport
class UsbTransport:
    """The camera's USB3 Vision endpoints (control OUT/IN, stream IN)."""

    def __init__(self, dev):
        self.dev = dev
        cfg = dev.get_active_configuration()
        self.ctrl_intf = self.stream_intf = None
        self.ctrl_out = self.ctrl_in = self.stream_in = None
        for intf in cfg:
            if intf.bInterfaceClass != 0xEF or intf.bInterfaceSubClass != 0x05:
                continue
            eps = list(intf)
            ins = [e.bEndpointAddress for e in eps if e.bEndpointAddress & 0x80]
            outs = [e.bEndpointAddress for e in eps if not e.bEndpointAddress & 0x80]
            if intf.bInterfaceProtocol == 0 and self.ctrl_intf is None:
                self.ctrl_intf, self.ctrl_in, self.ctrl_out = intf.bInterfaceNumber, ins[0], outs[0]
            elif intf.bInterfaceProtocol == 2 and self.stream_intf is None:
                self.stream_intf, self.stream_in = intf.bInterfaceNumber, ins[0]
        if self.ctrl_intf is None or self.stream_intf is None:
            raise U3VError("not a USB3 Vision camera with a streaming interface")
        self._claimed = []
        for n in (self.ctrl_intf, self.stream_intf):
            usb.util.claim_interface(dev, n)
            self._claimed.append(n)

    def ctrl_write(self, data: bytes, timeout_ms: int):
        self.dev.write(self.ctrl_out, data, timeout=timeout_ms)

    def ctrl_read(self, n: int, timeout_ms: int) -> bytes:
        return bytes(self.dev.read(self.ctrl_in, n, timeout=timeout_ms))

    def stream_read(self, n: int, timeout_ms: int) -> bytes:
        return bytes(self.dev.read(self.stream_in, n, timeout=timeout_ms))

    def reset_stream(self):
        try:
            self.dev.clear_halt(self.stream_in)
        except usb.core.USBError:
            pass

    def async_stream(self, sizes: list[int], depth: int = 4) -> "AsyncStream | None":
        """Queued stream reads through libusb's asynchronous API, or None if
        the backend doesn't allow it (then reads are synchronous)."""
        try:
            return AsyncStream(self.dev, self.stream_in, sizes, depth)
        except Exception:
            return None

    def close(self):
        for n in self._claimed:
            try:
                usb.util.release_interface(self.dev, n)
            except usb.core.USBError:
                pass
        usb.util.dispose_resources(self.dev)


class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class AsyncStream:
    """Keeps ``depth`` frames of stream transfers queued on the bulk IN
    endpoint, using libusb's asynchronous API through ctypes.

    With one synchronous read at a time the endpoint sits idle between
    transfers and the camera's buffer overflows: on a GP3 HD full frames
    (1280x1024) then arrive at 74.6 Hz instead of 150 Hz. A frame is a fixed
    sequence of transfers (leader, payload, final payload, trailer), so the
    queue is that sequence repeated; transfers complete in submission order
    and each is resubmitted as soon as its data has been taken.

    ``read(size)`` returns the next transfer's data and must be called with
    the sizes in the same order (a frame's sequence, repeated)."""

    LIBUSB_TRANSFER_TYPE_BULK = 2
    STATUS = {0: "completed", 1: "error", 2: "timed out", 3: "cancelled", 4: "stall",
              5: "no device", 6: "overflow"}

    def __init__(self, dev, endpoint: int, sizes: list[int], depth: int = 4):
        import usb.backend.libusb1 as lb
        backend = dev._ctx.backend
        handle = dev._ctx.handle
        if handle is None or not hasattr(backend, "lib"):
            raise U3VError("asynchronous transfers need pyusb's libusb1 backend")
        self.lib, self.ctx = backend.lib, backend.ctx
        lib = self.lib
        lib.libusb_cancel_transfer.argtypes = [ctypes.POINTER(lb._libusb_transfer)]
        lib.libusb_cancel_transfer.restype = ctypes.c_int
        lib.libusb_handle_events_timeout_completed.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_Timeval), ctypes.POINTER(ctypes.c_int)]
        lib.libusb_handle_events_timeout_completed.restype = ctypes.c_int
        self.sizes = list(sizes) * depth
        self.done = [True] * len(self.sizes)
        self.inflight: set[int] = set()
        self.pos = 0
        self.bufs = [ctypes.create_string_buffer(s) for s in self.sizes]
        self._cb = lb._libusb_transfer_cb_fn_p(self._callback)  # kept alive with self
        self._index: dict[int, int] = {}
        self.xfers = []
        for i, size in enumerate(self.sizes):
            t = lib.libusb_alloc_transfer(0)
            if not t:
                self._free()
                raise MemoryError("libusb_alloc_transfer failed")
            c = t.contents
            c.dev_handle = handle.handle
            c.flags = 0
            c.endpoint = endpoint
            c.type = self.LIBUSB_TRANSFER_TYPE_BULK
            c.timeout = 0          # no libusb timeout: read() has its own
            c.length = size
            c.buffer = ctypes.cast(self.bufs[i], ctypes.c_void_p)
            c.callback = self._cb
            c.num_iso_packets = 0
            self._index[ctypes.addressof(c)] = i
            self.xfers.append(t)

    def _callback(self, transfer):
        # runs in whichever thread handles libusb events (may be a control
        # transfer's thread): only flag the completion
        self.done[self._index[ctypes.addressof(transfer.contents)]] = True

    def _submit(self, i: int):
        self.done[i] = False
        r = self.lib.libusb_submit_transfer(self.xfers[i])
        if r:
            self.done[i] = True
            raise usb.core.USBError(f"libusb_submit_transfer failed ({r})", r)
        self.inflight.add(i)

    def submit_all(self):
        self.pos = 0
        for i in range(len(self.xfers)):
            self._submit(i)

    def _handle_events(self, seconds: float = 0.05):
        tv = _Timeval(0, int(seconds * 1e6))
        self.lib.libusb_handle_events_timeout_completed(self.ctx, ctypes.byref(tv), None)

    def read(self, size: int, timeout_ms: int, abort=None, into=None):
        """Next transfer's data as bytes, or, with ``into`` (a writable numpy
        uint8 array of at least ``size``), copied there and its length
        returned."""
        i = self.pos
        if self.sizes[i] != size:
            raise U3VError(f"stream read of {size} bytes at queue position {i} "
                           f"(expected {self.sizes[i]})")
        deadline = time.monotonic() + timeout_ms / 1000
        while not self.done[i]:
            if abort is not None and abort():
                raise usb.core.USBError("stream stopped")
            if time.monotonic() > deadline:
                raise usb.core.USBError("Operation timed out", None, 110)
            self._handle_events()
        c = self.xfers[i].contents
        self.inflight.discard(i)
        if c.status != 0:
            raise usb.core.USBError(f"stream transfer {self.STATUS.get(c.status, c.status)}")
        n = c.actual_length
        if into is not None:
            ctypes.memmove(into.ctypes.data, self.bufs[i], n)
            data = n
        else:
            data = ctypes.string_at(self.bufs[i], n)
        self.pos = (i + 1) % len(self.xfers)
        self._submit(i)
        return data

    def cancel(self, timeout_s: float = 1.0):
        """Cancel all queued transfers and wait for them to finish."""
        for i in list(self.inflight):
            self.lib.libusb_cancel_transfer(self.xfers[i])
        t_end = time.monotonic() + timeout_s
        while any(not self.done[i] for i in self.inflight) and time.monotonic() < t_end:
            self._handle_events(0.02)
        self.inflight = {i for i in self.inflight if not self.done[i]}

    def _free(self):
        for t in self.xfers:
            self.lib.libusb_free_transfer(t)
        self.xfers = []

    def close(self):
        self.cancel()
        if not self.inflight:  # never free a transfer libusb still owns
            self._free()


class GenCP:
    """GenCP register access over the control endpoints. Also the GenICam
    ``port`` of the node map (``read``/``write``)."""

    def __init__(self, transport, timeout_ms: int = 1000):
        self.t = transport
        self.timeout_ms = timeout_ms
        self.max_cmd = self.max_ack = 512   # until the SBRM is read
        self._req = 0
        self._lock = threading.Lock()

    def _transact(self, cmd: int, scd: bytes, expect: int) -> bytes:
        with self._lock:
            self._req = (self._req + 1) & 0xFFFF or 1
            req = self._req
            self.t.ctrl_write(struct.pack("<IHHHH", U3V_PREFIX, FLAG_REQUEST_ACK, cmd, len(scd),
                                          req) + scd, self.timeout_ms)
            timeout = self.timeout_ms
            while True:
                ack = self.t.ctrl_read(self.max_ack, timeout)
                if len(ack) < 12:
                    raise U3VError(f"short GenCP acknowledge ({len(ack)} bytes)")
                prefix, status, cmd_id, length, ack_req = struct.unpack_from("<IHHHH", ack)
                if prefix != U3V_PREFIX:
                    raise U3VError(f"bad GenCP prefix 0x{prefix:08x}")
                if ack_req != req:
                    continue  # stale acknowledge of an earlier, timed-out command
                if cmd_id == PENDING_ACK:
                    timeout = struct.unpack_from("<HH", ack, 12)[1] + 100
                    continue
                if status:
                    raise U3VError(f"GenCP error 0x{status:04x} "
                                   f"({GENCP_STATUS.get(status, 'unknown')})")
                if cmd_id != expect:
                    raise U3VError(f"unexpected GenCP acknowledge 0x{cmd_id:04x}")
                return ack[12:12 + length]

    def read(self, addr: int, n: int) -> bytes:
        out = bytearray()
        chunk = max(4, self.max_ack - 12)
        while len(out) < n:
            k = min(chunk, n - len(out))
            out += self._transact(READMEM_CMD, struct.pack("<QHH", addr + len(out), 0, k),
                                  READMEM_ACK)[:k]
        return bytes(out)

    def write(self, addr: int, data: bytes):
        data = bytes(data)
        chunk = max(4, self.max_cmd - 12 - 8)
        for i in range(0, len(data), chunk):
            self._transact(WRITEMEM_CMD, struct.pack("<Q", addr + i) + data[i:i + chunk],
                           WRITEMEM_ACK)

    def u32(self, addr: int) -> int:
        return struct.unpack("<I", self.read(addr, 4))[0]

    def u64(self, addr: int) -> int:
        return struct.unpack("<Q", self.read(addr, 8))[0]

    def set_u32(self, addr: int, v: int):
        self.write(addr, struct.pack("<I", v))

    def string(self, addr: int, n: int = 64) -> str:
        return self.read(addr, n).split(b"\0", 1)[0].decode("ascii", "replace").strip()


def _lines(value) -> list[str]:
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [str(s) for s in value]


def load_led_mode() -> str:
    """LED mode ("gazepoint" or "dark"): OPENGP3_HD_IR_MODE, saved config, or
    the default. A saved probe result (explicit lines) selects "dark"."""
    mode = DEFAULT_IR_MODE
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        mode = cfg.get("ir_mode", "dark" if ("ir_lines" in cfg or "ir_line" in cfg) else mode)
    except (OSError, ValueError):
        pass
    mode = os.environ.get("OPENGP3_HD_IR_MODE", mode).strip().lower()
    return mode if mode in LED_MODES else DEFAULT_IR_MODE


def load_led_config() -> tuple[list[str], bool]:
    """(lines, inverted) of the IR illuminators in "dark" mode: environment,
    saved probe result, or the default (Line3, not inverted)."""
    lines, invert = list(DEFAULT_IR_LINES), DEFAULT_IR_INVERT
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        lines = _lines(cfg.get("ir_lines", cfg.get("ir_line", lines)))
        invert = bool(cfg.get("ir_invert", invert))
    except (OSError, ValueError):
        pass
    env_lines = os.environ.get("OPENGP3_HD_IR_LINE")
    if env_lines:
        lines = _lines(env_lines)
    env_inv = os.environ.get("OPENGP3_HD_IR_INVERT")
    if env_inv is not None:
        invert = env_inv.strip().lower() in ("1", "true", "yes")
    return lines, invert


def save_led_config(lines, invert: bool, path: Path | None = None):
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # explicit lines are for the single-bank "dark" mode
    path.write_text(json.dumps({"ir_mode": "dark", "ir_lines": _lines(lines),
                                "ir_invert": invert}, indent=2))


def led_lines_from_probe(results, margin: float = 3.0) -> tuple[list[str], bool]:
    """From ``probe_leds`` results: every line that brightens the image
    clearly, with the polarity of the brightest one ([] if none does)."""
    off = next(b for line, _, b in results if line == "off")
    lit = [(line, inv, b) for line, inv, b in results if line != "off" and b > off + margin]
    if not lit:
        return [], False
    invert = lit[0][1]  # results are sorted brightest first
    lines = sorted({line for line, inv, _ in lit if inv == invert})
    return lines, invert


# -------------------------------------------------------------------- camera
class GP3HDCamera(FrameStream):
    """Direct access to the GP3 HD camera, with the same interface as
    ``GP3Camera``: video modes, features, IR LEDs and a threaded frame stream.

    Typical use::

        cam = GP3HDCamera.open()
        cam.set_full_sensor()      # whole sensor, Mono8
        cam.set_rate(150)          # or 60
        cam.set_leds(True)
        cam.start_stream()
        seq, t, img = cam.grab()   # (h, w) uint8
        cam.close()

    ``nodes`` is the camera's GenICam node map for anything else.
    """

    model = "GP3 HD"
    bus = "USB3"
    #: lens focal length in pixels, for head depth from the inter-pupillary
    #: distance (opengp3.eyemodel). Gazepoint doesn't publish the lens; its
    #: head box (35 x 22 cm at 65 cm) is the camera's field of view, which on
    #: the 6.14 x 4.92 mm sensor (4.8 um pixels) gives 2900-3025 px (horizontal
    #: / vertical). The same method gives 2614-2836 px for the GP3, whose
    #: measured 2750 px lies in the middle. So: ~2990 px (14.4 mm). A subject
    #: at ~60 cm read 56 cm with it; the rest is consistent with an IPD of
    #: ~67 mm instead of the model's 63 mm (measure it with the printed marker).
    focal_px = 2990.0

    #: GP3-style feature names -> (GenICam value, auto) features
    _FEATURES = {"gain": ("Gain", "GainAuto"), "shutter": ("ExposureTime", "ExposureAuto")}

    def __init__(self, transport, dev=None):
        self.dev = dev
        self.t = transport
        self.cp = GenCP(transport)
        self.rate = self._target_rate = 150.0
        self.bpp = 1
        self._acquiring = False
        #: queue stream transfers with libusb's asynchronous API (needed for
        #: 150 Hz at full frame); False forces one synchronous read at a time
        self.use_async = True
        self._async: AsyncStream | None = None
        self._read_bootstrap()
        self.xml = self._read_xml()
        self.nodes = NodeMap(self.xml, self.cp)
        self.ir_lines, self.ir_invert = load_led_config()
        #: "gazepoint" (alternating bright/dark-pupil banks) or "dark" (Line3)
        self.ir_mode = load_led_mode()
        self.nodes.try_set("TLParamsLocked", 0)
        self._stop_acquisition()
        self.width = int(self.nodes.get("Width"))
        self.height = int(self.nodes.get("Height"))

    # ---------------------------------------------------------------- setup
    @classmethod
    def open(cls, auto_setup: bool | None = None) -> "GP3HDCamera":
        """Open the first GP3 HD (USB3 Vision camera with VID 0x1e10)."""
        def opener():
            for dev in usb.core.find(find_all=True, idVendor=VID, backend=_libusb_backend()) or ():
                if is_u3v(dev):
                    return cls._open_device(dev)
            raise GP3NotFoundError("GP3 HD not found on USB (VID 0x1e10, USB3 Vision). "
                                   "Is it plugged in, and not in use by another program?")
        return _open_with_setup(opener, auto_setup)

    @classmethod
    def _open_device(cls, dev) -> "GP3HDCamera":
        prepare_device(dev, cls.model)
        try:
            transport = UsbTransport(dev)
        except usb.core.USBError as e:
            usb.util.dispose_resources(dev)
            raise GP3Error(f"cannot claim the GP3 HD's interfaces ({e}): is another program "
                           "using it?") from e
        try:
            return cls(transport, dev)
        except Exception:
            transport.close()
            raise

    def _read_bootstrap(self):
        cp = self.cp
        self.info_strings = {
            "manufacturer": cp.string(ABRM_MANUFACTURER), "model": cp.string(ABRM_MODEL),
            "family": cp.string(ABRM_FAMILY), "version": cp.string(ABRM_DEVICE_VERSION),
            "serial": cp.string(ABRM_SERIAL),
        }
        cp.timeout_ms = max(cp.timeout_ms, cp.u32(ABRM_MAX_RESPONSE_MS) + 100)
        sbrm = cp.u64(ABRM_SBRM)
        cp.max_cmd = cp.u32(sbrm + SBRM_MAX_CMD_LEN) or cp.max_cmd
        cp.max_ack = cp.u32(sbrm + SBRM_MAX_ACK_LEN) or cp.max_ack
        if cp.u32(sbrm + SBRM_N_STREAMS) < 1:
            raise U3VError("camera reports no stream channel")
        self.sirm = cp.u64(sbrm + SBRM_SIRM)
        self.manifest = cp.u64(ABRM_MANIFEST_TABLE)

    def _read_xml(self) -> str:
        """The camera's GenICam XML (first manifest entry, unzipped)."""
        n = self.cp.u64(self.manifest)
        if not n:
            raise U3VError("camera has no GenICam file")
        _, _, addr, size = struct.unpack("<IIQQ", self.cp.read(self.manifest + 8, 24))
        data = self.cp.read(addr, size)
        if data[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                name = next((m for m in z.namelist() if m.lower().endswith(".xml")),
                            z.namelist()[0])
                data = z.read(name)
        return data.decode("utf-8", "replace").rstrip("\0")

    def close(self):
        try:
            self.stop_stream()
            self.stop()
            self.set_leds(False)
        except Exception:
            pass
        self.t.close()

    @property
    def serial(self) -> str:
        return self.info_strings.get("serial", "")

    # --------------------------------------------------------- video modes
    def _set_roi(self, width: int, height: int, centre: bool = True):
        n = self.nodes
        self.stop()
        for f in ("BinningHorizontal", "BinningVertical", "DecimationHorizontal",
                  "DecimationVertical"):
            n.try_set(f, 1)
        n.try_set("PixelFormat", "Mono8")
        n.try_set("OffsetX", 0)
        n.try_set("OffsetY", 0)
        wmax = int(n.get("WidthMax")) if n.has("WidthMax") else int(n.range("Width")[1])
        hmax = int(n.get("HeightMax")) if n.has("HeightMax") else int(n.range("Height")[1])
        winc = int(n.range("Width")[2] or 1)
        hinc = int(n.range("Height")[2] or 1)
        w = min(wmax, width) // winc * winc
        h = min(hmax, height) // hinc * hinc
        n.set("Width", w)
        n.set("Height", h)
        if centre:
            xinc = int((n.range("OffsetX")[2] if n.has("OffsetX") else 1) or 1)
            yinc = int((n.range("OffsetY")[2] if n.has("OffsetY") else 1) or 1)
            n.try_set("OffsetX", (wmax - w) // 2 // xinc * xinc)
            n.try_set("OffsetY", (hmax - h) // 2 // yinc * yinc)
        self.width, self.height = int(n.get("Width")), int(n.get("Height"))

    def set_full_sensor(self, width: int = 1 << 16, height: int = 1 << 16):
        """The whole sensor (or a centred width x height region), Mono8,
        at the current rate (150 Hz by default)."""
        self._set_roi(width, height)
        self._apply_rate(self._target_rate)

    def set_video_mode(self, mode: int = 5, rate: int = 5):
        """GP3 compatibility: IIDC Format 0 mode 5 is a 640x480 centre crop;
        rate codes 4/5 are 30/60 Hz."""
        self._set_roi(640, 480)
        self._apply_rate({3: 15.0, 4: 30.0, 5: 60.0, 6: 120.0}.get(rate, 60.0))

    def set_rate(self, hz: float):
        """Fixed frame rate (Gazepoint runs the GP3 HD at 60 or 150 Hz), also
        used by later ``set_full_sensor()`` calls. Sets the exposure, and so
        the LED pulse, and the gain for the LED mode (see ``MODE_SETTINGS``)."""
        self._target_rate = float(hz)
        self._apply_rate(hz)

    def _apply_rate(self, hz: float):
        n = self.nodes
        period_us = 1e6 / hz
        n.try_set("AcquisitionMode", "Continuous")
        if n.has("ExposureAuto"):
            n.try_set("ExposureAuto", "Off")
        exposure_us, gain_db = MODE_SETTINGS.get(self.ir_mode, MODE_SETTINGS["dark"])
        if n.has("ExposureTime"):
            exp = min(exposure_us, 0.99 * period_us)
            lo, hi, _ = n.range("ExposureTime")
            exp = max(lo or 0, min(hi or exp, exp))
            self._abs_control("ExposureTime")
            n.try_set("ExposureTime", exp)
        if n.has("Gain"):
            n.try_set("GainAuto", "Off")
            self._abs_control("Gain")
            hi = n.range("Gain")[1]
            n.try_set("Gain", min(gain_db, hi) if hi else gain_db)
        n.try_set("AcquisitionFrameRateAuto", "Off")          # FLIR
        n.try_set("AcquisitionFrameRateEnabled", True)        # FLIR
        n.try_set("AcquisitionFrameRateEnable", True)         # SFNC
        if n.has("AcquisitionFrameRate"):
            hi = n.range("AcquisitionFrameRate")[1]
            n.try_set("AcquisitionFrameRate", min(hz, hi) if hi else hz)
            try:
                hz = float(n.get("AcquisitionFrameRate"))
            except Exception:
                pass
        self.rate = float(hz)

    # ------------------------------------------------------------ features
    def feature(self, name: str) -> dict:
        """GP3-compatible feature info: "gain" (dB) or "shutter" (exposure, us)."""
        value_name, auto_name = self._FEATURES[name]
        n = self.nodes
        if not n.has(value_name):
            return {"present": False, "min": 0, "max": 0, "value": 0, "auto": False,
                    "on": False, "auto_capable": False}
        lo, hi, _ = n.range(value_name)
        auto = n.has(auto_name) and n.get(auto_name) != "Off"
        return {"present": True, "min": lo or 0, "max": hi or 0, "value": n.get(value_name),
                "auto": auto, "on": True, "auto_capable": n.has(auto_name)}

    def set_feature(self, name: str, value: float | None = None, auto: bool = False):
        """value=None + auto=True -> auto mode, else manual with value."""
        value_name, auto_name = self._FEATURES[name]
        n = self.nodes
        if auto:
            n.set(auto_name, "Continuous")
            return
        n.try_set(auto_name, "Off")
        self._abs_control(value_name)
        f = self.feature(name)
        n.set(value_name, max(f["min"], min(f["max"], float(value))))

    # -------------------------------------------------------------- IR LEDs
    def set_leds(self, on: bool = True, line=None, invert: bool | None = None):
        """IR illuminators on the strobe output line(s) ``line`` (a name, a
        list, or default: the saved probe result, see ``probe_leds``).

        Turning the default LEDs on also switches off any other line still
        strobing (left by Gazepoint Control or an earlier run): on the GP3 HD
        that is the bright-pupil bank, which spoils dark-pupil detection.
        Turning the default LEDs off switches off every strobing line."""
        n = self.nodes
        if not n.has("LineSelector"):
            return
        if line is None and self.ir_mode == "gazepoint":
            return self._set_gazepoint_leds(on)
        lines = _lines(line) if line else self.ir_lines
        invert = self.ir_invert if invert is None else invert
        if line is None:
            for other in self._strobing_lines():
                if not on or other not in lines:
                    self._set_line(other, False, invert)
        for name in lines:
            self._set_line(name, on, invert)

    @property
    def illumination(self) -> str:
        """"bright-dark" (alternating bright/dark-pupil frames) or "dark"."""
        return "bright-dark" if self.ir_mode == "gazepoint" else "dark"

    def _set_gazepoint_leds(self, on: bool):
        """Gazepoint Control's illumination (see GAZEPOINT_STROBE_REGS)."""
        both = [BRIGHT_PUPIL_LINE, DARK_PUPIL_LINE]
        for other in self._strobing_lines():
            if not on or other not in both:
                self._set_line(other, False, False)
        if not on:
            return
        n = self.nodes
        for line in both:  # GenICam line setup first: it rewrites the GPIO control words
            n.set("LineSelector", line)
            n.try_set("LineMode", "Output")
            n.try_set("LineSource", "ExposureActive")
            n.try_set("LineInverter", False)
            # Gazepoint's explicit durations: with 0 (pulse = exposure) the
            # alternation stops and only Line2 fires
            n.try_set("StrobeDuration", GAZEPOINT_STROBE_US[line])
            n.try_set("StrobeDelay", 0)
        if not n.has("CamRegBaseAddress"):
            return
        base = int(n.get("CamRegBaseAddress"))
        for off, value in GAZEPOINT_STROBE_REGS.items():
            self.cp.set_u32(base + off, value)

    def _strobing_lines(self) -> list[str]:
        """Lines currently configured as exposure-strobe outputs."""
        n, out = self.nodes, []
        if not (n.has("LineSource") and n.has("LineMode")):
            return out
        for name in n.entries("LineSelector"):
            try:
                n.set("LineSelector", name)
                if n.get("LineMode") == "Output" and n.get("LineSource") == "ExposureActive":
                    out.append(name)
            except (GenICamError, GP3Error, usb.core.USBError):
                pass
        return out

    def choose_led_lines(self, lines: list[str], invert: bool, frames: int = 60):
        """Among LED lines (from ``probe_leds``), the combination with which
        both eyes are found most often: each line alone and all together.
        Returns (best combination or None if no eyes were found, {combo: count}).
        The stream must be running and a person in front of the tracker."""
        from .detection import detect_eyes
        combos = [(line,) for line in lines] + ([tuple(lines)] if len(lines) > 1 else [])
        counts = {}
        for combo in combos:
            self.set_leds(False, line=lines, invert=invert)
            self.set_leds(True, line=list(combo), invert=invert)
            self.grab(latest=True)
            time.sleep(0.1)
            self.grab(latest=True)
            counts[combo] = sum(len(detect_eyes(self.grab(timeout_s=1.0)[2])) == 2
                                for _ in range(frames))
        self.set_leds(False, line=lines, invert=invert)
        best = max(counts, key=lambda c: (counts[c], -len(c))) if counts else None
        return (best if best and counts[best] > 0 else None), counts

    def _abs_control(self, feature: str):
        """Point Grey cameras apply ExposureTime/Gain (absolute float values)
        only while the feature's IIDC control register has its abs-control bit
        set. Gazepoint Control leaves gain in relative mode, so writing Gain
        would be silently ignored."""
        off = {"Gain": PGR_GAIN, "ExposureTime": PGR_SHUTTER}.get(feature)
        n = self.nodes
        if off is None or not n.has("CamRegBaseAddress"):
            return
        try:
            addr = int(n.get("CamRegBaseAddress")) + off
            v = self.cp.u32(addr)
            if v & 0x8000_0000 and not v & 0x4000_0000:  # present, abs off
                self.cp.set_u32(addr, v | 0x4000_0000)
        except (GP3Error, GenICamError, usb.core.USBError):
            pass

    def _strobe_every_frame(self, line: str):
        """Make the line's strobe fire on every frame of the Point Grey strobe
        pattern (see PGR_STRPAT_MASK)."""
        n = self.nodes
        if not (n.has("CamRegBaseAddress") and line.startswith("Line")):
            return
        try:
            addr = int(n.get("CamRegBaseAddress")) + PGR_STRPAT_MASK + 0x10 * int(line[4:])
            if self.cp.u32(addr) & 0x8000_0000:  # pattern supported on this pin
                self.cp.set_u32(addr, PGR_STRPAT_ALL)
        except (ValueError, GP3Error, GenICamError, usb.core.USBError):
            pass

    def _set_line(self, line: str, on: bool, invert: bool):
        n = self.nodes
        n.set("LineSelector", line)
        if on:
            n.try_set("LineMode", "Output")
            src = n.entries("LineSource") if n.has("LineSource") else []
            for s in ("ExposureActive", "Strobe", "ExposureStart"):
                if s in src:
                    n.try_set("LineSource", s)
                    break
            n.try_set("LineInverter", invert)
            n.try_set("StrobeEnabled", True)   # FLIR legacy name
            n.try_set("StrobeDuration", 0)     # pulse = exposure (Gazepoint leaves ~6.7 ms)
            n.try_set("StrobeDelay", 0)
            self._strobe_every_frame(line)
        else:
            n.try_set("StrobeEnabled", False)
            if not n.try_set("LineMode", "Input"):
                n.try_set("LineInverter", not invert)

    def probe_leds(self, frames: int = 8) -> list[tuple[str, bool, float]]:
        """Find which output lines drive the IR illuminators: for every (line,
        inverted) setting, the brightness of the *darkest* of ``frames``
        frames, brightest first (see ``led_lines_from_probe``). Scoring the
        darkest frame matters: the GP3 HD's LED board cycles over 3 frames,
        and the wrong polarity leaves one of them unlit while still raising
        the average. The stream must be running."""
        n = self.nodes
        lines = n.entries("LineSelector") if n.has("LineSelector") else []

        def brightness():
            self.grab(latest=True)
            time.sleep(0.1)
            self.grab(latest=True)
            return float(min(self.grab(timeout_s=1.0)[2].mean() for _ in range(frames)))

        for line in lines:
            self.set_leds(False, line=line)
        base = brightness()
        out = [("off", False, base)]
        for line in lines:
            for inv in (False, True):
                try:
                    self.set_leds(True, line=line, invert=inv)
                    out.append((line, inv, brightness()))
                except (GenICamError, GP3Error, usb.core.USBError, TimeoutError):
                    pass
                finally:
                    try:
                        self.set_leds(False, line=line, invert=inv)
                    except Exception:
                        pass
        return sorted(out, key=lambda r: -r[2])

    # ------------------------------------------------------------ streaming
    def _stop_acquisition(self):
        try:
            self.nodes.execute("AcquisitionStop")
        except Exception:
            pass
        try:
            self.cp.set_u32(self.sirm + SIRM_CONTROL, 0)
        except Exception:
            pass
        self.nodes.try_set("TLParamsLocked", 0)

    def start(self):
        """Lock the transport parameters, size the stream transfers and start
        acquisition."""
        if self._acquiring:
            return
        cp, n = self.cp, self.nodes
        n.try_set("TLParamsLocked", 1)
        align = 1 << (cp.u32(self.sirm + SIRM_INFO) >> 24)
        payload = cp.u64(self.sirm + SIRM_REQ_PAYLOAD)
        if not payload and n.has("PayloadSize"):
            payload = int(n.get("PayloadSize"))
        payload = payload or self.width * self.height
        self._leader = _align(max(cp.u32(self.sirm + SIRM_REQ_LEADER), 52), align)
        self._trailer = _align(max(cp.u32(self.sirm + SIRM_REQ_TRAILER), 32), align)
        size = min(_align(MAX_TRANSFER, align), _align(payload, align))
        self._count = payload // size
        self._final1 = _align(payload % size, align)
        self._size = size
        cp.set_u32(self.sirm + SIRM_MAX_LEADER, self._leader)
        cp.set_u32(self.sirm + SIRM_PAYLOAD_SIZE, size)
        cp.set_u32(self.sirm + SIRM_PAYLOAD_COUNT, self._count)
        cp.set_u32(self.sirm + SIRM_FINAL1, self._final1)
        cp.set_u32(self.sirm + SIRM_FINAL2, 0)
        cp.set_u32(self.sirm + SIRM_MAX_TRAILER, self._trailer)
        self.t.reset_stream()
        self._async = None
        if self.use_async and hasattr(self.t, "async_stream"):
            sizes = [self._leader] + [size] * self._count + ([self._final1] if self._final1 else [])
            self._async = self.t.async_stream(sizes + [self._trailer])
            if self._async is not None:
                self._async.submit_all()  # transfers wait before the first frame starts
        cp.set_u32(self.sirm + SIRM_CONTROL, 1)
        n.execute("AcquisitionStart")
        self._acquiring = True

    def stop(self):
        if self._acquiring:
            self._acquiring = False
            self._stop_acquisition()
            if self._async is not None:
                self._async.close()
                self._async = None
            self.t.reset_stream()

    def stop_stream(self):
        # the reader thread must leave the queued transfers before they are freed
        if getattr(self, "_thread", None):
            self._stop_evt.set()
            self._thread.join(timeout=2)
            self._thread = None
        self.stop()

    def _stopping(self) -> bool:
        evt = getattr(self, "_stop_evt", None)
        return evt is not None and evt.is_set()

    def _chunk(self, size: int, timeout_ms: int) -> bytes:
        if self._async is not None:
            return self._async.read(size, timeout_ms, abort=self._stopping)
        return self.t.stream_read(size, timeout_ms)

    def _read_frame(self, timeout_ms: int = 1000) -> np.ndarray:
        """Leader, payload transfers, trailer -> (h, w) uint8 image. With
        queued transfers, a broken frame restarts acquisition so the queue
        lines up with frame boundaries again; synchronous reads skip to the
        next leader."""
        try:
            return self._read_one_frame(timeout_ms)
        except (U3VError, usb.core.USBError):
            if self._async is not None and self._acquiring and not self._stopping():
                self.stop()
                self.start()
            raise

    def _read_one_frame(self, timeout_ms: int) -> np.ndarray:
        while True:
            leader = self._chunk(self._leader, timeout_ms)
            if len(leader) >= 4 and struct.unpack_from("<I", leader)[0] == LEADER_MAGIC:
                break
            if self._async is not None:
                raise U3VError("lost frame sync")
        if len(leader) < 50:
            raise U3VError(f"short leader ({len(leader)} bytes)")
        _, _, block, _, ptype = struct.unpack_from("<HHQHH", leader, 4)
        self._frame_id = block + 1  # camera frame counter: gaps are lost frames
        _ts, pix, w, h = struct.unpack_from("<QIII", leader, 20)
        pad_x = struct.unpack_from("<H", leader, 48)[0]
        # payload goes straight into the frame array: one copy per byte (at
        # 150 Hz x 1.3 MB, extra copies cost the detection thread real time)
        frame = np.empty(self._count * self._size + self._final1, np.uint8)
        got = 0
        for k in range(self._count + (1 if self._final1 else 0)):
            want = self._size if k < self._count else self._final1
            dest = frame[got:got + want]
            if self._async is not None:
                n = self._async.read(want, timeout_ms, abort=self._stopping, into=dest)
            else:
                chunk = self.t.stream_read(want, timeout_ms)
                n = len(chunk)
                dest[:n] = np.frombuffer(chunk, np.uint8)
            if 4 <= n <= self._trailer and \
                    struct.unpack_from("<I", dest[:4].tobytes())[0] == TRAILER_MAGIC:
                raise U3VError("frame ended early")
            got += n
        trailer = self._chunk(self._trailer, timeout_ms)
        if len(trailer) < 20 or struct.unpack_from("<I", trailer)[0] != TRAILER_MAGIC:
            raise U3VError("missing frame trailer")
        status = struct.unpack_from("<H", trailer, 16)[0]
        if status:
            raise U3VError(f"frame status 0x{status:04x}")
        if ptype != 1 or pix != PIXEL_MONO8:
            raise U3VError(f"unexpected payload type {ptype} / pixel format 0x{pix:08x}")
        stride = w + pad_x
        if got < stride * h:
            raise U3VError(f"incomplete frame ({got} of {stride * h} bytes)")
        img = frame[:stride * h].reshape(h, stride)
        return img[:, :w] if pad_x else img

    # ----------------------------------------------------------------- info
    def info(self) -> dict:
        d = {"usb": f"{self.dev.idVendor:04x}:{self.dev.idProduct:04x} GP3 HD (USB3 Vision)"
                    if self.dev is not None else "GP3 HD (USB3 Vision)"}
        from . import camera
        d["usb_driver"] = camera.usb_driver
        d.update(self.info_strings)
        d["genicam"] = f"{self.nodes.vendor} {self.nodes.model}".strip()
        d["ir_leds"] = ({"mode": "gazepoint", "dark_pupil": DARK_PUPIL_LINE,
                         "bright_pupil": BRIGHT_PUPIL_LINE} if self.ir_mode == "gazepoint" else
                        {"mode": "dark", "lines": self.ir_lines, "inverted": self.ir_invert})
        feats = {}
        for f in ("Width", "Height", "WidthMax", "HeightMax", "OffsetX", "OffsetY", "PixelFormat",
                  "AcquisitionFrameRate", "ExposureAuto", "ExposureTime", "GainAuto", "Gain",
                  "DeviceTemperature"):
            if self.nodes.has(f):
                try:
                    feats[f] = self.nodes.get(f)
                except Exception as e:
                    feats[f] = f"error: {e}"
        d["features"] = feats
        if self.nodes.has("LineSelector"):
            d["lines"] = self.nodes.entries("LineSelector")
        return d

    def __enter__(self) -> "GP3HDCamera":
        return self

    def __exit__(self, *exc):
        self.close()
