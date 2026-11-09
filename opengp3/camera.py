"""Low-level user-space driver for the Gazepoint GP3 camera.

The GP3 is a Point Grey (FLIR) Firefly MV USB 2.0 camera (VID 0x1e10,
PID 0x2001, "USB 2.0 Digital Imaging Camera") behind an IR filter, with IR
LEDs wired to the camera's strobe output. It speaks IIDC/DCAM tunnelled over
USB, the same mapping libdc1394 uses for its USB backend:

* register read  -> control transfer bmRequestType=0xC0
* register write -> control transfer bmRequestType=0x40
  (bRequest = high 16 bits of the 48-bit 1394 address, 0xffff -> 0x7f;
  wValue/wIndex = address bits 0..15 / 16..31; little-endian quadlets)
* image data     -> bulk IN endpoint 0x81, a continuous stream without frame
  markers (frames stay aligned because each read is exactly one frame)

So it can be driven from macOS user space via libusb, with no kernel driver.
"""
from __future__ import annotations

import os
import queue
import struct
import threading
import time

import numpy as np
import usb.core
import usb.util


class GP3Error(RuntimeError):
    """Base class for OpenGP3 hardware errors."""


class GP3NotFoundError(GP3Error):
    """No GP3 / Firefly MV camera on the USB bus."""


class LibUSBNotFoundError(GP3Error):
    """libusb-1.0 is not installed (brew install libusb)."""

VID = 0x1E10
PIDS = {
    0x2000: "Firefly MV Color",
    0x2001: "Firefly MV Mono (Gazepoint GP3)",
    0x2002: "Firefly MV HiRes Color",
    0x2003: "Firefly MV HiRes Mono",
    0x2004: "Chameleon Color",
    0x2005: "Chameleon Mono",
}

CONFIG_ROM_BASE = 0xFFFF_F000_0000
DEFAULT_CMD_BASE = 0xFFFF_F0F0_0000
BULK_EP = 0x81

# --- IIDC register offsets (relative to command register base) -------------
INITIALIZE = 0x000
V_FORMAT_INQ = 0x100
V_MODE_INQ_0 = 0x180  # +4*format
V_RATE_INQ_0 = 0x200  # +0x20*format + 4*mode
FEATURE_HI_INQ = 0x404
FEATURE_INQ_BASE = 0x500
CUR_V_FRM_RATE = 0x600
CUR_V_MODE = 0x604
CUR_V_FORMAT = 0x608
ISO_CHANNEL = 0x60C
CAMERA_POWER = 0x610
ISO_EN = 0x614
FEATURE_BASE = 0x800

FEATURES = {  # name: offset (control reg = FEATURE_BASE + off)
    "brightness": 0x00, "auto_exposure": 0x04, "sharpness": 0x08,
    "gamma": 0x18, "shutter": 0x1C, "gain": 0x20, "frame_rate": 0x3C,
}

# Point Grey strobe registers (used for IR-LED sync on some setups)
PGR_STROBE_CTRL_INQ = 0x1300
PGR_STROBE_CNT = 0x1500  # +4*n

# Format 0 modes: (width, height, bytes per pixel, name)
FORMAT0_MODES = {
    0: (160, 120, 2, "YUV444"), 1: (320, 240, 2, "YUV422"),
    2: (640, 480, 1.5, "YUV411"), 3: (640, 480, 2, "YUV422"),
    4: (640, 480, 3, "RGB8"), 5: (640, 480, 1, "MONO8"),
    6: (640, 480, 2, "MONO16"),
}
FRAMERATES = {0: 1.875, 1: 3.75, 2: 7.5, 3: 15, 4: 30, 5: 60, 6: 120, 7: 240}


def _libusb_backend():
    """Locate libusb-1.0 (Homebrew/MacPorts paths aren't on the default
    dyld search path for python.org / venv interpreters)."""
    import usb.backend.libusb1 as lb
    be = lb.get_backend()
    if be is not None:
        return be
    for p in ("/opt/homebrew/lib/libusb-1.0.dylib", "/usr/local/lib/libusb-1.0.dylib",
              "/opt/local/lib/libusb-1.0.dylib"):
        if os.path.exists(p):
            be = lb.get_backend(find_library=lambda _, p=p: p)
            if be is not None:
                return be
    raise LibUSBNotFoundError("libusb-1.0 not found - install it with: brew install libusb")


def _bit(value: int, n: int) -> bool:
    """IIDC numbers bits from the MSB: bit 0 == 0x80000000."""
    return bool(value & (0x8000_0000 >> n))


class GP3Camera:
    """Direct access to the GP3 camera: registers, video modes, IR LEDs and a
    threaded 60 fps frame stream.

    Typical use::

        cam = GP3Camera.open()
        cam.set_full_sensor()      # 752x480, widest field of view
        cam.set_leds(True)         # IR illuminators
        cam.start_stream()
        seq, t, img = cam.grab()   # (480, 752) uint8
        cam.close()
    """

    def __init__(self, dev: usb.core.Device):
        self.dev = dev
        self.cmd_base = DEFAULT_CMD_BASE
        self.width, self.height, self.bpp = 640, 480, 1
        self._claimed = False

    # ---------------------------------------------------------------- setup
    @classmethod
    def open(cls) -> "GP3Camera":
        backend = _libusb_backend()
        for pid in PIDS:
            dev = usb.core.find(idVendor=VID, idProduct=pid, backend=backend)
            if dev is not None:
                break
        else:
            raise GP3NotFoundError("GP3 / Firefly MV not found on USB (VID 0x1e10). "
                                   "Is it plugged in, and not in use by another program?")
        cam = cls(dev)
        try:
            dev.set_configuration(1)
        except usb.core.USBError as e:
            # already configured / busy is fine; anything else is fatal
            if e.errno == 19:
                raise GP3NotFoundError(
                    "GP3 disappeared or is held by another process "
                    "(close other OpenGP3 programs)") from e
            if e.errno not in (16, None):
                raise
        cam._parse_config_rom()
        return cam

    def close(self):
        try:
            self.stop_stream()
            self.stop()
            self.set_leds(False)
        except Exception:
            pass
        if self._claimed:
            usb.util.release_interface(self.dev, 0)
        usb.util.dispose_resources(self.dev)

    # ------------------------------------------------------- raw registers
    @staticmethod
    def _addr_to_request(addr: int) -> int:
        hi = addr >> 32
        try:
            return {0xFFFF: 0x7F, 0xD000: 0x80, 0xD0001: 0x81}[hi]
        except KeyError:
            raise ValueError(f"unsupported high address 0x{hi:x}")

    def read_abs(self, addr: int, nquads: int = 1) -> list[int]:
        data = self.dev.ctrl_transfer(
            0xC0, self._addr_to_request(addr), addr & 0xFFFF,
            (addr >> 16) & 0xFFFF, nquads * 4, timeout=1000)
        return list(struct.unpack(f"<{len(data)//4}I", bytes(data)))

    def write_abs(self, addr: int, *quads: int):
        payload = struct.pack(f"<{len(quads)}I", *quads)
        self.dev.ctrl_transfer(
            0x40, self._addr_to_request(addr), addr & 0xFFFF,
            (addr >> 16) & 0xFFFF, payload, timeout=1000)

    def read(self, off: int) -> int:
        return self.read_abs(self.cmd_base + off)[0]

    def write(self, off: int, value: int):
        self.write_abs(self.cmd_base + off, value)

    # ------------------------------------------------------ config ROM
    def _parse_config_rom(self):
        """Walk the IEEE-1212 config ROM to find the IIDC command base."""
        rom_addr = CONFIG_ROM_BASE + 0x400
        self.rom = []
        for i in range(64):
            try:
                q = self.read_abs(rom_addr + 4 * i)
            except usb.core.USBError:
                break
            if not q:  # past the end of the ROM
                break
            self.rom.append(q[0])
        try:
            bus_info_len = (self.rom[0] >> 24) & 0xFF
            root = 1 + bus_info_len
            root_len = self.rom[root] >> 16
            unit_dir = None
            for i in range(root + 1, root + 1 + root_len):
                key, val = self.rom[i] >> 24, self.rom[i] & 0xFFFFFF
                if key == 0xD1:  # unit directory offset
                    unit_dir = i + val
            if unit_dir is None:
                return
            ud_len = self.rom[unit_dir] >> 16
            for i in range(unit_dir + 1, unit_dir + 1 + ud_len):
                key, val = self.rom[i] >> 24, self.rom[i] & 0xFFFFFF
                if key == 0xD4:  # unit dependent directory
                    dep = i + val
                    for j in range(dep + 1, dep + 1 + (self.rom[dep] >> 16)):
                        k, v = self.rom[j] >> 24, self.rom[j] & 0xFFFFFF
                        if k == 0x40:  # command_regs_base
                            self.cmd_base = CONFIG_ROM_BASE + 4 * v
                            return
        except IndexError:
            pass  # keep default 0xFFFFF0F00000

    # ------------------------------------------------------ high level
    def feature(self, name: str) -> dict:
        off = FEATURES[name]
        inq = self.read(FEATURE_INQ_BASE + off)
        cur = self.read(FEATURE_BASE + off)
        return {
            "present": _bit(inq, 0), "min": (inq >> 12) & 0xFFF,
            "max": inq & 0xFFF, "auto_capable": _bit(inq, 6),
            "on": _bit(cur, 6), "auto": _bit(cur, 7), "value": cur & 0xFFF,
        }

    def set_feature(self, name: str, value: int | None = None, auto=False):
        """value=None + auto=True -> auto mode, else manual with value."""
        reg = 0x8200_0000  # presence + ON
        if auto:
            reg |= 0x0100_0000
        else:
            f = self.feature(name)
            value = max(f["min"], min(f["max"], int(value)))
            reg |= value & 0xFFF
        self.write(FEATURE_BASE + FEATURES[name], reg)

    def set_video_mode(self, mode: int = 5, rate: int = 5):
        """Format 0, mode 5 = 640x480 MONO8; rate 5 = 60 fps (GP3 native)."""
        self.stop()
        self.write(CUR_V_FORMAT, 0 << 29)
        self.write(CUR_V_MODE, mode << 29)
        self.write(CUR_V_FRM_RATE, rate << 29)
        self.write(ISO_CHANNEL, (0 << 28) | (2 << 24))  # ch 0, S400
        w, h, bpp, _ = FORMAT0_MODES[mode]
        self.width, self.height, self.bpp = w, h, bpp

    def set_full_sensor(self, width: int = 752, height: int = 480):
        """Format 7 mode 0, MONO8: the whole 752x480 sensor (Format 0's
        640x480 is a centre crop, so this is ~17% wider field of view).
        Frame rate follows shutter (FRAME_RATE feature), 60 fps max."""
        self.stop()
        self.write(CUR_V_FORMAT, 7 << 29)
        self.write(CUR_V_MODE, 0 << 29)
        self.write(ISO_CHANNEL, (0 << 28) | (2 << 24))
        base = CONFIG_ROM_BASE + 4 * self.read(0x2E0)  # mode-0 CSR
        f7 = lambda off: base + off
        mx = self.read_abs(f7(0x000))[0]
        unit = self.read_abs(f7(0x004))[0]
        uw, uh = unit >> 16, unit & 0xFFFF
        width = min(mx >> 16, width - width % uw)
        height = min(mx & 0xFFFF, height - height % uh)
        self.write_abs(f7(0x008), 0)                      # IMAGE_POSITION left/top
        self.write_abs(f7(0x00C), (width << 16) | height)  # IMAGE_SIZE
        self.write_abs(f7(0x010), 0 << 24)                 # COLOR_CODING_ID = MONO8
        self.write_abs(f7(0x07C), 0x4000_0000)             # VALUE_SETTING: apply
        time.sleep(0.05)
        para = self.read_abs(f7(0x040))[0]                 # PACKET_PARA_INQ
        max_bpp = para & 0xFFFF
        if max_bpp:
            self.write_abs(f7(0x044), max_bpp << 16)       # BYTE_PER_PACKET
        self.width, self.height, self.bpp = width, height, 1

    def set_strobe(self, n: int = 0, on: bool = True, polarity_high=True,
                   delay: int = 0, duration: int = 0):
        """Point Grey strobe output; may be what drives the GP3 IR LEDs."""
        reg = 0x8000_0000
        if on:
            reg |= 0x0200_0000
        if polarity_high:
            reg |= 0x0100_0000
        reg |= (delay & 0xFFF) << 12 | (duration & 0xFFF)
        self.write(PGR_STROBE_CNT + 4 * n, reg)

    def set_leds(self, on: bool = True):
        """GP3 IR illuminators are wired to PGR strobe 2, active-low.
        (Found empirically: mean frame brightness 14 -> ~190 when enabled.)"""
        if on:
            self.set_strobe(2, on=True, polarity_high=False)
        else:
            self.write(PGR_STROBE_CNT + 4 * 2, 0x8000_0000)

    @property
    def frame_bytes(self) -> int:
        return int(self.width * self.height * self.bpp)

    def start(self):
        if not self._claimed:
            usb.util.claim_interface(self.dev, 0)
            self._claimed = True
        self.write(CAMERA_POWER, 0x8000_0000)
        # The bulk stream has no frame markers: frames are aligned only
        # because the first byte after ISO_EN is the start of a frame. So
        # stop, drain any stale bytes, then enable and always read exactly
        # one frame per grab().
        self.stop()
        self._drain()
        self.write(ISO_EN, 0x8000_0000)

    def stop(self):
        self.write(ISO_EN, 0)

    def _drain(self, max_s: float = 1.0):
        """Discard leftover bytes in the endpoint (stream must be stopped)."""
        t_end = time.time() + max_s
        while time.time() < t_end:
            try:
                self.dev.read(BULK_EP, 1 << 20, timeout=100)
            except usb.core.USBError:  # timeout -> pipe empty
                return

    def _read_frame(self, timeout_ms: int = 1000) -> np.ndarray:
        n = self.frame_bytes
        buf = bytearray()
        while len(buf) < n:
            buf += self.dev.read(BULK_EP, n - len(buf), timeout=timeout_ms)
        return np.frombuffer(bytes(buf), np.uint8).reshape(self.height, self.width)

    # The Firefly MV drops a frame if no bulk transfer is pending when it
    # starts sending, so a reader thread keeps a read in flight back-to-back
    # and the consumer pulls frames from a queue.
    def _reader(self):
        while not self._stop_evt.is_set():
            try:
                img = self._read_frame()
            except usb.core.USBError as e:
                if self._stop_evt.is_set():
                    break
                self._err = e
                continue
            ts = time.time()
            self._seq += 1
            try:
                self._q.put_nowait((self._seq, ts, img))
            except queue.Full:  # consumer too slow: drop oldest
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put_nowait((self._seq, ts, img))

    def start_stream(self, buffer_frames: int = 120):
        self._q = queue.Queue(maxsize=buffer_frames)
        self._stop_evt = threading.Event()
        self._seq, self._err = 0, None
        self.start()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop_stream(self):
        if getattr(self, "_thread", None):
            self._stop_evt.set()
            self.stop()
            self._thread.join(timeout=2)
            self._thread = None

    def grab(self, timeout_s: float = 2.0, latest: bool = False
             ) -> tuple[int, float, np.ndarray]:
        """Return (frame_seq, host_timestamp, (h, w) uint8 image).

        latest=False: next frame in order (every frame; use for recording,
            consumer must keep up with 60 fps or frames queue up).
        latest=True: newest frame, discarding older queued ones (use for
            live display / interactive loops slower than 60 fps, otherwise
            they run up to `buffer_frames` behind real time).
        frame_seq gaps mean frames were skipped or dropped."""
        try:
            item = self._q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(f"no frame from GP3 ({self._err})")
        if latest:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
        return item

    def info(self) -> dict:
        d = {
            "usb": f"{self.dev.idVendor:04x}:{self.dev.idProduct:04x} "
                   f"{PIDS.get(self.dev.idProduct, '?')}",
            "command_base": hex(self.cmd_base),
            "config_rom_quads": len(self.rom),
        }
        try:
            d["manufacturer"] = usb.util.get_string(self.dev, self.dev.iManufacturer)
            d["product"] = usb.util.get_string(self.dev, self.dev.iProduct)
            d["serial"] = usb.util.get_string(self.dev, self.dev.iSerialNumber)
        except Exception:
            pass
        fmt_inq = self.read(V_FORMAT_INQ)
        d["formats"] = [f for f in range(8) if _bit(fmt_inq, f)]
        if 0 in d["formats"]:
            mode_inq = self.read(V_MODE_INQ_0)
            modes = {}
            for m in range(8):
                if _bit(mode_inq, m):
                    r = self.read(V_RATE_INQ_0 + 4 * m)
                    modes[FORMAT0_MODES.get(m, (0, 0, 0, f"mode{m}"))[3]
                          + f"(m{m})"] = [FRAMERATES[i] for i in range(8) if _bit(r, i)]
            d["format0_modes"] = modes
        d["current"] = {
            "format": self.read(CUR_V_FORMAT) >> 29,
            "mode": self.read(CUR_V_MODE) >> 29,
            "rate": FRAMERATES.get(self.read(CUR_V_FRM_RATE) >> 29),
            "iso_en": bool(self.read(ISO_EN) >> 31),
        }
        d["features"] = {}
        for name in FEATURES:
            try:
                f = self.feature(name)
                if f["present"]:
                    d["features"][name] = f
            except usb.core.USBError:
                pass
        try:
            d["strobe_inq"] = hex(self.read(PGR_STROBE_CTRL_INQ))
            d["strobe0"] = hex(self.read(PGR_STROBE_CNT))
        except usb.core.USBError:
            pass
        return d

    # context manager -------------------------------------------------------
    def __enter__(self) -> "GP3Camera":
        return self

    def __exit__(self, *exc):
        self.close()

