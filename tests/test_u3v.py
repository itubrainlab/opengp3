"""GP3 HD (USB3 Vision) driver against a simulated camera: GenCP register
access, the GenICam node map, stream setup, frame parsing, IR LED probing and
the full tracking pipeline - no hardware needed."""
import io
import struct
import time
import zipfile

import cv2
import numpy as np
import pytest
import usb.core

from conftest import synth_eye_image
from opengp3 import GazeTracker, Screen
from opengp3.genicam import Formula, GenICamError, NodeMap
from opengp3 import u3v
from opengp3.u3v import GP3HDCamera

SBRM, SIRM, MANIFEST, XML_ADDR = 0x1000, 0x2000, 0x3000, 0x8000
W_REG, H_REG, OFFX_REG, OFFY_REG, PIX_REG = 0x20000, 0x20004, 0x20008, 0x2000C, 0x20010
TL_REG, START_REG, STOP_REG, MODE_REG, EXP_REG = 0x20014, 0x20018, 0x2001C, 0x20020, 0x20024
EXPAUTO_REG, GAIN_REG, GAINAUTO_REG, RATE_REG, RATEEN_REG = 0x20028, 0x2002C, 0x20030, 0x20034, 0x20038
LINESEL_REG, TEMP_REG, STRUCT_REG, LINE_BASE = 0x2003C, 0x20040, 0x20044, 0x20100
PGR_BASE = 0x10000


def _int(name, reg, extra=""):
    return (f'<Integer Name="{name}"><pValue>{name}Reg</pValue>{extra}</Integer>'
            f'<IntReg Name="{name}Reg"><Address>{hex(reg)}</Address><Length>4</Length>'
            f'<AccessMode>RW</AccessMode><pPort>Device</pPort><Sign>Unsigned</Sign>'
            f'<Endianess>LittleEndian</Endianess></IntReg>')


def _enum(name, entries, pvalue):
    e = "".join(f'<EnumEntry Name="{k}"><Value>{v}</Value></EnumEntry>' for k, v in entries.items())
    return f'<Enumeration Name="{name}">{e}<pValue>{pvalue}</pValue></Enumeration>'


def _line_reg(name, extra):
    """Per-line register, address = LINE_BASE + 0x10 * LineSelector."""
    return (f'<MaskedIntReg Name="{name}"><Address>{hex(LINE_BASE)}</Address>'
            f'<IntSwissKnife Name="{name}Addr"><pVariable Name="SEL">LineSelectorReg</pVariable>'
            f'<Formula>SEL * 0x10</Formula></IntSwissKnife><Length>4</Length><AccessMode>RW</AccessMode>'
            f'<pPort>Device</pPort>{extra}<Endianess>LittleEndian</Endianess></MaskedIntReg>')


XML = f"""<?xml version="1.0" encoding="utf-8"?>
<RegisterDescription xmlns="http://www.genicam.org/GenApi/Version_1_1" ModelName="FakeChameleon3"
    VendorName="FLIR">
  <Category Name="Root"><pFeature>Width</pFeature></Category>
  {_int("Width", W_REG, "<Min>16</Min><pMax>WidthMax</pMax><Inc>16</Inc>")}
  {_int("Height", H_REG, "<Min>2</Min><pMax>HeightMax</pMax><Inc>2</Inc>")}
  <Integer Name="WidthMax"><Value>1280</Value></Integer>
  <Integer Name="HeightMax"><Value>1024</Value></Integer>
  <Integer Name="CamRegBaseAddress"><Value>0x10000</Value></Integer>
  {_int("OffsetX", OFFX_REG, "<Min>0</Min><Max>1264</Max><Inc>16</Inc>")}
  {_int("OffsetY", OFFY_REG, "<Min>0</Min><Max>1022</Max><Inc>2</Inc>")}
  {_int("PixelFormatRaw", PIX_REG)}
  {_enum("PixelFormat", {"Mono8": "0x01080001", "Mono16": "0x01100007"}, "PixelFormatRawReg")}
  {_int("TLParamsLocked", TL_REG)}
  {_int("AcqStart", START_REG)}
  {_int("AcqStop", STOP_REG)}
  <Command Name="AcquisitionStart"><pValue>AcqStartReg</pValue><CommandValue>1</CommandValue></Command>
  <Command Name="AcquisitionStop"><pValue>AcqStopReg</pValue><CommandValue>1</CommandValue></Command>
  {_int("AcqMode", MODE_REG)}
  {_enum("AcquisitionMode", {"Continuous": 0, "SingleFrame": 1}, "AcqModeReg")}
  <Float Name="ExposureTime"><pValue>ExposureConv</pValue><Min>10</Min><Max>30000</Max></Float>
  <Converter Name="ExposureConv"><pVariable Name="TICK">TickNs</pVariable>
    <FormulaTo>FROM * 1000 / TICK</FormulaTo><FormulaFrom>TO * TICK / 1000</FormulaFrom>
    <pValue>ExposureRaw</pValue><Slope>Increasing</Slope></Converter>
  <Integer Name="TickNs"><Value>500</Value></Integer>
  <IntReg Name="ExposureRaw"><Address>{hex(EXP_REG)}</Address><Length>4</Length>
    <AccessMode>RW</AccessMode><pPort>Device</pPort><Sign>Unsigned</Sign>
    <Endianess>LittleEndian</Endianess></IntReg>
  {_enum("ExposureAuto", {"Off": 0, "Once": 1, "Continuous": 2}, "ExposureAutoReg")}
  <MaskedIntReg Name="ExposureAutoReg"><Address>{hex(EXPAUTO_REG)}</Address><Length>4</Length>
    <AccessMode>RW</AccessMode><pPort>Device</pPort><LSB>31</LSB><MSB>30</MSB>
    <Endianess>BigEndian</Endianess></MaskedIntReg>
  <Float Name="Gain"><pValue>GainReg</pValue><Min>0</Min><Max>24</Max></Float>
  <FloatReg Name="GainReg"><Address>{hex(GAIN_REG)}</Address><Length>4</Length>
    <AccessMode>RW</AccessMode><pPort>Device</pPort><Endianess>BigEndian</Endianess></FloatReg>
  {_enum("GainAuto", {"Off": 0, "Continuous": 2}, "GainAutoReg")}
  <MaskedIntReg Name="GainAutoReg"><Address>{hex(GAINAUTO_REG)}</Address><Length>4</Length>
    <AccessMode>RW</AccessMode><pPort>Device</pPort><LSB>4</LSB><MSB>5</MSB>
    <Endianess>LittleEndian</Endianess></MaskedIntReg>
  <Float Name="AcquisitionFrameRate"><pValue>RateReg</pValue><Min>1</Min><Max>149.5</Max></Float>
  <FloatReg Name="RateReg"><Address>{hex(RATE_REG)}</Address><Length>4</Length>
    <AccessMode>RW</AccessMode><pPort>Device</pPort><Endianess>LittleEndian</Endianess></FloatReg>
  <Boolean Name="AcquisitionFrameRateEnable"><pValue>RateEnReg</pValue></Boolean>
  <MaskedIntReg Name="RateEnReg"><Address>{hex(RATEEN_REG)}</Address><Length>4</Length>
    <AccessMode>RW</AccessMode><pPort>Device</pPort><Bit>0</Bit>
    <Endianess>LittleEndian</Endianess></MaskedIntReg>
  {_enum("LineSelector", {"Line0": 0, "Line1": 1, "Line2": 2, "Line3": 3}, "LineSelectorReg")}
  <IntReg Name="LineSelectorReg"><Address>{hex(LINESEL_REG)}</Address><Length>4</Length>
    <AccessMode>RW</AccessMode><pPort>Device</pPort><Sign>Unsigned</Sign>
    <Endianess>LittleEndian</Endianess></IntReg>
  {_enum("LineMode", {"Input": 0, "Output": 1}, "LineModeReg")}
  {_line_reg("LineModeReg", "<Bit>0</Bit>")}
  <Boolean Name="LineInverter"><pValue>LineInvReg</pValue></Boolean>
  {_line_reg("LineInvReg", "<Bit>1</Bit>")}
  {_enum("LineSource", {"Off": 0, "ExposureActive": 5}, "LineSrcReg")}
  <MaskedIntReg Name="LineSrcReg"><Address>{hex(LINE_BASE)}</Address>
    <pIndex Offset="0x10">LineSelectorReg</pIndex><Length>4</Length><AccessMode>RW</AccessMode>
    <pPort>Device</pPort><LSB>8</LSB><MSB>15</MSB><Endianess>LittleEndian</Endianess></MaskedIntReg>
  <SwissKnife Name="DeviceTemperature"><pVariable Name="R">TempRaw</pVariable>
    <Formula>(R &gt; 0x7FF) ? (R - 0x1000) / 16.0 : R / 16.0</Formula></SwissKnife>
  <IntReg Name="TempRaw"><Address>{hex(TEMP_REG)}</Address><Length>4</Length>
    <AccessMode>RO</AccessMode><pPort>Device</pPort><Sign>Unsigned</Sign>
    <Endianess>LittleEndian</Endianess></IntReg>
  <StructReg Comment="strobe"><Address>{hex(STRUCT_REG)}</Address><Length>4</Length>
    <AccessMode>RW</AccessMode><pPort>Device</pPort><Endianess>LittleEndian</Endianess>
    <StructEntry Name="StrobeEnabledReg"><Bit>0</Bit></StructEntry>
    <StructEntry Name="StrobeDelay"><LSB>8</LSB><MSB>15</MSB></StructEntry>
  </StructReg>
  <Boolean Name="StrobeEnabled"><pValue>StrobeEnabledReg</pValue></Boolean>
  <IntSwissKnife Name="PayloadSize"><pVariable Name="W">WidthReg</pVariable>
    <pVariable Name="H">HeightReg</pVariable><Formula>W * H</Formula></IntSwissKnife>
</RegisterDescription>
"""

TIMEOUT = usb.core.USBError("Operation timed out", None, 110)


class FakeU3V:
    """Simulated GP3 HD camera behind the UsbTransport interface."""

    def __init__(self, banks=None, ir_inverted=False, pending_on_start=True):
        self.mem = bytearray(0x21000)
        self.acks = []
        # as measured on a GP3 HD: Line3 is the off-axis (dark pupil) illuminator;
        # Line2 is the bank next to the lens, which makes the pupils glow
        self.banks = banks or {3: 1.0}
        self.bright_pupil_line = 2
        self.ir_inverted = ir_inverted
        self.pending_on_start = pending_on_start
        self.acquiring = False
        self.chunks = [b"\x00garbage"]  # stale bytes before the first frame: driver must resync
        self.block = 0
        self.frames_sent = 0
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("FakeChameleon3.xml", XML)
        xml = zbuf.getvalue()
        self.mem[XML_ADDR:XML_ADDR + len(xml)] = xml
        self._s(0x0004, b"FLIR"), self._s(0x0044, b"Chameleon3 CM3-U3-13Y3M")
        self._s(0x0084, b"Chameleon3"), self._s(0x00C4, b"1.0"), self._s(0x0144, b"18123456")
        self._u32(0x01CC, 200)
        self._u64(0x01D0, MANIFEST), self._u64(0x01D8, SBRM)
        self._u32(SBRM + 0x14, 256), self._u32(SBRM + 0x18, 256)   # small: forces chunking
        self._u32(SBRM + 0x1C, 1), self._u64(SBRM + 0x20, SIRM)
        self._u32(SIRM + 0x00, 3 << 24)                             # 8-byte alignment
        self._u32(SIRM + 0x10, 52), self._u32(SIRM + 0x14, 32)
        self._u64(MANIFEST, 1)
        self.mem[MANIFEST + 8:MANIFEST + 32] = struct.pack("<IIQQ", 0x01000000, 1 << 10,
                                                           XML_ADDR, len(xml))
        self._u32(W_REG, 1280), self._u32(H_REG, 1024), self._u32(PIX_REG, 0x01080001)
        self._u32(EXP_REG, 20000)                                   # 10 ms
        self._u32(EXPAUTO_REG, 2 << 24)                             # Continuous (big-endian bits 30-31)
        self._u32(TEMP_REG, 0xFF0)                                  # -1.0 degC
        # Point Grey strobe pattern as Gazepoint Control leaves it: 3 frames, pins 1-3
        # firing on one frame only
        self._u32(PGR_BASE + 0x110C, 0x80000301)
        self._u32(PGR_BASE + 0x1118, 0x8000FFFF)
        for pin in (1, 2, 3):
            self._u32(PGR_BASE + 0x1118 + 0x10 * pin, 0x80009F00)
        self._update()
        self.image = synth_eye_image([(520, 500, 3, -2), (780, 498, 3, -2)], size=(1280, 1024))
        self.bright_image = self.image.copy()   # bright pupil: retina lit through the pupil
        for px, py in ((520, 500), (780, 498)):
            disc = np.zeros(self.image.shape, np.uint8)
            cv2.circle(disc, (px, py), 9, 1, -1)
            self.bright_image[(disc > 0) & (self.image < 230)] = 200
        # Gazepoint illumination, bright-pupil frame (on-axis bank): darker face, glowing
        # pupils, one glint per eye 2 px below the dark-frame pair's midpoint
        b = (self.image.astype(np.float32) * 0.65)
        b[self.image >= 200] = 30                      # no off-axis glint pair
        b = b.astype(np.uint8)
        for px, py in ((520, 500), (780, 498)):
            cv2.circle(b, (px, py), 9, 170, -1, cv2.LINE_AA)
            cv2.circle(b, (px - 3, py + 4), 2, 255, -1)
        self.gp_bright = b

    # memory helpers
    def _s(self, a, b):
        self.mem[a:a + len(b)] = b

    def _u32(self, a, v):
        self.mem[a:a + 4] = struct.pack("<I", v)

    def _u64(self, a, v):
        self.mem[a:a + 8] = struct.pack("<Q", v)

    def r32(self, a):
        return struct.unpack_from("<I", self.mem, a)[0]

    def _update(self):
        self._u64(SIRM + 0x08, self.r32(W_REG) * self.r32(H_REG))

    def gazepoint_active(self) -> bool:
        """Gazepoint Control's strobe registers and both LED lines strobing:
        3-frame cycle, dark-pupil (Line3) on frame 0, bright-pupil (Line2) on 1-2."""
        lines_ok = all((self.r32(LINE_BASE + 0x10 * l) & 0xFF03) == 0x0501 for l in (2, 3))
        return (lines_ok and self.r32(PGR_BASE + 0x110C) & 0xF00 == 0x300
                and self.r32(PGR_BASE + 0x1130) == 0x80080000
                and self.r32(PGR_BASE + 0x1140) == 0x80080001
                and self.r32(PGR_BASE + 0x1138) == self.r32(PGR_BASE + 0x1148) == 0x80009F00)

    def strobing(self, line) -> bool:
        """Line strobes on every frame: output, ExposureActive source, the
        LED board's polarity and an all-frames strobe-pattern mask."""
        reg = self.r32(LINE_BASE + 0x10 * line)
        every_frame = self.r32(PGR_BASE + 0x1118 + 0x10 * line) == 0x8000FFFF
        return ((reg & 1) == 1 and (reg >> 8) & 0xFF == 5 and bool(reg & 2) == self.ir_inverted
                and every_frame)

    def light(self) -> float:
        """Fraction of the IR illumination that is on (strobed banks)."""
        return min(1.0, sum(share for line, share in self.banks.items() if self.strobing(line)))

    def leds_on(self):
        return self.light() > 0.99

    # control endpoints
    def ctrl_write(self, data, timeout_ms):
        prefix, flags, cmd, length, req = struct.unpack_from("<IHHHH", data)
        assert prefix == u3v.U3V_PREFIX and flags & u3v.FLAG_REQUEST_ACK
        scd = data[12:12 + length]
        if cmd == u3v.READMEM_CMD:
            addr, _, n = struct.unpack("<QHH", scd)
            assert 12 + n <= 256, "read larger than the max acknowledge length"
            payload, ack_cmd = bytes(self.mem[addr:addr + n]), u3v.READMEM_ACK
        elif cmd == u3v.WRITEMEM_CMD:
            addr = struct.unpack_from("<Q", scd)[0]
            value = scd[8:]
            assert len(data) <= 256, "command larger than the max command length"
            self.mem[addr:addr + len(value)] = value
            self._on_write(addr)
            payload, ack_cmd = struct.pack("<HH", 0, len(value)), u3v.WRITEMEM_ACK
            if addr == START_REG and self.pending_on_start:
                self.acks.append(struct.pack("<IHHHHHH", u3v.U3V_PREFIX, 0, u3v.PENDING_ACK, 4,
                                             req, 0, 50))
        else:
            raise AssertionError(cmd)
        self.acks.append(struct.pack("<IHHHH", u3v.U3V_PREFIX, 0, ack_cmd, len(payload), req)
                         + payload)

    def ctrl_read(self, n, timeout_ms):
        return self.acks.pop(0)[:n]

    def _on_write(self, addr):
        if addr in (W_REG, H_REG):
            self._update()
        if addr == START_REG and self.r32(START_REG):
            assert self.r32(SIRM + 0x04) == 1, "stream channel must be enabled before AcquisitionStart"
            assert self.r32(TL_REG) == 1, "TLParamsLocked must be set before AcquisitionStart"
            self.acquiring, self.block = True, 0
            self._u32(START_REG, 0)
        if addr == STOP_REG and self.r32(STOP_REG):
            self.acquiring = False
            self._u32(STOP_REG, 0)

    # stream endpoint
    def _frame_chunks(self):
        w, h = self.r32(W_REG), self.r32(H_REG)
        ox, oy = self.r32(OFFX_REG), self.r32(OFFY_REG)
        if self.gazepoint_active():
            src = self.image if self.block % 3 == 0 else self.gp_bright
            img = src[oy:oy + h, ox:ox + w]
        else:
            src = self.bright_image if self.strobing(self.bright_pupil_line) else self.image
            img = src[oy:oy + h, ox:ox + w]
            light = self.light()
            if light < 0.99:  # unlit: ambient only (1/8); partly lit: proportionally dimmer
                img = (img.astype(np.float32) * (0.125 + 0.875 * light)).astype(np.uint8)
        payload = np.ascontiguousarray(img).tobytes()
        self.block += 1
        leader = struct.pack("<IHHQHHQIIIIIHH", u3v.LEADER_MAGIC, 0, 52, self.block, 0, 1,
                             time.monotonic_ns(), 0x01080001, w, h, ox, oy, 0, 0)
        size, count = self.r32(SIRM + 0x1C), self.r32(SIRM + 0x20)
        final1 = self.r32(SIRM + 0x24)
        assert size and size % 8 == 0 and final1 % 8 == 0
        assert count * size + final1 >= len(payload)
        chunks = [leader]
        for i in range(count):
            chunks.append(payload[i * size:(i + 1) * size])
        if final1:
            chunks.append(payload[count * size:])
        trailer = struct.pack("<IHHQHHQI", u3v.TRAILER_MAGIC, 0, 32, self.block, 0, 0,
                              len(payload), h)
        return chunks + [trailer]

    def stream_read(self, n, timeout_ms):
        if getattr(self, "junk", 0):  # injected stray transfer (tests resync)
            self.junk -= 1
            return b"\x00junk"
        if not self.chunks:
            if not (self.acquiring and self.r32(SIRM + 0x04)):
                time.sleep(0.01)
                raise TIMEOUT
            time.sleep(1 / 300)
            self.chunks = self._frame_chunks()
            self.frames_sent += 1
        return self.chunks.pop(0)[:n]

    def reset_stream(self):
        pass

    def close(self):
        self.closed = True


def _make_hd(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(u3v, "CONFIG_PATH", tmp_path / "gp3hd.json")
    for k in ("OPENGP3_HD_IR_LINE", "OPENGP3_HD_IR_INVERT"):
        monkeypatch.delenv(k, raising=False)
    if mode is None:
        monkeypatch.delenv("OPENGP3_HD_IR_MODE", raising=False)
    else:
        monkeypatch.setenv("OPENGP3_HD_IR_MODE", mode)
    fake = FakeU3V()
    return GP3HDCamera(fake), fake


@pytest.fixture
def hd(monkeypatch, tmp_path):
    """Driver tests: single-bank "dark" LED mode (every frame alike)."""
    cam, fake = _make_hd(monkeypatch, tmp_path, "dark")
    yield cam, fake
    cam.close()


@pytest.fixture
def hd_gazepoint(monkeypatch, tmp_path):
    """Default LED mode: Gazepoint Control's alternating bright/dark pupil."""
    cam, fake = _make_hd(monkeypatch, tmp_path, None)
    yield cam, fake
    cam.close()


# ------------------------------------------------------------------ formulas
@pytest.mark.parametrize("text,vars,integer,expected", [
    ("1 + 2 * 3", {}, False, 7),
    ("(1 + 2) * 3", {}, False, 9),
    ("2 ** 3 ** 2", {}, False, 512),
    ("X > 0x7FF ? X - 0x1000 : X", {"X": 0xFFF}, True, -1),
    ("A = 1 ? 10 : 20", {"A": 2}, True, 20),
    ("7 / 2", {}, True, 3),
    ("7 / 2", {}, False, 3.5),
    ("(V >> 4) & 0xF | 1 << 8", {"V": 0xAB}, True, 0x10A),
    ("-X + ABS(-3)", {"X": 2}, False, 1),
    ("ROUND(PI, 2)", {}, False, 3.14),
    ("A <> B && !(A < 0) || 0", {"A": 1, "B": 2}, True, 1),
])
def test_formula(text, vars, integer, expected):
    assert Formula(text, integer)(vars) == pytest.approx(expected)


def test_formula_errors():
    with pytest.raises(GenICamError):
        Formula("1 +")
    with pytest.raises(GenICamError):
        Formula("Y * 2")({})


# ----------------------------------------------------------------- node map
def test_node_map_registers(hd):
    cam, fake = hd
    n = cam.nodes
    assert n.model == "FakeChameleon3"
    assert n.get("ExposureTime") == pytest.approx(10000.0)      # converter: ticks -> us
    n.set("ExposureTime", 4000)
    assert fake.r32(EXP_REG) == 8000
    assert n.get("ExposureAuto") == "Continuous"                # big-endian bits 30-31 = last byte
    fake.mem[EXPAUTO_REG:EXPAUTO_REG + 4] = bytes([0xFC, 0xFF, 0xFF, 0xFE])
    n.set("ExposureAuto", "Once")
    assert bytes(fake.mem[EXPAUTO_REG:EXPAUTO_REG + 4]) == bytes([0xFC, 0xFF, 0xFF, 0xFD])
    n.set("Gain", 6.5)
    assert struct.unpack(">f", bytes(fake.mem[GAIN_REG:GAIN_REG + 4]))[0] == 6.5
    n.set("GainAuto", "Continuous")
    assert fake.r32(GAINAUTO_REG) == 2 << 4
    assert n.get("GainAuto") == "Continuous"
    assert n.get("DeviceTemperature") == pytest.approx(-1.0)    # SwissKnife with ternary
    assert n.range("Width") == (16, 1280, 16)
    n.set("LineSelector", "Line3")
    n.set("LineMode", "Output")
    n.set("LineSource", "ExposureActive")                       # pIndex addressing
    n.set("LineInverter", True)                                 # IntSwissKnife addressing
    assert fake.r32(LINE_BASE + 0x30) == 1 | 2 | (5 << 8)
    assert n.get("LineSource") == "ExposureActive"
    n.set("StrobeEnabled", True)
    n.set("StrobeDelay", 0x42)                                  # StructEntry inherits address
    assert fake.r32(STRUCT_REG) == 1 | (0x42 << 8)
    assert n.get("PayloadSize") == 1280 * 1024
    with pytest.raises(GenICamError):
        n.set("PixelFormat", "RGB8")
    assert not n.try_set("NoSuchFeature", 1)


# ------------------------------------------------------------------- camera
def test_open_reads_bootstrap_and_xml(hd):
    cam, fake = hd
    assert cam.serial == "18123456"
    assert cam.info_strings["model"] == "Chameleon3 CM3-U3-13Y3M"
    assert (cam.width, cam.height) == (1280, 1024)
    assert cam.cp.max_ack == 256 and cam.cp.timeout_ms >= 300
    info = cam.info()
    assert info["features"]["Width"] == 1280 and info["lines"][2] == "Line2"


def test_modes_and_rate(hd):
    cam, fake = hd
    cam.set_video_mode(5, 5)                                    # GP3-style 640x480 centre crop, 60 Hz
    assert (cam.width, cam.height) == (640, 480)
    assert (fake.r32(OFFX_REG), fake.r32(OFFY_REG)) == (320, 272)
    assert cam.rate == 60
    cam.set_full_sensor()
    assert (cam.width, cam.height) == (1280, 1024) and fake.r32(OFFX_REG) == 0
    assert cam.rate == pytest.approx(149.5)                     # 150 clamped to the camera maximum
    assert fake.r32(RATEEN_REG) & 1
    assert cam.nodes.get("ExposureAuto") == "Off"
    assert cam.nodes.get("ExposureTime") <= 0.8e6 / 150 + 1    # exposure fits the frame period


def test_features_gp3_compatible(hd):
    cam, fake = hd
    f = cam.feature("gain")
    assert f["present"] and (f["min"], f["max"]) == (0, 24)
    cam.set_feature("gain", 99)                                 # clamped
    assert cam.feature("gain")["value"] == 24 and not cam.feature("gain")["auto"]
    cam.set_feature("shutter", None, auto=True)
    assert cam.feature("shutter")["auto"]


def test_stream_frames(hd):
    cam, fake = hd
    cam.set_full_sensor()
    cam.set_leds(True)
    assert fake.leds_on()
    cam.start_stream()
    seqs = []
    for _ in range(5):
        seq, t, img = cam.grab(timeout_s=2)
        seqs.append(seq)
        assert img.shape == (1024, 1280) and img.dtype == np.uint8
        np.testing.assert_array_equal(img, fake.image)
    assert seqs == sorted(seqs)
    # transfer sizes: 1 MiB transfers + a final transfer, aligned
    assert fake.r32(SIRM + 0x1C) == 1 << 20 and fake.r32(SIRM + 0x20) == 1
    assert fake.r32(SIRM + 0x24) == 1280 * 1024 - (1 << 20)
    cam.stop_stream()
    assert not fake.acquiring and fake.r32(TL_REG) == 0 and fake.r32(SIRM + 0x04) == 0
    cam.set_leds(False)
    assert not fake.leds_on()


def test_cropped_stream(hd):
    cam, fake = hd
    cam.set_video_mode()
    cam.set_leds(True)
    cam.start_stream()
    _, _, img = cam.grab(timeout_s=2)
    np.testing.assert_array_equal(img, fake.image[272:752, 320:960])
    # payload < 1 MiB: a single transfer of exactly the payload, no final transfer
    assert (fake.r32(SIRM + 0x1C), fake.r32(SIRM + 0x20), fake.r32(SIRM + 0x24)) == (640 * 480, 1, 0)


def test_default_leds_dark_pupil_bank_only(hd):
    cam, fake = hd
    assert (cam.ir_lines, cam.ir_invert) == (["Line3"], False)
    # Gazepoint Control leaves both banks strobing, with a one-in-three pattern
    n = cam.nodes
    for line in ("Line2", "Line3"):
        n.set("LineSelector", line); n.set("LineMode", "Output"); n.set("LineSource", "ExposureActive")
    fake._u32(PGR_BASE + 0x1138, 0x8000FFFF)                          # Line2 every frame
    assert fake.r32(PGR_BASE + 0x1148) == 0x80009F00                  # Line3 1 frame in 3
    cam.set_leds(True)
    assert fake.r32(PGR_BASE + 0x1148) == 0x8000FFFF                  # Line3 every frame
    assert not fake.strobing(2) and fake.strobing(3)                  # bright-pupil bank off
    n.set("LineSelector", "Line2")
    assert n.get("LineMode") == "Input"
    assert fake.r32(PGR_BASE + 0x1128) == 0x80009F00                  # Line1 untouched
    assert fake.light() == pytest.approx(1.0)
    cam.set_leds(False)
    assert fake.light() == 0


@pytest.mark.slow
def test_choose_led_lines_avoids_bright_pupil_bank(hd):
    cam, fake = hd
    fake.banks = {2: 1.0, 3: 1.0}
    cam.set_full_sensor()
    cam.start_stream()
    best, counts = cam.choose_led_lines(["Line2", "Line3"], False, frames=6)
    assert best == ("Line3",)
    assert counts[("Line2",)] == 0 and counts[("Line2", "Line3")] == 0 and counts[("Line3",)] == 6


@pytest.mark.slow
def test_probe_leds_finds_both_banks_and_saves(hd):
    cam, fake = hd
    fake.banks, fake.ir_inverted = {1: 0.3, 3: 0.7}, False     # not the default wiring
    cam.set_full_sensor()
    cam.start_stream()
    results = cam.probe_leds(frames=2)
    assert results[0][:2] == ("Line3", False)
    lines, inv = u3v.led_lines_from_probe(results)
    assert (lines, inv) == (["Line1", "Line3"], False)
    u3v.save_led_config(lines, inv)
    assert u3v.load_led_config() == (["Line1", "Line3"], False)
    cam.ir_lines, cam.ir_invert = u3v.load_led_config()
    cam.set_leds(True)
    assert fake.light() == pytest.approx(1.0)


def test_led_config_env_override_and_old_format(monkeypatch, tmp_path):
    path = tmp_path / "gp3hd.json"
    monkeypatch.setattr(u3v, "CONFIG_PATH", path)
    path.write_text('{"ir_line": "Line3", "ir_invert": true}')    # single-line format
    monkeypatch.delenv("OPENGP3_HD_IR_LINE", raising=False)
    monkeypatch.delenv("OPENGP3_HD_IR_INVERT", raising=False)
    assert u3v.load_led_config() == (["Line3"], True)
    monkeypatch.setenv("OPENGP3_HD_IR_LINE", "Line1, Line2")
    monkeypatch.setenv("OPENGP3_HD_IR_INVERT", "0")
    assert u3v.load_led_config() == (["Line1", "Line2"], False)


def test_gencp_error_status(hd):
    cam, fake = hd
    orig = fake.ctrl_write

    def failing(data, timeout_ms):
        orig(data, timeout_ms)
        a = bytearray(fake.acks[-1])
        a[4:6] = struct.pack("<H", 0x8003)
        fake.acks[-1] = bytes(a)
    fake.ctrl_write = failing
    with pytest.raises(u3v.U3VError, match="invalid address"):
        cam.cp.read(0x0, 4)


def test_tracker_pipeline_on_gp3hd(hd):
    """The whole tracker runs unchanged on the GP3 HD's frames."""
    cam, fake = hd
    gt = GazeTracker(Screen(1920, 1080, width_cm=53.0, distance_cm=65.0), camera=cam,
                     input_monitor=False, rate_hz=150)
    with gt:
        assert gt.model == "GP3 HD" and gt.image_size == (1280, 1024)
        assert gt.rate == pytest.approx(149.5)
        assert gt.eye_model.w == 1280
        t_end = time.time() + 5
        s = None
        while time.time() < t_end:
            s = gt.latest()
            if s is not None and len(s.eyes) == 2:
                break
            time.sleep(0.05)
        assert s is not None and set(s.eyes) == {"R", "L"}
        assert s.eyes["R"].pupil_x == pytest.approx(520, abs=1.5)
        assert s.eyes["L"].pupil_x == pytest.approx(780, abs=1.5)
        from opengp3.server import OpenGazeServer, _Client
        srv = OpenGazeServer(gt, "127.0.0.1", port=0)
        ack = srv.handle(_Client(sock=None, addr=None), "GET", "PRODUCT_ID", {})[0]
        assert 'VALUE="GP3HD" BUS="USB3" RATE="150"' in ack
        ack = srv.handle(_Client(sock=None, addr=None), "GET", "SERIAL_ID", {})[0]
        assert 'VALUE="18123456"' in ack


class FakeAsync:
    """Stands in for AsyncStream on the simulated camera: enforces the
    queued transfer-size cycle and records restarts."""

    def __init__(self, fake, sizes):
        self.fake, self.sizes, self.pos = fake, list(sizes), 0
        self.submitted = self.closed = False

    def submit_all(self):
        self.submitted, self.pos = True, 0

    def read(self, size, timeout_ms, abort=None, into=None):
        assert self.submitted and size == self.sizes[self.pos], "read out of the queued cycle"
        self.pos = (self.pos + 1) % len(self.sizes)
        data = self.fake.stream_read(size, timeout_ms)
        if into is None:
            return data
        into[:len(data)] = np.frombuffer(data, np.uint8)
        return len(data)

    def close(self):
        self.closed = True


def test_queued_stream_restarts_on_lost_sync(hd):
    cam, fake = hd
    streams = []

    def async_stream(sizes, depth=4):
        streams.append(FakeAsync(fake, sizes))
        return streams[-1]
    fake.async_stream = async_stream
    cam.set_full_sensor()
    cam.set_leds(True)
    cam.start_stream()                # first queued read hits stale bytes -> restart
    _, _, img = cam.grab(timeout_s=3)
    np.testing.assert_array_equal(img, fake.image)
    assert streams[0].sizes == [56, 1 << 20, 1280 * 1024 - (1 << 20), 32]
    assert len(streams) >= 2 and streams[0].closed      # resynced by restarting acquisition
    n_before = len(streams)
    fake.junk = 1                                        # break sync mid-stream
    t_end = time.time() + 5
    while len(streams) == n_before and time.time() < t_end:
        time.sleep(0.02)
    assert len(streams) > n_before
    cam.grab(latest=True)
    _, _, img = cam.grab(timeout_s=3)                    # frames keep coming after the restart
    np.testing.assert_array_equal(img, fake.image)
    cam.stop_stream()
    assert streams[-1].closed and not fake.acquiring


# ------------------------------------------------- Gazepoint illumination
def test_gazepoint_leds_reproduce_gazepoint_registers(hd_gazepoint):
    cam, fake = hd_gazepoint
    assert cam.ir_mode == "gazepoint" and cam.illumination == "bright-dark"
    cam.set_full_sensor()
    assert cam.nodes.get("ExposureTime") == pytest.approx(6600, abs=1)   # Gazepoint's pulse
    cam.set_leds(True)
    assert fake.gazepoint_active()
    assert fake.r32(PGR_BASE + 0x1128) == 0x80009F00                     # Line1 untouched
    cam.set_leds(False)
    assert not fake.gazepoint_active() and fake.light() == 0


def test_bright_dark_detector_on_gazepoint_cycle(hd_gazepoint):
    from opengp3.brightdark import BrightDarkDetector
    cam, fake = hd_gazepoint
    cam.set_full_sensor()
    cam.set_leds(True)
    cam.start_stream()
    det = BrightDarkDetector()
    kinds, pcrs = [], {"B": [], "D": []}
    for _ in range(12):
        seq, _, img = cam.grab(timeout_s=3)
        eyes = det.update(img)
        kinds.append(det.last_kind)
        if len(eyes) == 2:
            pcrs[det.last_kind].append([e.pcr for e in eyes])
    assert "".join(kinds[3:12]).count("D") == 3                         # one dark frame per cycle
    for k in ("B", "D"):
        assert pcrs[k], f"no eyes in {k} frames"
        mean = np.mean(pcrs[k], axis=0)
        # bright frames: pupil - on-axis glint = (3, -4); dark frames are shifted to
        # the same reference (their own glint pair alone would give (3, 2))
        np.testing.assert_allclose(mean, [[3, -4], [3, -4]], atol=0.6)
        for e in det.update(img) or []:
            assert 7 <= e.pupil_r <= 11


def test_tracker_uses_bright_dark_detection_on_gp3hd(hd_gazepoint):
    cam, fake = hd_gazepoint
    gt = GazeTracker(Screen(1920, 1080, width_cm=53.0, distance_cm=65.0), camera=cam,
                     input_monitor=False)
    with gt:
        assert gt.illumination == "bright-dark"
        t_end, s = time.time() + 5, None
        while time.time() < t_end:
            s = gt.latest()
            if s is not None and set(s.eyes) == {"R", "L"}:
                break
            time.sleep(0.05)
        assert s is not None and set(s.eyes) == {"R", "L"}
        assert s.eyes["R"].pupil_x == pytest.approx(520, abs=1.5)
        assert s.eyes["R"].pcr == pytest.approx((3, -4), abs=0.8)


def test_pupil_gain_control_steps():
    from opengp3.brightdark import BrightDarkDetector, PupilGainControl

    class Cam:
        gain = 12.0

        def feature(self, name):
            return {"min": 0.0, "max": 18.0, "value": self.gain}

        def set_feature(self, name, value=None, auto=False):
            self.gain = value

    cam, det = Cam(), BrightDarkDetector()
    gc = PupilGainControl(cam)
    det.pupil_level = 255.0                       # saturated bright pupil: step down fast
    gc.update(det, 0.0)
    assert cam.gain == 11.0 and det.pupil_level is None
    det.pupil_level = 255.0
    gc.update(det, 0.1)                           # rate-limited
    assert cam.gain == 11.0
    gc.update(det, 0.3)
    assert cam.gain == 10.0
    det.pupil_level = 60.0                        # dim: step up slowly
    gc.update(det, 0.6)
    assert cam.gain == 10.5
    det.pupil_level = 180.0                       # in range: unchanged
    gc.update(det, 0.9)
    assert cam.gain == 10.5


def test_gp3hd_focal_length_used_by_eye_model(hd):
    from opengp3.eyemodel import DEFAULT_FOCAL_PX
    cam, fake = hd
    assert cam.focal_px != DEFAULT_FOCAL_PX
    gt = GazeTracker(Screen(1920, 1080, width_cm=53.0), camera=cam, input_monitor=False)
    with gt:
        assert gt.eye_model.focal_px == cam.focal_px


@pytest.mark.slow
def test_display_frames_are_one_type_on_gazepoint_illumination(hd_gazepoint):
    """latest_frame() (the tracker display) only shows dark-pupil frames, so it
    doesn't flicker between the cycle's bright and dark frames."""
    cam, fake = hd_gazepoint
    gt = GazeTracker(Screen(1920, 1080, width_cm=53.0), camera=cam, input_monitor=False)
    with gt:
        means, t_end = [], time.time() + 3
        while time.time() < t_end and len(means) < 15:
            img, s = gt.latest_frame()
            if img is not None:
                means.append(float(img[::8, ::8].mean()))
            time.sleep(0.03)
    assert len(means) >= 10
    assert max(means) - min(means) < 1.0          # all the same (dark-pupil) frame type
    assert np.mean(means) == pytest.approx(float(fake.image[::8, ::8].mean()), abs=1.0)
