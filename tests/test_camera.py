"""GP3 driver against a simulated USB device: frame reads, stream glitches
(the camera re-sends part of a frame when the host leaves it without a
pending read), desync detection and realignment - no hardware needed."""
import struct

import numpy as np
import pytest
import usb.core

from opengp3.camera import GP3Camera, ISO_EN

W, H = 60, 40
CMD_BASE_LOW16 = 0xF0F0   # (cmd_base >> 16) & 0xFFFF, for offsets < 0x10000


def synth_frame(n, w=W, h=H, seed_edge=False):
    """A 'scene' with a smooth horizontal brightness gradient (like an
    IR-lit face), a slanted dark/bright edge (no perfectly vertical image
    feature) and sensor noise. With ``seed_edge`` the edge is razor-vertical:
    a scene feature that flags the seam detector but is not a roll of the
    previous scene."""
    rng = np.random.default_rng(n)
    img = (110 + 1.5 * np.arange(w)).astype(np.int16) * np.ones((h, 1), np.int16)
    if seed_edge:
        img[:, :30] = 40
    else:
        for i in range(h):
            img[i, :15 + i // 4] = 40
    img = np.clip(img + rng.integers(-4, 5, (h, w)), 0, 255)
    return img.astype(np.uint8)


class FakeDev:
    """pyusb Device stand-in serving a canned bulk byte stream.

    Registers are not interpreted except ISO_EN (offset 0x614): enabling it
    restarts the "camera" at the next frame boundary, like the real one;
    disabling it empties the pipe (reads time out). ``glitch_before`` injects
    ``glitch_len`` extra bytes into the stream before that frame, so every
    following read starts that many bytes early - a frame read then spans the
    seam and comes out as the scene rolled horizontally (the measured hardware
    failure). With ``compensate`` the same number of bytes is dropped from the
    end of the glitched frame, so exactly one read is slipped and the stream
    re-aligns itself; without it every following read stays slipped."""

    def __init__(self, frames, glitch_before=None, glitch_len=0, compensate=False,
                 fail_at=None):
        parts, self.frame_starts, pos = [], [], 0
        for i, f in enumerate(frames):
            self.frame_starts.append(pos)
            raw = f.tobytes()
            if i == glitch_before:
                # what the host sees: its read window starts early, so it first
                # gets the tail of the previous frame again
                parts.append(frames[i - 1].tobytes()[-glitch_len:])
                pos += glitch_len
                if compensate:     # the camera drops the tail: one slipped read
                    raw = raw[:len(raw) - glitch_len]
            parts.append(raw)
            pos += len(raw)
        self.stream = b"".join(parts)
        self.pos = 0
        self.iso = False
        self.iso_writes = []
        self.fail_at = fail_at    # stream position where a read fails mid-frame

    def read(self, ep, n, timeout=0):
        if not self.iso or self.pos >= len(self.stream):
            raise usb.core.USBError("timed out")
        if self.fail_at is not None and self.pos < self.fail_at < self.pos + n:
            self.pos = self.fail_at
            self.fail_at = None
            raise usb.core.USBError("timed out")
        data = self.stream[self.pos:self.pos + n]
        self.pos += len(data)
        if len(data) < n:
            raise usb.core.USBError("timed out")
        return data

    def ctrl_transfer(self, rtype, request, wvalue, windex, payload, timeout=0):
        if windex == CMD_BASE_LOW16 and wvalue == ISO_EN:
            on = bool(struct.unpack("<I", bytes(payload))[0] & 0x8000_0000)
            self.iso_writes.append(on)
            self.iso = on
            if on:  # the camera restarts streaming at a frame boundary
                self.pos = min([b for b in self.frame_starts if b >= self.pos]
                               or [len(self.stream)])
        return b"\0\0\0\0"


def make_cam(frames, **kw):
    cam = GP3Camera(FakeDev(frames, **kw))
    cam.width, cam.height, cam.bpp = W, H, 1
    cam.use_async = False          # no libusb in tests: the sync read path
    return cam


def iso_cycles(dev):
    """ISO_EN off->on pairs after the first enable."""
    ons = dev.iso_writes
    return sum(1 for i in range(2, len(ons)) if ons[i] and not ons[i - 1])


def test_aligned_frames_roundtrip():
    frames = [synth_frame(i) for i in range(4)]
    cam = make_cam(frames)
    cam.realign()
    for f in frames:
        assert np.array_equal(cam._read_frame(), f)
        assert cam._seam_streak == 0


def test_glitched_stream_realigns():
    """A re-sent frame prefix (the hardware bug: 'the camera sees double')
    slips every following frame; the driver verifies the rolled image against
    the recent scene and restarts the stream at a frame boundary."""
    frames = [synth_frame(i) for i in range(6)]
    cam = make_cam(frames, glitch_before=2, glitch_len=1237)
    cam.realign()
    cam._read_frame(), cam._read_frame()          # aligned
    for _ in range(2):                             # slipped frames
        img = cam._read_frame()
        assert not any(np.array_equal(img, f) for f in frames)   # doubled
    assert iso_cycles(cam.dev) == 1                # realigned after 2 frames
    assert cam._seam_streak == 0
    img = cam._read_frame()                        # stream back on boundaries
    assert any(np.array_equal(img, f) for f in frames)


def test_scene_edge_does_not_realign():
    """A static vertical scene edge flags the seam detector but is not a roll
    of the previous scene: no realignment, and the detector re-arms."""
    frames = [synth_frame(0), synth_frame(1)] + [synth_frame(2, seed_edge=True)] * 5
    cam = make_cam(frames)
    cam.realign()
    for _ in frames:
        cam._read_frame()
    assert iso_cycles(cam.dev) == 0
    assert cam._seam_streak == 0


def test_single_glitch_is_ignored():
    """A glitch the camera compensates within the same frame (re-sends its
    head, drops its tail) yields one slipped read that heals by itself:
    verified, but below the two-frame threshold, no stream restart."""
    frames = [synth_frame(i) for i in range(6)]
    cam = make_cam(frames, glitch_before=2, glitch_len=500, compensate=True)
    cam.realign()
    cam._read_frame(), cam._read_frame()          # aligned
    img = cam._read_frame()                       # the one slipped frame
    assert not any(np.array_equal(img, f) for f in frames)
    assert cam._seam_streak == 1                  # flagged (and verified)
    img = cam._read_frame()                       # aligned again
    assert any(np.array_equal(img, f) for f in frames)
    assert cam._seam_streak == 0
    assert iso_cycles(cam.dev) == 0


def test_usb_error_mid_frame_realigns():
    """A failed read leaves the stream at an unknown position: the driver
    restarts it instead of reading doubled frames forever."""
    frames = [synth_frame(i) for i in range(6)]
    cam = make_cam(frames, fail_at=W * H + 700)
    cam.realign()
    cam._read_frame()
    with pytest.raises(usb.core.USBError):
        cam._read_frame()                         # fails part-way into frame 2
    assert iso_cycles(cam.dev) == 1               # realigned already
    img = cam._read_frame()
    assert any(np.array_equal(img, f) for f in frames)


def test_realign_rate_limited():
    """Back-to-back persistent glitches realign at most once per interval."""
    frames = [synth_frame(i) for i in range(20)]
    cam = make_cam(frames, glitch_before=2, glitch_len=1237)
    cam.realign()
    for _ in range(12):
        try:
            cam._read_frame()
        except usb.core.USBError:
            break
    assert iso_cycles(cam.dev) == 1
