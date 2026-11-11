import opengp3.platform as plat


def test_input_monitor_transitions(monkeypatch):
    state = {"buttons": (False, False), "keys": set()}
    monkeypatch.setattr(plat, "mouse_position", lambda: (10.0, 20.0))
    monkeypatch.setattr(plat, "mouse_buttons", lambda: state["buttons"])
    monkeypatch.setattr(plat, "pressed_keys", lambda: state["keys"])
    m = plat.InputMonitor()
    assert m.poll() == (10.0, 20.0, 0, " ", 0)
    state["buttons"] = (True, False)
    state["keys"] = {"A"}
    assert m.poll()[2:] == (1, "A", 1)              # left down, A down
    assert m.poll()[2:] == (0, " ", 0)
    state["buttons"] = (False, True)                # left up + right down: queued
    state["keys"] = set()
    assert m.poll()[2:] == (2, "A", 2)
    assert m.poll()[2:] == (3, " ", 0)


def test_backend_never_raises():
    plat.displays()
    plat.mouse_position()
    plat.pressed_keys()


def test_edid_width_mm():
    edid = bytearray(128)
    edid[:8] = b"\x00\xff\xff\xff\xff\xff\xff\x00"
    edid[21] = 53                                   # basic display parameters: 53 cm
    assert plat.edid_width_mm(bytes(edid)) == 530.0
    edid[54], edid[55] = 0x02, 0x3A                 # detailed timing descriptor (pixel clock)
    edid[66], edid[68] = 0x0F, 0x20                 # 0x20F = 527 mm (upper 4 bits in byte 68)
    assert plat.edid_width_mm(bytes(edid)) == 527.0
    assert plat.edid_width_mm(b"\x00" * 128) is None   # not an EDID
    assert plat.edid_width_mm(b"") is None


def test_plausible_width_mm():
    assert plat.plausible_width_mm(527, 1920, 1080, 296) == 527
    assert plat.plausible_width_mm(None) is None and plat.plausible_width_mm(0) is None
    assert plat.plausible_width_mm(160, 1920, 1080, 90) is None      # EDID aspect placeholder
    assert plat.plausible_width_mm(40) is None and plat.plausible_width_mm(9000) is None
    assert plat.plausible_width_mm(677, 1920, 1080, 600) is None     # aspect doesn't match pixels
    # macOS without EDID: 72 dpi guess (1920 px -> 677.3 mm)
    assert plat.plausible_width_mm(1920 / 72 * 25.4, 1920, dpi_guess=72) is None
    assert plat.plausible_width_mm(597, 2560, dpi_guess=72) == 597


def test_linux_edid_widths(tmp_path):
    def edid(cm):
        e = bytearray(128)
        e[:8] = b"\x00\xff\xff\xff\xff\xff\xff\x00"
        e[21] = cm
        return bytes(e)
    for name, status, data in (("card0-HDMI-A-1", "connected", edid(53)),
                               ("card0-DP-2", "connected", edid(60)),
                               ("card0-DP-3", "disconnected", edid(40)),
                               ("card0-eDP-1", "connected", b"")):
        d = tmp_path / name
        d.mkdir()
        (d / "status").write_text(status + "\n")
        (d / "edid").write_bytes(data)
    (tmp_path / "card0").mkdir()
    w = plat.linux_edid_widths(str(tmp_path))
    assert w == {"HDMI-A-1": 530.0, "HDMI-1": 530.0, "DP-2": 600.0}
    assert plat.linux_edid_widths(str(tmp_path / "missing")) == {}
