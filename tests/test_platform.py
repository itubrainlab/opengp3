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
