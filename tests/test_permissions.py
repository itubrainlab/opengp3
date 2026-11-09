"""Linux automatic USB permission setup (mocked: no root, no real udev)."""
import subprocess
import sys

import pytest

import opengp3.camera as camera
import opengp3.permissions as perm


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("OPENGP3_NO_AUTO_SETUP", raising=False)
    monkeypatch.setattr(perm.os, "geteuid", lambda: 1000, raising=False)


def test_elevation_prefers_pkexec_in_desktop_session(linux, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(perm.shutil, "which", lambda c: f"/usr/bin/{c}")
    cmd = perm.elevation_command()
    assert cmd[:3] == ["pkexec", "sh", "-c"]
    assert cmd[-1] == perm.RULE                    # rule passed as data, not in the script
    assert perm.RULE not in cmd[3]


def test_elevation_uses_sudo_on_terminal(linux, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(perm.shutil, "which", lambda c: "/usr/bin/sudo" if c == "sudo" else None)
    monkeypatch.setattr(perm.sys.stdin, "isatty", lambda: True, raising=False)
    assert perm.elevation_command()[:3] == ["sudo", "sh", "-c"]


def test_elevation_root_and_none(linux, monkeypatch):
    monkeypatch.setattr(perm.os, "geteuid", lambda: 0, raising=False)
    assert perm.elevation_command()[0] == "sh"
    monkeypatch.setattr(perm.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(perm.shutil, "which", lambda c: None)
    assert perm.elevation_command() is None
    with pytest.raises(perm.SetupError, match="sudo tee"):
        perm.install(announce=lambda *_: None)


def test_install_runs_command(linux, monkeypatch):
    calls = []
    monkeypatch.setattr(perm, "elevation_command", lambda: ["pkexec", "sh", "-c", "x", "n", perm.RULE])
    monkeypatch.setattr(perm.subprocess, "run",
                        lambda cmd, timeout: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    perm.install(announce=lambda *_: None)
    assert calls and calls[0][0] == "pkexec"
    monkeypatch.setattr(perm.subprocess, "run",
                        lambda cmd, timeout: subprocess.CompletedProcess(cmd, 126))  # cancelled
    with pytest.raises(perm.SetupError, match="cancelled"):
        perm.install(announce=lambda *_: None)


def test_open_installs_rule_then_retries(linux, monkeypatch):
    attempts = {"n": 0}

    def fake_open():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise camera._NoPermission()
        return "camera"

    installed = []
    monkeypatch.setattr(camera.GP3Camera, "_open", staticmethod(fake_open))
    monkeypatch.setattr(perm, "rule_installed", lambda: False)
    monkeypatch.setattr(perm, "install", lambda: installed.append(1))
    assert camera.GP3Camera.open() == "camera"
    assert installed == [1] and attempts["n"] == 2


def test_open_respects_opt_out_and_existing_rule(linux, monkeypatch):
    def denied():
        raise camera._NoPermission()

    monkeypatch.setattr(camera.GP3Camera, "_open", staticmethod(denied))
    monkeypatch.setattr(perm, "install", lambda: pytest.fail("must not install"))
    monkeypatch.setattr(perm, "rule_installed", lambda: False)
    monkeypatch.setenv("OPENGP3_NO_AUTO_SETUP", "1")
    with pytest.raises(camera.GP3Error, match="sudo tee"):
        camera.GP3Camera.open()
    monkeypatch.delenv("OPENGP3_NO_AUTO_SETUP")
    with pytest.raises(camera.GP3Error, match="sudo tee"):
        camera.GP3Camera.open(auto_setup=False)
    monkeypatch.setattr(perm, "rule_installed", lambda: True)
    with pytest.raises(camera.GP3Error, match="replug"):
        camera.GP3Camera.open()
