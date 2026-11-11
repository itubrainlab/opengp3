"""Windows UsbDk backend selection (mocked libusb: no driver needed)."""
import sys

import pytest

import opengp3.camera as camera


class FakeLib:
    def __init__(self, result=0):
        self.calls, self.result = [], result

    def libusb_set_option(self, ctx, option):
        self.calls.append((ctx, option.value))
        return self.result


class FakeBackend:
    def __init__(self, result=0):
        self.lib, self.ctx = FakeLib(result), object()


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(camera, "usb_driver", "winusb")
    monkeypatch.delenv("OPENGP3_USBDK", raising=False)


def test_usbdk_enabled_once_when_installed(windows, monkeypatch):
    monkeypatch.setattr(camera, "usbdk_installed", lambda: True)
    be = FakeBackend()
    monkeypatch.setattr(camera, "_find_libusb", lambda: be)
    assert camera._libusb_backend() is be
    camera._libusb_backend()                         # cached backend: option set only once
    assert be.lib.calls == [(be.ctx, camera.LIBUSB_OPTION_USE_USBDK)]
    assert camera.usb_driver == "usbdk"
    assert "UsbDk could not take over" in camera._access_help("GP3 HD")


def test_usbdk_not_installed_falls_back_to_winusb(windows, monkeypatch):
    monkeypatch.setattr(camera, "usbdk_installed", lambda: False)
    be = FakeBackend()
    monkeypatch.setattr(camera, "_find_libusb", lambda: be)
    camera._libusb_backend()
    assert be.lib.calls == [] and camera.usb_driver == "winusb"
    hint = camera._access_help("GP3 HD")
    assert "UsbDk" in hint and "1E10 3300" in hint


def test_usbdk_disabled_by_env(windows, monkeypatch):
    monkeypatch.setattr(camera, "usbdk_installed", lambda: True)
    monkeypatch.setenv("OPENGP3_USBDK", "0")
    be = FakeBackend()
    monkeypatch.setattr(camera, "_find_libusb", lambda: be)
    camera._libusb_backend()
    assert be.lib.calls == [] and camera.usb_driver == "winusb"


def test_usbdk_option_failure_keeps_winusb(windows, monkeypatch):
    monkeypatch.setattr(camera, "usbdk_installed", lambda: True)
    be = FakeBackend(result=-5)                      # LIBUSB_ERROR_NOT_FOUND
    monkeypatch.setattr(camera, "_find_libusb", lambda: be)
    camera._libusb_backend()
    assert camera.usb_driver == "winusb"


def test_open_waits_for_reenumeration_under_usbdk(windows, monkeypatch):
    monkeypatch.setattr(camera, "usb_driver", "usbdk")
    monkeypatch.setattr(camera.time, "sleep", lambda s: None)
    tries = []

    def opener():
        tries.append(1)
        if len(tries) < 4:
            raise camera.GP3NotFoundError("gone")
        return "cam"
    assert camera._open_with_setup(opener) == "cam" and len(tries) == 4


def test_open_does_not_wait_without_usbdk(windows):
    def opener():
        raise camera.GP3NotFoundError("gone")
    with pytest.raises(camera.GP3NotFoundError):
        camera._open_with_setup(opener)


def test_usbdk_never_on_other_platforms(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    be = FakeBackend()
    monkeypatch.setattr(camera, "_find_libusb", lambda: be)
    camera._libusb_backend()
    assert be.lib.calls == [] and not camera.usbdk_installed()
