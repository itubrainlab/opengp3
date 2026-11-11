"""One-time USB permission setup on Linux.

Linux only lets normal users read raw USB device files; libusb needs write
access to control the GP3 camera. OpenGP3 fixes this automatically the first
time it hits the permission error: it installs a udev rule granting access to
Gazepoint's cameras (VID 1e10: the GP3 and the GP3 HD) and re-applies the rules to the plugged-in
device. This needs administrator rights once per machine: a graphical
password prompt (pkexec) in a desktop session, or sudo in a terminal.

Disable the automatic attempt with OPENGP3_NO_AUTO_SETUP=1 (e.g. for
services); run it explicitly with ``opengp3 setup``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

RULE_PATH = "/etc/udev/rules.d/60-opengp3.rules"
RULE = ('# OpenGP3: user access to the Gazepoint GP3 / GP3 HD cameras (Point Grey / FLIR)\n'
        'SUBSYSTEM=="usb", ATTR{idVendor}=="1e10", MODE="0666", TAG+="uaccess"\n')

# Runs as root. The rule text is a constant passed as an argument ($1), so no
# user-controlled data reaches the shell.
_SCRIPT = ('printf "%s" "$1" > ' + RULE_PATH + ' && udevadm control --reload-rules && '
           'udevadm trigger --action=change --subsystem-match=usb --attr-match=idVendor=1e10 && '
           'udevadm settle --timeout=5')


class SetupError(RuntimeError):
    pass


def needed() -> bool:
    """True on Linux when the rule is not installed yet."""
    return sys.platform.startswith("linux") and not rule_installed()


def rule_installed() -> bool:
    try:
        with open(RULE_PATH) as fh:
            text = fh.read()
        # rules written before GP3 HD support matched the GP3's product ID only
        return 'ATTR{idVendor}=="1e10"' in text and "idProduct" not in text
    except OSError:
        return False


def auto_setup_enabled() -> bool:
    return os.environ.get("OPENGP3_NO_AUTO_SETUP", "").strip() not in ("1", "true", "yes")


def elevation_command() -> list[str] | None:
    """How to run the setup script as root, or None if no way is available."""
    if os.geteuid() == 0:
        return ["sh", "-c", _SCRIPT, "opengp3-setup", RULE]
    graphical = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if graphical and shutil.which("pkexec"):
        return ["pkexec", "sh", "-c", _SCRIPT, "opengp3-setup", RULE]
    if shutil.which("sudo") and sys.stdin is not None and sys.stdin.isatty():
        return ["sudo", "sh", "-c", _SCRIPT, "opengp3-setup", RULE]
    return None


def manual_instructions() -> str:
    rule = RULE.splitlines()[1]
    return ("Allow non-root access to the GP3 by running:\n\n"
            f"  echo '{rule}' | sudo tee {RULE_PATH}\n"
            "  sudo udevadm control --reload-rules && sudo udevadm trigger\n")


def install(announce=print) -> None:
    """Install the udev rule (prompts for the administrator password once)."""
    if not sys.platform.startswith("linux"):
        raise SetupError("USB permission setup is only needed on Linux")
    cmd = elevation_command()
    if cmd is None:
        raise SetupError("cannot ask for administrator rights here (no pkexec in a desktop "
                         "session, no sudo on a terminal).\n" + manual_instructions())
    announce("OpenGP3: the GP3 needs a one-time USB permission rule "
             f"({RULE_PATH}); asking for the administrator password...")
    try:
        r = subprocess.run(cmd, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SetupError(f"permission setup failed ({e}).\n" + manual_instructions()) from e
    if r.returncode != 0:
        raise SetupError("permission setup was cancelled or failed.\n" + manual_instructions())
    announce("OpenGP3: USB permission rule installed.")
