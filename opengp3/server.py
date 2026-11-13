"""Open Gaze API server: lets existing Gazepoint clients (Gazepoint's demo
programs, the LSL bridge, PsychoPy/OpenSesame/MATLAB plugins, ...) use OpenGP3
in place of Gazepoint Control.

Protocol (Open Gaze API v2): TCP, default port 4242; one XML element per
line, terminated by CR LF. Clients send ``<GET ID="..." />`` and
``<SET ID="..." .../>``; the server answers ``<ACK .../>`` or ``<NACK .../>``,
streams ``<REC .../>`` data records once ENABLE_SEND_DATA is on, and sends
``<CAL .../>`` records during calibration. Field selection (ENABLE_SEND_*) and
the record counter are per client.

The calibration window must run on the main thread (macOS GUI rule): the
server only queues requests; the application's main loop executes them
(see ``opengp3.cli.cmd_server``), or a client may draw its own targets with
CALIBRATE_SHOW=0, following the CAL records.
"""
from __future__ import annotations

import queue
import re
import socket
import threading
from dataclasses import dataclass, field

import cv2

from . import __version__
from .calibration import (API_DEFAULT_DELAY, API_DEFAULT_POINTS, API_DEFAULT_TIMEOUT, EYES,
                          NINE_POINTS)
from .screen import Screen
from .tracker import GazeSample, GazeTracker

API_VERSION = "2.8"

ALL_NETWORKS = "0.0.0.0"           # default: reachable from other computers (multi-machine setups)
LOCAL_ONLY = ("127.0.0.1", "localhost", "::1")


def local_ipv4_addresses() -> list[str]:
    """This computer's IPv4 addresses on its networks (no loopback): what
    clients on other machines connect to."""
    found: list[str] = []
    try:  # the address of the default route (connect() on UDP sends nothing)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    out = []
    for a in found:
        if a and not a.startswith("127.") and a not in out:
            out.append(a)
    return out

# ENABLE_SEND_* names in record field order
DATA_FIELDS = ["COUNTER", "TIME", "TIME_TICK", "POG_FIX", "POG_LEFT", "POG_RIGHT", "POG_BEST",
               "POG_AAC", "PUPIL_LEFT", "PUPIL_RIGHT", "EYE_LEFT", "EYE_RIGHT", "CURSOR", "KB",
               "BLINK", "PUPILMM", "DIAL", "GSR", "HR", "HR_PULSE", "HR_IBI", "TTL", "PIX",
               "USER_DATA"]

_TAG = re.compile(r"<\s*(GET|SET)\b(.*?)/?\s*>", re.S)
_ATTR = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"')


def parse_command(text: str):
    """'<SET ID="X" STATE="1" />' -> ("SET", "X", {"STATE": "1"}) or None."""
    m = _TAG.search(text)
    if not m:
        return None
    attrs = {k.upper(): v for k, v in _ATTR.findall(m.group(2))}
    ident = attrs.pop("ID", "").strip().upper()
    return m.group(1), ident, attrs


def _q(v) -> str:
    """Escape a value for an XML attribute."""
    return (str(v).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def element(tag: str, attrs: list[tuple[str, object]]) -> str:
    return "<" + tag + "".join(f' {k}="{_q(v)}"' for k, v in attrs) + " />\r\n"


def _f(v: float, n: int = 5) -> str:
    return f"{v:.{n}f}"


@dataclass
class _Client:
    sock: socket.socket
    addr: tuple
    enabled: set = field(default_factory=set)
    send_data: bool = False
    counter: int = 0
    out: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2000))
    alive: bool = True


class OpenGazeServer:
    """Serves a running ``GazeTracker`` over the Open Gaze API.

    ``requests`` is a queue of GUI actions for the main thread:
    ("calibrate", None), ("calibrate_stop", None), ("calibrate_show", bool),
    ("tracker_display", bool).
    """

    def __init__(self, tracker: GazeTracker, host: str = ALL_NETWORKS, port: int = 4242,
                 product: str = "GP3", default_points=None):
        self.tracker, self.host, self.port, self.product = tracker, host, port, product
        self.clients: list[_Client] = []
        self._lock = threading.Lock()
        self._srv: socket.socket | None = None
        self._running = False
        self.requests: queue.Queue = queue.Queue()
        # server state (Gazepoint Control equivalents)
        #: calibration points restored by CALIBRATE_RESET (default: 9-point grid)
        self.default_points = list(default_points or NINE_POINTS)
        self.cal_points = list(self.default_points)
        self.cal_timeout = API_DEFAULT_TIMEOUT
        self.cal_delay = API_DEFAULT_DELAY
        self.cal_show = False
        self.cal_running = False
        self.tracker_display = False
        self.tray = False
        self.user_dur = 0
        self.marker_mm = 11.0
        self.marker_on = False
        tracker.add_callback(self._on_sample)

    # ----------------------------------------------------------- networking
    def start(self) -> "OpenGazeServer":
        self._running = True
        self._listen(self.host)
        return self

    def _listen(self, host: str):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, self.port))
        except OSError:
            sock.close()
            raise
        sock.listen(8)
        self._srv, self.host = sock, host
        self.port = sock.getsockname()[1]
        self._accept_thread = threading.Thread(target=self._accept, args=(sock,),
                                               name="opengaze-accept", daemon=True)
        self._accept_thread.start()

    def _close_listener(self):
        """Close the listening socket and wait for its accept thread. On Linux
        close() alone neither wakes a thread blocked in accept() nor frees the
        port; shutdown() does both."""
        sock, self._srv = self._srv, None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
        t = getattr(self, "_accept_thread", None)
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2)

    @property
    def local_only(self) -> bool:
        """True when only programs on this computer can connect."""
        return self.host in LOCAL_ONLY or self.host.startswith("127.")

    def set_local_only(self, local_only: bool):
        """Switch between accepting connections from this computer only and
        from all networks, while running. Connected clients stay connected."""
        self.set_host(LOCAL_ONLY[0] if local_only else ALL_NETWORKS)

    def set_host(self, host: str):
        """Listen on another address (same port), keeping connected clients."""
        if host == self.host or not self._running:
            self.host = host
            return
        old = self.host
        self._close_listener()
        try:
            self._listen(host)
        except OSError:
            self._listen(old)  # keep serving on the previous address
            raise

    def addresses(self) -> list[str]:
        """host:port strings clients can use to reach this server."""
        if self.local_only:
            return [f"127.0.0.1:{self.port}"]
        if self.host not in (ALL_NETWORKS, ""):
            return [f"{self.host}:{self.port}"]
        return [f"{a}:{self.port}" for a in local_ipv4_addresses()] or [f"127.0.0.1:{self.port}"]

    def stop(self):
        self._running = False
        self.tracker.remove_callback(self._on_sample)
        self._close_listener()
        with self._lock:
            for c in self.clients:
                self._drop(c)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _accept(self, listener: socket.socket):
        while self._running:
            try:
                sock, addr = listener.accept()
            except OSError:  # listener closed (stop or address change)
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c = _Client(sock, addr)
            with self._lock:
                self.clients.append(c)
            threading.Thread(target=self._reader, args=(c,), daemon=True).start()
            threading.Thread(target=self._writer, args=(c,), daemon=True).start()

    def _reader(self, c: _Client):
        buf = ""
        while c.alive:
            try:
                data = c.sock.recv(4096)
            except OSError:
                break
            if not data:
                break
            buf += data.decode("utf-8", "replace")
            # commands end with '>' (CR LF optional, as some clients omit it)
            while ">" in buf:
                end = buf.index(">") + 1
                cmd, buf = buf[:end], buf[end:]
                parsed = parse_command(cmd)
                if parsed:
                    for line in self.handle(c, *parsed):
                        self._send(c, line)
        with self._lock:
            self._drop(c)

    def _writer(self, c: _Client):
        while c.alive:
            try:
                line = c.out.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                c.sock.sendall(line.encode())
            except OSError:
                break
        with self._lock:
            self._drop(c)

    def _send(self, c: _Client, line: str):
        try:
            c.out.put_nowait(line)
        except queue.Full:  # client not reading: drop it rather than stall tracking
            c.alive = False

    def _drop(self, c: _Client):
        c.alive = False
        try:
            c.sock.close()
        except OSError:
            pass
        if c in self.clients:
            self.clients.remove(c)

    def broadcast(self, line: str):
        with self._lock:
            for c in list(self.clients):
                self._send(c, line)

    # ----------------------------------------------------------- commands
    def handle(self, c: _Client, verb: str, ident: str, a: dict) -> list[str]:
        """Execute one GET/SET; returns the response lines."""
        setting = verb == "SET"
        ack = lambda *kv: [element("ACK", [("ID", ident), *kv])]
        nack = [element("NACK", [("ID", ident)])]
        tr = self.tracker

        def flag(name, default=None):
            v = a.get(name, default)
            return None if v is None else v.strip() not in ("0", "", "false", "FALSE")

        if ident == "ENABLE_SEND_DATA":
            if setting:
                c.send_data = bool(flag("STATE", "0"))
            return ack(("STATE", int(c.send_data)))
        if ident.startswith("ENABLE_SEND_"):
            name = ident[len("ENABLE_SEND_"):]
            if name not in DATA_FIELDS:
                return nack
            if setting:
                (c.enabled.add if flag("STATE", "0") else c.enabled.discard)(name)
            return ack(("STATE", int(name in c.enabled)))

        if ident == "CALIBRATE_SHOW":
            if setting:
                self.cal_show = bool(flag("STATE", "0"))
                self.requests.put(("calibrate_show", self.cal_show))
            return ack(("STATE", int(self.cal_show)))
        if ident == "CALIBRATE_START":
            if setting:
                start = bool(flag("STATE", "0"))
                if start and not self.cal_points:
                    return nack
                self.requests.put(("calibrate" if start else "calibrate_stop", None))
                self.cal_running = start
            return ack(("STATE", int(self.cal_running)))
        if ident in ("CALIBRATE_TIMEOUT", "CALIBRATE_DELAY"):
            attr = "cal_timeout" if ident == "CALIBRATE_TIMEOUT" else "cal_delay"
            if setting:
                try:
                    v = float(a.get("VALUE", ""))
                except ValueError:
                    return nack
                if v < 0 or (v == 0 and attr == "cal_timeout"):
                    return nack
                setattr(self, attr, v)
            return ack(("VALUE", f"{getattr(self, attr):g}"))
        if ident == "CALIBRATE_RESULT_SUMMARY":
            cal = tr.calibration
            if cal is None:
                return ack(("AVE_ERROR", "0.00"), ("VALID_POINTS", 0))
            err = cal.average_error
            return ack(("AVE_ERROR", f"{err:.2f}" if err == err else "0.00"),
                       ("VALID_POINTS", cal.valid_points))
        if ident == "CALIBRATE_CLEAR":
            if setting:
                self.cal_points = []
            return ack(("PTS", len(self.cal_points)))
        if ident == "CALIBRATE_RESET":
            if setting:
                self.cal_points = list(self.default_points)
            return ack(("PTS", len(self.cal_points)))
        if ident == "CALIBRATE_ADDPOINT":
            if setting:
                try:
                    self.cal_points.append((float(a["X"]), float(a["Y"])))
                except (KeyError, ValueError):
                    return nack
            pts = [("PTS", len(self.cal_points))]
            for i, (x, y) in enumerate(self.cal_points, 1):
                pts += [(f"X{i}", _f(x)), (f"Y{i}", _f(y))]
            return ack(*pts)

        if ident == "USER_DATA":
            if setting:
                tr.user_data = a.get("VALUE", "")
                try:
                    self.user_dur = int(a.get("DUR", "0") or 0)
                except ValueError:
                    self.user_dur = 0
            return ack(("VALUE", tr.user_data or "0"), *([("DUR", self.user_dur)] if setting else []))
        if ident == "TRACKER_DISPLAY":
            if setting:
                if "STATE" in a:
                    self.tracker_display = bool(flag("STATE"))
                if "TRAY" in a:
                    self.tray = bool(flag("TRAY"))
                self.requests.put(("tracker_display", self.tracker_display))
            return ack(("STATE", int(self.tracker_display)), ("TRAY", int(self.tray)))
        if ident == "TIME_TICK_FREQUENCY":
            return ack(("FREQ", int(cv2.getTickFrequency())))
        if ident == "SCREEN_SIZE":
            if setting:
                try:
                    x, y = float(a.get("X", 0)), float(a.get("Y", 0))
                    w, h = int(float(a["WIDTH"])), int(float(a["HEIGHT"]))
                except (KeyError, ValueError):
                    return nack
                s = tr.screen
                tr.set_screen(Screen(w, h, s.width_cm, s.distance_cm, x, y))
            s = tr.screen
            return ack(("X", int(s.x)), ("Y", int(s.y)), ("WIDTH", s.width), ("HEIGHT", s.height))
        if ident == "CAMERA_SIZE":
            return ack(("WIDTH", tr.image_size[0]), ("HEIGHT", tr.image_size[1]))
        if ident == "PRODUCT_ID":
            cam = tr.camera
            model = getattr(cam, "model", "GP3")
            product = (getattr(cam, "product_id", None)
                       or (self.product if self.product != "GP3" else model.replace(" ", "")))
            return ack(("VALUE", product), ("BUS", getattr(cam, "bus", "USB2")),
                       ("RATE", f"{getattr(cam, 'rate', 60.0):.0f}"), *self._warning())
        if ident == "SERIAL_ID":
            return ack(("VALUE", self._serial()), *self._warning())
        if ident == "COMPANY_ID":
            # manufacturer of the eye-tracker hardware (none for webcam / simulated)
            return ack(("VALUE", getattr(tr.camera, "company_id",
                                         "OPENGP3" if tr.warning else "GAZEPOINT")),
                       *self._warning())
        if ident == "API_ID":
            return ack(("VALUE", API_VERSION), ("IMPL", f"OpenGP3 {__version__}"),
                       *self._warning())
        if ident == "TRACKER_ID":
            return ack(("ACTIVE_ID", 0), ("MAX_ID", 1), ("SEARCH", "NONE"))
        if ident == "MARKER_PIX":
            if setting:
                try:
                    if "VALUE" in a:
                        self.marker_mm = float(a["VALUE"])
                except ValueError:
                    return nack
                if "STATE" in a:
                    self.marker_on = bool(flag("STATE"))
                tr.enable_marker(self.marker_mm if self.marker_on else None)
            return ack(("VALUE", f"{self.marker_mm:g}"), ("STATE", int(self.marker_on)))
        if ident == "AAC_FILTER":
            if setting:
                try:
                    tr.set_aac_length(int(float(a["VALUE"])))
                except (KeyError, ValueError):
                    return nack
            return ack(("VALUE", tr.aac.length))
        # TTL_WRITE and anything else: not available (no biometrics hardware)
        return nack

    def _warning(self) -> list[tuple[str, str]]:
        """Extra WARNING attribute when the data is degraded or not real (webcam
        fallback, simulated camera); standard clients ignore unknown attributes."""
        w = self.tracker.warning
        return [("WARNING", w)] if w else []

    def _serial(self) -> str:
        from .lsl import _serial
        return _serial(self.tracker)

    # ----------------------------------------------------------- calibration
    def calibration_listener(self, event: str, n: int, fx: float, fy: float):
        ident = "CALIB_START_PT" if event == "start_pt" else "CALIB_RESULT_PT"
        self.broadcast(element("CAL", [("ID", ident), ("PT", n), ("CALX", _f(fx)),
                                       ("CALY", _f(fy))]))

    def send_calibration_result(self):
        """CALIB_RESULT: per point, target and each eye's calibrated gaze
        (fractions of the screen) with validity flags."""
        cal = self.tracker.calibration
        attrs = [("ID", "CALIB_RESULT")]
        if cal is not None and cal.points:
            s = cal.screen
            for k, p in enumerate(sorted(cal.points, key=lambda p: p.target), 1):
                attrs += [(f"CALX{k}", _f(p.x / s.width)), (f"CALY{k}", _f(p.y / s.height))]
                for eye, pre in (("L", "L"), ("R", "R")):
                    g = p.gaze.get(eye)
                    attrs += [(f"{pre}X{k}", _f(g[0] / s.width) if g else _f(0)),
                              (f"{pre}Y{k}", _f(g[1] / s.height) if g else _f(0)),
                              (f"{pre}V{k}", int(bool(p.valid.get(eye))))]
        self.broadcast(element("CAL", attrs))
        self.cal_running = False

    # ----------------------------------------------------------- data records
    def _on_sample(self, s: GazeSample):
        with self._lock:
            clients = [c for c in self.clients if c.send_data and c.alive]
        if clients:
            for c in clients:
                c.counter += 1
                self._send(c, self.format_record(s, c.enabled, c.counter))
        if self.user_dur == 1 and self.tracker.user_data:  # one-time USER value
            self.tracker.user_data = ""

    def format_record(self, s: GazeSample, enabled: set, counter: int = 0) -> str:
        W, H = self.tracker.screen.width, self.tracker.screen.height
        iw, ih = self.tracker.image_size
        r: list[tuple[str, object]] = []
        on = enabled.__contains__
        if on("COUNTER"):
            r.append(("CNT", counter))
        if on("TIME"):
            r.append(("TIME", _f(s.time)))
        if on("TIME_TICK"):
            r.append(("TIME_TICK", s.tick))
        if on("POG_FIX"):
            f = s.fixation
            r += [("FPOGX", _f(f.x / W)), ("FPOGY", _f(f.y / H)), ("FPOGS", _f(f.start)),
                  ("FPOGD", _f(f.duration)), ("FPOGID", f.id), ("FPOGV", int(f.valid))]
        for eye, side in (("POG_LEFT", "L"), ("POG_RIGHT", "R")):
            if on(eye):
                g = s.per_eye.get(side)
                r += [(f"{side}POGX", _f(g[0] / W if g else 0)), (f"{side}POGY", _f(g[1] / H if g else 0)),
                      (f"{side}POGV", int(g is not None))]
        if on("POG_BEST"):
            v = s.raw_x is not None
            r += [("BPOGX", _f(s.raw_x / W if v else 0)), ("BPOGY", _f(s.raw_y / H if v else 0)),
                  ("BPOGV", int(v))]
        if on("POG_AAC"):
            a = s.aac
            r += [("APOGX", _f(a[0] / W if a else 0)), ("APOGY", _f(a[1] / H if a else 0)),
                  ("APOGV", int(a is not None))]
        for name, side in (("PUPIL_LEFT", "L"), ("PUPIL_RIGHT", "R")):
            if on(name):
                m = s.metrics.get(side)
                v = bool(m and m.valid)
                r += [(f"{side}PCX", _f(m.cx if v else 0)), (f"{side}PCY", _f(m.cy if v else 0)),
                      (f"{side}PD", _f(m.diameter_px if v else 0)),
                      (f"{side}PS", _f(m.scale if v else 0)), (f"{side}PV", int(v))]
        for name, side in (("EYE_LEFT", "L"), ("EYE_RIGHT", "R")):
            if on(name):
                m = s.metrics.get(side)
                v = bool(m and m.valid and m.z_m)
                r += [(f"{side}EYEX", _f(m.x_m if v else 0)), (f"{side}EYEY", _f(m.y_m if v else 0)),
                      (f"{side}EYEZ", _f(m.z_m if v else 0)),
                      (f"{side}PUPILD", _f(m.pupil_mm / 1000 if v else 0)), (f"{side}PUPILV", int(v))]
        if on("CURSOR"):
            if s.cursor:
                sc = self.tracker.screen
                cx, cy, cs = s.cursor
                r += [("CX", _f((cx - sc.x) / W)), ("CY", _f((cy - sc.y) / H)), ("CS", cs)]
            else:
                r += [("CX", _f(0)), ("CY", _f(0)), ("CS", 0)]
        if on("KB"):
            r += [("KB", s.key[0]), ("KBS", s.key[1])]
        if on("BLINK"):
            b = s.blink
            r += [("BKID", b.id), ("BKDUR", _f(b.duration)), ("BKPMIN", b.per_minute)]
        if on("PUPILMM"):
            for side in ("L", "R"):
                m = s.metrics.get(side)
                v = bool(m and m.valid and m.pupil_mm)
                r += [(f"{side}PMM", _f(m.pupil_mm if v else 0)), (f"{side}PMMV", int(v))]
        # biometrics: separate Gazepoint hardware, not supported -> invalid
        if on("DIAL"):
            r += [("DIAL", _f(0)), ("DIALV", 0)]
        if on("GSR"):
            r += [("GSR", 0), ("GSRV", 0)]
        if on("HR"):
            r += [("HR", _f(0)), ("HRV", 0)]
        if on("HR_PULSE"):
            r.append(("HRP", 0))
        if on("HR_IBI"):
            r.append(("HRIBI", _f(0, 3)))
        if on("TTL"):
            r += [("TTL0", 0), ("TTL1", "000000"), ("TTLV", 0)]
        if on("PIX"):
            mk = s.marker
            v = bool(mk and mk.valid)
            r += [("PIXX", _f(mk.x if v else 0)), ("PIXY", _f(mk.y if v else 0)),
                  ("PIXS", _f(mk.mm_per_px if v else 0)), ("PIXV", int(v))]
        if on("USER_DATA"):
            r.append(("USER", s.user))
        return element("REC", r)


class OpenGazeClient:
    """Minimal Open Gaze API client (works with OpenGP3 and Gazepoint Control).

        with OpenGazeClient() as c:
            c.set("ENABLE_SEND_POG_FIX", STATE=1)
            c.set("ENABLE_SEND_DATA", STATE=1)
            for rec in c.records():
                print(rec["FPOGX"], rec["FPOGY"])
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4242, timeout: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._buf = ""
        self.pending: list[tuple[str, dict]] = []  # messages received while waiting for ACKs

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _send(self, verb, ident, **attrs):
        self.sock.sendall(element(verb, [("ID", ident), *attrs.items()]).encode())

    def _next(self) -> tuple[str, dict]:
        while "\n" not in self._buf:
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError("server closed the connection")
            self._buf += data.decode("utf-8", "replace")
        line, self._buf = self._buf.split("\n", 1)
        m = re.match(r"\s*<\s*(\w+)(.*?)/?\s*>", line)
        if not m:
            return "", {}
        return m.group(1), dict(_ATTR.findall(m.group(2)))

    def _command(self, verb, ident, **attrs) -> dict:
        self._send(verb, ident, **attrs)
        while True:
            tag, a = self._next()
            if tag in ("ACK", "NACK") and a.get("ID") == ident:
                if tag == "NACK":
                    raise ValueError(f"server rejected {verb} {ident}")
                return a
            self.pending.append((tag, a))

    def get(self, ident: str) -> dict:
        return self._command("GET", ident)

    def set(self, ident: str, **attrs) -> dict:
        return self._command("SET", ident, **attrs)

    def messages(self):
        """All incoming (tag, attrs): REC data, CAL calibration records, ..."""
        while self.pending:
            yield self.pending.pop(0)
        while True:
            yield self._next()

    def records(self):
        """Only the REC data records, as dicts of strings."""
        for tag, a in self.messages():
            if tag == "REC":
                yield a
