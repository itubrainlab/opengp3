"""Open Gaze API server, end to end: fake camera -> detection -> tracker ->
TCP server -> client, including API-driven (headless) calibration."""
import socket
import threading
import time

import pytest

from opengp3 import GazeTracker, Screen
from opengp3.server import (DATA_FIELDS, OpenGazeClient, OpenGazeServer, _Client, element,
                            parse_command)

SCREEN = Screen(1920, 1080, width_cm=53.0, distance_cm=65.0)


def test_parse_command_variants():
    assert parse_command('<SET ID="CALIBRATE_SHOW" STATE="1" />') == ("SET", "CALIBRATE_SHOW", {"STATE": "1"})
    assert parse_command('<SET ID="CALIBRATE_START" STATE ="1" />')[2] == {"STATE": "1"}
    assert parse_command('<GET ID="api_id"/>') == ("GET", "API_ID", {})
    assert parse_command("garbage") is None


def test_element_escapes():
    assert element("ACK", [("ID", "USER_DATA"), ("VALUE", 'a"<b>')]) == \
        '<ACK ID="USER_DATA" VALUE="a&quot;&lt;b&gt;" />\r\n'


@pytest.fixture
def server(fake_camera):
    gt = GazeTracker(SCREEN, camera=fake_camera, input_monitor=False)
    gt.start()
    srv = OpenGazeServer(gt, port=0).start()
    yield gt, srv, fake_camera
    srv.stop()
    gt.stop()


def test_configuration_commands(server):
    gt, srv, cam = server
    c = _Client(sock=None, addr=None)
    h = lambda v, i, **a: srv.handle(c, v, i, {k: str(x) for k, x in a.items()})[0]
    assert 'VALUE="2.8"' in h("GET", "API_ID")
    assert 'WIDTH="752" HEIGHT="480"' in h("GET", "CAMERA_SIZE")
    assert 'VALUE="GP3"' in h("GET", "PRODUCT_ID")
    assert 'WIDTH="1920" HEIGHT="1080"' in h("GET", "SCREEN_SIZE")
    assert 'FREQ="' in h("GET", "TIME_TICK_FREQUENCY")
    assert h("GET", "CALIBRATE_ADDPOINT").startswith('<ACK ID="CALIBRATE_ADDPOINT" PTS="9" X1="0.50000"')
    assert 'PTS="0"' in h("SET", "CALIBRATE_CLEAR")
    assert 'PTS="1" X1="0.20000" Y1="0.30000"' in h("SET", "CALIBRATE_ADDPOINT", X=0.2, Y=0.3)
    assert 'PTS="9"' in h("SET", "CALIBRATE_RESET")             # server default: 9 points
    assert 'VALUE="2"' in h("SET", "CALIBRATE_TIMEOUT", VALUE=2)
    assert h("SET", "CALIBRATE_TIMEOUT", VALUE=0).startswith("<NACK")
    assert 'VALUE="7"' in h("SET", "AAC_FILTER", VALUE=7) and gt.aac.length == 7
    assert 'VALUE="TRIG1"' in h("SET", "USER_DATA", VALUE="TRIG1")
    assert h("SET", "TTL_WRITE", CHANNEL=1, VALUE=0).startswith("<NACK")   # no biometrics hw
    assert h("GET", "NOT_A_COMMAND").startswith("<NACK")
    for f in DATA_FIELDS:
        assert 'STATE="1"' in h("SET", f"ENABLE_SEND_{f}", STATE=1)


def test_streams_all_fields_and_gazepoint_demo_parsing(server):
    gt, srv, cam = server
    with OpenGazeClient(port=srv.port) as c:
        for f in DATA_FIELDS:
            c.set(f"ENABLE_SEND_{f}", STATE=1)
        c.set("USER_DATA", VALUE="MARK")
        c.set("ENABLE_SEND_DATA", STATE=1)
        recs = []
        for r in c.records():
            recs.append(r)
            if len(recs) >= 20:
                break
    r = recs[-1]
    for key in ["CNT", "TIME", "TIME_TICK", "FPOGX", "FPOGV", "LPOGX", "RPOGV", "BPOGX", "APOGV",
                "LPCX", "LPD", "LPS", "RPV", "LEYEZ", "RPUPILD", "CX", "CS", "KB", "KBS", "BKID",
                "LPMM", "RPMMV", "DIALV", "GSRV", "HRV", "HRP", "HRIBI", "TTL0", "TTL1", "PIXV",
                "USER"]:
        assert key in r, key
    assert [int(x["CNT"]) for x in recs] == list(range(1, 21))     # per-client counter, no gaps
    assert r["LPV"] == "1" and r["RPV"] == "1"                     # both eyes detected
    assert float(r["LEYEZ"]) > 0.3
    assert r["USER"] == "MARK"
    assert float(recs[-1]["TIME"]) > float(recs[0]["TIME"])


def test_api_calibration_headless_then_gaze(server):
    """Client-drawn calibration (CALIBRATE_SHOW=0): the server times the
    points and announces them; the simulated subject looks at each one."""
    gt, srv, cam = server
    srv.cal_delay, srv.cal_timeout = 0.3, 1.0   # generous: CI runners are slow

    def listener(event, n, fx, fy):  # the subject follows the announced point
        srv.calibration_listener(event, n, fx, fy)
        if event == "start_pt":
            cam.look = (fx, fy)

    from opengp3.calibration import API_DEFAULT_POINTS
    with OpenGazeClient(port=srv.port) as c:
        c.set("CALIBRATE_CLEAR")                   # client-defined 5-point calibration
        for x, y in API_DEFAULT_POINTS:
            c.set("CALIBRATE_ADDPOINT", X=x, Y=y)
        assert c.get("CALIBRATE_ADDPOINT")["PTS"] == "5"
        c.set("CALIBRATE_SHOW", STATE=0)
        assert c.set("CALIBRATE_START", STATE=1)["STATE"] == "1"
        assert srv.requests.get(timeout=1)[0] == "calibrate_show"
        assert srv.requests.get(timeout=1)[0] == "calibrate"
        # what ControlApp does on the main thread, headless:
        run = gt.start_calibration(srv.cal_points, delay=srv.cal_delay, timeout=srv.cal_timeout,
                                   retry=False, listener=listener, keep_order=True, min_samples=5)
        while not run.tick(None):
            time.sleep(0.005)
        per_point = {i: sum(1 for j, *_ in run.labelled() if j == i) for i in range(5)}
        print("samples per calibration point:", per_point, "tracking fps:", gt.fps)
        gt.finish_calibration(run)
        srv.send_calibration_result()

        cal_msgs = []
        for tag, a in c.messages():
            if tag == "CAL":
                cal_msgs.append(a)
                if a["ID"] == "CALIB_RESULT":
                    break
        ids = [(m["ID"], m.get("PT")) for m in cal_msgs[:-1]]
        assert ids == [(k, str(n)) for n in range(1, 6) for k in ("CALIB_START_PT", "CALIB_RESULT_PT")]
        assert (cal_msgs[0]["CALX"], cal_msgs[0]["CALY"]) == ("0.50000", "0.50000")
        res = cal_msgs[-1]
        assert all(res[f"LV{k}"] == "1" and res[f"RV{k}"] == "1" for k in range(1, 6))
        # 24" test screen: points are placed inside the centred 22" area and the
        # CAL records give their true screen positions
        area = SCREEN.certified_area()
        x2, y2 = area.to_screen(0.85, 0.15)
        assert float(cal_msgs[2]["CALX"]) == pytest.approx(x2 / SCREEN.width, abs=1e-4)
        assert float(res["CALX2"]) == pytest.approx(x2 / SCREEN.width, abs=1e-4)
        assert abs(float(res["LX2"]) - x2 / SCREEN.width) < 0.05
        assert abs(float(res["RY2"]) - y2 / SCREEN.height) < 0.05
        summary = c.get("CALIBRATE_RESULT_SUMMARY")
        assert summary["VALID_POINTS"] == "5" and float(summary["AVE_ERROR"]) < 60

        # after calibration: the subject looks at an unseen point
        cam.look = (0.3, 0.7)
        time.sleep(0.4)
        c.set("ENABLE_SEND_POG_BEST", STATE=1)
        c.set("ENABLE_SEND_POG_FIX", STATE=1)
        c.set("ENABLE_SEND_DATA", STATE=1)
        recs = []
        for r in c.records():
            recs.append(r)
            if len(recs) >= 30:
                break
    r = recs[-1]
    assert r["BPOGV"] == "1"
    assert abs(float(r["BPOGX"]) - 0.3) < 0.05 and abs(float(r["BPOGY"]) - 0.7) < 0.05
    assert r["FPOGV"] == "1" and abs(float(r["FPOGX"]) - 0.3) < 0.05


def test_gazepoint_cc0_demo_style_client(server):
    """The parsing approach of Gazepoint's public-domain Python demo (raw
    recv, split on spaces, find the field name) must work against us."""
    gt, srv, cam = server
    s = socket.create_connection(("127.0.0.1", srv.port), timeout=3)
    s.send(b'<SET ID="ENABLE_SEND_CURSOR" STATE="1" />\r\n')
    s.send(b'<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n')
    s.send(b'<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n')
    seen = set()
    t_end = time.time() + 3
    while time.time() < t_end and not {"FPOGX", "CX"} <= seen:
        data = s.recv(1024).decode()
        for el in data.split(" "):
            for name in ("FPOGX", "FPOGY", "FPOGV", "CX", "CY"):
                if el.startswith(name + "="):
                    float(el.split('"')[1])
                    seen.add(name)
    s.close()
    assert {"FPOGX", "CX"} <= seen


def test_user_data_one_shot(server):
    gt, srv, cam = server
    with OpenGazeClient(port=srv.port) as c:
        c.set("ENABLE_SEND_USER_DATA", STATE=1)
        c.set("ENABLE_SEND_DATA", STATE=1)
        c.set("USER_DATA", VALUE="ONCE", DUR=1)
        users = []
        for r in c.records():
            users.append(r["USER"])
            if len(users) >= 15:
                break
    assert users.count("ONCE") == 1


def test_default_points_option(server):
    gt, srv, cam = server
    from opengp3.calibration import API_DEFAULT_POINTS, NINE_POINTS
    assert srv.cal_points == NINE_POINTS and NINE_POINTS[0] == (0.5, 0.5)
    five = OpenGazeServer(gt, port=0, default_points=API_DEFAULT_POINTS)
    c = _Client(sock=None, addr=None)
    five.cal_points = []
    assert 'PTS="5"' in five.handle(c, "SET", "CALIBRATE_RESET", {})[0]
    five.stop()
