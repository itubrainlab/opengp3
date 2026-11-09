"""``opengp3`` command line.

    opengp3 info                                  camera registers / modes
    opengp3 view                                  live IR image + eye overlay
    opengp3 snapshot eye.png                      save one frame
    opengp3 record eyes.csv [--duration 10]       raw pupil/glint per frame
    opengp3 calibrate --save cal.json [--points 9] [--area-diag-in 22] [--validate]
    opengp3 validate --calibration cal.json
    opengp3 track --calibration cal.json [--csv gaze.csv] [--show]
    opengp3 accuracy [--points 9] [--area-diag-in 22] [--plot result.png]
    opengp3 server [--port 4242] [--host 0.0.0.0]  Open Gaze API server + control window
    opengp3 marker marker.png [--mm 11]            printable marker for pupil size in mm
    opengp3 setup                                  Linux: one-time USB permission rule

Screen options (all gaze commands): --screen-width-cm (needed for degrees),
--distance-cm (default 65).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time

import cv2
import numpy as np

from .calibration import Calibration, CalibrationError, ValidationResult
from .camera import GP3Camera, GP3Error
from .detection import EyeTracker, detect_eyes
from .display import Cancelled
from .screen import Screen
from .tracker import GazeTracker


# ------------------------------------------------------------------ camera
def _configure(cam: GP3Camera, a):
    if a.crop:
        cam.set_video_mode(mode=5, rate=a.rate)  # 640x480 centre crop
    else:
        cam.set_full_sensor()  # 752x480, widest field of view
    if a.shutter is not None:
        cam.set_feature("shutter", a.shutter)
    if a.gain is not None:
        cam.set_feature("gain", a.gain)
    cam.set_leds(not a.no_leds)
    cam.start_stream()


def cmd_info(a):
    with GP3Camera.open() as cam:
        print(json.dumps(cam.info(), indent=2, default=str))


def cmd_view(a):
    """Live camera view. Eye detection runs on every camera frame in the
    tracker's background thread; this loop only draws the newest frame, so
    slow drawing never costs tracking frames."""
    from .control import render_tracker_view
    gt = GazeTracker(Screen(1, 1), smoothing=False, input_monitor=False,
                     full_sensor=not a.crop)
    with gt:
        cam = gt.camera
        if a.shutter is not None:
            cam.set_feature("shutter", a.shutter)
        if a.gain is not None:
            cam.set_feature("gain", a.gain)
        if a.no_leds:
            cam.set_leds(False)
        win = "GP3 (q / Esc / close to quit)"
        last_frame, shown, t0, disp_fps = -1, 0, time.time(), 0.0
        seq0, t_cam = None, time.time()
        try:
            while True:
                img, s = gt.latest_frame()
                if img is not None and s is not None and s.frame != last_frame:
                    last_frame = s.frame
                    shown += 1
                    now = time.time()
                    if now - t0 >= 1:
                        disp_fps, shown, t0 = shown / (now - t0), 0, now
                    if seq0 is None:
                        seq0, t_cam = s.frame, now
                    cam_fps = (s.frame - seq0) / max(1e-6, now - t_cam)
                    if now - t_cam > 2:  # sliding estimate
                        seq0, t_cam = s.frame, now
                    vis = render_tracker_view(img, s, gt.fps, [
                        f"camera {cam_fps:.1f} fps   tracking {gt.fps:.1f} fps   "
                        f"display {disp_fps:.1f} fps   dropped {gt.dropped_frames}"])
                    cv2.imshow(win, vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1 and last_frame >= 0:
                        break
                except cv2.error:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def cmd_snapshot(a):
    with GP3Camera.open() as cam:
        _configure(cam, a)
        for _ in range(10):  # let exposure settle
            _, _, img = cam.grab()
        cv2.imwrite(a.output, img)
        print(f"saved {img.shape} frame to {a.output} (mean {img.mean():.1f})")


def cmd_record(a):
    with GP3Camera.open() as cam:
        _configure(cam, a)
        tracker = EyeTracker()
        i, seq = 0, 0
        t_end = time.time() + a.duration if a.duration else float("inf")
        with open(a.output, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["t", "frame", "eye", "pupil_x", "pupil_y", "pupil_r", "glint_x",
                         "glint_y", "pcr_x", "pcr_y", "confidence"])
            print(f"recording to {a.output} (Ctrl-C to stop)")
            try:
                while time.time() < t_end:
                    seq, t, img = cam.grab()
                    for side, e in sorted(tracker.update(detect_eyes(img)).items()):
                        wr.writerow([f"{t:.4f}", seq, side, f"{e.pupil_x:.2f}", f"{e.pupil_y:.2f}",
                                     f"{e.pupil_r:.2f}", f"{e.glint_x:.2f}", f"{e.glint_y:.2f}",
                                     f"{e.pcr[0]:.3f}", f"{e.pcr[1]:.3f}", f"{e.confidence:.2f}"])
                    i += 1
            except KeyboardInterrupt:
                pass
        print(f"{i} frames processed, last camera frame {seq}")


# ------------------------------------------------------------------ gaze
def _screen(a) -> Screen:
    return Screen.detect(width_cm=a.screen_width_cm, distance_cm=a.distance_cm,
                         display=getattr(a, "display", 0))


def _area(a, screen: Screen):
    """Explicit --area-diag-in, else None (tracker default: whole screen up to
    22", centred 22" area on larger screens, unless --full-screen)."""
    return screen.area(a.area_diag_in, _aspect(a.area_aspect)) if a.area_diag_in else None


def _max_diag(a):
    return None if getattr(a, "full_screen", False) else 22.0


def _report_area(gt: GazeTracker, area=None):
    r = area or gt.default_area()
    sc = gt.screen
    if (r.w, r.h) != (sc.width, sc.height):
        print(f"calibrating a centred {r.w:.0f}x{r.h:.0f} area of the {sc.width}x{sc.height} screen "
              f"(screen is {sc.diagonal_in:.0f}\"; the GP3 is specified up to 22\"; "
              "--full-screen to override)", flush=True)


def _aspect(s: str) -> float:
    w, h = (float(v) for v in s.split(":"))
    return w / h


def plot_validation(result: ValidationResult, cal: Calibration, path: str, width: int = 1600):
    sc = result.screen
    s = width / sc.width
    c = np.full((int(sc.height * s), width, 3), 30, np.uint8)
    a = cal.area
    cv2.rectangle(c, (int(a.x * s), int(a.y * s)), (int((a.x + a.w) * s), int((a.y + a.h) * s)),
                  (70, 70, 70), 1)
    for p in result.points:
        for gx, gy in p.samples:
            cv2.circle(c, (int(gx * s), int(gy * s)), 1, (255, 180, 80), -1)
        t, m = (int(p.x * s), int(p.y * s)), (int(p.gaze_x * s), int(p.gaze_y * s))
        cv2.line(c, t, m, (0, 200, 255), 1)
        cv2.circle(c, t, 6, (255, 255, 255), 1)
        cv2.circle(c, m, 4, (0, 200, 255), -1)
        label = f"{p.accuracy_deg:.1f}" if p.accuracy_deg is not None else f"{p.offset:.0f}pt"
        cv2.putText(c, label, (t[0] + 8, t[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
    cv2.putText(c, "white ring = target, orange = median gaze, blue = samples",
                (10, c.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.imwrite(path, c)


def dump_samples(gt: GazeTracker, path: str):
    """All labelled calibration/validation samples, for diagnostics."""
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["phase", "target", "target_x", "target_y", "t", "eye", "pupil_x", "pupil_y",
                     "pupil_r", "pcr_x", "pcr_y", "gaze_x", "gaze_y"])
        for phase, rows in (("calibration", gt.last_calibration_samples),
                            ("validation", gt.last_validation_samples)):
            for i, x, y, s in rows:
                for side, e in sorted(s.eyes.items()):
                    g = s.per_eye.get(side, (None, None))
                    wr.writerow([phase, i, f"{x:.1f}", f"{y:.1f}", f"{s.t:.4f}", side,
                                 f"{e.pupil_x:.2f}", f"{e.pupil_y:.2f}", f"{e.pupil_r:.2f}",
                                 f"{e.pcr[0]:.3f}", f"{e.pcr[1]:.3f}",
                                 "" if g[0] is None else f"{g[0]:.1f}",
                                 "" if g[1] is None else f"{g[1]:.1f}"])


def cmd_calibrate(a):
    screen = _screen(a)
    with GazeTracker(screen, max_diagonal_in=_max_diag(a)) as gt:
        _report_area(gt, _area(a, screen))
        try:
            cal = gt.calibrate(points=a.points, area=_area(a, screen))
            print(cal.summary())
            if a.save:
                cal.save(a.save)
                print(f"calibration saved to {a.save}")
            if a.validate or a.plot:
                res = gt.validate()
                print("\nvalidation:\n" + res.report())
                if a.plot:
                    plot_validation(res, cal, a.plot)
                    print(f"plot saved to {a.plot}")
            if a.show:
                gt.show_gaze(duration=a.show)
        finally:
            if a.dump:
                dump_samples(gt, a.dump)
                print(f"samples written to {a.dump}")


def cmd_validate(a):
    cal = Calibration.load(a.calibration)
    screen = _screen(a)
    with GazeTracker(screen, calibration=cal) as gt:
        res = gt.validate()
        print(res.report())
        if a.plot:
            plot_validation(res, cal, a.plot)
            print(f"plot saved to {a.plot}")


def cmd_track(a):
    cal = Calibration.load(a.calibration)
    screen = _screen(a)
    with GazeTracker(screen, calibration=cal, smoothing=not a.raw) as gt:
        gt.gaze_pointer = a.pointer
        if a.show:
            if a.csv:
                _record_gaze_bg(gt, a)
            gt.show_gaze(duration=a.duration or None)
            return
        _record_gaze(gt, a)


def _gaze_row(s):
    pe = s.per_eye
    f = lambda v: "" if v is None else f"{v:.1f}"
    return [f"{s.t:.4f}", s.frame, int(s.valid), f(s.x), f(s.y), f(s.raw_x), f(s.raw_y),
            f(pe.get("R", (None,))[0]), f(pe.get("R", (None, None))[1]),
            f(pe.get("L", (None,))[0]), f(pe.get("L", (None, None))[1]),
            "".join(sorted(s.eyes))]


GAZE_HEADER = ["t", "frame", "valid", "x", "y", "raw_x", "raw_y", "right_x", "right_y",
               "left_x", "left_y", "eyes"]


def _record_gaze(gt: GazeTracker, a):
    fh = open(a.csv, "w", newline="") if a.csv else None
    wr = csv.writer(fh) if fh else None
    if wr:
        wr.writerow(GAZE_HEADER)
    n = valid = 0
    try:
        for s in gt.gaze(duration=a.duration or None):
            n += 1
            valid += s.valid
            if wr:
                wr.writerow(_gaze_row(s))
            elif n % 6 == 0:  # ~10 Hz to the console
                print(f"{s.t:.3f}  " + (f"x={s.x:7.1f}  y={s.y:7.1f}  eyes={''.join(sorted(s.eyes))}"
                                        if s.valid else "-- no gaze --"), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()
    print(f"{n} samples, {100 * valid / max(n, 1):.1f}% valid" + (f", written to {a.csv}" if fh else ""))


def _record_gaze_bg(gt: GazeTracker, a):
    """CSV logging from the tracking thread while the main thread shows the cursor."""
    fh = open(a.csv, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(GAZE_HEADER)
    gt.add_callback(lambda s: wr.writerow(_gaze_row(s)))
    import atexit
    atexit.register(fh.close)


def cmd_accuracy(a):
    a.validate, a.save = True, a.save
    cmd_calibrate(a)


def cmd_server(a):
    from .control import ControlApp, default_calibration_path
    from .server import OpenGazeServer
    screen = _screen(a)
    path = default_calibration_path()
    cal = None
    src = a.calibration or (str(path) if path.exists() else None)
    if src:
        try:
            cal = Calibration.load(src)
            print(f"loaded calibration {src}", flush=True)
        except Exception as e:
            print(f"could not load calibration {src}: {e}")
    gt = GazeTracker(screen, calibration=cal, monocular=a.monocular, auto_gain=a.auto_gain,
                     aac_length=a.aac_filter, max_diagonal_in=_max_diag(a))
    from .calibration import API_DEFAULT_POINTS, NINE_POINTS
    with gt:
        _report_area(gt)
        srv = OpenGazeServer(gt, a.host, a.port,
                             default_points=API_DEFAULT_POINTS if a.points == 5 else NINE_POINTS).start()
        lsl = None
        if a.lsl:
            from .lsl import LSLStreamer
            lsl = LSLStreamer(gt)
            print("LSL streams: 'OpenGP3 Gaze' (60 Hz) and 'OpenGP3 Markers'", flush=True)
        print(f"Open Gaze API server listening on {a.host}:{srv.port}  "
              f"(screen {screen.width}x{screen.height}"
              + (f", {screen.width_cm:.1f} cm" if screen.width_cm else "") + ")", flush=True)
        try:
            ControlApp(gt, srv, show_tracker=not a.no_display, media_dir=a.media,
                       mirror=not a.no_mirror, lsl=lsl).loop()
        except KeyboardInterrupt:
            pass
        finally:
            if lsl is not None:
                lsl.close()
            srv.stop()


def cmd_marker(a):
    from .eyemodel import MarkerTracker
    img = MarkerTracker.image(a.mm, a.dpi)
    cv2.imwrite(a.output, img)
    print(f"{a.output}: ArUco 4x4 id 0, black square {a.mm} mm at {a.dpi} dpi "
          f"({img.shape[1]}x{img.shape[0]} px incl. white border). Print at 100% scale, measure "
          "the black square and use that size (laser printers: inkjet ink can be invisible in IR).")


def cmd_setup(a):
    """One-time OS setup for USB access (Linux: udev rule)."""
    import sys as _sys
    from . import permissions
    if not _sys.platform.startswith("linux"):
        if _sys.platform == "win32":
            print("Windows: bind the WinUSB driver to the GP3 camera with Zadig "
                  "(https://zadig.akeo.ie, 'USB 2.0 Digital Imaging Camera', 1E10 2001).")
        else:
            print("No setup needed on this system.")
        return
    if a.print_only:
        print(permissions.manual_instructions())
        return
    if permissions.rule_installed() and not a.force:
        print(f"Already set up ({permissions.RULE_PATH}). If access is still denied, replug the GP3.")
        return
    permissions.install()


# ------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(prog="opengp3", description="Gazepoint GP3 driver and gaze tracker")
    sub = p.add_subparsers(dest="cmd", required=True)

    def cam_opts(s):
        s.add_argument("--crop", action="store_true",
                       help="640x480 centre crop instead of the full 752x480 sensor")
        s.add_argument("--rate", type=int, default=5, help="IIDC rate code with --crop (4=30, 5=60 fps)")
        s.add_argument("--shutter", type=int)
        s.add_argument("--gain", type=int)
        s.add_argument("--no-leds", action="store_true", help="leave the IR illuminators off")

    def screen_opts(s):
        s.add_argument("--screen-width-cm", type=float,
                       help="physical screen width (default: as reported by the OS)")
        s.add_argument("--distance-cm", type=float, default=65.0, help="eye-screen distance")
        s.add_argument("--display", type=int, default=0, help="display index (0 = main)")

    def area_opts(s):
        s.add_argument("--points", type=int, default=9, choices=[5, 9, 13, 16, 25])
        s.add_argument("--full-screen", action="store_true",
                       help="use the whole screen even if it is larger than 22\" (beyond the GP3 spec)")
        s.add_argument("--area-diag-in", type=float, default=0,
                       help="calibrate a centred area of this monitor diagonal (inches) "
                            "instead of the full screen; needs --screen-width-cm")
        s.add_argument("--area-aspect", default="16:9")

    sub.add_parser("info", help="camera registers and modes").set_defaults(fn=cmd_info)
    s = sub.add_parser("view", help="live IR view with eye overlay"); cam_opts(s); s.set_defaults(fn=cmd_view)
    s = sub.add_parser("snapshot", help="save one frame"); s.add_argument("output"); cam_opts(s)
    s.set_defaults(fn=cmd_snapshot)
    s = sub.add_parser("record", help="record pupil/glint features to CSV"); s.add_argument("output")
    s.add_argument("--duration", type=float, default=0); cam_opts(s); s.set_defaults(fn=cmd_record)

    s = sub.add_parser("calibrate", help="fullscreen calibration")
    screen_opts(s); area_opts(s)
    s.add_argument("--save", default="calibration.json")
    s.add_argument("--validate", action="store_true", help="run validation afterwards")
    s.add_argument("--plot", help="save a validation plot (implies --validate)")
    s.add_argument("--show", type=float, default=0, metavar="SEC",
                   help="afterwards, show the live gaze cursor for SEC seconds")
    s.add_argument("--dump", help="write all labelled calibration/validation samples to CSV")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("validate", help="accuracy of a saved calibration")
    screen_opts(s); s.add_argument("--calibration", default="calibration.json")
    s.add_argument("--plot"); s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("track", help="stream gaze with a saved calibration")
    screen_opts(s); s.add_argument("--calibration", default="calibration.json")
    s.add_argument("--csv", help="write every sample to this CSV")
    s.add_argument("--duration", type=float, default=0, help="seconds (0 = until Ctrl-C / Esc)")
    s.add_argument("--show", action="store_true", help="fullscreen gaze cursor")
    s.add_argument("--raw", action="store_true", help="disable smoothing")
    s.add_argument("--pointer", action="store_true", help="move the mouse cursor with your gaze")
    s.set_defaults(fn=cmd_track)

    s = sub.add_parser("accuracy", help="calibrate + validate + report")
    screen_opts(s); area_opts(s)
    s.add_argument("--save", default="calibration.json")
    s.add_argument("--plot", default="accuracy.png")
    s.add_argument("--show", type=float, default=0, metavar="SEC")
    s.add_argument("--dump")
    s.set_defaults(fn=cmd_accuracy)

    s = sub.add_parser("server", help="Open Gaze API server with tracker display and calibration")
    screen_opts(s)
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept remote clients")
    s.add_argument("--port", type=int, default=4242)
    s.add_argument("--calibration", help="calibration to load (default: the last one)")
    s.add_argument("--no-display", action="store_true", help="start without the tracker window")
    s.add_argument("--monocular", choices=["R", "L"], help="track only this eye")
    s.add_argument("--auto-gain", action="store_true", help="gain sweep when no eyes are found")
    s.add_argument("--aac-filter", type=int, default=15, help="AAC moving-average length")
    s.add_argument("--media", help="folder with CalibPoint<N>.png/.mp3 calibration targets")
    s.add_argument("--full-screen", action="store_true",
                   help="calibrate on the whole screen even if it is larger than 22\"")
    s.add_argument("--points", type=int, choices=[5, 9], default=9,
                   help="default calibration (also restored by CALIBRATE_RESET)")
    s.add_argument("--lsl", action="store_true",
                   help="stream gaze and markers on Lab Streaming Layer (needs opengp3[lsl])")
    s.add_argument("--no-mirror", action="store_true", help="show the camera image unmirrored")
    s.set_defaults(fn=cmd_server)

    s = sub.add_parser("marker", help="printable marker for real-world pupil size")
    s.add_argument("output")
    s.add_argument("--mm", type=float, default=11.0)
    s.add_argument("--dpi", type=int, default=600)
    s.set_defaults(fn=cmd_marker)

    for name in ("setup", "udev"):
        s = sub.add_parser(name, help="one-time USB permission setup (Linux udev rule)")
        s.add_argument("--print-only", action="store_true", help="only print the manual commands")
        s.add_argument("--force", action="store_true", help="reinstall the rule")
        s.set_defaults(fn=cmd_setup)

    a = p.parse_args(argv)

    try:
        a.fn(a)
    except Cancelled:
        print("cancelled")
        return 1
    except (GP3Error, CalibrationError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
