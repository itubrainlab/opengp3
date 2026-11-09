"""``opengp3`` command line.

    opengp3 info                                  camera registers / modes
    opengp3 view                                  live IR image + eye overlay
    opengp3 snapshot eye.png                      save one frame
    opengp3 record eyes.csv [--duration 10]       raw pupil/glint per frame
    opengp3 calibrate --save cal.json [--points 9] [--area-diag-in 22] [--validate]
    opengp3 validate --calibration cal.json
    opengp3 track --calibration cal.json [--csv gaze.csv] [--show]
    opengp3 accuracy [--points 9] [--area-diag-in 22] [--plot result.png]

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
    with GP3Camera.open() as cam:
        _configure(cam, a)
        t0, n, fps = time.time(), 0, 0.0
        win = "GP3 (q / Esc / close to quit)"
        tracker = EyeTracker()
        try:
            while True:
                _, _, img = cam.grab(latest=True)
                n += 1
                if time.time() - t0 >= 1:
                    fps, n, t0 = n / (time.time() - t0), 0, time.time()
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                eyes = tracker.update(detect_eyes(img))
                for side, e in eyes.items():
                    cv2.circle(vis, (int(e.pupil_x), int(e.pupil_y)), int(e.pupil_r) + 1,
                               (0, 255, 0), 1)
                    cv2.drawMarker(vis, (int(e.glint_x), int(e.glint_y)), (0, 0, 255),
                                   cv2.MARKER_CROSS, 8)
                    cv2.putText(vis, side, (int(e.pupil_x) - 5, int(e.pupil_y) - int(e.pupil_r) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                h, w = img.shape
                if len(eyes) == 2:
                    status, col = "both eyes", (0, 255, 0)
                elif eyes:
                    status, col = f"only {'right' if 'R' in eyes else 'left'} eye", (0, 200, 255)
                else:
                    status, col = "no eyes", (0, 0, 255)
                if eyes:
                    xs = [e.pupil_x for e in eyes.values()]
                    ys = [e.pupil_y for e in eyes.values()]
                    m, hint = 0.08, []
                    if min(xs) < m * w: hint.append("move head toward image right")
                    if max(xs) > (1 - m) * w: hint.append("move head toward image left")
                    if min(ys) < m * h: hint.append("lower head / tilt GP3 up")
                    if max(ys) > (1 - m) * h: hint.append("raise head / tilt GP3 down")
                    if hint:
                        status += " - " + ", ".join(hint)
                cv2.putText(vis, f"{fps:.1f} fps  {status}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, col, 1)
                cv2.imshow(win, vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()


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
    return Screen.detect(width_cm=a.screen_width_cm, distance_cm=a.distance_cm)


def _area(a, screen: Screen):
    return screen.area(a.area_diag_in, _aspect(a.area_aspect)) if a.area_diag_in else None


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
    with GazeTracker(screen) as gt:
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
        s.add_argument("--screen-width-cm", type=float, help="physical screen width (for degrees)")
        s.add_argument("--distance-cm", type=float, default=65.0, help="eye-screen distance")

    def area_opts(s):
        s.add_argument("--points", type=int, default=9, choices=[5, 9, 13, 16, 25])
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
    s.set_defaults(fn=cmd_track)

    s = sub.add_parser("accuracy", help="calibrate + validate + report")
    screen_opts(s); area_opts(s)
    s.add_argument("--save", default="calibration.json")
    s.add_argument("--plot", default="accuracy.png")
    s.add_argument("--show", type=float, default=0, metavar="SEC")
    s.add_argument("--dump")
    s.set_defaults(fn=cmd_accuracy)

    a = p.parse_args(argv)
    if getattr(a, "area_diag_in", 0) and not a.screen_width_cm:
        p.error("--area-diag-in needs --screen-width-cm")
    try:
        a.fn(a)
    except Cancelled:
        print("cancelled")
        return 1
    except (GP3Error, CalibrationError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
