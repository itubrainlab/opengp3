# OpenGP3

A cross-platform (macOS, Linux, Windows) Python library and application for the **Gazepoint GP3** eye
tracker. It drives the camera with its own user-space driver (libusb) and does its own pupil/glint
detection, calibration, fixation and blink detection. It also serves the **Open Gaze API**, the TCP/XML
protocol Gazepoint's software uses, so existing Gazepoint clients (Gazepoint's demo programs, its LSL
bridge, PsychoPy/OpenSesame/MATLAB integrations) can use OpenGP3 in place of Gazepoint Control.

```python
from opengp3 import GazeTracker

with GazeTracker() as gt:                 # screen size (and physical width) detected from the OS
    gt.calibrate(points=9)                # fullscreen calibration (run on the main thread)
    print(gt.validate().report())         # accuracy / precision in degrees
    for s in gt.gaze(duration=10):        # every frame, ~60 Hz
        if s.valid:
            print(s.x, s.y, s.fixation.valid, s.metrics["R"].pupil_mm)
```

## Install

```bash
pip install git+https://github.com/itubrainlab/opengp3          # latest
pip install git+https://github.com/itubrainlab/opengp3@v0.2.0   # a release
```

libusb comes bundled (via `libusb-package`). USB access per OS:

| OS | Setup |
|---|---|
| macOS | nothing |
| Linux | automatic: the first time OpenGP3 is denied access to the GP3, it installs a udev rule after one administrator password prompt (pkexec in a desktop session, sudo in a terminal) and continues. `opengp3 setup` does this explicitly; set `OPENGP3_NO_AUTO_SETUP=1` to disable the automatic prompt (e.g. for services) |
| Windows | bind the **WinUSB** driver to the GP3 camera with [Zadig](https://zadig.akeo.ie) ("USB 2.0 Digital Imaging Camera", `1E10 2001`). Gazepoint's own driver can't be used through libusb, so Gazepoint Control can't use the camera while WinUSB is bound (Zadig/Device Manager can switch back). |

Requirements: Python ≥ 3.9. Only one program can use the GP3 at a time. Mouse/keyboard data and the
gaze pointer use CoreGraphics (macOS), user32 (Windows) or X11 (Linux). Under Wayland these features
report "not available", because Wayland doesn't allow global pointer access.

From source, for development:

```bash
git clone https://github.com/itubrainlab/opengp3 && cd opengp3
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest               # offline tests, no hardware needed
```

## Command line

```bash
opengp3 server                                 # Open Gaze API server (port 4242) + tracker window
opengp3 view                                   # live IR image, eye overlay, head-position hints
opengp3 calibrate --validate --plot accuracy.png                    # full screen, 9 points
opengp3 calibrate --points 13 --area-diag-in 22                     # centred 22" area
opengp3 validate  --calibration calibration.json
opengp3 track     --calibration calibration.json --csv gaze.csv --duration 60
opengp3 track     --calibration calibration.json --show             # fullscreen gaze cursor
opengp3 track     --calibration calibration.json --pointer          # mouse follows your gaze
opengp3 record eyes.csv --duration 10          # raw pupil/glint features per frame
opengp3 marker marker.png --mm 11              # printable marker for pupil size in mm
opengp3 info | snapshot eye.png
opengp3 setup                                  # Linux: one-time USB permission rule (also automatic)
```

- **22" limit:** the GP3 is specified for screens up to 22". On a larger display, calibration and
  validation targets go in a centred 22" (16:9) area by default. This applies to the Python API,
  `calibrate`/`accuracy` and API-driven calibration in `server`. `--full-screen` (or
  `GazeTracker(max_diagonal_in=None)`) overrides it; `--area-diag-in` picks another size.
- The screen's size and physical width come from the OS. `--screen-width-cm` and `--distance-cm`
  (default 65) override them; the physical width is needed for the 22" limit and for degrees.
  `--display N` selects a monitor.
- `--points` can be 5, 9, 13, 16 or 25. More points model the screen edges better.
- `--dump samples.csv` (on `calibrate`) writes every labelled sample, for diagnostics.

### `opengp3 server`: Gazepoint Control replacement

```bash
opengp3 server [--host 0.0.0.0] [--port 4242] [--monocular R|L] [--auto-gain] [--media DIR]
```

- **API:** serves Open Gaze API v2.8 on port 4242. Clients choose fields with `ENABLE_SEND_*`, stream
  `REC` records at 60 Hz, run calibration via `CALIBRATE_*` and receive `CAL` records. With
  `CALIBRATE_SHOW=0`, a client can draw its own targets and follow the server's timing.
- **Large screens:** on a display over 22", the calibration points (default or `CALIBRATE_ADDPOINT`)
  are placed inside the centred 22" area. The `CAL` records report their actual screen positions, so
  client-drawn targets land in the right place.
- **Tracker window keys:** `c` calibrate, `p` gaze pointer, `g` gain sweep for hard-to-track eyes,
  `q` quit. The window shows a close/far head-distance bar.
- **Calibration window keys:** `c` recalibrate, `5`/`9` points, `1` one-point drift correction,
  `u` undo (toggles the previous calibration), `s` tone per point, `+`/`-`/`d` speed.
- **Custom targets:** `--media DIR` uses `CalibPoint<N>.png` images and `.mp3` sounds as targets.
- **Persistence:** the last calibration is saved to `~/.opengp3/calibration.json` and reloaded.
- **Library use:** `OpenGazeServer(tracker).start()` serves a `GazeTracker` from your own code, and
  `OpenGazeClient` is a small client for any Open Gaze API server.

Tested against Gazepoint's own public-domain Python demo client (`demo/python/GazepointAPI.py`),
unmodified.

## Python API

| Object | What it does |
|---|---|
| `GazeTracker(screen, calibration=None, smoothing=True)` | Owns the camera and runs detection on every frame in a background thread. `start()`/`stop()` or `with`. |
| `.calibrate(points=9, area=None)` | Fullscreen calibration: a head-positioning screen, then the targets. Targets with too little data are retried. Returns and stores a `Calibration`. |
| `.validate(points=VALIDATION_POINTS)` | Accuracy on points the calibration didn't use. Returns a `ValidationResult` (`.accuracy_deg`, `.precision_rms_s2s_deg`, `.report()`). |
| `.gaze(duration)` | Iterator over every `GazeSample`. Also `.latest()`, `.subscribe()` (queue), `.add_callback(fn)`. |
| `.show_gaze(duration)` | Fullscreen gaze-cursor demo. |
| `GazeSample` | `t` (capture time), `frame`, `x`/`y` (smoothed), `raw_x`/`raw_y`, `per_eye`, `eyes` (pupil/glint features), `valid`. |
| `Calibration` | Per-eye mapping. `.save(path)`, `Calibration.load(path)`, `.predict(eyes)`, `.summary()`. |
| `Screen` / `Rect` | Screen geometry in points. `Screen.area(22)` gives a centred 22" area; `angle_deg(p, q)`. |
| `OneEuroFilter` | The gaze smoothing filter (pass your own to `GazeTracker(smoothing=...)` to tune it). |
| `GP3Camera`, `detect_eyes`, `EyeTracker` | Low-level access: registers, raw frames, IR LEDs, per-frame eye features. |

Notes:
- **OpenCV windows must run on the main thread on macOS.** `calibrate`, `validate` and `show_gaze` use
  one shared fullscreen window per session. The window checks that it really covers the screen before
  showing targets; a window that isn't truly fullscreen draws targets in the wrong place and silently
  ruins calibration.
- Samples are matched to targets by *capture* time, so a slow display loop can't mislabel data.
- Gaze is the mean of the calibrated eyes that are visible. Each eye is calibrated separately, so
  tracking continues when one eye leaves the camera's view.
- Calibration is valid for roughly the head position it was made in. Recalibrate if you move a lot.

## Feature equivalence with Gazepoint Control

Derived from Gazepoint's published documentation (Control and Open Gaze API manuals, release notes).
OpenGP3 is an independent implementation, not derived from Gazepoint's code. Where Gazepoint documents
how something is computed, OpenGP3 does the same:
- best POG = mean of the valid eyes;
- AAC POG = moving average of fixation data, default 15 samples;
- time tick = OpenCV tick count;
- pupil scale = 1 at calibration depth;
- the default 5-point calibration's positions, order and timing.

Undocumented internals (pupil detection, gaze mapping, fixation filter, eye model) are OpenGP3's own,
so equivalence there is behavioural, measured on the results below.

| Feature | Status |
|---|---|
| Camera, 60 Hz, IR illuminators | ✅ (752×480 full sensor) |
| Open Gaze API server: all configuration commands, `REC`/`CAL` records, per-client field selection | ✅ API 2.8 |
| Calibration: 5/9 points (also 13/16/25, custom points via API), animation/collection timing, results per point, summary | ✅ |
| Client-drawn calibration (`CALIBRATE_SHOW=0`) | ✅ |
| One-point calibration, undo, tone, speed keys, custom target images/sounds | ✅ |
| Gaze: left / right / best POG, fixation POG (start, duration, ID, valid), AAC POG + `AAC_FILTER` | ✅ |
| Pupil: image position, diameter (px), depth scale, diameter (mm), 3-D eye position | ✅ (mm and metres via an IPD model, about ±10 % absolute unless the IPD is measured) |
| Blink ID, duration, blinks per minute | ✅ |
| Cursor position + button states, keystrokes, `USER_DATA` (incl. one-shot) | ✅ |
| Time, counter, time tick + frequency, device info (`PRODUCT_ID`, `SERIAL_ID`, `CAMERA_SIZE`, …) | ✅ |
| Marker tracking for real-world scale (`MARKER_PIX`, `PIX` records), IPD measurement | ✅ uses a printable **ArUco** marker (`opengp3 marker`), not Gazepoint's marker design |
| Gaze pointer (mouse follows gaze) | ✅ |
| Screen selection (`SCREEN_SIZE`, `--display`) | ✅ (multi-monitor untested: one display available) |
| Monocular tracking (force L/R), automatic gain sweep, head-distance feedback | ✅ |
| Biometrics: GSR, heart rate, dial, TTL I/O | ❌ separate Gazepoint hardware with an undocumented protocol. Fields are sent as invalid; `TTL_WRITE` returns NACK |
| Multiple eye trackers, automatic switching | ❌ one tracker (`TRACKER_ID` reports 1) |
| Mobile-device calibration web page, Windows Eye Control, tray icon | ❌ not applicable (Windows/app-specific) |

Live check through the API only (34" ultrawide, 5-point default calibration on the *full* screen, run
before the 22" default existed):
- 61.2 records/s with 0 counter gaps;
- both eyes calibrated;
- fixation valid 91 % of the time while viewing; 16 fixations in 6 s, median 0.30 s;
- pupil diameter about 3.0–3.3 mm; eye depth 0.65 m.

## Results on this unit

34" LG UltraWide (3440×1440, ~80 cm wide), ~65 cm viewing distance, head free (no chin rest), unfiltered
60 Hz gaze. Validation always uses 8 points the calibration never saw.

| Calibration | Validation accuracy mean / median / worst | Precision RMS-S2S |
|---|---|---|
| **Library, 13 points, 22" area** (`opengp3 calibrate --points 13 --area-diag-in 22`) | **0.35° / 0.28° / 0.63°** (≈ 0.4 cm) | 0.50° |
| Earlier script, 9 points, 22" area | 0.58° / 0.61° / 0.83° | 0.62° |
| Earlier script, 9 points, full 34" ultrawide | 1.07° / 0.95° / 2.01° | 0.81° |

- **Full ultrawide:** its edges are about ±26° off-centre, beyond the GP3's range. The glint is lost at
  the outer points and errors grow toward the sides. On a screen this wide, calibrate a smaller area
  (`--area-diag-in 22`), or use `--points 16/25` for the full screen.
- **Speed:** detection takes about 9 ms per frame. The pipeline delivers 60.7 Hz with 0 dropped frames
  and about 8 ms from frame capture to gaze sample.

## How it works

**Hardware.** Inside, the GP3 is a **Point Grey Firefly MV FMVU-03MTM** USB 2.0 camera (`1e10:2001`).
macOS reports it as "USB 2.0 Digital Imaging Camera". It speaks IIDC/DCAM tunnelled over USB, the same
mapping as libdc1394's USB backend:

| What | USB transfer |
|---|---|
| Register read / write | control, `bmRequestType=0xC0` / `0x40`, `bRequest=0x7F`, `wValue/wIndex` = address low/high 16 bits, little-endian quadlets |
| Image data | bulk IN `0x81`. A continuous stream without frame markers; frames stay aligned because each read is exactly one frame after `ISO_EN`. A reader thread keeps a transfer pending, otherwise the camera drops frames. |
| IR illuminators | Point Grey **strobe 2, active-low** (found empirically) |

The sensor is 752×480, which is also the maximum resolution. The standard 640×480 mode is a centre crop,
so the full sensor (Format 7) is used by default for 17% more field of view. The lens is fixed. At
normal distance the eyes are about 270 px apart, giving about 16 cm of sideways head movement with one
eye visible, or 10 cm with both.

**Eye detection** works from the glints first. It finds the GP3's pair of LED reflections on a dark
pupil, inpaints them, segments the pupil with an edge-scored threshold sweep, and fits an ellipse. The
pupil–glint vector (PCR) is the gaze signal. `EyeTracker` keeps the subject's R/L eye identity across
frames when one eye is lost.

**Calibration** maps each eye's median PCR per target to the target position. The model (affine,
x²/y², or quadratic) is chosen per eye by leave-one-out error on the calibration points, so an eye that
lost some points gets a simpler model instead of being dropped.

## Alternative: Gazepoint's own software

To use Gazepoint's own calibrated gaze output instead, run Gazepoint Control on a Windows PC and read
the Open Gaze API (XML over TCP port 4242) from the Mac. A Windows VM is unreliable for this because it
needs USB passthrough.

## Linux and Windows status

The driver, detection, calibration and API server are pure Python on top of libusb and OpenCV.
The unit tests run on macOS, Linux and Windows in CI. Hardware tests so far have been on macOS only.

## License

MIT © Paolo Burelli. OpenGP3 is not affiliated with or endorsed by Gazepoint.
