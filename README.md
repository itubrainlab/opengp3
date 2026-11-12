# OpenGP3

A cross-platform (macOS, Linux, Windows) Python library and application for the **Gazepoint GP3** and
**GP3 HD** eye trackers. It drives the camera with its own user-space driver (libusb) and does its own pupil/glint
detection, calibration, fixation and blink detection. It also serves the **Open Gaze API**, the TCP/XML
protocol Gazepoint's software uses, so existing Gazepoint clients (Gazepoint's demo programs, its LSL
bridge, PsychoPy/OpenSesame/MATLAB integrations) can use OpenGP3 in place of Gazepoint Control.

Run `opengp3` to start **OpenGP3 Control**, a Qt application that replaces Gazepoint Control: tracker
display with head-positioning guidance, calibration and its results, gaze pointer, screen selection,
and the Open Gaze API server with Lab Streaming Layer output (see [OpenGP3 Control](#opengp3-control)).
From your own code, the library does the same:

```python
from opengp3 import GazeTracker

with GazeTracker(display=0) as gt:        # main screen; size and physical width from the OS
    gt.calibrate(points=9)                # fullscreen calibration (run on the main thread)
    print(gt.validate().report())         # accuracy / precision in degrees
    for s in gt.gaze(duration=10):        # every frame, ~60 Hz (GP3 HD: 150 Hz)
        if s.valid:
            print(s.x, s.y, s.fixation.valid, s.metrics["R"].pupil_mm)
```

The tracker model is detected automatically: a GP3 (USB 2) or a GP3 HD (USB 3, see
[GP3 HD](#gp3-hd)). Everything below works with both unless noted.

## Install

```bash
pip install git+https://github.com/itubrainlab/opengp3          # latest
pip install git+https://github.com/itubrainlab/opengp3@v0.4.1   # a release
```

libusb comes bundled (via `libusb-package`). USB access per OS:

| OS | Setup |
|---|---|
| macOS | nothing |
| Linux | automatic: the first time OpenGP3 is denied access to the GP3, it installs a udev rule after one administrator password prompt (pkexec in a desktop session, sudo in a terminal) and continues. `opengp3 setup` does this explicitly; set `OPENGP3_NO_AUTO_SETUP=1` to disable the automatic prompt (e.g. for services) |
| Windows | **Recommended:** install [UsbDk](https://github.com/daynix/UsbDk/releases) once (the x64 `.msi`, one administrator prompt). OpenGP3 uses it automatically. It leaves Gazepoint's camera driver installed and takes the camera over only while OpenGP3 has it open, so Gazepoint Control keeps working (just not at the same time). `OPENGP3_USBDK=0` turns it off; `opengp3 info` shows `usb_driver`. **Alternative:** bind the **WinUSB** driver to the camera with [Zadig](https://zadig.akeo.ie). GP3: "USB 2.0 Digital Imaging Camera", `1E10 2001`. GP3 HD: enable *Options > List All Devices*, untick *Ignore Hubs or Composite Parents*, and pick the composite device `1E10 3300` (not its "Interface 0"). This replaces Gazepoint's driver, so Gazepoint Control can't use the camera until you switch back (Zadig/Device Manager). |

Requirements: Python ≥ 3.9. PySide6 (Qt 6, for OpenGP3 Control), OpenCV, pylsl and libusb are installed
with OpenGP3. Only one program can use the eye tracker at a time. Mouse/keyboard data and the gaze
pointer use CoreGraphics (macOS), user32 (Windows) or X11 (Linux). Under Wayland these features report
"not available", because Wayland doesn't allow global pointer access.

From source, for development:

```bash
git clone https://github.com/itubrainlab/opengp3 && cd opengp3
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest               # offline tests, no hardware needed
```

## Command line

```bash
opengp3                                        # OpenGP3 Control: API server (port 4242), LSL, control window
opengp3 --display 1                            # ... tracking and calibrating on the second screen
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
- **Screen:** `--display N` picks the screen to track and calibrate on (0 = main; `python -c "import
  opengp3.platform as p; print(p.displays())"` lists them). Its size and physical width come from the
  OS: CoreGraphics on macOS, the monitor's EDID on Windows, XRandR on Linux with the EDID in
  `/sys/class/drm` as fallback (also under Wayland). Values an OS reports when it doesn't know (0, the
  EDID "16×9 cm" aspect placeholder, macOS's 72-dpi guess, sizes whose aspect ratio doesn't match the
  pixels) are ignored. `--screen-width-cm` and `--distance-cm` (default 65) override them; the
  physical width is needed for the 22" limit and for degrees.
- `--points` can be 5, 9, 13, 16 or 25. More points model the screen edges better.
- `--dump samples.csv` (on `calibrate`) writes every labelled sample, for diagnostics.

### OpenGP3 Control

`opengp3` without a command (or `opengp3 server`, `python -m opengp3`) runs the Open Gaze API server
with its control window, the counterpart of Gazepoint Control:

```bash
opengp3 [--display N] [--no-lsl] [--points 9|5] [--local-only] [--port 4242] [--monocular R|L]
        [--auto-gain] [--media DIR] [--no-mirror] [--hz 60|150] [--opencv-ui] [--simulate]
```

The window (Qt) has the tracker display on the left and these panels on the right:

| Panel | Contents |
|---|---|
| Tracker display | Camera image with pupils and glints, mirrored like a mirror (switchable), status and head hints; can be hidden (also via the API's `TRACKER_DISPLAY`) |
| Eye tracker | Model and serial, camera image size, frame rate (GP3 HD: 60 / 150 Hz), illumination, samples/s and dropped frames, eyes found, **head position box** (eyes in the camera's view, green inside the safe zone, orange near the edge) and **distance scale** (good 55-75 cm) |
| Calibration | Points (5, 9, 13, 16, 25), speed, tone per point, custom target images; **Calibrate** / Stop, **1-point** drift correction, **Undo** (toggle with the previous calibration), **Validate** (accuracy and precision report), **Show results**; summary per eye (points, model, error in px and degrees) and a table of every point's error per eye |
| Gaze | Gaze pointer (mouse follows gaze), AAC filter length, both eyes / right only / left only, gain sweep, printed marker for pupil size in mm |
| Screen | Display (with the OS-reported size), physical width, viewing distance, 22" calibration area on larger screens |
| Open Gaze API server | Address(es) to connect to, **Allow connections from other computers** switch, API version, connected clients, LSL streaming |

Calibration and validation run in a fullscreen window on the selected screen. After a calibration it
shows the targets, each eye's calibrated gaze (blue right, orange left) and the live gaze point; keys
there: `c` recalibrate, `1` one-point, `u` undo, `Esc` close. API clients can start calibrations and
show or hide both windows as with Gazepoint Control. `--opencv-ui` (also used automatically without
PySide6) runs the earlier OpenCV windows with keyboard controls instead: tracker window `c` calibrate,
`p` gaze pointer, `g` gain sweep, `q` quit; calibration window `c`, `5`/`9`, `1`, `u`, `s` tone,
`+`/`-`/`d` speed.

- **Network access:** the API listens on all networks by default (port 4242), so clients on other
  computers can connect, for multi-machine setups. The startup message and the app show the address
  to use. The API has **no authentication**: anyone who can reach the port can read gaze data and
  send commands. On untrusted networks use `--local-only`, or turn off "Allow connections from other
  computers" in the app, which switches while running and keeps connected clients. Your OS firewall
  may ask once whether to allow incoming connections.
- **No hardware?** `--simulate` (or `OPENGP3_SIMULATE=1`) runs everything on a simulated GP3: the
  simulated person looks at your mouse cursor and follows calibration targets by themselves. It
  keeps its own calibration file, so your real calibration is never overwritten.
- **Lab Streaming Layer, on by default** (`--no-lsl` turns it off; pylsl is installed with OpenGP3):
  - **`OpenGP3 Gaze`** (type `Gaze`, 60 Hz or the GP3 HD's 150 Hz, 26 float channels, named after the Open Gaze API fields):
    binocular, fixation and per-eye gaze (screen fractions), pupil size in mm, pupil positions in the
    camera image, eye distance, blinks and cursor.
  - **`OpenGP3 Markers`** (type `Markers`): user markers from API clients (`USER_DATA`) and
    calibration events.
  - Timestamps are the frames' capture times on the LSL clock. From Python, use
    `opengp3.lsl.LSLStreamer(tracker)`.
- **Default calibration:** 9 points (3×3 at 15/50/85 %, centre first). `CALIBRATE_RESET` restores it;
  `--points 5` makes the classic 5-point layout the default instead.
- **Tracker display:** the camera image is mirrored, so moving your head left moves it left on
  screen, and the hints say "move to your left/right". `--no-mirror` (or the checkbox) shows the raw
  camera image. Data coordinates are never mirrored.

- **API:** serves Open Gaze API v2.8 on port 4242. Clients choose fields with `ENABLE_SEND_*`, stream
  `REC` records at the camera rate (60 Hz, GP3 HD 150 Hz), run calibration via `CALIBRATE_*` and receive `CAL` records. With
  `CALIBRATE_SHOW=0`, a client can draw its own targets and follow the server's timing.
- **Large screens:** on a display over 22", the calibration points (default or `CALIBRATE_ADDPOINT`)
  are placed inside the centred 22" area. The `CAL` records report their actual screen positions, so
  client-drawn targets land in the right place.
- **Custom targets:** `--media DIR` (or *Target images...*) uses `CalibPoint<N>.png` images and
  `.mp3` sounds as targets.
- **Persistence:** the last calibration is saved to `~/.opengp3/calibration.json` and reloaded.
- **Library use:** `OpenGazeServer(tracker).start()` serves a `GazeTracker` from your own code, and
  `OpenGazeClient` is a small client for any Open Gaze API server.

Tested against Gazepoint's own public-domain Python demo client (`demo/python/GazepointAPI.py`),
unmodified.

## Python API

| Object | What it does |
|---|---|
| `GazeTracker(screen=None, display=None, calibration=None, smoothing=True)` | Owns the camera (GP3 or GP3 HD) and runs detection on every frame in a background thread. `start()`/`stop()` or `with`. Gaze is reported on `screen`, or on display number `display` (0 = main) with its size from the OS. |
| `.set_display(n, width_cm=None, distance_cm=None)` / `.set_screen(screen)` | Track and calibrate on another screen (recalibrate afterwards). |
| `.calibrate(points=9, area=None)` | Fullscreen calibration: a head-positioning screen, then the targets. Targets with too little data are retried. Returns and stores a `Calibration`. |
| `.validate(points=VALIDATION_POINTS)` | Accuracy on points the calibration didn't use. Returns a `ValidationResult` (`.accuracy_deg`, `.precision_rms_s2s_deg`, `.report()`). |
| `.gaze(duration)` | Iterator over every `GazeSample`. Also `.latest()`, `.subscribe()` (queue), `.add_callback(fn)`. |
| `.show_gaze(duration)` | Fullscreen gaze-cursor demo. |
| `GazeSample` | `t` (capture time), `frame`, `x`/`y` (smoothed), `raw_x`/`raw_y`, `per_eye`, `eyes` (pupil/glint features), `valid`. |
| `Calibration` | Per-eye mapping. `.save(path)`, `Calibration.load(path)`, `.predict(eyes)`, `.summary()`. |
| `Screen` / `Rect` | Screen geometry in points. `Screen.area(22)` gives a centred 22" area; `angle_deg(p, q)`. |
| `OneEuroFilter` | The gaze smoothing filter (pass your own to `GazeTracker(smoothing=...)` to tune it). |
| `open_camera()`, `GP3Camera`, `GP3HDCamera` | Low-level access: raw frames, features, IR LEDs; `open_camera()` returns whichever tracker is connected. |
| `detect_eyes`, `EyeTracker`, `BrightDarkDetector` | Per-frame eye features: the GP3's dark-pupil detector, eye identity across frames, the GP3 HD's bright/dark-pupil detector. |
| `opengp3.platform.displays()` | The screens, with position, size and physical width from the OS. |

Notes:
- **Windows must run on the main thread on macOS.** `calibrate`, `validate` and `show_gaze` use one
  shared fullscreen window per session, on the tracked screen: OpenCV from the library, Qt in OpenGP3
  Control (`GazeTracker.window_factory` chooses). The window checks that it really covers the screen before
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
| GP3 HD: USB3 Vision camera, 60/150 Hz, Gazepoint's bright/dark-pupil illumination | 🧪 tracks both eyes at 150 Hz on hardware; 0.44° calibrated accuracy, one subject (see [GP3 HD](#gp3-hd)) |
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
| Screen selection (`SCREEN_SIZE`, `--display`, display picker), physical size from the OS | ✅ (tested with two monitors on Windows) |
| Control application: tracker display, head position and distance, calibration controls and results, settings, server status | ✅ OpenGP3 Control (Qt), see [OpenGP3 Control](#opengp3-control) |
| Monocular tracking (force L/R), automatic gain sweep, head-distance feedback | ✅ |
| Biometrics: GSR, heart rate, dial, TTL I/O | ❌ separate Gazepoint hardware with an undocumented protocol. Fields are sent as invalid; `TTL_WRITE` returns NACK |
| Multiple eye trackers, automatic switching | ❌ one tracker (`TRACKER_ID` reports 1) |
| Mobile-device calibration web page, Windows Eye Control, tray icon | ❌ not implemented (Windows/app-specific) |

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
| GP3 HD, 13 points, 24" screen, 150 Hz (see [GP3 HD](#gp3-hd)) | 0.44° / 0.50° / 0.77° | 0.33° |
| Earlier script, 9 points, 22" area | 0.58° / 0.61° / 0.83° | 0.62° |
| Earlier script, 9 points, full 34" ultrawide | 1.07° / 0.95° / 2.01° | 0.81° |

- **Full ultrawide:** its edges are about ±26° off-centre, beyond the GP3's range. The glint is lost at
  the outer points and errors grow toward the sides. On a screen this wide, calibrate a smaller area
  (`--area-diag-in 22`), or use `--points 16/25` for the full screen.
- **Speed:** detection takes about 9 ms per frame. The pipeline delivers 60.7 Hz with 0 dropped frames
  and about 8 ms from frame capture to gaze sample.

## GP3 HD

The GP3 HD has a different camera: a FLIR (Point Grey) **Chameleon3** USB 3 camera (`1e10:3300`) at
60 or 150 Hz. It speaks the standard **USB3 Vision** protocol instead of the GP3's IIDC tunnel, so
OpenGP3 has a second driver for it (`opengp3.u3v.GP3HDCamera`, also returned by `open_camera()`):

- **Control:** GenCP register reads/writes on the control interface (USB class `EF/05/00`).
- **Features:** the camera stores a GenICam XML file describing its features. OpenGP3 reads it and
  sets `Width`, `ExposureTime`, `AcquisitionFrameRate`, ... by name (`opengp3.genicam.NodeMap`,
  `cam.nodes`).
- **Images:** the stream interface (`EF/05/02`) sends each frame as a leader, the Mono8 payload and
  a trailer. Four frames' worth of transfers stay queued (libusb asynchronous API); a broken frame
  restarts acquisition to realign the queue.

Defaults: the whole sensor, 150 Hz, Gazepoint's illumination (6.6 ms pulses) with automatic gain. `--hz 60` (CLI) or `GazeTracker(rate_hz=60)` selects 60 Hz; `PRODUCT_ID` in the Open Gaze API
reports `GP3HD`, `USB3` and the rate, and the LSL stream's nominal rate follows the camera.

**Illumination and detection: Gazepoint Control's method.** The GP3 HD has two IR LED banks, on the
camera's GPIO `Line2` and `Line3`. `Line2` sits next to the lens and lights the retina, so the pupil
glows (*bright pupil*); `Line3` is off axis and leaves the pupil black with a pair of glints (*dark
pupil*). Gazepoint Control drives them through a Point Grey *strobe pattern* that isn't in the
camera's GenICam file: a 3-frame cycle of one dark-pupil frame and two bright-pupil frames, 6.6 ms
pulses. OpenGP3 read those registers after a Gazepoint Control session and reproduces them value for
value (LED mode `gazepoint`, the default), so the IR exposure is the same as with Gazepoint Control.

Gazepoint's image processing is not published. OpenGP3 applies the standard bright/dark-pupil
technique this illumination is built for (`opengp3.brightdark.BrightDarkDetector`, used automatically
by `GazeTracker` on a GP3 HD):

1. **Frame type**: the dark-pupil frame is the brightest of each cycle.
2. **Finding the eyes**: latest bright frame minus latest dark frame. Only the pupils are much
   brighter in the first, so skin, lashes, dark irises, brows and glasses cancel out. Recently found
   eyes are also followed from their last position for a few frames, through fast eye movements
   that smear the difference image.
3. **Pupil fit in the current frame** (bright or dark disc), so no sample mixes two moments.
4. **One glint reference**: the on-axis glint of bright frames. Dark frames' glint-pair midpoint is
   shifted to it by an offset learned per eye from neighbouring frames, so both frame types give the
   same pupil-glint vector and share one calibration.
5. **Gain control**: the camera gain follows the bright pupil's brightness, keeping it below
   saturation (a saturated pupil hides the on-axis glint).

`OPENGP3_HD_IR_MODE=dark` selects the single-bank alternative instead: `Line3` on every frame, 3 ms
pulses, 18 dB, detected with the GP3's dark-pupil detector. `opengp3 probe-leds` finds the LED lines
on a differently wired unit and saves them for that mode (`~/.opengp3/gp3hd.json`).

**Status, measured on a GP3 HD** (Chameleon3 CM3-U3-13Y3M, PYTHON1300 sensor, firmware 1.13.3,
Windows 11 with UsbDk), one subject at about 55-65 cm, no calibration yet:

- **150 Hz at full frame (1280×1024)**: the camera's frame period is 6.70 ms with no missing frames;
  the stream keeps four frames of USB transfers queued (libusb asynchronous API).
- **Tracking (`gazepoint` mode)**: 149.4 samples/s processed with no dropped frames, detection 3-4 ms
  per frame. Over 8 s of free viewing: both eyes in 92 % of samples, one eye in 7 %. On recorded
  steady viewing: both eyes in 99-100 % of bright and dark frames. Pupil-glint vector noise
  0.11-0.14 px sample to sample. Pupil 2.9-3.6 mm. Bright and dark frames agree on the pupil-glint
  vector to within 0.05 px.
- **UsbDk re-enumerates the camera when a program closes it**, so it is off the bus for about 17 s
  afterwards. Opening waits for it (up to 30 s).
- **Calibrated accuracy** (`opengp3 accuracy --points 13`, 24" 1920×1080 screen, 52.7 cm wide,
  65 cm assumed viewing distance, head free, unfiltered 150 Hz gaze; 8 validation points the
  calibration never saw): **0.44° mean / 0.50° median / 0.77° worst**, precision **RMS-S2S 0.33°**
  (SD 0.65°). Both eyes in 98 % of validation samples. One subject, one session.
- **Head depth:** the eye model's focal length for the GP3 HD is 2990 px. Gazepoint doesn't publish
  the lens; its head box (35 × 22 cm at 65 cm) is the camera's field of view, which gives 2900-3025 px
  on the 4.8 µm sensor (the same method brackets the GP3's measured 2750 px to within 1 %). Depth in
  metres also depends on the subject's IPD (63 mm assumed): one subject at about 60 cm read 56 cm,
  consistent with a 67 mm IPD. The printed marker measures the IPD.
- Not measured yet: more subjects (glasses, lighter or darker irises, children), and
  sample-count-based filter settings (fixation confirmation, AAC length), which span less time at
  150 Hz.

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

The driver, detection, calibration and API server are pure Python on top of libusb and OpenCV; the
control app uses Qt (PySide6). The unit tests, including the control app on Qt's offscreen platform,
run on macOS, Linux and Windows in CI. Hardware tests: the GP3 on macOS; the GP3 HD and OpenGP3
Control on Windows 11 (UsbDk, two monitors). Physical screen sizes on macOS and Linux are covered by
unit tests but not yet checked on hardware.

## License

MIT © Paolo Burelli. OpenGP3 is not affiliated with or endorsed by Gazepoint.
