# OpenGP3

A Python library and command-line tool for the **Gazepoint GP3** eye tracker on macOS. It has its own
user-space driver (libusb), so it needs no Windows, VM or kernel extension. It includes pupil/glint
detection, fullscreen calibration, validation, and gaze streaming in screen coordinates at 60 Hz.

```python
from opengp3 import GazeTracker, Screen

screen = Screen.detect(width_cm=53, distance_cm=65)   # physical size only needed for degrees
with GazeTracker(screen) as gt:
    gt.calibrate(points=9)                # fullscreen calibration (run on the main thread)
    print(gt.validate().report())         # accuracy / precision in degrees
    gt.calibration.save("calibration.json")
    for s in gt.gaze(duration=10):        # every frame, ~60 Hz
        if s.valid:
            print(s.t, s.x, s.y)          # screen points, top-left origin
```

## Install

```bash
brew install libusb
pip install opengp3
```

Requirements: macOS, Python ≥ 3.9 (`pyusb`, `numpy` and `opencv-python` are installed automatically).
Only one program can use the GP3 at a time.

From source, for development:

```bash
git clone https://github.com/itubrainlab/opengp3 && cd opengp3
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest               # offline tests, no hardware needed
```

## Command line

```bash
opengp3 view                                   # live IR image, eye overlay, head-position hints
opengp3 calibrate --screen-width-cm 80 --validate --plot accuracy.png    # full screen, 9 points
opengp3 calibrate --points 13 --screen-width-cm 80 --area-diag-in 22    # centred 22" area
opengp3 validate  --calibration calibration.json --screen-width-cm 80
opengp3 track     --calibration calibration.json --csv gaze.csv --duration 60
opengp3 track     --calibration calibration.json --show                 # fullscreen gaze cursor
opengp3 record eyes.csv --duration 10          # raw pupil/glint features per frame
opengp3 info | snapshot eye.png
```

- `--screen-width-cm` and `--distance-cm` (default 65) are only used to report errors in degrees.
- `--points` can be 5, 9, 13, 16 or 25. More points model the screen edges better.
- `--dump samples.csv` (on `calibrate`) writes every labelled sample, for diagnostics.

`python gp3.py …` (in a source checkout) is an alias for `opengp3 …`.

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

## License

MIT © Paolo Burelli. OpenGP3 is not affiliated with or endorsed by Gazepoint.
