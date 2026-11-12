#!/usr/bin/env python3
"""Live check of OpenGP3's webcam fallback (run it from a terminal that has
camera permission, e.g. macOS Terminal):

    .venv/bin/python scripts/webcam_check.py [results.json]

Sit in front of the webcam, look at the screen and keep still for ~8 s.
Measures frame rate, detector speed, how often both eyes are found and the
noise of the gaze feature, and writes them (plus an annotated frame) next to
the results file.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from opengp3.webcam import WebcamEyeDetector, WebcamUnavailable, open_webcam


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "webcam_check.json")
    res = {}
    try:
        cam = open_webcam()
    except WebcamUnavailable as e:
        res["error"] = str(e)
        out.write_text(json.dumps(res, indent=2))
        print("webcam unavailable:", e)
        return 1
    res["size"] = [cam.width, cam.height]
    res["camera_fps"] = cam.rate
    det = WebcamEyeDetector()
    cam.start_stream()
    print("look at the screen and keep still for 8 s...", flush=True)
    for _ in range(15):  # warm-up (auto exposure, detector)
        det.update(cam.grab()[2])
    n = both = 0
    times, feats, opens = [], [], []
    last_img, last_eyes = None, None
    t0 = time.time()
    while time.time() - t0 < 8:
        _, _, img = cam.grab()
        a = time.perf_counter()
        eyes = det.update(img)
        times.append(time.perf_counter() - a)
        n += 1
        if len(eyes) == 2:
            both += 1
            feats.append([v for e in eyes for v in e.pcr])
            opens.append([e.openness for e in eyes])
            last_img, last_eyes = img, eyes
    cam.close()
    res["frames"] = n
    res["processed_fps"] = round(n / 8, 1)
    res["detector_ms"] = round(1000 * float(np.median(times)), 1)
    res["both_eyes_percent"] = round(100 * both / max(n, 1), 1)
    if feats:
        f = np.array(feats)
        res["feature_mean_R_L"] = np.round(f.mean(0), 2).tolist()
        res["feature_sd_R_L"] = np.round(f.std(0), 2).tolist()
        res["openness_median"] = np.round(np.median(opens, axis=0), 2).tolist()
    if last_img is not None:
        vis = cv2.cvtColor(last_img, cv2.COLOR_GRAY2BGR)
        for e in last_eyes:
            cv2.circle(vis, (int(e.pupil_x), int(e.pupil_y)), 3, (0, 255, 0), -1)   # iris
            cv2.circle(vis, (int(e.glint_x), int(e.glint_y)), 3, (0, 0, 255), -1)   # eye centre
        img_path = out.with_suffix(".jpg")
        cv2.imwrite(str(img_path), vis)
        res["frame"] = str(img_path)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
