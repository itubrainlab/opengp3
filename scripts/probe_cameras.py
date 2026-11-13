#!/usr/bin/env python3
"""List which camera indices deliver real images (read-only diagnostic).

    .venv/bin/python scripts/probe_cameras.py [out.json] [frame-prefix]

For each OpenCV camera index 0-4: opened?, frames received, mean brightness
(black frames = a virtual camera with no source, or no permission).
"""
import json
import sys
import time

import cv2


def probe(i, seconds=2.0, save=None):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(i, backend)
    r = {"index": i, "opened": bool(cap.isOpened())}
    if r["opened"]:
        n, best, size, frame = 0, 0.0, None, None
        t_end = time.time() + seconds
        while time.time() < t_end:
            ok, f = cap.read()
            if ok and f is not None and f.size:
                n += 1
                if float(f.mean()) >= best:
                    best, frame = float(f.mean()), f
                size = [int(f.shape[1]), int(f.shape[0])]
        if save and frame is not None:
            cv2.imwrite(f"{save}_{i}.jpg", cv2.resize(frame, (480, 270)))
        r.update(frames=n, max_mean_brightness=round(best, 1), size=size,
                 real_image=n > 0 and best > 2)
    cap.release()
    return r


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "camera_probe.json"
    save = sys.argv[2] if len(sys.argv) > 2 else None   # prefix for one small frame per camera
    results = [probe(i, save=save) for i in range(5)]
    try:
        from opengp3.sources import webcams
        names = {w.index: (w.name, w.kind) for w in webcams()}
        for r in results:
            r["name"], r["kind"] = names.get(r["index"], (None, None))
    except Exception:
        pass
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
