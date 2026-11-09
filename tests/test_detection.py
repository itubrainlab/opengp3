import cv2
import numpy as np
import pytest

from opengp3 import EyeTracker, detect_eyes


def synth_eye_image(eyes, size=(752, 480), seed=0):
    """Grey face, darker iris, dark pupil, and a pair of saturated glints
    (the GP3's two LED banks) per eye. eyes: [(pupil_x, pupil_y, glint_dx, glint_dy)]"""
    w, h = size
    img = np.full((h, w), 150, np.uint8)
    glints = np.zeros((h, w), np.float32)
    for px, py, gdx, gdy in eyes:
        cv2.circle(img, (int(px), int(py)), 28, 80, -1, cv2.LINE_AA)
        # sub-pixel pupil via supersampling
        big = np.zeros((h * 4, w * 4), np.uint8)
        cv2.circle(big, (int(px * 4), int(py * 4)), 36, 255, -1)
        mask = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA) / 255.0
        img = (img * (1 - mask) + 35 * mask).astype(np.uint8)
        gx, gy = px - gdx, py - gdy
        for ox in (-3.5, 3.5):
            cv2.circle(glints, (int(round(gx + ox)), int(round(gy))), 2, 1.0, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    img = img + rng.normal(0, 2, img.shape)
    # real GP3 glints: saturated core (5-20 px) with a ~2 px bloom halo
    halo = cv2.GaussianBlur(glints, (7, 7), 1.5)
    img = np.maximum(img, 255 * np.minimum(1.0, 2.2 * halo))
    img[glints > 0] = 255
    return np.clip(img, 0, 255).astype(np.uint8)


def test_detects_both_eyes_and_pcr():
    truth = [(260.3, 240.6, 2.0, -1.0), (520.7, 236.2, -1.5, 1.0)]
    eyes = detect_eyes(synth_eye_image(truth))
    assert len(eyes) == 2
    for e, (px, py, gdx, gdy) in zip(eyes, truth):
        assert abs(e.pupil_x - px) < 0.8 and abs(e.pupil_y - py) < 0.8
        assert 7 < e.pupil_r < 11
        assert e.n_glints == 2
        assert abs(e.pcr[0] - gdx) < 1.0 and abs(e.pcr[1] - gdy) < 1.0


def test_pcr_follows_pupil_shift():
    """Moving the pupil relative to the glints must change PCR by the same amount
    (guards against the pupil fit locking onto the glints)."""
    a = detect_eyes(synth_eye_image([(300, 240, 0.0, 0.0)]))[0]
    b = detect_eyes(synth_eye_image([(300, 243, 0.0, 3.0)]))[0]
    assert b.pcr[1] - a.pcr[1] == pytest.approx(3.0, abs=0.8)


def test_no_eyes_in_empty_image():
    img = np.full((480, 752), 120, np.uint8)
    assert detect_eyes(img) == []


def test_eye_tracker_keeps_identity_with_one_eye():
    tr = EyeTracker()
    both = detect_eyes(synth_eye_image([(260, 240, 0, 0), (520, 240, 0, 0)]))
    assert set(tr.update(both)) == {"R", "L"}
    only_right_of_image = detect_eyes(synth_eye_image([(525, 241, 0, 0)]))
    assert set(tr.update(only_right_of_image)) == {"L"}   # subject's left eye
    only_left_of_image = detect_eyes(synth_eye_image([(262, 239, 0, 0)]))
    assert set(tr.update(only_left_of_image)) == {"R"}
