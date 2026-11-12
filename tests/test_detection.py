import cv2
import numpy as np
import pytest

from conftest import synth_eye_image
from opengp3 import EyeTracker, detect_eyes


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
    both = detect_eyes(synth_eye_image([(260, 240, 0, -1.5), (520, 240, 0, -1.5)]))
    assert set(tr.update(both)) == {"R", "L"}
    only_right_of_image = detect_eyes(synth_eye_image([(525, 241, 0, -1.5)]))
    assert set(tr.update(only_right_of_image)) == {"L"}   # subject's left eye
    only_left_of_image = detect_eyes(synth_eye_image([(262, 239, 0, -1.5)]))
    assert set(tr.update(only_left_of_image)) == {"R"}
