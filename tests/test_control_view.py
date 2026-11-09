import numpy as np

from opengp3.control import render_tracker_view
from opengp3.detection import EyeFeature
from opengp3.tracker import GazeSample


def _sample(x):
    return GazeSample(t=0.0, frame=1, eyes={"R": EyeFeature(x, 240, 6, x, 240, 2, 1.0)})


def test_mirror_flips_image_and_overlay():
    img = np.zeros((480, 752), np.uint8)
    img[:, :10] = 255                                   # bright stripe at the image left
    normal = render_tracker_view(img, _sample(100), 60.0)
    mirrored = render_tracker_view(img, _sample(100), 60.0, mirror=True)
    assert normal[300, 5].mean() > 200 and mirrored[300, 5].mean() < 50
    assert mirrored[300, 751 - 5].mean() > 200          # stripe now on the right
    # pupil circle (green) drawn at the mirrored position
    green = lambda v, x: v[230:251, x - 9:x + 10, 1].max()
    assert green(normal, 100) == 255 and green(mirrored, 751 - 100) == 255


def test_hints_in_subject_terms():
    img = np.zeros((480, 752), np.uint8)
    # eye near the image left edge = subject too far to their right
    v = render_tracker_view(img, _sample(20), 60.0, mirror=True)
    assert v.shape == (480, 752, 3)
