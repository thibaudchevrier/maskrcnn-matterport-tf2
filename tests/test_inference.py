"""mrcnn.inference must work without TensorFlow, on numpy 1.x and 2.x."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mrcnn import inference

CONFIG = SimpleNamespace(
    NUM_CLASSES=47,
    IMAGE_MIN_DIM=800,
    IMAGE_MAX_DIM=1024,
    IMAGE_MIN_SCALE=0,
    IMAGE_RESIZE_MODE="square",
    MEAN_PIXEL=np.array([123.7, 116.8, 103.9]),
    BACKBONE="resnet101",
    BACKBONE_STRIDES=[4, 8, 16, 32, 64],
    RPN_ANCHOR_SCALES=[32, 64, 128, 256, 512],
    RPN_ANCHOR_RATIOS=[0.5, 1, 2],
    RPN_ANCHOR_STRIDE=1,
)


def test_module_does_not_import_tensorflow():
    # Fresh interpreter: other tests import mrcnn.model (and TensorFlow) in this process.
    """mrcnn.inference imports without TensorFlow (checked in a fresh interpreter)."""
    code = "import sys, mrcnn.inference; assert 'tensorflow' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_resize_image_square_keeps_aspect_ratio_and_pads():
    """Square mode keeps the aspect ratio and zero-pads around the image."""
    image = np.full((400, 200, 3), 255, dtype=np.uint8)
    molded, window, scale, _, _ = inference.resize_image(image, 800, 1024, 0, "square")
    assert molded.shape == (1024, 1024, 3)
    assert scale == pytest.approx(2.56)  # capped by max_dim: 400 * 2.56 = 1024
    y1, x1, y2, x2 = window
    assert (y2 - y1, x2 - x1) == (1024, 512)
    assert molded[:, :x1].max() == 0 and molded[y1:y2, x1:x2].min() > 0


def test_box_normalization_roundtrip():
    """Normalizing then denormalizing boxes gives them back."""
    boxes = np.array([[10, 20, 110, 220], [0, 0, 1024, 1024]])
    normalized = inference.norm_boxes(boxes, (1024, 1024))
    np.testing.assert_array_equal(inference.denorm_boxes(normalized, (1024, 1024)), boxes)


def test_pyramid_anchor_count():
    """The pyramid has 3 anchors per feature map pixel at every level."""
    anchors = inference.pyramid_anchors(CONFIG, (1024, 1024))
    expected = sum(3 * (1024 // s) ** 2 for s in CONFIG.BACKBONE_STRIDES)
    assert anchors.shape == (expected, 4)


def test_mold_inputs_shapes():
    """Molded inputs have the network's image, meta and window shapes."""
    molded, metas, windows = inference.mold_inputs([np.zeros((300, 200, 3), np.uint8)], CONFIG)
    assert molded.shape == (1, 1024, 1024, 3)
    assert metas.shape == (1, 1 + 3 + 3 + 4 + 1 + 47)
    assert windows.shape == (1, 4)


def test_unmold_detections_maps_back_to_original_image():
    """A detection covering the window maps to the whole original image."""
    original, molded_shape = (400, 200, 3), (1024, 1024, 3)
    _, window, _, _, _ = inference.resize_image(np.zeros(original, np.uint8), 800, 1024, 0)
    # One detection covering the full real image, class 5, then zero padding.
    y1, x1, y2, x2 = inference.norm_boxes(np.array(window), molded_shape[:2])
    detections = np.zeros((100, 6), np.float32)
    detections[0] = [y1, x1, y2, x2, 5, 0.9]
    masks = np.zeros((100, 28, 28, 47), np.float32)
    masks[0, :, :, 5] = 1.0

    boxes, class_ids, _, full_masks = inference.unmold_detections(
        detections, masks, original, molded_shape, window
    )
    np.testing.assert_array_equal(boxes, [[0, 0, 400, 200]])
    np.testing.assert_array_equal(class_ids, [5])
    assert full_masks.shape == (400, 200, 1) and full_masks.all()


def test_unmold_detections_without_detections():
    """No detection gives empty boxes and masks of the image size."""
    boxes, _, _, masks = inference.unmold_detections(
        np.zeros((100, 6), np.float32),
        np.zeros((100, 28, 28, 47), np.float32),
        (50, 60, 3),
        (1024, 1024, 3),
        (0, 0, 1024, 1024),
    )
    assert boxes.shape == (0, 4) and masks.shape == (50, 60, 0)
