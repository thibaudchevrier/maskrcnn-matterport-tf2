"""Train a tiny Mask R-CNN on synthetic shapes, then reload it for inference.

Guards the compatibility fixes (TF 2.15 / Keras 2, numpy >= 1.24, scikit-image >= 0.19):
this fails if data loading, graph-mode training, checkpointing or detection breaks.
"""

import subprocess
import sys

import numpy as np

from mrcnn import config as mconfig
from mrcnn import model as modellib
from mrcnn import utils


class SmokeConfig(mconfig.Config):
    """Tiny configuration: 128 px images, ResNet-50, 2 steps per epoch."""

    NAME = "smoke"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3  # background + 2 shapes
    BACKBONE = "resnet50"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 16
    STEPS_PER_EPOCH = 2
    VALIDATION_STEPS = 1


class ShapesDataset(utils.Dataset):
    """Synthetic dataset: every image has a square and a bar."""

    def load(self, count):
        """Register the two classes and ``count`` images."""
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "bar")
        for i in range(count):
            self.add_image("shapes", image_id=i, path=None)

    def load_image(self, image_id):
        """Draw the same square and bar on a gray background."""
        image = np.full((128, 128, 3), 60, np.uint8)
        image[20:70, 30:80] = 200
        image[80:120, 10:110] = 120
        return image

    def load_mask(self, image_id):
        """Return the square and bar masks with their class ids."""
        mask = np.zeros((128, 128, 2), bool)
        mask[20:70, 30:80, 0] = True
        mask[80:120, 10:110, 1] = True
        return mask, np.array([1, 2], np.int32)


def test_train_checkpoint_and_detect(tmp_path):
    """Train one epoch, reload the checkpoint, detect, export, and serve the export identically."""
    train, val = ShapesDataset(), ShapesDataset()
    train.load(4)
    train.prepare()
    val.load(2)
    val.prepare()

    model = modellib.MaskRCNN(mode="training", config=SmokeConfig(), model_dir=str(tmp_path))
    model.train(train, val, learning_rate=1e-3, epochs=1, layers="all")
    checkpoint = model.find_last()
    assert checkpoint.endswith("_0001.h5")

    class InferenceConfig(SmokeConfig):
        """Inference configuration keeping every detection."""

        DETECTION_MIN_CONFIDENCE = 0.0

    inference = modellib.MaskRCNN(
        mode="inference", config=InferenceConfig(), model_dir=str(tmp_path)
    )
    inference.load_weights(checkpoint, by_name=True)
    [result] = inference.detect([train.load_image(0)])
    assert result["masks"].shape[:2] == (128, 128)
    assert result["rois"].shape[1] == 4

    # Export, then serve the export with mrcnn.serving in a fresh process, as a service would
    # (mrcnn.model switches this process to graph mode): same detections as MaskRCNN.detect.
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    inference.save(str(export_dir))
    np.save(tmp_path / "image.npy", train.load_image(0))
    serve = (
        "import sys, numpy as np; from mrcnn.serving import SavedModelPredictor; "
        "r = SavedModelPredictor(sys.argv[1]).detect(np.load(sys.argv[2])); "
        "np.savez(sys.argv[3], **r)"
    )
    subprocess.run(
        [sys.executable, "-c", serve, export_dir, tmp_path / "image.npy", tmp_path / "served.npz"],
        check=True,
    )
    served = np.load(tmp_path / "served.npz")
    np.testing.assert_allclose(served["scores"], result["scores"], rtol=1e-5)
    np.testing.assert_array_equal(served["class_ids"], result["class_ids"])
    np.testing.assert_array_equal(served["masks"], result["masks"])
