"""Run a model exported with ``MaskRCNN.save`` (``config.json`` + TF SavedModel).

Only TensorFlow is needed to load the SavedModel (the ``serve`` extra, any TensorFlow 2.x): the
pre/post-processing comes from ``mrcnn.inference``, so services don't need the training stack.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details).

Examples
--------
Run an export on one RGB image::

    predictor = SavedModelPredictor("export_dir")
    result = predictor.detect(image)  # rois, class_ids, scores, masks, like MaskRCNN.detect
"""

import json
import os
from types import SimpleNamespace

import numpy as np

from mrcnn import inference


def load_config(export_dir):
    """Read an exported ``config.json`` as an object with the ``Config`` attribute names.

    Parameters
    ----------
    export_dir : str | os.PathLike
        Export directory written by ``MaskRCNN.save``.

    Returns
    -------
    SimpleNamespace
        The settings, with ``MEAN_PIXEL`` as a numpy array.

    Raises
    ------
    ValueError
        If the export uses the training-only ``"crop"`` resize mode.
    """
    with open(os.path.join(export_dir, "config.json"), encoding="utf-8") as f:
        values = json.load(f)
    if values["IMAGE_RESIZE_MODE"] == "crop":
        raise ValueError("IMAGE_RESIZE_MODE 'crop' is for training only")
    values["MEAN_PIXEL"] = np.array(values["MEAN_PIXEL"])
    return SimpleNamespace(**values)


class SavedModelPredictor:
    """Detect objects in one image at a time with an exported SavedModel.

    Parameters
    ----------
    export_dir : str | os.PathLike
        Export directory written by ``MaskRCNN.save``.

    Attributes
    ----------
    config : SimpleNamespace
        The exported settings (``config.json``).
    """

    config: SimpleNamespace

    def __init__(self, export_dir):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel  # optional: `serve` extra

        self.config = load_config(export_dir)
        self._tf = tf
        self._model = tf.saved_model.load(str(export_dir))
        self._infer = self._model.signatures["serving_default"]
        self._anchors = {}

    def detect(self, image):
        """Detect the objects in one image.

        Parameters
        ----------
        image : np.ndarray
            RGB image, shape ``(height, width, 3)``.

        Returns
        -------
        dict
            Like ``MaskRCNN.detect``, in original image coordinates: ``rois``
            ``[N, (y1, x1, y2, x2)]``, ``class_ids`` ``[N]``, ``scores`` ``[N]`` and ``masks``
            ``[height, width, N]``.

        Raises
        ------
        ValueError
            If the image is not ``(height, width, 3)``.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB image of shape [H, W, 3], got {image.shape}")
        molded, metas, windows = inference.mold_inputs([image], self.config)
        outputs = self._infer(
            input_image=self._tf.constant(molded.astype(np.float32)),
            input_image_meta=self._tf.constant(metas.astype(np.float32)),
            input_anchors=self._tf.constant(self._get_anchors(molded.shape[1:3])[np.newaxis]),
        )
        rois, class_ids, scores, masks = inference.unmold_detections(
            outputs["mrcnn_detection"].numpy()[0],
            outputs["mrcnn_mask"].numpy()[0],
            image.shape,
            molded.shape[1:],
            windows[0],
        )
        return {"rois": rois, "class_ids": class_ids, "scores": scores, "masks": masks}

    def _get_anchors(self, image_shape):
        """Get the normalized anchor pyramid for a molded image shape, computed once per shape.

        Parameters
        ----------
        image_shape : tuple
            ``(height, width)`` of the molded image.

        Returns
        -------
        np.ndarray
            ``[N, (y1, x1, y2, x2)]`` anchors in normalized coordinates, float32.
        """
        key = tuple(image_shape)
        if key not in self._anchors:
            anchors = inference.pyramid_anchors(self.config, image_shape)
            self._anchors[key] = inference.norm_boxes(anchors, image_shape).astype(np.float32)
        return self._anchors[key]
