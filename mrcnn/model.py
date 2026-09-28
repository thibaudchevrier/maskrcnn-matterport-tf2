"""The Mask R-CNN model (TensorFlow 2.15, Keras 2, graph mode).

Backbone, region proposal network, heads, losses, data generator, and the ``MaskRCNN`` class that
builds, trains and runs the model.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details).
Written by Waleed Abdulla.
"""

import datetime
import errno
import logging
import multiprocessing
import os
import random
import re
from collections import OrderedDict
from collections.abc import Iterator

import h5py
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import numpy as np
import tensorflow as tf
from keras.utils import get_file

from mrcnn import inference, utils
from mrcnn.inference import compose_image_meta, compute_backbone_shapes, mold_image
from mrcnn.parallel_model import ParallelModel

# Requires TensorFlow 1.3+ and Keras 2.0.8+.

tf.compat.v1.disable_eager_execution()

############################################################
#  Utility Functions
############################################################


def log(text, array=None):
    """Print a message, and optionally the shape, min and max of an array.

    Parameters
    ----------
    text : str
        Message.
    array : np.ndarray | None
        Array to describe. By default ``None``.
    """
    if array is not None:
        text = text.ljust(25)
        text += f"shape: {str(array.shape):20}  "
        if array.size:
            text += f"min: {array.min():10.5f}  max: {array.max():10.5f}"
        else:
            text += f"min: {'':10}  max: {'':10}"
        text += f"  {array.dtype}"
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Keras ``BatchNormalization`` with one place to change its behaviour.

    Batch normalization hurts training with small batches, so this layer is often frozen (see
    ``Config.TRAIN_BN``) and then works as a linear layer.
    """

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ  # Keras layers take their inputs only
        """Normalize the inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool | None
            ``None`` trains the layer (normal mode); ``False`` freezes it (good with small batches);
            ``True`` forces training mode even at inference (don't use). By default ``None``.

        Returns
        -------
        tf.Tensor
            The normalized tensor.
        """
        return super().call(inputs, training=training)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    """Build a ResNet identity block (no convolution on the shortcut).

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor.
    kernel_size : int
        Kernel size of the middle convolution of the main path.
    filters : list[int]
        Number of filters of the three convolutions of the main path.
    stage : int
        Stage label, used in layer names.
    block : str
        Block label (``"a"``, ``"b"``...), used in layer names.
    use_bias : bool
        Use a bias in the convolutions. By default ``True``.
    train_bn : bool
        Train (``True``) or freeze the batch normalization layers. By default ``True``.

    Returns
    -------
    tf.Tensor
        The block's output.
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + "2a", use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + "2a")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        use_bias=use_bias,
    )(x)
    x = BatchNorm(name=bn_name_base + "2b")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + "2c")(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation("relu", name="res" + str(stage) + block + "_out")(x)
    return x


def conv_block(
    input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True
):
    """Build a ResNet convolution block (a convolution on the shortcut).

    From stage 3, the first convolution of the main path and the shortcut use ``strides=(2, 2)``.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor.
    kernel_size : int
        Kernel size of the middle convolution of the main path.
    filters : list[int]
        Number of filters of the three convolutions of the main path.
    stage : int
        Stage label, used in layer names.
    block : str
        Block label (``"a"``, ``"b"``...), used in layer names.
    strides : tuple
        Strides of the first convolution and the shortcut. By default ``(2, 2)``.
    use_bias : bool
        Use a bias in the convolutions. By default ``True``.
    train_bn : bool
        Train (``True``) or freeze the batch normalization layers. By default ``True``.

    Returns
    -------
    tf.Tensor
        The block's output.
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = KL.Conv2D(
        nb_filter1, (1, 1), strides=strides, name=conv_name_base + "2a", use_bias=use_bias
    )(input_tensor)
    x = BatchNorm(name=bn_name_base + "2a")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        use_bias=use_bias,
    )(x)
    x = BatchNorm(name=bn_name_base + "2b")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + "2c")(x, training=train_bn)

    shortcut = KL.Conv2D(
        nb_filter3, (1, 1), strides=strides, name=conv_name_base + "1", use_bias=use_bias
    )(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + "1")(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation("relu", name="res" + str(stage) + block + "_out")(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build the ResNet backbone.

    Parameters
    ----------
    input_image : tf.Tensor
        ``[batch, height, width, 3]`` input images.
    architecture : str
        ``"resnet50"`` or ``"resnet101"``.
    stage5 : bool
        Build stage 5 of the network. By default ``False``.
    train_bn : bool
        Train (``True``) or freeze the batch normalization layers. By default ``True``.

    Returns
    -------
    list
        Outputs ``[C1, C2, C3, C4, C5]`` of the stages (``C5`` is ``None`` without stage 5).
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name="conv1", use_bias=True)(x)
    x = BatchNorm(name="bn_conv1")(x, training=train_bn)
    x = KL.Activation("relu")(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b", train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block="c", train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b", train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c", train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block="d", train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a", train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a", train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b", train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c", train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################


def apply_box_deltas_graph(boxes, deltas):
    """Apply refinement deltas to boxes (TensorFlow).

    Parameters
    ----------
    boxes : tf.Tensor
        ``[N, (y1, x1, y2, x2)]`` boxes to update.
    deltas : tf.Tensor
        ``[N, (dy, dx, log(dh), log(dw))]`` refinements to apply.

    Returns
    -------
    tf.Tensor
        ``[N, (y1, x1, y2, x2)]`` refined boxes.
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """Clip boxes to a window (TensorFlow).

    Parameters
    ----------
    boxes : tf.Tensor
        ``[N, (y1, x1, y2, x2)]`` boxes.
    window : tf.Tensor
        ``[4]`` window ``(y1, x1, y2, x2)``.

    Returns
    -------
    tf.Tensor
        ``[N, (y1, x1, y2, x2)]`` clipped boxes.
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class AnchorsLayer(KL.Layer):
    """Layer outputting fixed anchors, so the anchors are part of the Keras graph.

    Parameters
    ----------
    anchors : np.ndarray
        ``[num_anchors, (y1, x1, y2, x2)]`` anchors in normalized coordinates.
    name : str
        Layer name. By default ``"anchors"``.
    **kwargs : dict
        Keyword arguments of ``keras.layers.Layer``.

    Attributes
    ----------
    anchors : tf.Variable
        The anchors.
    """

    anchors: tf.Variable

    def __init__(self, anchors, name="anchors", **kwargs):
        super().__init__(name=name, **kwargs)
        self.anchors = tf.Variable(anchors)

    # pylint: disable-next=arguments-differ,unused-argument  # input only connects the layer
    def call(self, dummy):
        """Return the anchors, whatever the input.

        Parameters
        ----------
        dummy : tf.Tensor
            Any tensor; it only connects the layer to the graph.

        Returns
        -------
        tf.Variable
            The anchors.
        """
        return self.anchors

    def get_config(self):
        """Serialize the layer.

        Returns
        -------
        dict
            The layer configuration.
        """
        config = super().get_config()
        return config


class ProposalLayer(KL.Layer):
    """Select the anchors to pass as proposals to the second stage.

    Filtering uses the anchor scores and non-max suppression to remove overlaps; the bounding box
    refinement deltas are applied to the anchors.

    Parameters
    ----------
    proposal_count : int
        Number of proposals to keep per image.
    nms_threshold : float
        IoU threshold of the non-max suppression.
    config : Config | None
        Model configuration. By default ``None``.
    **kwargs : dict
        Keyword arguments of ``keras.layers.Layer``.

    Attributes
    ----------
    config : Config
        Model configuration.
    proposal_count : int
        Number of proposals to keep per image.
    nms_threshold : float
        IoU threshold of the non-max suppression.
    """

    config: object
    proposal_count: int
    nms_threshold: float

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):  # pylint: disable=arguments-differ  # Keras layers take their inputs only
        """Compute the proposals.

        Parameters
        ----------
        inputs : list
            ``rpn_probs`` ``[batch, num_anchors, (bg prob, fg prob)]``, ``rpn_bbox``
            ``[batch, num_anchors, (dy, dx, log(dh), log(dw))]`` and ``anchors`` ``[batch,
            num_anchors, (y1, x1, y2, x2)]`` in
            normalized coordinates.

        Returns
        -------
        tf.Tensor
            ``[batch, proposal_count, (y1, x1, y2, x2)]`` proposals in normalized coordinates,
            zero-padded.
        """
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], tf.gather, self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], tf.gather, self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice(
            [anchors, ix],
            tf.gather,
            self.config.IMAGES_PER_GPU,
            names=["pre_nms_anchors"],
        )

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice(
            [pre_nms_anchors, deltas],
            apply_box_deltas_graph,
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors"],
        )

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(
            boxes,
            lambda x: clip_boxes_graph(x, window),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors_clipped"],
        )

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            """Run non-max suppression on the boxes of one image and pad the result.

            Parameters
            ----------
            boxes : tf.Tensor
                ``[N, (y1, x1, y2, x2)]`` boxes.
            scores : tf.Tensor
                ``[N]`` scores.

            Returns
            -------
            tf.Tensor
                ``[proposal_count, (y1, x1, y2, x2)]`` kept boxes, zero-padded.
            """
            indices = tf.image.non_max_suppression(
                boxes,
                scores,
                self.proposal_count,
                self.nms_threshold,
                name="rpn_non_max_suppression",
            )
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        """Compute the output shape.

        Parameters
        ----------
        input_shape : list
            Shapes of the inputs.

        Returns
        -------
        tuple
            ``(None, proposal_count, 4)``.
        """
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################


def log2_graph(x):
    """Compute the base-2 logarithm (TensorFlow has no native one).

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.

    Returns
    -------
    tf.Tensor
        ``log2(x)``.
    """
    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KL.Layer):
    """ROI pooling on several levels of the feature pyramid.

    Each box is pooled from the pyramid level matching its size.

    Parameters
    ----------
    pool_shape : tuple
        ``(pool_height, pool_width)`` of the pooled regions, usually ``(7, 7)``.
    **kwargs : dict
        Keyword arguments of ``keras.layers.Layer``.

    Attributes
    ----------
    pool_shape : tuple
        ``(pool_height, pool_width)`` of the pooled regions.
    """

    pool_shape: tuple

    def __init__(self, pool_shape, **kwargs):
        super().__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):  # pylint: disable=arguments-differ  # Keras layers take their inputs only
        """Pool the regions of the boxes.

        Parameters
        ----------
        inputs : list
            ``boxes`` ``[batch, num_boxes, (y1, x1, y2, x2)]`` in normalized coordinates (possibly
            zero-padded),
            ``image_meta`` ``[batch, meta length]`` (see ``inference.compose_image_meta``), then the
            feature maps of the pyramid levels, each ``[batch, height, width, channels]``.

        Returns
        -------
        tf.Tensor
            ``[batch, num_boxes, pool_height, pool_width, channels]`` pooled regions.
        """
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)["image_shape"][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(
                tf.image.crop_and_resize(
                    feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear"
                )
            )

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        """Compute the output shape.

        Parameters
        ----------
        input_shape : list
            Shapes of the inputs.

        Returns
        -------
        tuple
            ``input_shape[0][:2] + pool_shape + (channels,)``.
        """
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


############################################################
#  Detection Target Layer
############################################################


def overlaps_graph(boxes1, boxes2):
    """Compute the IoU overlaps between two sets of boxes (TensorFlow).

    Parameters
    ----------
    boxes1 : tf.Tensor
        ``[N, (y1, x1, y2, x2)]`` boxes.
    boxes2 : tf.Tensor
        ``[M, (y1, x1, y2, x2)]`` boxes.

    Returns
    -------
    tf.Tensor
        ``[N, M]`` IoU overlaps.
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate the detection targets of one image.

    Subsamples the proposals and generates their target class ids, box deltas and masks. The outputs
    are zero-padded when there are not enough target ROIs.

    Parameters
    ----------
    proposals : tf.Tensor
        ``[POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)]`` in normalized coordinates, possibly
        zero-padded.
    gt_class_ids : tf.Tensor
        ``[MAX_GT_INSTANCES]`` class ids.
    gt_boxes : tf.Tensor
        ``[MAX_GT_INSTANCES, (y1, x1, y2, x2)]`` in normalized coordinates.
    gt_masks : tf.Tensor
        ``[height, width, MAX_GT_INSTANCES]`` boolean masks.
    config : Config
        Model configuration.

    Returns
    -------
    rois : tf.Tensor
        ``[TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]`` in normalized coordinates.
    class_ids : tf.Tensor
        ``[TRAIN_ROIS_PER_IMAGE]`` class ids.
    deltas : tf.Tensor
        ``[TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]``.
    masks : tf.Tensor
        ``[TRAIN_ROIS_PER_IMAGE, height, width]`` masks cropped to their box and resized to the
        network's mask size.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = crowd_iou_max < 0.001

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64),
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(
        tf.cast(roi_masks, tf.float32), boxes, box_ids, config.MASK_SHAPE
    )
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KL.Layer):
    """Subsample the proposals and generate their targets (training only).

    Parameters
    ----------
    config : Config
        Model configuration.
    **kwargs : dict
        Keyword arguments of ``keras.layers.Layer``.

    Attributes
    ----------
    config : Config
        Model configuration.
    """

    config: object

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def call(self, inputs):  # pylint: disable=arguments-differ  # Keras layers take their inputs only
        """Generate the targets of each image of the batch.

        Parameters
        ----------
        inputs : list
            ``proposals`` ``[batch, N, (y1, x1, y2, x2)]`` (normalized, possibly zero-padded),
            ``gt_class_ids``
            ``[batch, MAX_GT_INSTANCES]``, ``gt_boxes`` ``[batch, MAX_GT_INSTANCES, (y1, x1, y2,
            x2)]``
            (normalized) and ``gt_masks`` ``[batch, height, width, MAX_GT_INSTANCES]``.

        Returns
        -------
        list
            ``rois`` ``[batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]``, ``target_class_ids``
            ``[batch, TRAIN_ROIS_PER_IMAGE]``, ``target_deltas`` ``[batch, TRAIN_ROIS_PER_IMAGE,
            (dy, dx, log(dh), log(dw))]``
            and ``target_mask`` ``[batch, TRAIN_ROIS_PER_IMAGE, height, width]``, zero-padded.
        """
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU,
            names=names,
        )
        return outputs

    def compute_output_shape(self, input_shape):
        """Compute the output shapes.

        Parameters
        ----------
        input_shape : list
            Shapes of the inputs.

        Returns
        -------
        list
            Shapes of the rois, class ids, deltas and masks.
        """
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (
                None,
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.MASK_SHAPE[0],
                self.config.MASK_SHAPE[1],
            ),  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        """Compute the output masks (none).

        Parameters
        ----------
        inputs : list
            Inputs of the layer.
        mask : list | None
            Input masks. By default ``None``.

        Returns
        -------
        list
            ``[None, None, None, None]``.
        """
        return [None, None, None, None]


############################################################
#  Detection Layer
############################################################


def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine the classified proposals, filter overlaps and return the final detections.

    Parameters
    ----------
    rois : tf.Tensor
        ``[N, (y1, x1, y2, x2)]`` in normalized coordinates.
    probs : tf.Tensor
        ``[N, num_classes]`` class probabilities.
    deltas : tf.Tensor
        ``[N, num_classes, (dy, dx, log(dh), log(dw))]`` class-specific box deltas.
    window : tf.Tensor
        ``(y1, x1, y2, x2)`` of the real image inside the padded one, in normalized coordinates.
    config : Config
        Model configuration.

    Returns
    -------
    tf.Tensor
        ``[DETECTION_MAX_INSTANCES, (y1, x1, y2, x2, class_id, score)]`` in normalized
        coordinates, zero-padded.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Run non-max suppression on the ROIs of one class.

        Parameters
        ----------
        class_id : tf.Tensor
            The class id.

        Returns
        -------
        tf.Tensor
            ``[DETECTION_MAX_INSTANCES]`` indices of the kept ROIs, padded with -1.
        """
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD,
        )
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode="CONSTANT", constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat(
        [
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), np.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis],
        ],
        axis=1,
    )

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KL.Layer):
    """Turn classified proposals and their box deltas into the final detections.

    Parameters
    ----------
    config : Config | None
        Model configuration. By default ``None``.
    **kwargs : dict
        Keyword arguments of ``keras.layers.Layer``.

    Attributes
    ----------
    config : Config
        Model configuration.
    """

    config: object

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def call(self, inputs):  # pylint: disable=arguments-differ  # Keras layers take their inputs only
        """Compute the detections of each image of the batch.

        Parameters
        ----------
        inputs : list
            ``rois``, ``mrcnn_class``, ``mrcnn_bbox`` and ``image_meta``.

        Returns
        -------
        tf.Tensor
            ``[batch, DETECTION_MAX_INSTANCES, (y1, x1, y2, x2, class_id, score)]`` in normalized
            coordinates.
        """
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m["image_shape"][0]
        window = norm_boxes_graph(m["window"], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU,
        )

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6]
        )

    def compute_output_shape(self, input_shape):
        """Compute the output shape.

        Parameters
        ----------
        input_shape : list
            Shapes of the inputs.

        Returns
        -------
        tuple
            ``(None, DETECTION_MAX_INSTANCES, 6)``.
        """
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Build the region proposal network on one feature map.

    Parameters
    ----------
    feature_map : tf.Tensor
        ``[batch, height, width, depth]`` backbone features.
    anchors_per_location : int
        Number of anchors per feature map pixel.
    anchor_stride : int
        Density of the anchors: 1 for every feature map pixel, 2 for every other pixel.

    Returns
    -------
    rpn_class_logits : tf.Tensor
        ``[batch, H * W * anchors_per_location, 2]`` anchor classifier logits (before softmax).
    rpn_probs : tf.Tensor
        ``[batch, H * W * anchors_per_location, 2]`` anchor classifier probabilities.
    rpn_bbox : tf.Tensor
        ``[batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))]`` deltas to apply to the
        anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        strides=anchor_stride,
        name="rpn_conv_shared",
    )(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(
        2 * anchors_per_location, (1, 1), padding="valid", activation="linear", name="rpn_class_raw"
    )(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(
        anchors_per_location * 4, (1, 1), padding="valid", activation="linear", name="rpn_bbox_pred"
    )(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Build the region proposal network as a Keras model, to share its weights across levels.

    Parameters
    ----------
    anchor_stride : int
        Density of the anchors: 1 for every feature map pixel, 2 for every other pixel.
    anchors_per_location : int
        Number of anchors per feature map pixel.
    depth : int
        Depth of the backbone feature maps.

    Returns
    -------
    keras.Model
        Model outputting ``rpn_class_logits``, ``rpn_probs`` and ``rpn_bbox`` (see ``rpn_graph``).
    """
    input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################


def fpn_classifier_graph(
    rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024
):
    """Build the classifier and box regressor heads of the feature pyramid network.

    Parameters
    ----------
    rois : tf.Tensor
        ``[batch, num_rois, (y1, x1, y2, x2)]`` proposals in normalized coordinates.
    feature_maps : list
        Feature maps ``[P2, P3, P4, P5]`` of the pyramid.
    image_meta : tf.Tensor
        ``[batch, meta length]`` (see ``inference.compose_image_meta``).
    pool_size : int
        Width of the square feature map produced by ROI pooling.
    num_classes : int
        Number of classes, the depth of the outputs.
    train_bn : bool
        Train (``True``) or freeze the batch normalization layers. By default ``True``.
    fc_layers_size : int
        Size of the two fully-connected layers. By default 1024.

    Returns
    -------
    logits : tf.Tensor
        ``[batch, num_rois, num_classes]`` classifier logits (before softmax).
    probs : tf.Tensor
        ``[batch, num_rois, num_classes]`` classifier probabilities.
    bbox_deltas : tf.Tensor
        ``[batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]`` deltas to apply to the
        proposals.
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")(
        [rois, image_meta] + feature_maps
    )
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(
        KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1"
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_class_bn1")(x, training=train_bn)
    x = KL.Activation("relu")(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_class_bn2")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name="mrcnn_class_logits")(
        shared
    )
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(
        mrcnn_class_logits
    )

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation="linear"), name="mrcnn_bbox_fc")(
        shared
    )
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)

    mrcnn_bbox = KL.Reshape((s[1] if s[1] else -1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
    """Build the mask head of the feature pyramid network.

    Parameters
    ----------
    rois : tf.Tensor
        ``[batch, num_rois, (y1, x1, y2, x2)]`` proposals in normalized coordinates.
    feature_maps : list
        Feature maps ``[P2, P3, P4, P5]`` of the pyramid.
    image_meta : tf.Tensor
        ``[batch, meta length]`` (see ``inference.compose_image_meta``).
    pool_size : int
        Width of the square feature map produced by ROI pooling.
    num_classes : int
        Number of classes, the depth of the outputs.
    train_bn : bool
        Train (``True``) or freeze the batch normalization layers. By default ``True``.

    Returns
    -------
    tf.Tensor
        ``[batch, num_rois, MASK_POOL_SIZE * 2, MASK_POOL_SIZE * 2, num_classes]`` masks.
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")(
        [rois, image_meta] + feature_maps
    )

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn1")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn2")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn3")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn4")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(
        KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv"
    )(x)
    x = KL.TimeDistributed(
        KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask"
    )(x)
    return x


############################################################
#  Loss Functions
############################################################


def smooth_l1_loss(y_true, y_pred):
    """Compute the smooth-L1 loss element-wise.

    Parameters
    ----------
    y_true : tf.Tensor
        Targets, typically ``[N, 4]``.
    y_pred : tf.Tensor
        Predictions, same shape.

    Returns
    -------
    tf.Tensor
        The loss, same shape as the inputs.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """Compute the RPN anchor classifier loss.

    Parameters
    ----------
    rpn_match : tf.Tensor
        ``[batch, anchors, 1]`` anchor match type: 1 positive, -1 negative, 0 neutral.
    rpn_class_logits : tf.Tensor
        ``[batch, anchors, 2]`` background/foreground logits.

    Returns
    -------
    tf.Tensor
        Scalar loss.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(
        target=anchor_class, output=rpn_class_logits, from_logits=True
    )
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Compute the RPN bounding box loss.

    Parameters
    ----------
    config : Config
        Model configuration.
    target_bbox : tf.Tensor
        ``[batch, max positive anchors, (dy, dx, log(dh), log(dw))]``, zero-padded.
    rpn_match : tf.Tensor
        ``[batch, anchors, 1]`` anchor match type: 1 positive, -1 negative, 0 neutral.
    rpn_bbox : tf.Tensor
        ``[batch, anchors, (dy, dx, log(dh), log(dw))]`` predicted deltas.

    Returns
    -------
    tf.Tensor
        Scalar loss.
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """Compute the loss of the classifier head.

    Parameters
    ----------
    target_class_ids : tf.Tensor
        ``[batch, num_rois]`` class ids, zero-padded.
    pred_class_logits : tf.Tensor
        ``[batch, num_rois, num_classes]`` logits.
    active_class_ids : tf.Tensor
        ``[batch, num_classes]``: 1 for the classes of the image's dataset, 0 otherwise.

    Returns
    -------
    tf.Tensor
        Scalar loss.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, "int64")

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits
    )

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Compute the loss of the box refinement head.

    Parameters
    ----------
    target_bbox : tf.Tensor
        ``[batch, num_rois, (dy, dx, log(dh), log(dw))]``.
    target_class_ids : tf.Tensor
        ``[batch, num_rois]`` class ids.
    pred_bbox : tf.Tensor
        ``[batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]``.

    Returns
    -------
    tf.Tensor
        Scalar loss.
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(
        tf.size(target_bbox) > 0,
        smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
        tf.constant(0.0),
    )
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Compute the binary cross-entropy loss of the mask head.

    Parameters
    ----------
    target_masks : tf.Tensor
        ``[batch, num_rois, height, width]`` float32 masks of 0 and 1, zero-padded.
    target_class_ids : tf.Tensor
        ``[batch, num_rois]`` class ids, zero-padded.
    pred_masks : tf.Tensor
        ``[batch, proposals, height, width, num_classes]`` float32 masks between 0 and 1.

    Returns
    -------
    tf.Tensor
        Scalar loss.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(
        tf.size(y_true) > 0, K.binary_crossentropy(target=y_true, output=y_pred), tf.constant(0.0)
    )
    loss = K.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################


def load_image_gt(dataset, config, image_id, augment=False, augmentation=None, use_mini_mask=False):
    """Load the image and ground truth (masks, boxes, class ids) of one image.

    Parameters
    ----------
    dataset : Dataset
        Dataset to load from.
    config : Config
        Model configuration.
    image_id : int
        Internal image id.
    augment : bool
        Deprecated (use ``augmentation``): random horizontal flips. By default ``False``.
    augmentation : object | None
        imgaug-style augmenter, e.g. ``imgaug.augmenters.Fliplr(0.5)``. By default ``None``.
    use_mini_mask : bool
        Return masks cropped to their box and resized to ``MINI_MASK_SHAPE`` instead of full-size
        ones. By default ``False``.

    Returns
    -------
    image : np.ndarray
        ``[height, width, 3]`` resized image.
    image_meta : np.ndarray
        Image meta (see ``inference.compose_image_meta``).
    class_ids : np.ndarray
        ``[instance_count]`` class ids.
    bbox : np.ndarray
        ``[instance_count, (y1, x1, y2, x2)]`` boxes.
    mask : np.ndarray
        ``[height, width, instance_count]`` masks (``MINI_MASK_SHAPE`` with mini masks).
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = inference.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE,
    )
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        # pylint: disable-next=import-outside-toplevel,import-error  # optional, not a dependency
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = [
            "Sequential",
            "SomeOf",
            "OneOf",
            "Sometimes",
            "Fliplr",
            "Flipud",
            "CropAndPad",
            "Affine",
            "PiecewiseAffine",
        ]

        # pylint: disable-next=unused-argument  # signature imposed by imgaug.HooksImages
        def hook(images, augmenter, parents, default):
            """Tell imgaug which augmenters to apply to masks.

            Parameters
            ----------
            images : np.ndarray
                The masks being augmented.
            augmenter : object
                The imgaug augmenter about to run.
            parents : list
                Parent augmenters.
            default : bool
                imgaug's default decision.

            Returns
            -------
            bool
                Whether ``augmenter`` also applies to masks (geometric augmenters only).
            """
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(
        image_id, original_shape, image.shape, window, scale, active_class_ids
    )

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate the targets of the classifier and mask heads (debugging or heads-only training).

    Parameters
    ----------
    rpn_rois : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` proposals.
    gt_class_ids : np.ndarray
        ``[instance_count]`` class ids.
    gt_boxes : np.ndarray
        ``[instance_count, (y1, x1, y2, x2)]`` boxes.
    gt_masks : np.ndarray
        ``[height, width, instance_count]`` full-size or mini masks.
    config : Config
        Model configuration.

    Returns
    -------
    rois : np.ndarray
        ``[TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]``.
    class_ids : np.ndarray
        ``[TRAIN_ROIS_PER_IMAGE]`` class ids.
    bboxes : np.ndarray
        ``[TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]`` class-specific refinements.
    masks : np.ndarray
        ``[TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES]`` class-specific masks.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, f"Expected int but got {gt_class_ids.dtype}"
    assert gt_boxes.dtype == np.int32, f"Expected int but got {gt_boxes.dtype}"
    assert gt_masks.dtype == np.bool_, f"Expected bool but got {gt_masks.dtype}"

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, (
        f"keep doesn't match ROI batch size {keep.shape[0]}, {config.TRAIN_ROIS_PER_IMAGE}"
    )

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4]
    )
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros(
        (
            config.TRAIN_ROIS_PER_IMAGE,
            config.MASK_SHAPE[0],
            config.MASK_SHAPE[1],
            config.NUM_CLASSES,
        ),
        dtype=np.float32,
    )
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = np.round(
                inference.resize(class_mask, (gt_h, gt_w))
            ).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = inference.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


# pylint: disable-next=unused-argument  # image_shape kept: public Matterport API
def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Find the positive and negative anchors, and the deltas refining the positive ones.

    Parameters
    ----------
    image_shape : tuple
        ``(height, width, channels)`` of the molded image.
    anchors : np.ndarray
        ``[num_anchors, (y1, x1, y2, x2)]`` anchors.
    gt_class_ids : np.ndarray
        ``[num_gt_boxes]`` class ids.
    gt_boxes : np.ndarray
        ``[num_gt_boxes, (y1, x1, y2, x2)]`` boxes.
    config : Config
        Model configuration.

    Returns
    -------
    rpn_match : np.ndarray
        ``[num_anchors]`` int32: 1 positive, -1 negative, 0 neutral anchor.
    rpn_bbox : np.ndarray
        ``[RPN_TRAIN_ANCHORS_PER_IMAGE, (dy, dx, log(dh), log(dw))]`` deltas of the positive
        anchors.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    # TODO: use box_refinement() rather than duplicating the code here
    for ix, (i, a) in enumerate(zip(ids, anchors[ids], strict=True)):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV

    return rpn_match, rpn_bbox


# pylint: disable-next=unused-argument  # gt_class_ids kept: public Matterport API
def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generate random ROI proposals, similar to what the region proposal network would output.

    Parameters
    ----------
    image_shape : tuple
        ``(height, width, channels)`` of the image.
    count : int
        Number of ROIs to generate.
    gt_class_ids : np.ndarray
        ``[N]`` class ids.
    gt_boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` boxes, in pixels.

    Returns
    -------
    np.ndarray
        ``[count, (y1, x1, y2, x2)]`` ROIs, in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        # pylint: disable-next=unbalanced-tuple-unpacking  # np.split(..., 2) gives 2 arrays
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        # pylint: disable-next=unbalanced-tuple-unpacking  # np.split(..., 2) gives 2 arrays
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i : rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    # pylint: disable-next=unbalanced-tuple-unpacking  # np.split(..., 2) gives 2 arrays
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    # pylint: disable-next=unbalanced-tuple-unpacking  # np.split(..., 2) gives 2 arrays
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(
    dataset,
    config,
    shuffle=True,
    augment=False,
    augmentation=None,
    random_rois=0,
    batch_size=1,
    detection_targets=False,
    no_augmentation_sources=None,
) -> Iterator[tuple[list, list]]:
    """Generate batches of images and training targets forever.

    Parameters
    ----------
    dataset : Dataset
        Dataset to take the images from.
    config : Config
        Model configuration.
    shuffle : bool
        Shuffle the images before every epoch. By default ``True``.
    augment : bool
        Deprecated (use ``augmentation``): random horizontal flips. By default ``False``.
    augmentation : object | None
        imgaug-style augmenter, e.g. ``imgaug.augmenters.Fliplr(0.5)``. By default ``None``.
    random_rois : int
        If positive, also generate that many random proposals, to train the heads without the
        region proposal network. By default 0.
    batch_size : int
        Images per batch. By default 1.
    detection_targets : bool
        Also generate detection targets (class ids, box deltas, masks), for debugging.
        By default ``False``.
    no_augmentation_sources : list[str] | None
        Dataset sources to leave un-augmented. By default ``None``.

    Yields
    ------
    tuple
        ``(inputs, outputs)``. ``inputs``: images ``[batch, H, W, C]``, image metas, ``rpn_match``
        ``[batch, N]``, ``rpn_bbox`` ``[batch, N, (dy, dx, log(dh), log(dw))]``, ``gt_class_ids``
        ``[batch, MAX_GT_INSTANCES]``, ``gt_boxes`` ``[batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]``
        and
        ``gt_masks`` ``[batch, height, width, MAX_GT_INSTANCES]``, plus the random ROIs and
        detection
        targets when requested. ``outputs`` is empty unless ``detection_targets`` is set.

    Raises
    ------
    Exception
        The error of an image, once more than 5 images have failed (failing images are logged and
        skipped).
    """
    b = 0  # batch item index
    # Batch arrays, allocated when a batch starts (b == 0)
    batch_image_meta = batch_rpn_match = batch_rpn_bbox = batch_images = None
    batch_gt_class_ids = batch_gt_boxes = batch_gt_masks = None
    batch_rpn_rois = batch_rois = batch_mrcnn_class_ids = batch_mrcnn_bbox = None
    batch_mrcnn_mask = None
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = inference.generate_pyramid_anchors(
        config.RPN_ANCHOR_SCALES,
        config.RPN_ANCHOR_RATIOS,
        backbone_shapes,
        config.BACKBONE_STRIDES,
        config.RPN_ANCHOR_STRIDE,
    )

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]["source"] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
                    dataset,
                    config,
                    image_id,
                    augment=augment,
                    augmentation=None,
                    use_mini_mask=config.USE_MINI_MASK,
                )
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
                    dataset,
                    config,
                    image_id,
                    augment=augment,
                    augmentation=augmentation,
                    use_mini_mask=config.USE_MINI_MASK,
                )

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(
                image.shape, anchors, gt_class_ids, gt_boxes, config
            )

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = build_detection_targets(
                        rpn_rois, gt_class_ids, gt_boxes, gt_masks, config
                    )

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype
                )
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype
                )
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1], config.MAX_GT_INSTANCES),
                    dtype=gt_masks.dtype,
                )
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype
                    )
                    if detection_targets:
                        batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype
                        )
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype
                        )
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype
                        )

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False
                )
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, : gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, : gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, : gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [
                    batch_images,
                    batch_image_meta,
                    batch_rpn_match,
                    batch_rpn_bbox,
                    batch_gt_class_ids,
                    batch_gt_boxes,
                    batch_gt_masks,
                ]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                        outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except Exception:  # pylint: disable=broad-exception-caught  # log and skip bad images
            # Log it and skip the image
            logging.exception("Error processing image %s", dataset.image_info[image_id])
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class
############################################################


class MaskRCNN:
    """The Mask R-CNN model: builds the Keras model, trains it, and runs detection.

    Parameters
    ----------
    mode : str
        ``"training"`` or ``"inference"``.
    config : Config
        Model configuration (a sub-class of ``Config``).
    model_dir : str
        Directory for training logs and checkpoints.

    Attributes
    ----------
    mode : str
        ``"training"`` or ``"inference"``.
    config : Config
        Model configuration.
    model_dir : str
        Directory for training logs and checkpoints.
    keras_model : keras.Model
        The Keras model.
    epoch : int
        Last completed training epoch.
    log_dir : str
        Directory of the current training run.
    checkpoint_path : str
        Path template of the checkpoints of the current run.
    anchors : np.ndarray
        Last anchor pyramid computed by ``get_anchors``, in pixels.
    """

    mode: str
    config: object
    model_dir: str
    keras_model: KM.Model
    epoch: int
    log_dir: str
    checkpoint_path: str
    anchors: np.ndarray

    def __init__(self, mode, config, model_dir):
        assert mode in ["training", "inference"]
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self._anchor_cache = {}
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build the Mask R-CNN architecture.

        Parameters
        ----------
        mode : str
            ``"training"`` or ``"inference"``: the inputs and outputs of the model differ.
        config : Config
            Model configuration.

        Returns
        -------
        keras.Model
            The model.

        Raises
        ------
        ValueError
            If the image size can't be divided by 2 six times (the pyramid levels).
        """
        assert mode in ["training", "inference"]

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise ValueError(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. "
            )

        # Inputs
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(*x))(
                [input_gt_boxes, K.shape(input_image)[1:3]]
            )
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks",
                    dtype=bool,
                )
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks",
                    dtype=bool,
                )
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True, train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(
                input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN
            )
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c5p5")(C5)
        P4 = KL.Add(name="fpn_p4add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c4p4")(C4),
            ]
        )
        P3 = KL.Add(name="fpn_p3add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c3p3")(C3),
            ]
        )
        P2 = KL.Add(name="fpn_p2add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c2p2")(C2),
            ]
        )
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = AnchorsLayer(anchors, name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(
            config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE
        )
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs, strict=True))
        outputs = [
            KL.Concatenate(axis=1, name=n)(list(o))
            for o, n in zip(outputs, output_names, strict=True)
        ]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = (
            config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        )
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config,
        )([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(
                input_image_meta
            )

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(
                    shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32
                )
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(*x))(
                    [input_rois, K.shape(input_image)[1:3]]
                )
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                config, name="proposal_targets"
            )([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
                rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,
            )

            mrcnn_mask = build_fpn_mask_graph(
                rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
            )

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits]
            )
            rpn_bbox_loss = KL.Lambda(
                lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss"
            )([input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids]
            )
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox]
            )
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask]
            )

            # Model
            inputs = [
                input_image,
                input_image_meta,
                input_rpn_match,
                input_rpn_bbox,
                input_gt_class_ids,
                input_gt_boxes,
                input_gt_masks,
            ]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [
                rpn_class_logits,
                rpn_class,
                rpn_bbox,
                mrcnn_class_logits,
                mrcnn_class,
                mrcnn_bbox,
                mrcnn_mask,
                rpn_rois,
                output_rois,
                rpn_class_loss,
                rpn_bbox_loss,
                class_loss,
                bbox_loss,
                mask_loss,
            ]
            model = KM.Model(inputs, outputs, name="mask_rcnn")
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
                rpn_rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,
            )

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
            )

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(
                detection_boxes,
                mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
            )

            model = KM.Model(
                [input_image, input_image_meta, input_anchors],
                [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                name="mask_rcnn",
            )

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Find the last checkpoint of the last training run in the model directory.

        Returns
        -------
        str
            Path of the checkpoint.

        Raises
        ------
        FileNotFoundError
            If there is no training run or no checkpoint.
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            raise FileNotFoundError(
                errno.ENOENT, f"Could not find model directory under {self.model_dir}"
            )
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            raise FileNotFoundError(errno.ENOENT, f"Could not find weight files in {dir_name}")
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Load weights, with multi-GPU support and layers that can be excluded.

        Parameters
        ----------
        filepath : str
            Path of the ``.h5`` weights.
        by_name : bool
            Match layers by name instead of by order. By default ``False``.
        exclude : list[str] | None
            Names of layers not to load. By default ``None``.
        """
        # TF-internal module (TF 2.15): loads Matterport's by-name .h5 checkpoints.
        # pylint: disable-next=import-outside-toplevel,no-name-in-module
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        f = h5py.File(filepath, mode="r")
        if "layer_names" not in f.attrs and "model_weights" in f:
            f = f["model_weights"]

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = (
            keras_model.inner_model.layers
            if hasattr(keras_model, "inner_model")
            else keras_model.layers
        )

        # Exclude some layers
        if exclude:
            layers = [layer for layer in layers if layer.name not in exclude]

        if by_name:
            hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            hdf5_format.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, "close"):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Download the ImageNet weights of the ResNet-50 backbone from Keras.

        Returns
        -------
        str
            Path of the weights file.
        """
        TF_WEIGHTS_PATH_NO_TOP = (
            "https://github.com/fchollet/deep-learning-models/"
            "releases/download/v0.2/"
            "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )
        weights_path = get_file(
            "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
            md5_hash="a268eb855778b3df3c7506639542a6af",
        )
        return weights_path

    def compile(self, learning_rate, momentum):
        """Get the model ready for training: losses, regularization, metrics, then Keras compile.

        Parameters
        ----------
        learning_rate : float
            Learning rate of the SGD optimizer.
        momentum : float
            Momentum of the SGD optimizer.
        """
        # Optimizer object
        optimizer = keras.optimizers.legacy.SGD(
            lr=learning_rate, momentum=momentum, clipnorm=self.config.GRADIENT_CLIP_NORM
        )
        # Add Losses
        # First, clear previously set losses to avoid duplication
        # Keras 2 internals: drop the losses of a previous compile before adding them again.
        if self.keras_model._losses:  # pylint: disable=protected-access
            self.keras_model._losses = []  # pylint: disable=protected-access
        # self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",
            "rpn_bbox_loss",
            "mrcnn_class_loss",
            "mrcnn_bbox_loss",
            "mrcnn_mask_loss",
        ]
        added_loss_name = []
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output.name in added_loss_name:
                continue
            loss = tf.math.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(
                name, 1.0
            )
            self.keras_model.add_loss(loss)
            added_loss_name.append(layer.output.name)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if "gamma" not in w.name and "beta" not in w.name
        ]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            loss = tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(
                name, 1.0
            )
            self.keras_model.add_metric(loss, name=name, aggregation="mean")

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Make the layers whose name matches a regular expression trainable, and freeze the others.

        Parameters
        ----------
        layer_regex : str
            Regular expression of the layer names to train.
        keras_model : keras.Model | None
            Model to walk (for nested models); ``None`` for this model. By default ``None``.
        indent : int
            Indentation of the log lines (nesting depth). By default 0.
        verbose : int
            Log the trainable layers when positive. By default 1.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = (
            keras_model.inner_model.layers
            if hasattr(keras_model, "inner_model")
            else keras_model.layers
        )

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == "Model":
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == "TimeDistributed":
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log(f"{' ' * indent}{layer.name:20}   ({layer.__class__.__name__})")

    def set_log_dir(self, model_path=None):
        """Set the log directory and the epoch counter.

        Parameters
        ----------
        model_path : str | None
            Checkpoint to resume from: the log directory and epoch are read from its name. ``None``,
            or a
            name in another format, starts a new directory at epoch 0. By default ``None``.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = (
                r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})"
                r"[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            )
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                    int(m.group(5)),
                )
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print(f"Re-starting from epoch {self.epoch}")

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, f"{self.config.NAME.lower()}{now:%Y%m%dT%H%M}")

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(
            self.log_dir, f"mask_rcnn_{self.config.NAME.lower()}_*epoch*.h5"
        )
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(
        self,
        train_dataset,
        val_dataset,
        learning_rate,
        epochs,
        layers,
        augmentation=None,
        custom_callbacks=None,
        no_augmentation_sources=None,
    ):
        """Train the model.

        Parameters
        ----------
        train_dataset : Dataset
            Training set.
        val_dataset : Dataset
            Validation set.
        learning_rate : float
            Learning rate.
        epochs : int
            Total number of epochs: previous epochs count as done, so this is the epoch to reach,
            not
            the number of epochs of this call.
        layers : str
            Layers to train: a regular expression of layer names, or ``"heads"`` (region proposal
            network, classifier and mask heads), ``"all"``, ``"3+"``, ``"4+"`` or ``"5+"`` (ResNet
            stage
            3, 4 or 5 and up).
        augmentation : object | None
            imgaug-style augmenter, e.g. ``imgaug.augmenters.Fliplr(0.5)``. By default ``None``.
        custom_callbacks : list | None
            Extra ``keras.callbacks`` for the fit. By default ``None``.
        no_augmentation_sources : list[str] | None
            Dataset sources to leave un-augmented. By default ``None``.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": (
                r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)"
                r"|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)"
            ),
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        layers = layer_regex.get(layers, layers)

        # Data generators
        train_generator = data_generator(
            train_dataset,
            self.config,
            shuffle=True,
            augmentation=augmentation,
            batch_size=self.config.BATCH_SIZE,
            no_augmentation_sources=no_augmentation_sources,
        )
        val_generator = data_generator(
            val_dataset, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE
        )

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=False,
                update_freq="batch",
            ),
            keras.callbacks.ModelCheckpoint(
                self.checkpoint_path, verbose=0, save_weights_only=True
            ),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log(f"\nStarting at epoch {self.epoch}. LR={learning_rate}\n")
        log(f"Checkpoint Path: {self.checkpoint_path}")
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        workers = 0 if os.name == "nt" else multiprocessing.cpu_count()

        print("NB workers", workers)
        self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Resize and normalize images for the network (see ``inference.mold_inputs``).

        Parameters
        ----------
        images : list
            RGB images, possibly of different sizes.

        Returns
        -------
        tuple
            ``(molded_images, image_metas, windows)``.
        """
        return inference.mold_inputs(images, self.config)

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape, image_shape, window):
        """Convert the network outputs of one image (see ``inference.unmold_detections``).

        Parameters
        ----------
        detections : np.ndarray
            ``[N, (y1, x1, y2, x2, class_id, score)]`` in normalized coordinates.
        mrcnn_mask : np.ndarray
            ``[N, height, width, num_classes]`` masks.
        original_image_shape : tuple
            Shape of the image before resizing.
        image_shape : tuple
            Shape of the molded image.
        window : np.ndarray
            ``(y1, x1, y2, x2)`` of the real image inside the molded one, in pixels.

        Returns
        -------
        tuple
            ``(boxes, class_ids, scores, masks)`` in original image coordinates.
        """
        return inference.unmold_detections(
            detections, mrcnn_mask, original_image_shape, image_shape, window
        )

    def detect(self, images, verbose=0):
        """Run the detection pipeline.

        Parameters
        ----------
        images : list
            RGB images, possibly of different sizes; ``BATCH_SIZE`` of them.
        verbose : int
            Log the inputs when positive. By default 0.

        Returns
        -------
        list[dict]
            One dict per image: ``rois`` ``[N, (y1, x1, y2, x2)]``, ``class_ids`` ``[N]``,
            ``scores``
            ``[N]`` and ``masks`` ``[height, width, N]``.
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log(f"Processing {len(images)} images")
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, (
                "After resizing, all images must have the same size. "
                "Check IMAGE_RESIZE_MODE and image sizes."
            )

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        # print("PPPPPPPPPPPP ", image_metas)
        # print("QQQQQQQQQ ", anchors)
        # np.save("image_metas.npy", image_metas)
        # np.save("anchors.npy", anchors)
        # np.save("molded_images.npy", molded_images)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)

        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict(
            [molded_images, image_metas, anchors], verbose=0
        )
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(
                detections[i], mrcnn_mask[i], image.shape, molded_images[i].shape, windows[i]
            )
            results.append(
                {
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                }
            )
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Run the detection pipeline on already molded images (debugging).

        Parameters
        ----------
        molded_images : np.ndarray
            Images as returned by ``load_image_gt``.
        image_metas : np.ndarray
            Their image metas, also returned by ``load_image_gt``.
        verbose : int
            Log the inputs when positive. By default 0.

        Returns
        -------
        list[dict]
            One dict per image: ``rois``, ``class_ids``, ``scores`` and ``masks``, like ``detect``.
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE, (
            "Number of images must be equal to BATCH_SIZE"
        )

        if verbose:
            log(f"Processing {len(molded_images)} images")
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict(
            [molded_images, image_metas, anchors], verbose=0
        )
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(
                detections[i], mrcnn_mask[i], image.shape, image.shape, window
            )
            results.append(
                {
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                }
            )
        return results

    def get_anchors(self, image_shape):
        """Get the normalized anchor pyramid of an image size, computed once per size.

        Parameters
        ----------
        image_shape : tuple
            ``(height, width, ...)`` of the molded image.

        Returns
        -------
        np.ndarray
            ``[N, (y1, x1, y2, x2)]`` anchors in normalized coordinates.
        """
        # Cache anchors and reuse if image shape is the same
        if tuple(image_shape) not in self._anchor_cache:
            # Generate Anchors
            a = inference.pyramid_anchors(self.config, image_shape)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = inference.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Find an ancestor of a TensorFlow tensor in the computation graph.

        Parameters
        ----------
        tensor : tf.Tensor
            Symbolic tensor.
        name : str
            Name of the ancestor to find.
        checked : list | None
            Tensors already searched, to avoid loops (internal). By default ``None``.

        Returns
        -------
        tf.Tensor | None
            The ancestor, or ``None`` if not found.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """Find the layer holding the weights of a possibly wrapped layer.

        Parameters
        ----------
        layer : keras.layers.Layer
            Layer, possibly wrapped (e.g. ``TimeDistributed``).

        Returns
        -------
        keras.layers.Layer
            The innermost layer.
        """
        if layer.__class__.__name__ == "TimeDistributed":
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """List the layers that have weights.

        Returns
        -------
        list
            The layers.
        """
        layers = []
        for wrapped in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            layer = self.find_trainable_layer(wrapped)
            # Include layer if it has weights
            if layer.get_weights():
                layers.append(layer)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Run part of the computation graph and return the requested outputs.

        Parameters
        ----------
        images : list
            Images; already molded if ``image_metas`` is given.
        outputs : list[tuple]
            ``(name, tensor)`` pairs to compute; the names label the results.
        image_metas : np.ndarray | None
            Image metas of already molded images. By default ``None``.

        Returns
        -------
        collections.OrderedDict
            The results as numpy arrays, keyed by the given names.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.0)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict(zip(outputs.keys(), outputs_np, strict=True))
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np

    def save(self, dir_path):
        """Export the model for serving: ``config.json`` and a TF SavedModel in ``dir_path``.

        Parameters
        ----------
        dir_path : str
            Existing export directory.
        """
        self.config.save_config(dir_path)
        self.keras_model.save(dir_path)


############################################################
#  Data Formatting
############################################################


def parse_image_meta_graph(meta):
    """Unpack a batch of image metas (see ``inference.compose_image_meta``) (TensorFlow).

    Parameters
    ----------
    meta : tf.Tensor
        ``[batch, meta length]``.

    Returns
    -------
    dict
        ``image_id``, ``original_image_shape``, ``image_shape``, ``window``, ``scale`` and
        ``active_class_ids`` tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


############################################################
#  Miscellenous Graph Functions
############################################################


def trim_zeros_graph(boxes, name="trim_zeros"):
    """Remove the all-zero (padding) boxes (TensorFlow).

    Parameters
    ----------
    boxes : tf.Tensor
        ``[N, 4]`` boxes.
    name : str
        Name of the operation. By default ``"trim_zeros"``.

    Returns
    -------
    boxes : tf.Tensor
        The non-zero boxes.
    non_zeros : tf.Tensor
        ``[N]`` boolean mask of the kept rows.
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Pick a different number of values from each row (TensorFlow).

    Parameters
    ----------
    x : tf.Tensor
        ``[rows, values]`` tensor.
    counts : tf.Tensor
        ``[rows]`` number of values to keep from each row.
    num_rows : int
        Number of rows.

    Returns
    -------
    tf.Tensor
        The kept values, concatenated.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, : counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Convert boxes from pixel to normalized coordinates (TensorFlow).

    In pixel coordinates ``(y2, x2)`` is outside the box; in normalized coordinates it's inside.

    Parameters
    ----------
    boxes : tf.Tensor
        ``[..., (y1, x1, y2, x2)]`` in pixels.
    shape : tf.Tensor
        ``(height, width)`` in pixels.

    Returns
    -------
    tf.Tensor
        ``[..., (y1, x1, y2, x2)]`` in normalized coordinates.
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0.0, 0.0, 1.0, 1.0])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Convert boxes from normalized to pixel coordinates (TensorFlow).

    Parameters
    ----------
    boxes : tf.Tensor
        ``[..., (y1, x1, y2, x2)]`` in normalized coordinates.
    shape : tf.Tensor
        ``(height, width)`` in pixels.

    Returns
    -------
    tf.Tensor
        ``[..., (y1, x1, y2, x2)]`` in pixels, int32.
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0.0, 0.0, 1.0, 1.0])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
