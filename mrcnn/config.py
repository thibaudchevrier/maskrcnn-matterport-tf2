"""Base configuration of Mask R-CNN.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details).
Written by Waleed Abdulla.
"""

import json
from pathlib import Path

import numpy as np


class Config:
    """Base configuration of Mask R-CNN.

    Don't use this class directly: sub-class it and override the attributes that need to
    change. The upper-case attributes are the settings; ``BATCH_SIZE``, ``IMAGE_SHAPE`` and
    ``IMAGE_META_SIZE`` are computed from them when the configuration is instantiated.

    Attributes
    ----------
    NAME : str | None
        Name the configurations. For example, 'COCO', 'Experiment 3', ...etc. Useful if your code
        needs to do things differently depending on which experiment is running. Override in
        sub-classes.
    GPU_COUNT : int
        NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    IMAGES_PER_GPU : int
        Number of images to train with on each GPU. A 12GB GPU can typically handle 2 images of
        1024x1024px. Adjust based on your GPU memory and image sizes. Use the highest number that
        your GPU can handle for best performance.
    STEPS_PER_EPOCH : int
        Number of training steps per epoch. This doesn't need to match the size of the training set.
        Tensorboard updates are saved at the end of each epoch, so setting this to a smaller number
        means getting more frequent TensorBoard updates. Validation stats are also calculated at
        each epoch end and they might take a while, so don't set this too small to avoid spending a
        lot of time on validation stats.
    VALIDATION_STEPS : int
        Number of validation steps to run at the end of every training epoch. A bigger number
        improves accuracy of validation stats, but slows down the training.
    BACKBONE : str
        Backbone network architecture. Supported values are: resnet50, resnet101. You can also
        provide a callable that should have the signature of model.resnet_graph. If you do so, you
        need to supply a callable to COMPUTE_BACKBONE_SHAPE as well.
    COMPUTE_BACKBONE_SHAPE : callable | None
        Only useful if you supply a callable to BACKBONE. Should compute the shape of each layer of
        the FPN Pyramid. See model.compute_backbone_shapes.
    BACKBONE_STRIDES : list
        The strides of each layer of the FPN Pyramid. These values are based on a Resnet101
        backbone.
    FPN_CLASSIF_FC_LAYERS_SIZE : int
        Size of the fully-connected layers in the classification graph.
    TOP_DOWN_PYRAMID_SIZE : int
        Size of the top-down layers used to build the feature pyramid.
    NUM_CLASSES : int
        Number of classification classes (including background). Override in sub-classes.
    RPN_ANCHOR_SCALES : tuple
        Length of square anchor side in pixels.
    RPN_ANCHOR_RATIOS : list
        Ratios of anchors at each cell (width/height) A value of 1 represents a square anchor, and
        0.5 is a wide anchor.
    RPN_ANCHOR_STRIDE : int
        Anchor stride. If 1 then anchors are created for each cell in the backbone feature map. If
        2, then anchors are created for every other cell, and so on.
    RPN_NMS_THRESHOLD : float
        Non-max suppression threshold to filter RPN proposals. You can increase this during training
        to generate more propsals.
    RPN_TRAIN_ANCHORS_PER_IMAGE : int
        How many anchors per image to use for RPN training.
    PRE_NMS_LIMIT : int
        ROIs kept after tf.nn.top_k and before non-maximum suppression.
    POST_NMS_ROIS_TRAINING : int
        ROIs kept after non-maximum suppression (training and inference).
    POST_NMS_ROIS_INFERENCE : int
        Described with ``POST_NMS_ROIS_TRAINING``.
    USE_MINI_MASK : bool
        If enabled, resizes instance masks to a smaller size to reduce memory load. Recommended when
        using high-resolution images.
    MINI_MASK_SHAPE : tuple
        Described with ``USE_MINI_MASK``. (height, width) of the mini-mask.
    IMAGE_RESIZE_MODE : str
        How input images are resized. ``"square"`` works well for training and prediction in most
        cases. The modes:

        - ``"none"``: no resizing or padding; the image is returned unchanged.
        - ``"square"``: scale up so the short side is ``IMAGE_MIN_DIM`` without the long side
          exceeding ``IMAGE_MAX_DIM``, then zero-pad to ``[IMAGE_MAX_DIM, IMAGE_MAX_DIM]`` so
          several images fit in one batch.
        - ``"pad64"``: zero-pad width and height to multiples of 64 (the 6 levels of the FPN
          pyramid, 2**6 = 64), scaling up first if ``IMAGE_MIN_DIM`` or ``IMAGE_MIN_SCALE`` is set;
          ``IMAGE_MAX_DIM`` is ignored.
        - ``"crop"``: scale with ``IMAGE_MIN_DIM`` and ``IMAGE_MIN_SCALE``, then take a random
          ``IMAGE_MIN_DIM x IMAGE_MIN_DIM`` crop. Training only; ``IMAGE_MAX_DIM`` is ignored.
    IMAGE_MIN_DIM : int
        Described with ``IMAGE_RESIZE_MODE``.
    IMAGE_MAX_DIM : int
        Described with ``IMAGE_RESIZE_MODE``.
    IMAGE_MIN_SCALE : int
        Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further up scaling. For
        example, if set to 2 then images are scaled up to double the width and height, or more, even
        if MIN_IMAGE_DIM doesn't require it. However, in 'square' mode, it can be overruled by
        IMAGE_MAX_DIM.
    IMAGE_CHANNEL_COUNT : int
        Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4. Changing this
        requires other changes in the code. See the WIKI for more details:
        https://github.com/matterport/Mask_RCNN/wiki.
    MEAN_PIXEL : np.ndarray
        Image mean (RGB).
    TRAIN_ROIS_PER_IMAGE : int
        Number of ROIs per image to feed to classifier/mask heads. The Mask RCNN paper uses 512 but
        often the RPN doesn't generate enough positive proposals to fill this and keep a
        positive:negative ratio of 1:3. You can increase the number of proposals by adjusting the
        RPN NMS threshold.
    ROI_POSITIVE_RATIO : float
        Percent of positive ROIs used to train classifier/mask heads.
    POOL_SIZE : int
        Pooled ROIs.
    MASK_POOL_SIZE : int
        Described with ``POOL_SIZE``.
    MASK_SHAPE : list
        Shape of output mask. To change this you also need to change the neural network mask branch.
    MAX_GT_INSTANCES : int
        Maximum number of ground truth instances to use in one image.
    RPN_BBOX_STD_DEV : np.ndarray
        Bounding box refinement standard deviation for RPN and final detections.
    BBOX_STD_DEV : np.ndarray
        Described with ``RPN_BBOX_STD_DEV``.
    DETECTION_MAX_INSTANCES : int
        Max number of final detections.
    DETECTION_MIN_CONFIDENCE : float
        Minimum probability value to accept a detected instance. ROIs below this threshold are
        skipped.
    DETECTION_NMS_THRESHOLD : float
        Non-maximum suppression threshold for detection.
    LEARNING_RATE : float
        Learning rate and momentum. The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
        weights to explode. Likely due to differences in optimizer implementation.
    LEARNING_MOMENTUM : float
        Described with ``LEARNING_RATE``.
    WEIGHT_DECAY : float
        Weight decay regularization.
    LOSS_WEIGHTS : dict[str, float]
        Loss weights for more precise optimization. Can be used for R-CNN training setup.
    USE_RPN_ROIS : bool
        Use RPN ROIs or externally generated ROIs for training. Keep this True for most situations.
        Set to False if you want to train the head branches on ROI generated by code rather than the
        ROIs from the RPN. For example, to debug the classifier head without having to train the
        RPN.
    TRAIN_BN : bool
        Train or freeze batch normalization layers. None: Train BN layers. This is the normal mode.
        False: Freeze BN layers. Good when using a small batch size. True: (don't use). Set layer in
        training mode even when predicting. Defaulting to False since batch size is often small.
    GRADIENT_CLIP_NORM : float
        Gradient norm clipping.
    BATCH_SIZE : int
        Computed: ``IMAGES_PER_GPU * GPU_COUNT``.
    IMAGE_SHAPE : np.ndarray
        Computed: shape of the molded images, ``[height, width, channels]``.
    IMAGE_META_SIZE : int
        Computed: length of the image meta vector (see ``inference.compose_image_meta``).
    """

    NAME = None

    GPU_COUNT = 1

    IMAGES_PER_GPU = 2

    STEPS_PER_EPOCH = 1000

    VALIDATION_STEPS = 50

    BACKBONE = "resnet101"

    COMPUTE_BACKBONE_SHAPE = None

    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    TOP_DOWN_PYRAMID_SIZE = 256

    NUM_CLASSES = 1

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    RPN_ANCHOR_STRIDE = 1

    RPN_NMS_THRESHOLD = 0.7

    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    PRE_NMS_LIMIT = 6000

    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 0
    IMAGE_CHANNEL_COUNT = 3

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    TRAIN_ROIS_PER_IMAGE = 200

    ROI_POSITIVE_RATIO = 0.33

    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    MASK_SHAPE = [28, 28]

    MAX_GT_INSTANCES = 100

    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    DETECTION_MAX_INSTANCES = 100

    DETECTION_MIN_CONFIDENCE = 0.7

    DETECTION_NMS_THRESHOLD = 0.3

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    WEIGHT_DECAY = 0.0001

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.0,
    }

    USE_RPN_ROIS = True

    TRAIN_BN = False

    GRADIENT_CLIP_NORM = 5.0

    BATCH_SIZE: int
    IMAGE_SHAPE: np.ndarray
    IMAGE_META_SIZE: int

    def __init__(self):
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array(
                [self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, self.IMAGE_CHANNEL_COUNT]
            )
        else:
            self.IMAGE_SHAPE = np.array(
                [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_CHANNEL_COUNT]
            )

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Print every setting and its value to stdout."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print(f"{a:30} {getattr(self, a)}")
        print("\n")

    def save_config(self, dir_path):
        """Write every setting to ``<dir_path>/config.json``, as read back by ``mrcnn.serving``.

        Parameters
        ----------
        dir_path : str | os.PathLike
            Existing directory, usually the export directory of ``MaskRCNN.save``.
        """
        data = {}
        for a in dir(self):
            type_data = getattr(self, a)
            if not a.startswith("__") and not callable(type_data):
                if isinstance(type_data, np.ndarray):
                    type_data = type_data.tolist()
                elif isinstance(type_data, tuple):
                    type_data = list(type_data)
                data[a] = type_data
        with open(Path(dir_path) / "config.json", "w", encoding="utf-8") as fp:
            json.dump(data, fp)
