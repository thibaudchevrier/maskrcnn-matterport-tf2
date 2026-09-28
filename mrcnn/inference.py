"""Inference-side helpers of Mask R-CNN: image resizing, anchors, image meta, and unmolding.

Turns images into network inputs and network outputs back into boxes and masks. numpy and
scikit-image only (no TensorFlow), so a service can run an exported model with any TensorFlow
version, without this package's training dependencies.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details).
Written by Waleed Abdulla.
"""

import math
import random

import numpy as np
import skimage.transform


def resize(
    image,
    output_shape,
    order=1,
    mode="constant",
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=False,
    anti_aliasing_sigma=None,
):
    """Resize an image with the package-wide scikit-image defaults.

    Parameters
    ----------
    image : np.ndarray
        Image or mask to resize.
    output_shape : tuple
        Target ``(height, width)``.
    order : int
        Spline interpolation order (0 nearest, 1 bilinear). By default 1.
    mode : str
        How points outside the boundaries are filled. By default ``"constant"``.
    cval : float
        Fill value for ``mode="constant"``. By default 0.
    clip : bool
        Clip the output to the input range. By default ``True``.
    preserve_range : bool
        Keep the input value range instead of converting to [0, 1]. By default ``False``.
    anti_aliasing : bool
        Gaussian-smooth before downsampling. By default ``False``.
    anti_aliasing_sigma : float | None
        Standard deviation of that smoothing. By default ``None``.

    Returns
    -------
    np.ndarray
        The resized image.
    """
    return skimage.transform.resize(
        image,
        output_shape,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma,
    )


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resize an image keeping its aspect ratio, then pad or crop it.

    Parameters
    ----------
    image : np.ndarray
        Image, shape ``(height, width, channels)``.
    min_dim : int | None
        Scale up so the short side is at least this long. By default ``None``.
    max_dim : int | None
        In ``"square"`` mode, the long side never exceeds it, and the output is
        ``max_dim x max_dim``. By default ``None``.
    min_scale : float | None
        Scale up by at least this factor, even if ``min_dim`` doesn't require it.
        By default ``None``.
    mode : str
        ``"none"`` (unchanged), ``"square"`` (resize and zero-pad to ``max_dim x max_dim``),
        ``"pad64"`` (zero-pad to multiples of 64, scaling up first with ``min_dim`` /
        ``min_scale``) or ``"crop"`` (scale, then take a random ``min_dim x min_dim`` crop; training
        only). By default ``"square"``.

    Returns
    -------
    image : np.ndarray
        The resized image, same dtype as the input.
    window : tuple
        ``(y1, x1, y2, x2)`` of the image inside the padded output; ``(y2, x2)`` excluded.
    scale : float
        Scale factor applied to the image.
    padding : list
        Padding added, ``[(top, bottom), (left, right), (0, 0)]``.
    crop : tuple | None
        ``(y, x, height, width)`` of the crop in ``"crop"`` mode, ``None`` otherwise.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the modes above.
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y : y + min_dim, x : x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise ValueError(f"Mode {mode} not supported")
    return image.astype(image_dtype), window, scale, padding, crop


def norm_boxes(boxes, shape):
    """Convert boxes from pixel to normalized coordinates.

    In pixel coordinates ``(y2, x2)`` is outside the box; in normalized coordinates it's inside.

    Parameters
    ----------
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` in pixels.
    shape : tuple
        ``(height, width)`` of the image, in pixels.

    Returns
    -------
    np.ndarray
        ``[N, (y1, x1, y2, x2)]`` in normalized coordinates, float32.

    Examples
    --------
    >>> norm_boxes(np.array([[0, 0, 11, 11]]), (11, 11)).tolist()
    [[0.0, 0.0, 1.0, 1.0]]
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Convert boxes from normalized to pixel coordinates, the inverse of ``norm_boxes``.

    Parameters
    ----------
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` in normalized coordinates.
    shape : tuple
        ``(height, width)`` of the image, in pixels.

    Returns
    -------
    np.ndarray
        ``[N, (y1, x1, y2, x2)]`` in pixels, int32; ``(y2, x2)`` excluded.

    Examples
    --------
    >>> denorm_boxes(np.array([[0.0, 0.0, 1.0, 1.0]]), (11, 11)).tolist()
    [[0, 0, 11, 11]]
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def unmold_mask(mask, bbox, image_shape):
    """Fit a small predicted mask into its box on a full-size canvas.

    Parameters
    ----------
    mask : np.ndarray
        ``[height, width]`` float mask predicted by the network, typically 28x28.
    bbox : np.ndarray
        ``(y1, x1, y2, x2)`` box to fit the mask in, in pixels.
    image_shape : tuple
        Shape of the original image, ``(height, width, ...)``.

    Returns
    -------
    np.ndarray
        Boolean mask of the original image's height and width (threshold 0.5).
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """Generate the anchors of one feature map.

    Parameters
    ----------
    scales : np.ndarray
        Anchor sizes in pixels, e.g. ``[32, 64, 128]``.
    ratios : np.ndarray
        Anchor width/height ratios, e.g. ``[0.5, 1, 2]``.
    shape : tuple
        ``(height, width)`` of the feature map.
    feature_stride : int
        Stride of the feature map relative to the image, in pixels.
    anchor_stride : int
        Stride of the anchors on the feature map (2 = every other feature map pixel).

    Returns
    -------
    np.ndarray
        ``[N, (y1, x1, y2, x2)]`` anchors in pixels.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate the anchors of every level of a feature pyramid.

    Each scale belongs to one pyramid level; every ratio is used at every level.

    Parameters
    ----------
    scales : tuple
        One anchor size (pixels) per pyramid level.
    ratios : list
        Anchor width/height ratios.
    feature_shapes : np.ndarray
        ``[levels, (height, width)]`` of the feature maps.
    feature_strides : list
        Stride of each feature map relative to the image, in pixels.
    anchor_stride : int
        Stride of the anchors on the feature maps.

    Returns
    -------
    np.ndarray
        ``[N, (y1, x1, y2, x2)]`` anchors in pixels, ordered by scale (``scales[0]`` first).
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for scale, shape, stride in zip(scales, feature_shapes, feature_strides, strict=True):
        anchors.append(generate_anchors(scale, ratios, shape, stride, anchor_stride))
    return np.concatenate(anchors, axis=0)


def compute_backbone_shapes(config, image_shape):
    """Compute the height and width of each stage of the backbone network.

    Parameters
    ----------
    config : Config
        Model configuration (``BACKBONE``, ``BACKBONE_STRIDES``, ``COMPUTE_BACKBONE_SHAPE``).
    image_shape : tuple
        ``(height, width, ...)`` of the molded image.

    Returns
    -------
    np.ndarray
        ``[stages, (height, width)]``.
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [
            [int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES
        ]
    )


def compose_image_meta(
    image_id, original_image_shape, image_shape, window, scale, active_class_ids
):
    """Pack the attributes of an image into one 1D array.

    Parameters
    ----------
    image_id : int
        Id of the image, useful for debugging.
    original_image_shape : tuple
        ``(height, width, channels)`` before resizing or padding.
    image_shape : tuple
        ``(height, width, channels)`` after resizing and padding.
    window : tuple
        ``(y1, x1, y2, x2)`` of the real image inside the padded one, in pixels.
    scale : float
        Scale factor applied to the original image.
    active_class_ids : np.ndarray
        Class ids available in the image's dataset, useful when training on several datasets.

    Returns
    -------
    np.ndarray
        ``[1 + 3 + 3 + 4 + 1 + num_classes]`` image meta, as ``parse_image_meta`` reads it.
    """
    meta = np.array(
        [image_id]  # size=1
        + list(original_image_shape)  # size=3
        + list(image_shape)  # size=3
        + list(window)  # size=4 (y1, x1, y2, x2) in image cooredinates
        + [scale]  # size=1
        + list(active_class_ids)  # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Unpack a batch of image metas (see ``compose_image_meta``).

    Parameters
    ----------
    meta : np.ndarray
        ``[batch, meta length]``.

    Returns
    -------
    dict
        ``image_id``, ``original_image_shape``, ``image_shape``, ``window``, ``scale`` and
        ``active_class_ids``, one row per image.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def mold_image(images, config):
    """Subtract the mean pixel from RGB images and convert them to float.

    Parameters
    ----------
    images : np.ndarray
        RGB image(s), channels last.
    config : Config
        Model configuration (``MEAN_PIXEL``).

    Returns
    -------
    np.ndarray
        The normalized image(s), float.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Undo ``mold_image``: add the mean pixel back and convert to uint8.

    Parameters
    ----------
    normalized_images : np.ndarray
        Image(s) normalized by ``mold_image``.
    config : Config
        Model configuration (``MEAN_PIXEL``).

    Returns
    -------
    np.ndarray
        The RGB image(s), uint8.
    """
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


def mold_inputs(images, config):
    """Resize and normalize images into the network's input format.

    Parameters
    ----------
    images : list
        RGB images ``[height, width, channels]``; they can have different sizes.
    config : Config
        Model configuration (resizing settings, ``MEAN_PIXEL``, ``NUM_CLASSES``).

    Returns
    -------
    molded_images : np.ndarray
        ``[N, height, width, 3]`` resized and normalized images.
    image_metas : np.ndarray
        ``[N, meta length]`` image metas (see ``compose_image_meta``).
    windows : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` of each real image inside its padded one.
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        molded_image, window, scale, _, _ = resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE,
        )
        molded_image = mold_image(molded_image, config)
        image_meta = compose_image_meta(
            0,
            image.shape,
            molded_image.shape,
            window,
            scale,
            np.zeros([config.NUM_CLASSES], dtype=np.int32),
        )
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    return np.stack(molded_images), np.stack(image_metas), np.stack(windows)


def pyramid_anchors(config, image_shape):
    """Build the anchor pyramid for a molded image shape.

    Parameters
    ----------
    config : Config
        Model configuration (anchor scales, ratios, stride; backbone strides).
    image_shape : tuple
        ``(height, width, ...)`` of the molded image.

    Returns
    -------
    np.ndarray
        ``[N, (y1, x1, y2, x2)]`` anchors in pixels.
    """
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    return generate_pyramid_anchors(
        config.RPN_ANCHOR_SCALES,
        config.RPN_ANCHOR_RATIOS,
        backbone_shapes,
        config.BACKBONE_STRIDES,
        config.RPN_ANCHOR_STRIDE,
    )


def unmold_detections(detections, mrcnn_mask, original_image_shape, image_shape, window):
    """Convert the network outputs of one image into boxes and full-size masks.

    Parameters
    ----------
    detections : np.ndarray
        ``[N, (y1, x1, y2, x2, class_id, score)]`` in normalized coordinates, zero-padded after the
        last detection.
    mrcnn_mask : np.ndarray
        ``[N, height, width, num_classes]`` predicted masks.
    original_image_shape : tuple
        ``(height, width, channels)`` of the image before resizing.
    image_shape : tuple
        ``(height, width, channels)`` of the molded image.
    window : np.ndarray
        ``(y1, x1, y2, x2)`` of the real image inside the molded one, in pixels.

    Returns
    -------
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` in pixels of the original image.
    class_ids : np.ndarray
        ``[N]`` class ids.
    scores : np.ndarray
        ``[N]`` confidences.
    masks : np.ndarray
        ``[height, width, N]`` boolean masks of the original image size.
    """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
        full_masks.append(full_mask)
    full_masks = (
        np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape[:2] + (0,))
    )

    return boxes, class_ids, scores, full_masks
