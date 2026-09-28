"""Common utilities of Mask R-CNN: boxes, masks, the ``Dataset`` base class, evaluation metrics.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details).
Written by Waleed Abdulla.
"""

import logging
import shutil
import urllib.request
import warnings

import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage import color, io

from mrcnn.inference import resize

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Bounding Boxes
############################################################


def extract_bboxes(mask):
    """Compute the bounding box of each instance mask.

    Parameters
    ----------
    mask : np.ndarray
        ``[height, width, num_instances]`` masks of 0 and 1.

    Returns
    -------
    np.ndarray
        ``[num_instances, (y1, x1, y2, x2)]`` int32 boxes; zeros for empty masks.
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Compute the IoU of one box with each of several boxes.

    The areas are passed in rather than computed here, so callers compute them once.

    Parameters
    ----------
    box : np.ndarray
        ``[y1, x1, y2, x2]``.
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]``.
    box_area : float
        Area of ``box``.
    boxes_area : np.ndarray
        ``[N]`` areas of ``boxes``.

    Returns
    -------
    np.ndarray
        ``[N]`` IoU values.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Compute the IoU overlaps between two sets of boxes.

    For better performance, pass the larger set first.

    Parameters
    ----------
    boxes1 : np.ndarray
        ``[N, (y1, x1, y2, x2)]``.
    boxes2 : np.ndarray
        ``[M, (y1, x1, y2, x2)]``.

    Returns
    -------
    np.ndarray
        ``[N, M]`` IoU overlaps.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Compute the IoU overlaps between two sets of masks.

    Parameters
    ----------
    masks1 : np.ndarray
        ``[height, width, N]`` masks.
    masks2 : np.ndarray
        ``[height, width, M]`` masks.

    Returns
    -------
    np.ndarray
        ``[N, M]`` IoU overlaps; zeros if either set is empty.
    """
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > 0.5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > 0.5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Run non-maximum suppression.

    Parameters
    ----------
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]``; ``(y2, x2)`` is outside the box.
    scores : np.ndarray
        ``[N]`` box scores.
    threshold : float
        IoU above which the lower-scored box is dropped.

    Returns
    -------
    np.ndarray
        Indices of the kept boxes, int32.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Apply refinement deltas to boxes.

    Parameters
    ----------
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]``; ``(y2, x2)`` is outside the box.
    deltas : np.ndarray
        ``[N, (dy, dx, log(dh), log(dw))]``.

    Returns
    -------
    np.ndarray
        ``[N, (y1, x1, y2, x2)]`` refined boxes.
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute the refinement that transforms boxes into ground-truth boxes (TensorFlow).

    Parameters
    ----------
    box : tf.Tensor
        ``[N, (y1, x1, y2, x2)]``.
    gt_box : tf.Tensor
        ``[N, (y1, x1, y2, x2)]`` ground-truth boxes.

    Returns
    -------
    tf.Tensor
        ``[N, (dy, dx, log(dh), log(dw))]`` deltas.
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute the refinement that transforms boxes into ground-truth boxes (numpy).

    Parameters
    ----------
    box : np.ndarray
        ``[N, (y1, x1, y2, x2)]``; ``(y2, x2)`` is outside the box.
    gt_box : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` ground-truth boxes.

    Returns
    -------
    np.ndarray
        ``[N, (dy, dx, log(dh), log(dw))]`` deltas.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################


class Dataset:
    """Base class for datasets.

    Sub-class it with a method that registers the images (``add_class``, ``add_image``) and
    override ``load_mask`` (and optionally ``image_reference``, ``load_image``), then call
    ``prepare()`` before use::

        class CatsAndDogsDataset(Dataset):
            def load_cats_and_dogs(self): ...
            def load_mask(self, image_id): ...

    Parameters
    ----------
    class_map : dict | None
        Not supported yet. By default ``None``.

    Attributes
    ----------
    image_info : list[dict]
        One dict per image: ``id``, ``source``, ``path``, and any extra ``add_image`` keywords.
    class_info : list[dict]
        One dict per class: ``source``, ``id``, ``name``; index 0 is the background.
    source_class_ids : dict
        Internal class ids of each source, set by ``prepare``.
    num_classes : int
        Number of classes, background included, set by ``prepare``.
    class_ids : np.ndarray
        Internal class ids, set by ``prepare``.
    class_names : list[str]
        Class names, set by ``prepare``.
    num_images : int
        Number of images, set by ``prepare``.
    class_from_source_map : dict
        ``"source.id"`` to internal class id, set by ``prepare``.
    image_from_source_map : dict
        ``"source.id"`` to internal image id, set by ``prepare``.
    sources : list[str]
        Dataset sources, set by ``prepare``.
    """

    image_info: list[dict]
    class_info: list[dict]
    source_class_ids: dict
    num_classes: int
    class_ids: np.ndarray
    class_names: list[str]
    num_images: int
    class_from_source_map: dict
    image_from_source_map: dict
    sources: list[str]

    # pylint: disable-next=unused-argument  # class_map: documented as not supported yet
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        """Register a class.

        Parameters
        ----------
        source : str
            Name of the dataset the class comes from.
        class_id : int
            Class id in that dataset.
        class_name : str
            Class name.

        Raises
        ------
        ValueError
            If ``source`` contains a dot (dots separate source and id in source class ids).
        """
        if "." in source:
            raise ValueError("Source name cannot contain a dot")
        # A source.class_id combination is only registered once
        known = any(info["source"] == source and info["id"] == class_id for info in self.class_info)
        if not known:
            self.class_info.append({"source": source, "id": class_id, "name": class_name})

    def add_image(self, source, image_id, path, **kwargs):
        """Register an image.

        Parameters
        ----------
        source : str
            Name of the dataset the image comes from.
        image_id : object
            Image id in that dataset.
        path : str | None
            Path of the image file.
        **kwargs : dict
            Extra information stored in ``image_info`` (e.g. annotations).
        """
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    # pylint: disable-next=unused-argument  # base implementation, overridden by datasets
    def image_reference(self, image_id):
        """Describe an image for debugging: its link in the source, or details to find it.

        Override it for your dataset.

        Parameters
        ----------
        image_id : int
            Internal image id.

        Returns
        -------
        str
            The reference; empty by default.
        """
        return ""

    # pylint: disable-next=unused-argument  # class_map: documented as not supported yet
    def prepare(self, class_map=None):
        """Build the internal class and image indexes; call it after registering the data.

        Parameters
        ----------
        class_map : dict | None
            Not supported yet: it would map classes of different datasets to the same id.
            By default ``None``.
        """

        def clean_name(name):
            """Shorten an object name for display (text before the first comma).

            Parameters
            ----------
            name : str
                Object name.

            Returns
            -------
            str
                The short name.
            """
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {
            f"{info['source']}.{info['id']}": id
            for info, id in zip(self.class_info, self.class_ids, strict=True)
        }
        self.image_from_source_map = {
            f"{info['source']}.{info['id']}": id
            for info, id in zip(self.image_info, self.image_ids, strict=True)
        }

        # Map sources to class_ids they support
        self.sources = list({i["source"] for i in self.class_info})
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info["source"]:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Map a source class id to the internal class id.

        Parameters
        ----------
        source_class_id : str
            ``"<source>.<id>"``, e.g. ``"coco.12"``.

        Returns
        -------
        int
            The internal class id.
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class id to its id in a source dataset.

        Parameters
        ----------
        class_id : int
            Internal class id.
        source : str
            Source dataset name.

        Returns
        -------
        object
            The class id in the source dataset.
        """
        info = self.class_info[class_id]
        assert info["source"] == source
        return info["id"]

    @property
    def image_ids(self):
        """Internal image ids, set by ``prepare``.

        Returns
        -------
        np.ndarray
            ``[num_images]`` ids.
        """
        return self._image_ids

    def source_image_link(self, image_id):
        """Get the path or URL of an image; override to link to an online copy.

        Parameters
        ----------
        image_id : int
            Internal image id.

        Returns
        -------
        str
            The image's path.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load an image as RGB.

        Parameters
        ----------
        image_id : int
            Internal image id.

        Returns
        -------
        np.ndarray
            ``[height, width, 3]`` image; grayscale is converted to RGB, alpha dropped.
        """
        # Load image
        image = io.imread(self.image_info[image_id]["path"])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    # pylint: disable-next=unused-argument  # base implementation, overridden by datasets
    def load_mask(self, image_id):
        """Load the instance masks of an image. Override it: the base class returns no masks.

        Parameters
        ----------
        image_id : int
            Internal image id.

        Returns
        -------
        masks : np.ndarray
            ``[height, width, instance_count]`` bool masks, one per instance.
        class_ids : np.ndarray
            ``[instance_count]`` class ids of the masks, int32.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning(
            "You are using the default load_mask(), maybe you need to define your own one."
        )
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_mask(mask, scale, padding, crop=None):
    """Resize and pad masks the way ``inference.resize_image`` resized their image.

    Parameters
    ----------
    mask : np.ndarray
        ``[height, width, num_instances]`` masks.
    scale : float
        Scale factor returned by ``resize_image``.
    padding : list
        ``[(top, bottom), (left, right), (0, 0)]`` returned by ``resize_image``.
    crop : tuple | None
        ``(y, x, height, width)`` returned by ``resize_image`` in ``"crop"`` mode.
        By default ``None``.

    Returns
    -------
    np.ndarray
        The resized masks.
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y : y + h, x : x + w]
    else:
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Shrink masks to their box and a small fixed size, to reduce memory load.

    ``expand_mask`` restores them to the image size.

    Parameters
    ----------
    bbox : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` boxes of the masks.
    mask : np.ndarray
        ``[height, width, N]`` masks.
    mini_shape : tuple
        ``(height, width)`` of the mini masks.

    Returns
    -------
    np.ndarray
        ``[mini_height, mini_width, N]`` bool mini masks.

    Raises
    ------
    ValueError
        If a box has an area of zero.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise ValueError("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m.astype(np.float32), mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Restore mini masks to the image size, reversing ``minimize_mask``.

    Parameters
    ----------
    bbox : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` boxes of the masks, in image pixels.
    mini_mask : np.ndarray
        ``[mini_height, mini_width, N]`` mini masks.
    image_shape : tuple
        ``(height, width, ...)`` of the image.

    Returns
    -------
    np.ndarray
        ``[height, width, N]`` bool masks.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m.astype(np.float32), (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(bool)
    return mask


############################################################
#  Anchors
############################################################


############################################################
#  Miscellaneous
############################################################


def trim_zeros(x):
    """Remove the all-zero rows used to pad an array.

    Parameters
    ----------
    x : np.ndarray
        ``[rows, columns]`` array.

    Returns
    -------
    np.ndarray
        The rows that are not all zeros.

    Examples
    --------
    >>> trim_zeros(np.array([[1, 2], [0, 0]])).tolist()
    [[1, 2]]
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(
    gt_boxes,
    gt_class_ids,
    gt_masks,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_threshold=0.5,
    score_threshold=0.0,
):
    """Match predicted instances to ground-truth instances, best scores first.

    Parameters
    ----------
    gt_boxes : np.ndarray
        ``[G, (y1, x1, y2, x2)]`` ground-truth boxes.
    gt_class_ids : np.ndarray
        ``[G]`` ground-truth class ids.
    gt_masks : np.ndarray
        ``[height, width, G]`` ground-truth masks.
    pred_boxes : np.ndarray
        ``[P, (y1, x1, y2, x2)]`` predicted boxes.
    pred_class_ids : np.ndarray
        ``[P]`` predicted class ids.
    pred_scores : np.ndarray
        ``[P]`` prediction confidences.
    pred_masks : np.ndarray
        ``[height, width, P]`` predicted masks.
    iou_threshold : float
        Mask IoU needed for a match. By default 0.5.
    score_threshold : float
        Predictions below this score are ignored. By default 0.0.

    Returns
    -------
    gt_match : np.ndarray
        ``[G]`` index of the matched prediction for each ground truth, -1 if none.
    pred_match : np.ndarray
        ``[P]`` index of the matched ground truth for each prediction, -1 if none.
    overlaps : np.ndarray
        ``[P, G]`` mask IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., : gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[: pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[: low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(
    gt_boxes,
    gt_class_ids,
    gt_masks,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_threshold=0.5,
):
    """Compute the average precision at one IoU threshold.

    Parameters
    ----------
    gt_boxes : np.ndarray
        ``[G, (y1, x1, y2, x2)]`` ground-truth boxes.
    gt_class_ids : np.ndarray
        ``[G]`` ground-truth class ids.
    gt_masks : np.ndarray
        ``[height, width, G]`` ground-truth masks.
    pred_boxes : np.ndarray
        ``[P, (y1, x1, y2, x2)]`` predicted boxes.
    pred_class_ids : np.ndarray
        ``[P]`` predicted class ids.
    pred_scores : np.ndarray
        ``[P]`` prediction confidences.
    pred_masks : np.ndarray
        ``[height, width, P]`` predicted masks.
    iou_threshold : float
        Mask IoU needed for a true positive. By default 0.5.

    Returns
    -------
    mAP : float
        Mean average precision.
    precisions : np.ndarray
        Precision at each score threshold.
    recalls : np.ndarray
        Recall at each score threshold.
    overlaps : np.ndarray
        ``[P, G]`` mask IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes,
        gt_class_ids,
        gt_masks,
        pred_boxes,
        pred_class_ids,
        pred_scores,
        pred_masks,
        iou_threshold,
    )

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(
    gt_box,
    gt_class_id,
    gt_mask,
    pred_box,
    pred_class_id,
    pred_score,
    pred_mask,
    iou_thresholds=None,
    verbose=1,
):
    """Compute the average precision over a range of IoU thresholds.

    Parameters
    ----------
    gt_box : np.ndarray
        ``[G, (y1, x1, y2, x2)]`` ground-truth boxes.
    gt_class_id : np.ndarray
        ``[G]`` ground-truth class ids.
    gt_mask : np.ndarray
        ``[height, width, G]`` ground-truth masks.
    pred_box : np.ndarray
        ``[P, (y1, x1, y2, x2)]`` predicted boxes.
    pred_class_id : np.ndarray
        ``[P]`` predicted class ids.
    pred_score : np.ndarray
        ``[P]`` prediction confidences.
    pred_mask : np.ndarray
        ``[height, width, P]`` predicted masks.
    iou_thresholds : np.ndarray | None
        IoU thresholds; ``None`` means 0.5 to 0.95 by steps of 0.05. By default ``None``.
    verbose : int
        Print the AP of each threshold when non-zero. By default 1.

    Returns
    -------
    float
        The average precision over the thresholds.
    """
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, _, _, _ = compute_ap(
            gt_box,
            gt_class_id,
            gt_mask,
            pred_box,
            pred_class_id,
            pred_score,
            pred_mask,
            iou_threshold=iou_threshold,
        )
        if verbose:
            print(f"AP @{iou_threshold:.2f}:\t {ap:.3f}")
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print(f"AP @{iou_thresholds[0]:.2f}-{iou_thresholds[-1]:.2f}:\t {AP:.3f}")
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall of predicted boxes at an IoU threshold.

    Parameters
    ----------
    pred_boxes : np.ndarray
        ``[P, (y1, x1, y2, x2)]`` predicted boxes, in pixels.
    gt_boxes : np.ndarray
        ``[G, (y1, x1, y2, x2)]`` ground-truth boxes, in pixels.
    iou : float
        IoU needed for a ground-truth box to count as found.

    Returns
    -------
    recall : float
        Fraction of ground-truth boxes found.
    positive_ids : np.ndarray
        Indices of the predicted boxes that found a ground-truth box.
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Run a graph written for one instance on each instance of a batch, and combine the results.

    Parameters
    ----------
    inputs : list
        Tensors with the same first dimension (the batch).
    graph_fn : callable
        Function building the graph for one instance; returns a tensor or a list of tensors.
    batch_size : int
        Number of slices (instances) to run.
    names : list[str] | None
        Names of the combined outputs. By default ``None``.

    Returns
    -------
    tf.Tensor | list
        The outputs stacked along the batch axis: one tensor, or a list when ``graph_fn`` returns
        several.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs, strict=True))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names, strict=True)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download the COCO trained weights of Matterport's v2.0 release.

    Parameters
    ----------
    coco_model_path : str
        Where to save the weights.
    verbose : int
        Print progress when non-zero. By default 1.
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")
