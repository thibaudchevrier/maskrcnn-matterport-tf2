"""Display and visualization helpers of Mask R-CNN (matplotlib; install the ``viz`` extra).

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details).
Written by Waleed Abdulla.
"""

import colorsys
import itertools
import random

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines, patches
from matplotlib.patches import Polygon
from skimage import measure

from mrcnn import inference, utils

############################################################
#  Visualization
############################################################


def display_images(images, titles=None, cols=4, cmap=None, norm=None, interpolation=None):
    """Show images in a grid, optionally with titles.

    Parameters
    ----------
    images : list
        Images, channels last.
    titles : list[str] | None
        One title per image. By default ``None``.
    cols : int
        Number of images per row. By default 4.
    cmap : str | None
        Matplotlib color map, e.g. ``"Blues"``. By default ``None``.
    norm : matplotlib.colors.Normalize | None
        Maps values to colors. By default ``None``.
    interpolation : str | None
        Matplotlib image interpolation. By default ``None``.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles, strict=True):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis("off")
        plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """Generate visually distinct random colors (evenly spaced hues, shuffled).

    Parameters
    ----------
    N : int
        Number of colors.
    bright : bool
        Bright (full value) or darker colors. By default ``True``.

    Returns
    -------
    list[tuple]
        ``N`` RGB colors with channels in [0, 1].
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Tint the pixels of a mask with a color, in place.

    Parameters
    ----------
    image : np.ndarray
        ``[height, width, 3]`` image, modified in place.
    mask : np.ndarray
        ``[height, width]`` mask of 0 and 1.
    color : tuple
        RGB color with channels in [0, 1].
    alpha : float
        Opacity of the tint. By default 0.5.

    Returns
    -------
    np.ndarray
        The tinted image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c]
        )
    return image


def display_instances(
    image,
    boxes,
    masks,
    class_ids,
    class_names,
    scores=None,
    title="",
    figsize=(16, 16),
    ax=None,
    show_mask=True,
    show_mask_polygon=True,
    show_bbox=True,
    colors=None,
    captions=None,
    show_caption=True,
    save_fig_path=None,
    filter_classes=None,
    min_score=None,
):
    """Draw instances (boxes, masks, captions) on an image.

    Parameters
    ----------
    image : np.ndarray
        ``[height, width, 3]`` image.
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` boxes in image coordinates.
    masks : np.ndarray
        ``[height, width, N]`` masks.
    class_ids : np.ndarray
        ``[N]`` class ids.
    class_names : list[str]
        Class names of the dataset.
    scores : np.ndarray | None
        ``[N]`` confidences. By default ``None``.
    title : str
        Figure title. By default ``""``.
    figsize : tuple
        Figure size. By default ``(16, 16)``.
    ax : matplotlib.axes.Axes | None
        Axes to draw on; a new figure is shown when ``None``. By default ``None``.
    show_mask : bool
        Draw the masks. By default ``True``.
    show_mask_polygon : bool
        Draw the mask outlines. By default ``True``.
    show_bbox : bool
        Draw the boxes. By default ``True``.
    colors : list | None
        One color per class id; random colors when ``None``. By default ``None``.
    captions : list[str] | None
        One caption per instance; ``"<class> <score>"`` when ``None``. By default ``None``.
    show_caption : bool
        Draw the captions. By default ``True``.
    save_fig_path : str | None
        Save the figure to this path. By default ``None``.
    filter_classes : list[int] | None
        Only draw instances of these class ids. By default ``None``.
    min_score : float | None
        Only draw instances scoring at least this. By default ``None``.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        if filter_classes is None or class_ids[i] in filter_classes:
            pass
        else:
            continue

        if min_score is None or scores is None:
            pass
        elif scores[i] < min_score:
            continue

        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,  # linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        if show_caption:
            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = f"{label} {score:.3f}" if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        if show_mask_polygon:
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = measure.find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches="tight")
    if auto_show:
        plt.show()


def display_differences(
    image,
    gt_box,
    gt_class_id,
    gt_mask,
    pred_box,
    pred_class_id,
    pred_score,
    pred_mask,
    class_names,
    title="",
    ax=None,
    show_mask=True,
    show_box=True,
    iou_threshold=0.5,
    score_threshold=0.5,
):
    """Show ground-truth and predicted instances on the same image, with their matches.

    Parameters
    ----------
    image : np.ndarray
        ``[height, width, 3]`` image.
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
    class_names : list[str]
        Class names of the dataset.
    title : str
        Figure title. By default ``""``.
    ax : matplotlib.axes.Axes | None
        Axes to draw on. By default ``None``.
    show_mask : bool
        Draw the masks. By default ``True``.
    show_box : bool
        Draw the boxes. By default ``True``.
    iou_threshold : float
        Mask IoU for a prediction to match a ground truth. By default 0.5.
    score_threshold : float
        Predictions below this score are ignored. By default 0.5.
    """
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box,
        gt_class_id,
        gt_mask,
        pred_box,
        pred_class_id,
        pred_score,
        pred_mask,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0, 0.8)] * len(gt_match) + [(1, 0, 0, 1)] * len(pred_match)
    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # Captions per instance show score/IoU
    pred_iou = [
        overlaps[i, int(pred_match[i])] if pred_match[i] > -1 else overlaps[i].max()
        for i in range(len(pred_match))
    ]
    captions = ["" for _ in gt_match] + [
        f"{pred_score[i]:.2f} / {pred_iou[i]:.2f}" for i in range(len(pred_match))
    ]
    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # Display
    display_instances(
        image,
        boxes,
        masks,
        class_ids,
        class_names,
        scores,
        ax=ax,
        show_bbox=show_box,
        show_mask=show_mask,
        colors=colors,
        captions=captions,
        title=title,
    )


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """Show anchors (or proposals) and their refinements on an image.

    Parameters
    ----------
    image : np.ndarray
        ``[height, width, 3]`` image.
    rois : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` anchors or proposals, in image coordinates.
    refined_rois : np.ndarray
        ``[N, 4]`` the same boxes refined to fit the objects.
    mask : np.ndarray
        ``[N, height, width]`` masks of the boxes.
    class_ids : np.ndarray
        ``[N]`` class ids.
    class_names : list[str]
        Class names of the dataset.
    limit : int
        Maximum number of boxes to draw (random sample). By default 10.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(ids, limit, replace=False) if ids.shape[0] > limit else ids

    _, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title(f"Showing {len(ids)} random ROIs out of {rois.shape[0]}")
    else:
        plt.title(f"{len(ids)} ROIs")

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis("off")

    for roi_id in ids:
        color = np.random.rand(3)
        class_id = class_ids[roi_id]
        # ROI
        y1, x1, y2, x2 = rois[roi_id]
        p = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color if class_id else "gray",
            facecolor="none",
            linestyle="dashed",
        )
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[roi_id]
            p = patches.Rectangle(
                (rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, f"{label}", color="w", size=11, backgroundcolor="none")

            # Mask
            m = inference.unmold_mask(mask[id], rois[id][:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print(f"Positive Ratio: {class_ids[class_ids > 0].shape[0] / class_ids.shape[0]:.2f}")


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw a 3-pixel wide box on an image array, in place.

    Parameters
    ----------
    image : np.ndarray
        ``[height, width, 3]`` image, modified in place.
    box : np.ndarray
        ``(y1, x1, y2, x2)`` box, in pixels.
    color : list[int]
        RGB color, channels in [0, 255].

    Returns
    -------
    np.ndarray
        The image with the box.
    """
    y1, x1, y2, x2 = box
    image[y1 : y1 + 2, x1:x2] = color
    image[y2 : y2 + 2, x1:x2] = color
    image[y1:y2, x1 : x1 + 2] = color
    image[y1:y2, x2 : x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Show an image and the masks of its most frequent classes.

    Parameters
    ----------
    image : np.ndarray
        ``[height, width, 3]`` image.
    mask : np.ndarray
        ``[height, width, N]`` instance masks.
    class_ids : np.ndarray
        ``[N]`` class ids of the masks.
    class_names : list[str]
        Class names of the dataset.
    limit : int
        Number of classes to show. By default 4.
    """
    to_display = []
    titles = []
    to_display.append(image)
    titles.append(f"H x W={image.shape[0]}x{image.shape[1]}")
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]]) for i in unique_class_ids]
    top_ids = [
        v[0]
        for v in sorted(
            zip(unique_class_ids, mask_area, strict=True), key=lambda r: r[1], reverse=True
        )
        if v[1] > 0
    ]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    Parameters
    ----------
    AP : float
        Average precision at IoU >= 0.5.
    precisions : np.ndarray
        Precision values.
    recalls : np.ndarray
        Recall values.
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title(f"Precision-Recall Curve. AP@50 = {AP:.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores, overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground-truth objects are classified.

    Parameters
    ----------
    gt_class_ids : np.ndarray
        ``[G]`` ground-truth class ids.
    pred_class_ids : np.ndarray
        ``[P]`` predicted class ids.
    pred_scores : np.ndarray
        ``[P]`` prediction confidences.
    overlaps : np.ndarray
        ``[P, G]`` IoU overlaps of predictions and ground-truth boxes.
    class_names : list[str]
        Class names of the dataset.
    threshold : float
        Confidence needed to predict a class. By default 0.5.
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation="nearest", cmap="Blues")
    plt.yticks(
        np.arange(len(pred_class_ids)),
        [f"{class_names[int(id)]} ({pred_scores[i]:.2f})" for i, id in enumerate(pred_class_ids)],
    )
    plt.xticks(
        np.arange(len(gt_class_ids)), [class_names[int(id)] for id in gt_class_ids], rotation=90
    )

    thresh = overlaps.max() / 2.0
    for i, j in itertools.product(range(overlaps.shape[0]), range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = "white" if overlaps[i, j] > thresh else "black" if overlaps[i, j] > 0 else "grey"
        plt.text(
            j,
            i,
            f"{overlaps[i, j]:.3f}\n{text}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
            color=color,
        )

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(
    image,
    boxes=None,
    refined_boxes=None,
    masks=None,
    captions=None,
    visibilities=None,
    title="",
    ax=None,
):
    """Draw boxes and masks, with optional refinements, captions and visibility levels.

    Parameters
    ----------
    image : np.ndarray
        ``[height, width, 3]`` image.
    boxes : np.ndarray | None
        ``[N, (y1, x1, y2, x2)]`` boxes in image coordinates, drawn dashed. By default ``None``.
    refined_boxes : np.ndarray | None
        Like ``boxes``, refined; drawn solid. By default ``None``.
    masks : np.ndarray | None
        ``[N, height, width]`` masks. By default ``None``.
    captions : list[str] | None
        One caption per box. By default ``None``.
    visibilities : list[int] | None
        One level per box: 0 (faint), 1 or 2 (prominent). By default ``None``.
    title : str
        Figure title. By default ``""``.
    ax : matplotlib.axes.Axes | None
        Axes to draw on. By default ``None``.

    Raises
    ------
    ValueError
        If a visibility is not 0, 1 or 2.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis("off")

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1
        else:
            raise ValueError(f"visibility must be 0, 1 or 2, got {visibility}")

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=alpha,
                linestyle=style,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle(
                (rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(
                x1,
                y1,
                caption,
                size=11,
                verticalalignment="top",
                color="w",
                backgroundcolor="none",
                bbox={"facecolor": color, "alpha": 0.5, "pad": 2, "edgecolor": "none"},
            )

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = measure.find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Show rows of values as an HTML table in a notebook.

    Parameters
    ----------
    table : list
        Rows, each an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += f"<td>{str(col):40}</td>"
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Show min, max and standard deviation of every weight of a model, flagging suspicious ones.

    Parameters
    ----------
    model : MaskRCNN
        The model whose Keras weights to inspect.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for layer in layers:
        weight_values = layer.get_weights()  # list of Numpy arrays
        weight_tensors = layer.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (layer.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append(
                [
                    weight_name + alert,
                    str(w.shape),
                    f"{w.min():+9.4f}",
                    f"{w.max():+10.4f}",
                    f"{w.std():+9.4f}",
                ]
            )
    display_table(table)
