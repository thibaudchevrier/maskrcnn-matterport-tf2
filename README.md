# maskrcnn-matterport

[Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN) (Keras/TensorFlow), ported to
TF2 graph mode in 2021 and packaged so it still trains and runs on current Python and hardware.

This is a **model library**: it defines the network, its training loop and inference. It knows
nothing about datasets, experiment tracking or serving. Those live in the project that uses it
(for the fashion model: [fashion-seg-train](https://github.com/thibaudchevrier/fashion-seg-train)).

## Compatibility

| | Supported |
|---|---|
| Python | 3.10, 3.11 |
| TensorFlow | 2.15 (the last release with Keras 2 built in; Keras 3 is not supported) |
| numpy | 1.x (< 2, required by TensorFlow 2.15) |
| Platforms | Linux x86_64, macOS arm64 (CPU) |

Changes from the 2021 port (v0.2.0):

- installable package (`pyproject.toml`), MIT `LICENSE` restored;
- no hard dependency on MLflow (tracking belongs to the training project);
- `distutils` removed (gone in Python 3.12), `np.bool` → `bool` (removed in numpy 1.24);
- masks cast to float before resizing (scikit-image ≥ 0.19 refuses to interpolate booleans);
- legacy SGD optimizer, required by the graph-mode (`tf.compat.v1`) training loop on TF ≥ 2.11.

Training augmentation still expects [imgaug](https://github.com/aleju/imgaug)-style augmenters.
imgaug itself is unmaintained and not a dependency; pass `augmentation=None` or a compatible object.

## Install

```bash
uv add "maskrcnn-matterport @ git+https://github.com/thibaudchevrier/maskrcnn-matterport-tf2@v0.2.0"
# optional plotting helpers (mrcnn.visualize):
uv add "maskrcnn-matterport[viz] @ git+https://github.com/thibaudchevrier/maskrcnn-matterport-tf2@v0.2.0"
```

## Usage

```python
from mrcnn import config, model as modellib, utils

class MyConfig(config.Config):
    NAME = "fashion"
    NUM_CLASSES = 1 + 46
    IMAGES_PER_GPU = 2

class MyDataset(utils.Dataset):
    ...  # implement load_image(image_id) and load_mask(image_id) -> (bool [H, W, N], class_ids [N])

model = modellib.MaskRCNN(mode="training", config=MyConfig(), model_dir="logs")
model.load_weights("mask_rcnn_coco.h5", by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model.train(train_dataset, val_dataset, learning_rate=1e-3, epochs=10, layers="heads")

inference = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir="logs")
inference.load_weights(model.find_last(), by_name=True)
result = inference.detect([image])[0]      # rois, class_ids, scores, masks
inference.save("export/")                  # config.json + TF SavedModel, for serving
```

COCO starting weights: `mask_rcnn_coco.h5` from the
[Matterport v2.0 release](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0).

## Development

```bash
uv sync
uv run pytest    # trains a tiny model on synthetic shapes, checkpoints it, reloads it for detection
```
