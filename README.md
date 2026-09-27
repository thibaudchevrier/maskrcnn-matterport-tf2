# maskrcnn-matterport

[Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN) (Keras/TensorFlow), ported to
TF2 graph mode in 2021 and packaged so it still trains and runs on current Python and hardware.

This is a **model library**: it defines the network, its training loop and inference. It knows
nothing about datasets, experiment tracking or serving. Those live in the project that uses it
(for the fashion model: [fashion-seg-train](https://github.com/thibaudchevrier/fashion-seg-train)).

## Install

The package is split by use, so a service running an exported model doesn't need the 2021
training stack:

| Install | Provides | Requires |
|---------|----------|----------|
| `maskrcnn-matterport` | `mrcnn.inference`: resizing, anchors, image meta, unmolding detections | numpy (1.x or 2.x), scikit-image; any Python ≥ 3.10 |
| `maskrcnn-matterport[serve]` | + `mrcnn.serving.SavedModelPredictor`: run a model exported with `MaskRCNN.save` | any TensorFlow 2.x that loads the SavedModel |
| `maskrcnn-matterport[train]` | + `mrcnn.model` / `mrcnn.utils`: build and train | TensorFlow 2.15 (last with Keras 2), numpy 1.x, Python 3.10–3.11 |
| `maskrcnn-matterport[viz]` | + `mrcnn.visualize` plotting helpers | matplotlib |

```toml
# pyproject.toml of a consumer
dependencies = ["maskrcnn-matterport[train]"]   # or [serve], or no extra

[tool.uv.sources]
maskrcnn-matterport = { git = "https://github.com/thibaudchevrier/maskrcnn-matterport-tf2", tag = "v0.3.0" }
```

Each release also attaches its wheel and sdist to the
[GitHub Release](https://github.com/thibaudchevrier/maskrcnn-matterport-tf2/releases); a wheel URL
works as a source too. GitHub Packages has no Python registry, so releases carry the built packages.

Changes from the 2021 port:

- installable package, MIT `LICENSE` restored, no hard dependency on MLflow (v0.2.0);
- `distutils` and `np.bool` removed, masks cast to float before resizing, legacy SGD optimizer for
  the graph-mode training loop on TF ≥ 2.11 (v0.2.0);
- inference helpers (`resize_image`, `norm_boxes`, `mold_image`, `compose_image_meta`,
  `unmold_mask`, anchors...) moved to `mrcnn.inference`: import them from there, no longer from
  `mrcnn.utils` / `mrcnn.model`. `mrcnn.serving` added, training dependencies moved to the `train`
  extra (v0.3.0).

Training augmentation still expects [imgaug](https://github.com/aleju/imgaug)-style augmenters.
imgaug itself is unmaintained and not a dependency; pass `augmentation=None` or a compatible object.

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

Serving an export, without the training stack (`[serve]` extra):

```python
from mrcnn.serving import SavedModelPredictor

result = SavedModelPredictor("export/").detect(image)   # rois, class_ids, scores, masks
```

COCO starting weights: `mask_rcnn_coco.h5` from the
[Matterport v2.0 release](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0).

## Development

```bash
uv sync --extra train
uv run pre-commit install --hook-type commit-msg   # once: checks commit messages locally
uv run pytest    # trains a tiny model, checkpoints, detects, exports and serves the export
```

### Commits, versions and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat: ...`, `fix(utils): ...`, `docs: ...`), checked locally by the hook and on every PR by CI.
`uv run cz commit` writes one interactively.

Releases are automatic. On every merge to `main`, [commitizen](https://commitizen-tools.github.io/commitizen/)
reads the commits since the last tag and, if there is a `feat` (minor), `fix`/`perf`/`refactor` (patch) or a
breaking change (minor while < 1.0), bumps the version in `pyproject.toml` and `uv.lock`, updates
`CHANGELOG.md`, tags `vX.Y.Z` and publishes a GitHub Release with the wheel and sdist. Other
types (`docs`, `ci`, `test`, `build`, `chore`...) never trigger a release.
