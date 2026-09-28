## v0.3.1 (2026-09-28)

### Refactor

- **mrcnn**: numpy docstrings and fixes found by the new lint

## v0.3.0 (2026-09-27)

### BREAKING CHANGE

- training dependencies moved to the `train` extra. Install
`maskrcnn-matterport[train]` to train (TensorFlow 2.15, Python < 3.12),
`[serve]` to run exports; the base install only provides mrcnn.inference.

### Feat

- **inference**: TensorFlow-free inference module and serving predictor

### Refactor

- **inference**: import moved helpers from mrcnn.inference

## v0.2.0 (2026-09-27)

### Feat

- package for Python 3.11 / TensorFlow 2.15 (v0.2.0)
