"""Matterport Mask R-CNN, packaged for TensorFlow 2.15 training and TensorFlow-free inference.

Modules: ``config`` (settings), ``model`` (network, training, detection), ``utils`` (boxes, masks,
datasets, metrics), ``inference`` (pre/post-processing without TensorFlow), ``serving`` (run an
export), ``visualize`` (plots), ``parallel_model`` (multi-GPU).
"""

from importlib.metadata import version

__version__ = version("maskrcnn-matterport")
