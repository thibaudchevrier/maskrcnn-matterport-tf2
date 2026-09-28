"""Multi-GPU support for Keras: replicate a model on several GPUs and merge the outputs.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details).
Written by Waleed Abdulla.

Ideas and small code snippets from:

- https://github.com/fchollet/keras/issues/2436
- https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
- https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
- https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py
"""

import keras.backend as K
import keras.layers as KL
import keras.models as KM
import tensorflow as tf


# pylint: disable-next=abstract-method  # functional model: call() comes from its graph
class ParallelModel(KM.Model):
    """Keras model running a copy of an inner model on each GPU.

    Each copy gets a slice of the inputs; the outputs are merged, and the loss is applied to the
    combined outputs. Loading and saving go to the inner model, which holds the weights.

    Parameters
    ----------
    keras_model : keras.Model
        The model to parallelize.
    gpu_count : int
        Number of GPUs, more than 1.

    Attributes
    ----------
    inner_model : keras.Model
        The parallelized model.
    gpu_count : int
        Number of GPUs.
    """

    inner_model: KM.Model
    gpu_count: int

    def __init__(self, keras_model, gpu_count):
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super().__init__(inputs=self.inner_model.inputs, outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model, which holds the weights.

        Parameters
        ----------
        attrname : str
            Name of the attribute to get.

        Returns
        -------
        object
            The inner model's attribute for loading/saving methods, this model's otherwise.
        """
        if "load" in attrname or "save" in attrname:
            return getattr(self.inner_model, attrname)
        return super().__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Print the summaries of both the wrapper and the inner model.

        Parameters
        ----------
        *args : tuple
            Positional arguments of ``keras.Model.summary``.
        **kwargs : dict
            Keyword arguments of ``keras.Model.summary``.
        """
        super().summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Build the replicas of the inner model, one per GPU, and merge their outputs.

        Returns
        -------
        list
            The merged output tensors: concatenated along the batch axis, or averaged for
            scalar outputs (losses and metrics).
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {
            name: tf.split(x, self.gpu_count)
            for name, x in zip(self.inner_model.input_names, self.inner_model.inputs, strict=True)
        }

        output_names = self.inner_model.output_names
        outputs_all = [[] for _ in self.inner_model.outputs]

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device(f"/gpu:{i}"), tf.name_scope(f"tower_{i}"):
                # Run a slice of inputs through this replica
                zipped_inputs = zip(
                    self.inner_model.input_names, self.inner_model.inputs, strict=True
                )
                inputs = [
                    KL.Lambda(
                        lambda s, name=name, i=i: input_slices[name][i],
                        output_shape=lambda s: (None,) + s[1:],
                    )(tensor)
                    for name, tensor in zipped_inputs
                ]
                # Create the model replica and get the outputs
                outputs = self.inner_model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                # Save the outputs for merging back together later
                for index, output in enumerate(outputs):
                    outputs_all[index].append(output)

        # Merge outputs on CPU
        with tf.device("/cpu:0"):
            merged = []
            for outputs, name in zip(outputs_all, output_names, strict=True):
                # Concatenate or average outputs?
                # Outputs usually have a batch dimension and we concatenate
                # across it. If they don't, then the output is likely a loss
                # or a metric value that gets averaged across the batch.
                # Keras expects losses and metrics to be scalars.
                if K.int_shape(outputs[0]) == ():
                    # Average
                    m = KL.Lambda(lambda o, outputs=outputs: tf.add_n(o) / len(outputs), name=name)(
                        outputs
                    )
                else:
                    # Concatenate
                    m = KL.Concatenate(axis=0, name=name)(outputs)
                merged.append(m)
        return merged
