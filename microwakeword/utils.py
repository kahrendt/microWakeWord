# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for operations on Model."""
import os.path
import numpy as np
import tensorflow as tf

from absl import logging

from microwakeword.layers import modes, stream, strided_drop


def _set_mode(model, mode):
    """Set model's inference type and disable training."""

    def _recursive_set_layer_mode(layer, mode):
        if isinstance(layer, tf.keras.layers.Wrapper):
            _recursive_set_layer_mode(layer.layer, mode)

        config = layer.get_config()
        # for every layer set mode, if it has it
        if "mode" in config:
            assert isinstance(
                layer,
                (stream.Stream, strided_drop.StridedDrop, strided_drop.StridedKeep),
            )
            layer.mode = mode
        # with any mode of inference - training is False
        if "training" in config:
            layer.training = False
        if mode == modes.Modes.NON_STREAM_INFERENCE:
            if "unroll" in config:
                layer.unroll = True

    for layer in model.layers:
        _recursive_set_layer_mode(layer, mode)
    return model


def _copy_weights(new_model, model):
    """Copy weights of trained model to an inference one."""

    def _same_weights(weight, new_weight):
        # Check that weights are the same
        # Note that states should be marked as non trainable
        return (
            weight.trainable == new_weight.trainable
            and weight.shape == new_weight.shape
            and weight.name[weight.name.rfind("/") : None]
            == new_weight.name[new_weight.name.rfind("/") : None]
        )

    if len(new_model.layers) != len(model.layers):
        raise ValueError(
            "number of layers in new_model: %d != to layers number in model: %d "
            % (len(new_model.layers), len(model.layers))
        )

    for i in range(len(model.layers)):
        layer = model.layers[i]
        new_layer = new_model.layers[i]

        # if number of weights in the layers are the same
        # then we can set weights directly
        if len(layer.get_weights()) == len(new_layer.get_weights()):
            new_layer.set_weights(layer.get_weights())
        elif layer.weights:
            k = 0  # index pointing to weights in the copied model
            new_weights = []
            # iterate over weights in the new_model
            # and prepare a new_weights list which will
            # contain weights from model and weight states from new model
            for k_new in range(len(new_layer.get_weights())):
                new_weight = new_layer.weights[k_new]
                new_weight_values = new_layer.get_weights()[k_new]
                same_weights = True

                # if there are weights which are not copied yet
                if k < len(layer.get_weights()):
                    weight = layer.weights[k]
                    weight_values = layer.get_weights()[k]
                    if (
                        weight.shape != weight_values.shape
                        or new_weight.shape != new_weight_values.shape
                    ):
                        raise ValueError("weights are not listed in order")

                    # if there are weights available for copying and they are the same
                    if _same_weights(weight, new_weight):
                        new_weights.append(weight_values)
                        k = k + 1  # go to next weight in model
                    else:
                        same_weights = False  # weights are different
                else:
                    same_weights = (
                        False  # all weights are copied, remaining is different
                    )

                if not same_weights:
                    # weight with index k_new is missing in model,
                    # so we will keep iterating over k_new until find similar weights
                    new_weights.append(new_weight_values)

            # check that all weights from model are copied to a new_model
            if k != len(layer.get_weights()):
                raise ValueError(
                    "trained model has: %d weights, but only %d were copied"
                    % (len(layer.get_weights()), k)
                )

            # now they should have the same number of weights with matched sizes
            # so we can set weights directly
            new_layer.set_weights(new_weights)
    return new_model


def save_model_summary(model, path, file_name="model_summary.txt"):
    """Saves model topology/summary in text format.

    Args:
      model: Keras model
      path: path where to store model summary
      file_name: model summary file name
    """
    with tf.io.gfile.GFile(os.path.join(path, file_name), "w") as fd:
        stringlist = []
        model.summary(
            print_fn=lambda x: stringlist.append(x)
        )  # pylint: disable=unnecessary-lambda
        model_summary = "\n".join(stringlist)
        fd.write(model_summary)


def convert_to_inference_model(model, input_tensors, mode):
    """Convert tf._keras_internal.engine.functional `Model` instance to a streaming inference.

    It will create a new model with new inputs: input_tensors.
    All weights will be copied. Internal states for streaming mode will be created
    Only tf._keras_internal.engine.functional Keras model is supported!

    Args:
        model: Instance of `Model`.
        input_tensors: list of input tensors to build the model upon.
        mode: is defined by modes.Modes

    Returns:
        An instance of streaming inference `Model` reproducing the behavior
        of the original model, on top of new inputs tensors,
        using copied weights.

    Raises:
        ValueError: in case of invalid `model` argument value or input_tensors
    """

    # scope is introduced for simplifiyng access to weights by names
    scope_name = "streaming"

    with tf.name_scope(scope_name):
        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                "Expected `model` argument to be a `Model` instance, got ", model
            )
        if isinstance(model, tf.keras.Sequential):
            raise ValueError(
                "Expected `model` argument "
                "to be a functional `Model` instance, "
                "got a `Sequential` instance instead:",
                model,
            )
        model = _set_mode(model, mode)
        new_model = tf.keras.models.clone_model(model, input_tensors)

    if mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
        return _copy_weights(new_model, model)
    elif mode == modes.Modes.NON_STREAM_INFERENCE:
        new_model.set_weights(model.get_weights())
        return new_model
    else:
        raise ValueError("non supported mode ", mode)


def to_streaming_inference(model_non_stream, config, mode):
    """Convert non streaming trained model to inference modes.

    Args:
      model_non_stream: trained Keras model non streamable
      config: dictionary containing microWakeWord training configuration
      mode: it supports Non streaming inference or Streaming inference with internal
        states

    Returns:
      Keras inference model of inference_type
    """

    input_data_shape = modes.get_input_data_shape(config, mode)

    # get input data type and use it for input streaming type
    if isinstance(model_non_stream.input, (tuple, list)):
        dtype = model_non_stream.input[0].dtype
    else:
        dtype = model_non_stream.input.dtype

    # For streaming, set the batch size to 1
    input_tensors = [
        tf.keras.layers.Input(
            shape=input_data_shape, batch_size=1, dtype=dtype, name="input_audio"
        )
    ]

    if (
        isinstance(model_non_stream.input, (tuple, list))
        and len(model_non_stream.input) > 1
    ):
        if len(model_non_stream.input) > 2:
            raise ValueError(
                "Maximum number of inputs supported is 2 (input_audio and "
                "cond_features), but got %d inputs" % len(model_non_stream.input)
            )

        input_tensors.append(
            tf.keras.layers.Input(
                shape=config["cond_shape"],
                batch_size=1,
                dtype=model_non_stream.input[1].dtype,
                name="cond_features",
            )
        )

    # Input tensors must have the same shape as the original
    if isinstance(model_non_stream.input, (tuple, list)):
        model_inference = convert_to_inference_model(
            model_non_stream, input_tensors, mode
        )
    else:
        model_inference = convert_to_inference_model(
            model_non_stream, input_tensors[0], mode
        )

    return model_inference


def model_to_saved(
    model_non_stream,
    config,
    mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
):
    """Convert Keras model to SavedModel.

    Depending on mode:
      1 Converted inference graph and model will be streaming statefull.
      2 Converted inference graph and model will be non streaming stateless.

    Args:
      model_non_stream: Keras non streamable model
      config: dictionary containing microWakeWord training configuration
      mode: inference mode it can be streaming with internal state or non
        streaming
    """

    if mode not in (
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        modes.Modes.NON_STREAM_INFERENCE,
    ):
        raise ValueError("mode %s is not supported " % mode)

    if mode == modes.Modes.NON_STREAM_INFERENCE:
        model = model_non_stream
    else:
        # convert non streaming Keras model to Keras streaming model, internal state
        model = to_streaming_inference(model_non_stream, config, mode)

    return model


def convert_saved_model_to_tflite(
    config, audio_processor, path_to_model, folder, fname, quantize=False
):
    """Convert SavedModel to TFLite and optionally quantize it.

    Args:
        config: dictionary containing microWakeWord training configuration
        audio_processor:  microWakeWord FeatureHandler object for retrieving spectrograms
        path_to_model: path to SavedModel
        folder: folder where converted model will be saved
        fname: output filename for TFLite file
        quantize: boolean selecting whether to quantize the model
    """

    def representative_dataset_gen():
        sample_fingerprints, _, _ = audio_processor.get_data(
            "training", 500, features_length=config["spectrogram_length"]
        )

        sample_fingerprints[0][
            0, 0
        ] = 0.0  # guarantee one pixel is the preprocessor min
        sample_fingerprints[0][
            0, 1
        ] = 26.0  # guarantee one pixel is the preprocessor max

        # for spectrogram in sample_fingerprints:
        #     yield spectrogram

        stride = config["stride"]

        for spectrogram in sample_fingerprints:
            assert spectrogram.shape[0] % stride == 0

            for i in range(0, spectrogram.shape[0] - stride, stride):
                sample = spectrogram[i : i + stride, :].astype(np.float32)
                yield [sample]

    converter = tf.lite.TFLiteConverter.from_saved_model(path_to_model)
    converter.optimizations = {tf.lite.Optimize.DEFAULT}

    # Without this flag, the Streaming layer `state` variables are left as float32,
    # resulting in Quantize and Dequantize operations before and after every `ReadVariable`
    # and `AssignVariable` operation.
    converter._experimental_variable_quantization = True

    if quantize:
        converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = tf.lite.RepresentativeDataset(
            representative_dataset_gen
        )

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, fname), "wb") as f:
        tflite_model = converter.convert()
        f.write(tflite_model)


def convert_model_saved(model, config, folder, mode):
    """Convert model to streaming and non streaming SavedModel.

    Args:
        model: model settings
        config: dictionary containing microWakeWord training configuration
        folder: folder where converted model will be saved
        mode: inference mode
    """

    path_model = os.path.join(config["train_dir"], folder)
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # Convert trained model to SavedModel
    converted_model = model_to_saved(model, config, mode)
    converted_model.summary()

    assert converted_model.input.shape[0] is not None

    # XXX: Using `converted_model.export(path_model)` results in obscure errors during
    # quantization, we create an export archive directly instead.
    export_archive = tf.keras.export.ExportArchive()
    export_archive.track(converted_model)
    export_archive.add_endpoint(
        name="serve",
        fn=converted_model.call,
        input_signature=[tf.TensorSpec(shape=converted_model.input.shape, dtype=tf.float32)],
    )
    export_archive.write_out(path_model)

    save_model_summary(converted_model, path_model)

    return converted_model
