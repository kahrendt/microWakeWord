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

import argparse
import json
import os
import yaml
from absl import logging

import microwakeword.inception as inception
import microwakeword.train as train
import microwakeword.test as test
import microwakeword.utils as utils

from microwakeword.layers import modes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_config",
        type=str,
        default="trained_models/model/training_parameters.yaml",
        help="""\
        Path to the training parameters yaml configuration.action=
        """,
    )
    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="If 1 run train and test, else run only test",
    )
    parser.add_argument(
        "--restore_checkpoint",
        type=int,
        default=0,
        help="If 1 it will restore a checkpoint and resume the training "
        "by initializing model weights and optimizer with checkpoint values. "
        "It will use learning rate and number of training iterations from "
        "--learning_rate and --how_many_training_steps accordinlgy. "
        "This option is useful in cases when training was interrupted. "
        "With it you should adjust learning_rate and how_many_training_steps.",
    )

    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
        value: A member of tf.logging.

        Returns:
        TF logging mode

        Raises:
        ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == "INFO":
            return logging.INFO
        elif value == "DEBUG":
            return logging.DEBUG
        elif value == "ERROR":
            return logging.ERROR
        elif value == "FATAL":
            return logging.FATAL
        elif value == "WARN":
            return logging.WARN
        else:
            raise argparse.ArgumentTypeError("Not an expected value")

    parser.add_argument(
        "--verbosity",
        type=verbosity_arg,
        default=logging.INFO,
        help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"',
    )

    # sub parser for model settings
    subparsers = parser.add_subparsers(dest="model_name", help="NN model name")

    # inception model settings
    parser_inception = subparsers.add_parser("inception")
    inception.model_parameters(parser_inception)

    flags, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError("Unknown argument: {}".format(unparsed))

    config = yaml.load(open(flags.training_config, "r").read(), yaml.Loader)

    config["summaries_dir"] = os.path.join(config["train_dir"], "logs/")

    desired_samples = int(config["sample_rate"] * config["clip_duration_ms"] / 1000)
    window_size_samples = int(config["sample_rate"] * config["window_size_ms"] / 1000)
    window_stride_samples = int(
        config["sample_rate"] * config["window_stride_ms"] / 1000
    )
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        config["spectrogram_length"] = 0
    else:
        config["spectrogram_length"] = 1 + int(
            length_minus_window / window_stride_samples
        )

    if flags.train:
        try:
            os.makedirs(config["train_dir"])
            os.makedirs(os.path.join(config["train_dir"], "restore"))  # DO I USE THIS?
            os.mkdir(config["summaries_dir"])
        except OSError as e:
            if flags.restore_checkpoint:
                pass
            else:
                raise ValueError(
                    "model already exists in folder %s" % config["train_dir"]
                ) from None

        train.train(flags, config)
    else:
        if not os.path.isdir(config["train_dir"]):
            raise ValueError('model is not trained set "--train 1" and retrain it')

    # write all flags settings into json TODO Switch to saving config.yaml?
    with open(os.path.join(config["train_dir"], "flags.json"), "wt") as f:
        json.dump(flags.__dict__, f)

    utils.convert_model_saved(
        flags, config, "non_stream", modes.Modes.NON_STREAM_INFERENCE
    )
    utils.convert_model_saved(
        flags,
        config,
        "stream_state_internal",
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
    )

    folder_name = "non_stream"
    test.tf_model_accuracy(config, folder_name)

    folder_name = "tflite_non_stream"
    file_name = "non_stream.tflite"
    utils.convert_saved_model_to_tflite(
        config,
        os.path.join(config["train_dir"], "non_stream"),
        os.path.join(config["train_dir"], folder_name),
        file_name,
    )
    test.tflite_model_accuracy(config, folder_name, file_name)

    folder_name = "tflite_stream_state_internal"
    file_name = "stream_state_internal.tflite"
    utils.convert_saved_model_to_tflite(
        config,
        os.path.join(config["train_dir"], "stream_state_internal"),
        os.path.join(config["train_dir"], folder_name),
        file_name,
    )
    test.tflite_model_accuracy(config, folder_name, file_name)
    # test.streaming_model_false_accept_rate(config, folder_name, file_name, 'dipco_features.npy')

    # quantize the internal streaming model here and then test it
    folder_name = "tflite_stream_state_internal_quant"
    file_name = "stream_state_internal_quantize.tflite"
    utils.convert_saved_model_to_tflite(
        config,
        os.path.join(config["train_dir"], "stream_state_internal"),
        os.path.join(config["train_dir"], folder_name),
        file_name,
        quantize=True,
    )
    test.tflite_model_accuracy(config, folder_name, file_name)
    test.streaming_model_false_accept_rate(
        config, folder_name, file_name, "dipco_features.npy"
    )
