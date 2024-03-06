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
import ast
import json
import os
import yaml
from absl import logging

import microwakeword.data as input_data
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
        "--test_tf_nonstreaming",
        type=int,
        default=0,
        help="Save the nonstreaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_nonstreaming",
        type=int,
        default=0,
        help="Save the TFLite nonstreaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_streaming",
        type=int,
        default=0,
        help="Save the (non-quantized) streaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_streaming_quantized",
        type=int,
        default=1,
        help="Save the quantized streaming model and test on the test datasets",
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
    parser.add_argument(
        "--use_weights",
        type=str,
        default="best_weights",
        help="Which set of weights to use when creating the model"
        "One of `best_weights`` or `last_weights`.",
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

    def parse(text):
        """Parse model parameters.

        Args:
        text: string with layer parameters: '128,128' or "'relu','relu'".

        Returns:
        list of parsed parameters
        """
        if not text:
            return []
        res = ast.literal_eval(text)
        if isinstance(res, tuple):
            return res
        else:
            return [res]

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

    config["spectrogram_length_final_layer"] = config["spectrogram_length"]

    # Load model
    if flags.model_name == "inception":
        model = inception.model(flags, config)
        spectrogram_slices_dropped = inception.spectrogram_slices_dropped(flags)
    else:
        raise ValueError("Unknown model type: {}".format(flags.model_name))

    config["spectrogram_length"] += spectrogram_slices_dropped

    logging.info(model.summary())
    
    logging.set_verbosity(flags.verbosity)

    data_processor = input_data.FeatureHandler(config)

    if flags.train:
        try:
            os.makedirs(config["train_dir"])
            os.mkdir(config["summaries_dir"])
        except OSError as e:
            if flags.restore_checkpoint:
                pass
            else:
                raise ValueError(
                    "model already exists in folder %s" % config["train_dir"]
                ) from None
        config_fname = os.path.join(config["train_dir"], "training_config.yaml")

        with open(config_fname, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        utils.save_model_summary(model, config["train_dir"])

        train.train(model, config, data_processor)
    else:
        if not os.path.isdir(config["train_dir"]):
            raise ValueError('model is not trained set "--train 1" and retrain it')

    # write all flags settings into json TODO Switch to saving config.yaml?
    with open(os.path.join(config["train_dir"], "flags.json"), "wt") as f:
        json.dump(flags.__dict__, f)

    if (
        flags.test_tf_nonstreaming
        or flags.test_tflite_nonstreaming
        or flags.test_tflite_streaming
        or flags.test_tflite_streaming_quantized
    ):
        # Reload the model with a batch size of 1 for inference
        config["batch_size"] = 1

        if flags.model_name == "inception":
            model = inception.model(flags, config)

        model.load_weights(
            os.path.join(config["train_dir"], flags.use_weights)
        ).expect_partial()

    if flags.test_tf_nonstreaming or flags.test_tflite_nonstreaming:
        # Save the nonstreaming model to disk
        logging.info("Saving nonstreaming model")

        utils.convert_model_saved(
            model,
            config,
            "non_stream",
            modes.Modes.NON_STREAM_INFERENCE,
        )

    if flags.test_tf_nonstreaming:
        # Test the nonstreaming model
        logging.info("Testing nonstreaming model")

        folder_name = "non_stream"
        test.tf_model_accuracy(
            config,
            folder_name,
            data_processor,
            data_set="testing",
            accuracy_name="testing_set_metrics.txt",
        )

    if flags.test_tflite_nonstreaming:
        # Convert the nonstreaming model to TFLite then test it
        logging.info("Converting nonstreaming model to TFLite")

        folder_name = "tflite_non_stream"
        file_name = "non_stream.tflite"
        utils.convert_saved_model_to_tflite(
            config,
            data_processor,
            os.path.join(config["train_dir"], "non_stream"),
            os.path.join(config["train_dir"], folder_name),
            file_name,
        )

        logging.info("Testing the TFLite nonstreaming model")
        test.tflite_model_accuracy(
            config,
            folder_name,
            data_processor,
            tflite_model_name=file_name,
            accuracy_name="testing_set_metrics.txt",
        )

    if flags.test_tflite_streaming or flags.test_tflite_streaming_quantized:
        # Save the internal streaming model to disk
        logging.info("Saving streaming model")

        utils.convert_model_saved(
            model,
            config,
            "stream_state_internal",
            modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )

    if flags.test_tflite_streaming:
        # Convert the internal streaming model to TFLite then test it
        logging.info("Converting streaming model (non-quantized) to TFLite")

        folder_name = "tflite_stream_state_internal"
        file_name = "stream_state_internal.tflite"
        utils.convert_saved_model_to_tflite(
            config,
            data_processor,
            os.path.join(config["train_dir"], "stream_state_internal"),
            os.path.join(config["train_dir"], folder_name),
            file_name,
        )

        logging.info("Testing the non-quantized TFLite streaming model")
        test.tflite_model_accuracy(
            config,
            folder_name,
            data_processor,
            tflite_model_name=file_name,
            accuracy_name="testing_set_metrics.txt",
        )
        if data_processor.get_mode_size("testing_ambient") > 0:
            test.tflite_model_accuracy(
                config,
                folder_name,
                data_processor,
                data_set="testing_ambient",
                tflite_model_name=file_name,
                accuracy_name="testing_ambient_set_false_accepts.txt",
            )

    if flags.test_tflite_streaming_quantized:
        # Quantize while converting the internal streaming model to TFLite and then test it
        logging.info("Quantizing and converting streaming model to TFLite")

        folder_name = "tflite_stream_state_internal_quant"
        file_name = "stream_state_internal_quantize.tflite"
        utils.convert_saved_model_to_tflite(
            config,
            data_processor,
            os.path.join(config["train_dir"], "stream_state_internal"),
            os.path.join(config["train_dir"], folder_name),
            file_name,
            quantize=True,
        )

        logging.info("Testing the quantized TFLite streaming model")
        test.tflite_model_accuracy(
            config,
            folder_name,
            data_processor,
            tflite_model_name=file_name,
            accuracy_name="testing_set_metrics.txt",
        )
        if data_processor.get_mode_size("testing_ambient") > 0:
            test.tflite_model_accuracy(
                config,
                folder_name,
                data_processor,
                data_set="testing_ambient",
                tflite_model_name=file_name,
                accuracy_name="testing_ambient_set_false_accepts.txt",
            )
