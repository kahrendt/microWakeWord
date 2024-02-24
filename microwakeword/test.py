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

"""Test utility functions for accuracy evaluation."""

import os
from absl import logging
import numpy as np
import tensorflow as tf


def compute_metrics(true_positives, true_negatives, false_positives, false_negatives):
    """Utility function to compute various metrics.

    Arguments:
        true_positives: Count of samples correctly predicted as positive
        true_negatives: Count of samples correctly predicted as negative
        false_positives: Count of samples incorrectly predicted as positive
        false_negatives: Count of samples incorrectly predicted as negative

    Returns:
        metric dictionary with keys for `accuracy`, `recall`, `precision`, `false_positive_rate`, `false_negative_rate`, and `count`
    """

    accuracy = float("nan")
    false_positive_rate = float("nan")
    false_negative_rate = float("nan")
    recall = float("nan")
    precision = float("nan")

    count = true_positives + true_negatives + false_positives + false_negatives

    if true_positives + true_negatives + false_positives + false_negatives > 0:
        accuracy = (true_positives + true_negatives) / count

    if false_positives + true_negatives > 0:
        false_positive_rate = false_positives / (false_positives + true_negatives)

    if true_positives + false_negatives > 0:
        false_negative_rate = false_negatives / (true_positives + false_negatives)
        recall = true_positives / (true_positives + false_negatives)

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "count": count,
    }


def metrics_to_string(metrics):
    """Utility function to return a string that describes various metrics.

    Arguments:
        metrics: metric dictionary with keys for `accuracy`, `recall`, `precision`, `false_positive_rate`, `false_negative_rate`, and `count`

    Returns:
        string describing the given metrics
    """

    return "accuracy = {accuracy:.4%}; recall = {recall:.4%}; precision = {precision:.4%}; fpr = {fpr:.4%}; fnr = {fnr:.4%}; (N={count})".format(
        accuracy=metrics["accuracy"],
        recall=metrics["recall"],
        precision=metrics["precision"],
        fpr=metrics["false_positive_rate"],
        fnr=metrics["false_negative_rate"],
        count=metrics["count"],
    )


def tf_model_accuracy(
    config,
    folder,
    audio_processor,
    data_set="testing",
    accuracy_name="tf_model_accuracy.txt",
):
    """Function to test a TF model on a specified data set.

    Arguments:
        config: dictionary containing microWakeWord training configuration
        folder: folder containing the TF model
        audio_processor:  microWakeWord FeatureHandler object for retrieving spectrograms
        data_set: data set to test the model on
        accuracy_name: filename to save metrics to

    Returns:
        metric dictionary with keys for `accuracy`, `recall`, `precision`, `false_positive_rate`, `false_negative_rate`, and `count`
    """

    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        data_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )

    with tf.device("/cpu:0"):
        model = tf.saved_model.load(os.path.join(config["train_dir"], folder))
        inference_batch_size = 1

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, len(test_fingerprints), inference_batch_size):
            spectrogram_features = test_fingerprints[i : i + inference_batch_size]
            sample_ground_truth = test_ground_truth[i]

            result = model(tf.convert_to_tensor(spectrogram_features, dtype=tf.float32))

            prediction = result.numpy()[0][0] > 0.5
            if sample_ground_truth == prediction:
                if sample_ground_truth:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if sample_ground_truth:
                    false_negatives += 1
                else:
                    false_positives += 1

            metrics = compute_metrics(
                true_positives, true_negatives, false_positives, false_negatives
            )

            if i % 1000 == 0 and i:
                logging.info(
                    "TensorFlow model on the {dataset} set: accuracy = {accuracy:.6}; recall = {recall:.6}; precision = {precision:.6}; fpr = {fpr:.6}; fnr = {fnr:.6} ({iteration} out of {length})".format(
                        dataset=data_set,
                        accuracy=metrics["accuracy"],
                        recall=metrics["recall"],
                        precision=metrics["precision"],
                        fpr=metrics["false_positive_rate"],
                        fnr=metrics["false_negative_rate"],
                        iteration=i,
                        length=len(test_fingerprints),
                    )
                )

    metrics_string = metrics_to_string(metrics)

    logging.info(
        "Final TensorFlow model on the " + data_set + " set: " + metrics_string
    )

    path = os.path.join(config["train_dir"], folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        fd.write(metrics_string)
    return metrics


def tflite_model_accuracy(
    config,
    folder,
    audio_processor,
    data_set="testing",
    tflite_model_name="stream_state_internal.tflite",
    accuracy_name="tflite_model_accuracy.txt",
):
    """Function to test a TFLite model on a specified data set.

    Model can be streaming or nonstreaming. If tested on an "_ambient" set,
    it detects a false accept if the previous probability was less than 0.5
    and the current probability is greater than 0.5.

    Arguments:
        config: dictionary containing microWakeWord training configuration
        folder: folder containing the TFLite model
        audio_processor:  microWakeWord FeatureHandler object for retrieving spectrograms
        data_set: data set to test the model on
        tflite_model_name: filename of the TFLite model
        accuracy_name: filename to save metrics to

    Returns:
        Metric dictionary with keys for `accuracy`, `recall`, `precision`, `false_positive_rate`, `false_negative_rate`, and `count`
    """

    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(config["train_dir"], folder, tflite_model_name)
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_quantized_model = input_details[0]["dtype"] == np.int8
    input_feature_slices = input_details[0]["shape"][1]

    for s in range(len(input_details)):
        if is_quantized_model:
            interpreter.set_tensor(
                input_details[s]["index"],
                np.zeros(input_details[s]["shape"], dtype=np.int8),
            )
        else:
            interpreter.set_tensor(
                input_details[s]["index"],
                np.zeros(input_details[s]["shape"], dtype=np.float32),
            )

    def quantize_input_data(data, input_details):
        """quantize the input data using scale and zero point

        Args:
            data (np.array in float): input data for the interpreter
            input_details : output of get_input_details from the tflm interpreter.

        Returns:
          np.ndarray: quantized data as int8 dtype
        """
        # Get input quantization parameters
        data_type = input_details["dtype"]

        input_quantization_parameters = input_details["quantization_parameters"]
        input_scale, input_zero_point = (
            input_quantization_parameters["scales"][0],
            input_quantization_parameters["zero_points"][0],
        )
        # quantize the input data
        data = data / input_scale + input_zero_point
        return data.astype(data_type)

    def dequantize_output_data(data: np.ndarray, output_details: dict) -> np.ndarray:
        """Dequantize the model output

        Args:
            data: integer data to be dequantized
            output_details: TFLM interpreter model output details

        Returns:
            np.ndarray: dequantized data as float32 dtype
        """
        output_quantization_parameters = output_details["quantization_parameters"]
        output_scale = output_quantization_parameters["scales"][0]
        output_zero_point = output_quantization_parameters["zero_points"][0]
        # Caveat: tflm_output_quant need to be converted to float to avoid integer
        # overflow during dequantization
        # e.g., (tflm_output_quant -output_zero_point) and
        # (tflm_output_quant + (-output_zero_point))
        # can produce different results (int8 calculation)
        return output_scale * (data.astype(np.float32) - output_zero_point)

    truncation_strategy = "truncate_start"
    if data_set.endswith("ambient"):
        truncation_strategy = "none"

    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        data_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy=truncation_strategy,
    )

    logging.info("Testing TFLite model on the {data_set} set".format(data_set=data_set))

    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    previous_probability = 0.0

    for i in range(0, len(test_fingerprints)):
        sample_fingerprint = test_fingerprints[i].astype(np.float32)
        sample_ground_truth = test_ground_truth[i]

        for feature_index in range(
            0, sample_fingerprint.shape[0], input_feature_slices
        ):
            new_data_to_input = sample_fingerprint[
                feature_index : feature_index + input_feature_slices, :
            ]

            if is_quantized_model:
                new_data_to_input = quantize_input_data(
                    new_data_to_input, input_details[0]
                )

            # Input new data and invoke the interpreter
            interpreter.set_tensor(
                input_details[0]["index"],
                np.reshape(new_data_to_input, input_details[0]["shape"]),
            )
            interpreter.invoke()

            # get output states and feed them as inputs
            # which will be fed in the next inference cycle for externally streaming models
            for s in range(1, len(input_details)):
                interpreter.set_tensor(
                    input_details[s]["index"],
                    interpreter.get_tensor(output_details[s]["index"]),
                )

            output = interpreter.get_tensor(output_details[0]["index"])
            if is_quantized_model:
                wakeword_probability = dequantize_output_data(
                    output[0][0], output_details[0]
                )
            else:
                wakeword_probability = output[0][0]

            if truncation_strategy == "none":
                if previous_probability <= 0.5 and wakeword_probability > 0.5:
                    false_positives += 1
                previous_probability = wakeword_probability

        if truncation_strategy != "none":
            prediction = wakeword_probability > 0.5
            if sample_ground_truth == prediction:
                if sample_ground_truth:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if sample_ground_truth:
                    false_negatives += 1
                else:
                    false_positives += 1

        metrics = compute_metrics(
            true_positives, true_negatives, false_positives, false_negatives
        )

        if i % 1000 == 0 and i:
            logging.info(
                "TFLite model on the {dataset} set: accuracy = {accuracy:.6}; recall = {recall:.6}; precision = {precision:.6}; fpr = {fpr:.6}; fnr = {fnr:.6} ({iteration} out of {length})".format(
                    dataset=data_set,
                    accuracy=metrics["accuracy"],
                    recall=metrics["recall"],
                    precision=metrics["precision"],
                    fpr=metrics["false_positive_rate"],
                    fnr=metrics["false_negative_rate"],
                    iteration=i,
                    length=len(test_fingerprints),
                )
            )

    if truncation_strategy != "none":
        metrics_string = metrics_to_string(metrics)
    else:
        metrics_string = "false accepts = {false_positives}; false accepts per hour = {faph:.4}".format(
            false_positives=false_positives,
            faph=false_positives
            / (audio_processor.get_mode_duration(data_set) / 3600.0),
        )

    logging.info("Final TFLite model on the " + data_set + " set: " + metrics_string)
    path = os.path.join(config["train_dir"], folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        fd.write(metrics_string)
    return metrics
