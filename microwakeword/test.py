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

import numpy as np
import tensorflow as tf

from absl import logging
from typing import list

from microwakeword.inference import Model


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


def compute_false_accepts_per_hour(
    streaming_probabilities_list: list[np.ndarray],
    cutoffs: np.array,
    ignore_slices_after_accept: int = 75,
    step_s: float = 0.02,
):
    """Computes the false accept per hour rates at various cutoffs given a list of streaming probabilities.

    Args:
        streaming_probabilities_list (List[numpy.ndarray]): A list containing streaming probabilities from negative audio clips
        cutoffs (numpy.array): An array of cutoffs/thresholds to test the false accpet rate at.
        ignore_slices_after_accept (int, optional): The number of probabililities slices to ignore after a false accept. Defaults to 75.
        step_s (float, optional): The duration between each probabilitiy in seconds. Defaults to 0.02.

    Returns:
        numpy.ndarray: The false accepts per hour corresponding to thresholds in `cutoffs`.
    """
    cutoffs_count = cutoffs.shape[0]

    false_accepts_at_cutoffs = np.zeros(cutoffs_count)
    probabilities_duration_h = 0

    for track_probabilities in streaming_probabilities_list:
        probabilities_duration_h += len(track_probabilities) * step_s / 3600.0

        cooldown_at_cutoffs = np.ones(cutoffs_count) * ignore_slices_after_accept

        for wakeword_probability in track_probabilities:
            # Decrease the cooldown cutoff by 1 with a minimum value of 0
            cooldown_at_cutoffs = np.maximum(
                cooldown_at_cutoffs - 1, np.zeros(cutoffs_count)
            )
            detection_boolean = (
                wakeword_probability > cutoffs
            )  # a list of detection states at each cutoff

            for index in range(cutoffs_count):
                if cooldown_at_cutoffs[index] == 0 and detection_boolean[index]:
                    false_accepts_at_cutoffs[index] += 1
                    cooldown_at_cutoffs[index] = ignore_slices_after_accept

    return false_accepts_at_cutoffs / probabilities_duration_h


def generate_roc_curve(
    false_accepts_per_hour: np.ndarray,
    positive_samples_probabilities: np.ndarray,
    cutoffs: np.ndarray,
    max_faph: float = 2.0,
):
    """Generates the coordinates for an ROC curve plotting false accepts per hour vs false rejections. Computes the false rejection rate at the specifiied cutoffs.

    Args:
        false_accepts_per_hour (np.ndarray): False accepts per hour rates for each threshold in `cutoffs`.
        positive_samples_probabilities (np.ndarray): Probabilities for each positive sample.
        cutoffs (np.ndarray): Thresholds used for `false_ccepts_per_hour`
        max_faph (float, optional): The maximum false accept per hour rate to include in curve's coordinates. Defaults to 2.0.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): (false accept per hour coordinates, false rejection rate coordinates, cutoffs for each coordinate)
    """

    # Compute the false negative rates at each cutoff
    false_negative_rate_at_cutoffs = []
    for cutoff in cutoffs:
        true_accepts = sum(i > cutoff for i in positive_samples_probabilities)
        false_negative_rate_at_cutoffs.append(
            1 - true_accepts / len(positive_samples_probabilities)
        )

    if false_accepts_per_hour[0] > max_faph:
        # Use linear interpolation to estimate false negative rate at max_faph

        # Increase the index until we find a faph less than max_faph
        index_of_first_viable = 1
        while false_accepts_per_hour[index_of_first_viable] > max_faph:
            index_of_first_viable += 1

        x0 = false_accepts_per_hour[index_of_first_viable - 1]
        y0 = false_negative_rate_at_cutoffs[index_of_first_viable - 1]
        x1 = false_accepts_per_hour[index_of_first_viable]
        y1 = false_negative_rate_at_cutoffs[index_of_first_viable]

        fnr_at_max_faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        cutoff_at_max_faph = (
            cutoffs[index_of_first_viable] + cutoffs[index_of_first_viable - 1]
        ) / 2.0
    else:
        # Smallest faph is less than max_faph, so assume the false negative rate is constant
        index_of_first_viable = 0
        fnr_at_max_faph = false_negative_rate_at_cutoffs[index_of_first_viable]
        cutoff_at_max_faph = cutoffs[index_of_first_viable]

    horizontal_coordinates = [max_faph]
    vertical_coordinates = [fnr_at_max_faph]
    cutoffs_at_coordinate = [cutoff_at_max_faph]

    for index in range(index_of_first_viable, len(false_negative_rate_at_cutoffs)):
        if false_accepts_per_hour[index] != horizontal_coordinates[-1]:
            # Only add a point if it is a new faph
            # This ensures if a faph rate is repeated, we use the small false negative rate
            horizontal_coordinates.append(false_accepts_per_hour[index])
            vertical_coordinates.append(false_negative_rate_at_cutoffs[index])
            cutoffs_at_coordinate.append(cutoffs[index])

    if horizontal_coordinates[-1] > 0:
        # If there isn't a cutoff with 0 faph, then add a coordinate at (0,1)
        horizontal_coordinates.append(0.0)
        vertical_coordinates.append(1.0)
        cutoffs_at_coordinate.append(0.0)

    # The points on the curve are listed in descending order, flip them before returning
    horizontal_coordinates = np.flip(horizontal_coordinates)
    vertical_coordinates = np.flip(vertical_coordinates)
    cutoffs_at_coordinate = np.flip(cutoffs_at_coordinate)
    return horizontal_coordinates, vertical_coordinates, cutoffs_at_coordinate


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

    model = Model(os.path.join(config["train_dir"], folder, tflite_model_name))

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

        probabilities = model.predict_spectrogram(sample_fingerprint)

        if truncation_strategy != "none":
            prediction = probabilities[-1] > 0.5
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
        else:
            previous_probability = 0
            last_positive_index = 0
            for index, prob in enumerate(probabilities):
                if (previous_probability <= 0.5 and prob > 0.5) and (
                    index - last_positive_index
                    > config["spectrogram_length_final_layer"]
                ):
                    false_positives += 1
                    last_positive_index = index
                previous_probability = prob

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
