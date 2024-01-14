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

import microwakeword.inception as inception
import microwakeword.data as input_data

# Tests internal or external streaming tflite models (updated)
# Can be quantized or not
#   - TODO: still need to update/fix comments


def streaming_model_false_accept_rate(flags, folder, tflite_model_name, features_fname):
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(flags.train_dir, folder, tflite_model_name)
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_quantized_model = input_details[0]["dtype"] == np.int8
    input_feature_slices = input_details[0]["shape"][1]

    window_stride = 1

    false_accept_count = 0

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

    features_data = np.load(features_fname)

    wake_word_detected = False

    start = 0
    end = 1
    ignore_slices = flags.spectrogram_length

    while end <= features_data.shape[0]:
        new_data_to_input = features_data[start:end, :]

        if is_quantized_model:
            new_data_to_input = quantize_input_data(new_data_to_input, input_details[0])

        # update indexes of streamed updates
        start += window_stride
        end += window_stride

        # Input new data and invoke the interpreter
        interpreter.set_tensor(
            input_details[0]["index"], np.expand_dims(new_data_to_input, 0)
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

        if wake_word_detected and wakeword_probability <= 0.5:
            wake_word_detected = False
            ignore_slices = flags.spectrogram_length

        if wakeword_probability > 0.5 and ignore_slices == 0:
            wake_word_detected = True
            false_accept_count += 1

        if ignore_slices > 0:
            ignore_slices -= 1

        if not start % 50000:
            logging.info(
                "DipCo false accepts = %d; false accepts per hour = %f",
                *(false_accept_count, false_accept_count / (start * 0.02 / 3600))
            )

    logging.info(
        "DipCo Total False Accepts = %d; total false accepts per hour = %f",
        *(
            false_accept_count,
            false_accept_count / (features_data.shape[0] * 0.02 / 3600.0),
        )
    )

def tf_model_accuracy(
    flags,
    folder,
    accuracy_name="tf_model_accuracy.txt",
    weights_name="best_weights",
):
    def compute_false_rates(
    true_positives, true_negatives, false_positives, false_negatives
    ):
        false_positive_rate = np.float64(false_positives) / (
            false_positives + true_negatives
        )
        false_negative_rate = np.float64(false_negatives) / (
            true_positives + false_negatives
        )

        return false_positive_rate, false_negative_rate

    audio_processor = input_data.FeatureHandler(
        general_negative_data_dir=flags.general_negative_dir,
        adversarial_negative_data_dir=flags.adversarial_negative_dir,
        positive_data_dir=flags.positive_dir,
    )
    
    test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
        "testing",
        batch_size=flags.batch_size,
        features_length=flags.spectrogram_length,
        truncation_strategy="truncate_start",
    )
    
    model = tf.saved_model.load(os.path.join(flags.train_dir, folder))
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

        count = true_positives + true_negatives + false_positives + false_negatives

        accuracy = (true_positives + true_negatives) / count
        recall = np.float64(true_positives) / (true_positives + false_negatives)
        precision = np.float64(true_positives) / (true_positives + false_positives)
        fpr = np.float64(false_positives) / (false_positives + true_negatives)
        fnr = np.float64(false_negatives) / (false_negatives + true_positives)

        if i % 1000 == 0 and i:
            logging.info(
                "TensorFlow model test: accuracy = %f; recall = %f; precision = %f; fpr = %f; fnr = %f; %d out of %d",
                *(accuracy, recall, precision, fpr, fnr, i, len(test_fingerprints))
            )

    false_positive_rate, false_negative_rate = compute_false_rates(
        true_positives, true_negatives, false_positives, false_negatives
    )    
    
    logging.info(
        "Final TensorFlow model test: accuracy = %f%%; recall = %f%%; precision = %f%%; fpr = %f%%; fnr = %f%%; (N=%d)",
        *(
            accuracy * 100,
            recall * 100,
            precision * 100,
            false_positive_rate * 100,
            false_negative_rate * 100,
            len(test_fingerprints),
        )
    )

    path = os.path.join(flags.train_dir, folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        fd.write("%f on set_size %d" % (accuracy * 100, len(test_fingerprints)))
    return accuracy * 100

def tflite_model_accuracy(
    flags,
    folder,
    tflite_model_name="stream_state_internal.tflite",
    accuracy_name="tflite_model_accuracy.txt",
):
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(flags.train_dir, folder, tflite_model_name)
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

    audio_processor = input_data.FeatureHandler(
        general_negative_data_dir=flags.general_negative_dir,
        adversarial_negative_data_dir=flags.adversarial_negative_dir,
        positive_data_dir=flags.positive_dir,
    )

    if input_feature_slices > 1:
        # If we have a nonstreaming model, truncate by removing the start of the spectrogram
        test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
            "testing",
            batch_size=flags.batch_size,
            features_length=flags.spectrogram_length,
            truncation_strategy="truncate_start",
        )
    else:
        # If a streaming model, fetch spectrograms twice the trained on length
        # This resets the internal variable states to match the blended in background noise
        test_fingerprints, test_ground_truth, _ = audio_processor.get_data(
            "testing",
            batch_size=flags.batch_size,
            features_length=flags.spectrogram_length * 2,
            truncation_strategy="truncate_start",
        )
    logging.info("Testing tflite model")

    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    accuracy = 0.0
    recall = 0.0
    precision = 0.0
    fpr = 0.0
    fnr = 0.0
    count = 0.0

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

        count = true_positives + true_negatives + false_positives + false_negatives

        accuracy = (true_positives + true_negatives) / count
        recall = np.float64(true_positives) / (true_positives + false_negatives)
        precision = np.float64(true_positives) / (true_positives + false_positives)
        fpr = np.float64(false_positives) / (false_positives + true_negatives)
        fnr = np.float64(false_negatives) / (false_negatives + true_positives)

        if i % 1000 == 0 and i:
            logging.info(
                "tflite model test: accuracy = %f; recall = %f; precision = %f; fpr = %f; fnr = %f; %d out of %d",
                *(accuracy, recall, precision, fpr, fnr, i, len(test_fingerprints))
            )

    logging.info(
        "true positives %d, true negatives %d, false positives %d, false negatives %d",
        *(true_positives, true_negatives, false_positives, false_negatives)
    )
    logging.info(
        "Final tflite model test: accuracy = %f%%; recall = %f%%; precision = %f%%; fpr = %f%%; fnr = %f%%; (N=%d)",
        *(
            accuracy * 100,
            recall * 100,
            precision * 100,
            fpr * 100,
            fnr * 100,
            len(test_fingerprints),
        )
    )

    path = os.path.join(flags.train_dir, folder)
    with open(os.path.join(path, accuracy_name), "wt") as fd:
        fd.write("%f on set_size %d" % (accuracy * 100, len(test_fingerprints)))
    return accuracy * 100
