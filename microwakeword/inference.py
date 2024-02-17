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

"""Functions and classes for using microwakeword models with audio files/data"""

# imports
import numpy as np
import tensorflow as tf
from feature_generation import generate_features_for_clip

class Model():
    """
    Class for loading and running tflite microwakeword models

    Args:
        tflite_model_path (str): path to tflite model file
    """
    def __init__(self, tflite_model_path):
        # Load tflite model
        interpreter = tf.lite.Interpreter(
            model_path=tflite_model_path,
        )
        interpreter.allocate_tensors()

        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

        self.is_quantized_model = self.input_details[0]["dtype"] == np.int8
        self.input_feature_slices = self.input_details[0]["shape"][1]

        for s in range(len(self.input_details)):
            if self.is_quantized_model:
                interpreter.set_tensor(
                    self.input_details[s]["index"],
                    np.zeros(self.input_details[s]["shape"], dtype=np.int8),
                )
            else:
                interpreter.set_tensor(
                    self.input_details[s]["index"],
                    np.zeros(self.input_details[s]["shape"], dtype=np.float32),
                )

        self.model = interpreter

    def predict_clip(self, data):
        """Run the model on a single clip of audio data

        Args:
            data (np.ndarray): input data for the model (16 khz, 16-bit PCM audio data)

        Returns:
            list: model predictions for the input audio data
        """

        # Get the spectrogram
        spec = generate_features_for_clip(data)

        # Slice the input data into the required number of chunks
        chunks = []
        for i in range(0, len(spec), self.input_feature_slices):
            chunk = spec[i : i + self.input_feature_slices]
            if len(chunk) == self.input_feature_slices:
                chunks.append(chunk)

        # Get the prediction for each chunk
        predictions = []
        for chunk in chunks:
            if self.is_quantized_model:
                new_data_to_input = self.quantize_input_data(chunk, self.input_details[0])
            self.model.set_tensor(
                self.input_details[0]["index"],
                np.reshape(new_data_to_input, self.input_details[0]["shape"]),
            )
            self.model.invoke()
            output = self.model.get_tensor(self.output_details[0]["index"])[0][0]
            if self.is_quantized_model:
                output = self.dequantize_output_data(output, self.output_details[0])

            predictions.append(output)

        return predictions

    def quantize_input_data(self, data, input_details) -> np.ndarray:
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

    def dequantize_output_data(self, data: np.ndarray, output_details: dict) -> np.ndarray:
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
