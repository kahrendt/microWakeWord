# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

"""Function to generate spectrograms and a class for loading and augmenting audio clips"""

import audiomentations
import audio_metadata
import datasets
import math
import os
import random
import wave

import numpy as np
import tensorflow as tf

from mmap_ninja.ragged import RaggedMmap
from pathlib import Path
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)


def generate_features_for_clip(audio, desired_spectrogram_length=None):
    """Generates spectrogram features for the given audio data.

    Args:
        clip (ndarray): audio data with sample rate 16 kHz and 16-bit samples
        desired_spectrogram_length (int, optional): Number of time features to include in the spectrogram.
                                                    Truncates earlier time features. Set to None to disable.

    Returns:
        (ndarray): spectrogram audio features
    """
    with tf.device("/cpu:0"):
        # The default settings match the TFLM preprocessor settings.
        # Preproccesor model is available from the tflite-micro repository, accessed December 2023.
        micro_frontend = frontend_op.audio_microfrontend(
            tf.convert_to_tensor(audio),
            sample_rate=16000,
            window_size=30,
            window_step=20,
            num_channels=40,
            upper_band_limit=7500,
            lower_band_limit=125,
            enable_pcan=True,
            min_signal_remaining=0.05,
            out_scale=1,
            out_type=tf.float32,
        )
        output = tf.multiply(micro_frontend, 0.0390625)

        spectrogram = output.numpy()
        if desired_spectrogram_length is not None:
            return spectrogram[
                -desired_spectrogram_length:
            ]  # truncate to match desired spectrogram size
        return spectrogram


class ClipsHandler:
    """ClipsHandler object that loads audio files from the disk, optionally filters by length, augments clips, and
    generates spectrogram features for augmented clips.

    Default augmentation settings and probabilities are borrowed from openWakeWord's data.py, accessed on February 23, 2024.

    Args:
        input_path (str): The path to audio files to be augmented.
        input_glob (str): The glob to choose audio files in `input_path`. Most audio types are supported, as clips
                          will automatically be converted to the appropriate format.
        impulse_paths (List[str], optional): The paths to room impulse response files.
                                   Set to None to disable the room impulse response augmentation.
        background_paths (List[str], optional): The paths to background audio files.
                                      Set to None to disable the background noise augmentaion.
        augmentation_probabilities (dict, optional): The individual probabilities of each augmentation. If all probabilities
                                           are zero, the input audio files will simply be padded with silence. The
                                           default values are:
                                            {
                                                "SevenBandParametricEQ": 0.25,
                                                "TanhDistortion": 0.25,
                                                "PitchShift": 0.25,
                                                "BandStopFilter": 0.25,
                                                "AddBackgroundNoise": 0.75,
                                                "Gain": 1.0,
                                                "RIR": 0.5
                                            }
        min_clip_duration_s (float, optional): The minimum clip duration (in seconds) of the input audio clips.
                                     Set to None to not filter clips.
        max_clip_duration_s (float, optional): The maximum clip duration (in seconds) of the input audio clips.
                                     Set to None to not filter clips.
        augmented_duration_s (float, optional): The final duration (in seconds) of the augmented file.
                                      Set to None to let spectrogram represent the clips actual duration.
        max_start_time_from_right_s (float, optional): The maximum time (in seconds) that the clip should start from the right.
                                             Only used if augmented_duration_s is set.
    """

    def __init__(
        self,
        input_path,
        input_glob,
        impulse_paths=None,
        background_paths=None,
        augmentation_probabilities: dict = {
            "SevenBandParametricEQ": 0.25,
            "TanhDistortion": 0.25,
            "PitchShift": 0.25,
            "BandStopFilter": 0.25,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        min_clip_duration_s=None,
        max_clip_duration_s=None,
        augmented_duration_s=None,
        max_start_time_from_right_s=None,
    ):
        #######################
        # Setup augmentations #
        #######################

        # If either the the background_paths or impulse_paths are not specified, use an identity transform instead
        def identity_transform(samples, sample_rate):
            return samples

        background_noise_augment = audiomentations.Lambda(
            transform=identity_transform, p=0.0
        )
        reverb_augment = audiomentations.Lambda(transform=identity_transform, p=0.0)

        if background_paths is not None:
            background_noise_augment = audiomentations.AddBackgroundNoise(
                p=augmentation_probabilities["AddBackgroundNoise"],
                sounds_path=background_paths,
                min_snr_in_db=-10,
                max_snr_in_db=15,
            )

        if impulse_paths is not None:
            reverb_augment = audiomentations.ApplyImpulseResponse(
                p=augmentation_probabilities["RIR"],
                ir_path=impulse_paths,
            )

        # Based on openWakeWord's augmentations, accessed on February 23, 2024.
        self.augment = audiomentations.Compose(
            transforms=[
                audiomentations.SevenBandParametricEQ(
                    p=augmentation_probabilities["SevenBandParametricEQ"],
                    min_gain_db=-6,
                    max_gain_db=6,
                ),
                audiomentations.TanhDistortion(
                    p=augmentation_probabilities["TanhDistortion"],
                    min_distortion=0.0001,
                    max_distortion=0.10,
                ),
                audiomentations.PitchShift(
                    p=augmentation_probabilities["PitchShift"],
                    min_semitones=-3,
                    max_semitones=3,
                ),
                audiomentations.BandStopFilter(
                    p=augmentation_probabilities["BandStopFilter"]
                ),
                background_noise_augment,
                audiomentations.Gain(
                    p=augmentation_probabilities["Gain"],
                    min_gain_in_db=-12,
                    max_gain_in_db=0,
                ),
                reverb_augment,
            ]
        )

        #####################################################
        # Load clips and optionally filter them by duration #
        #####################################################
        if min_clip_duration_s is None:
            min_clip_duration_s = 0

        if max_clip_duration_s is None:
            max_clip_duration_s = float("inf")

            if max_start_time_from_right_s is not None:
                max_clip_duration_s = min(
                    max_clip_duration_s, max_start_time_from_right_s
                )

            if augmented_duration_s is not None:
                max_clip_duration_s = min(max_clip_duration_s, augmented_duration_s)

        self.max_start_time_from_right_s = max_start_time_from_right_s
        self.augmented_duration_s = augmented_duration_s

        if (self.max_start_time_from_right_s is not None) and (
            self.augmented_duration_s is None
        ):
            raise ValueError(
                "max_start_time_from_right_s cannot be specified if augmented_duration_s is not configured."
            )

        if (
            (self.max_start_time_from_right_s is not None)
            and (self.augmented_duration_s is not None)
            and (self.max_start_time_from_right_s > self.augmented_duration_s)
        ):
            raise ValueError(
                "max_start_time_from_right_s cannot be greater than augmented_duration_s."
            )

        if self.augmented_duration_s is not None:
            self.desired_samples = int(augmented_duration_s * 16000)
        else:
            self.desired_samples = None

        paths_to_clips = [str(i) for i in Path(input_path).glob(input_glob)]

        # Filter audio clips by length
        if input_glob.endswith("wav"):
            # If it is a wave file, assume all wave files have the same parameters and filter by file size.
            # Based on openWakeWord's estimate_clip_duration and filter_audio_paths in data.py, accessed March 2, 2024.
            with wave.open(paths_to_clips[0], "rb") as input_wav:
                channels = input_wav.getnchannels()
                sample_width = input_wav.getsampwidth()
                sample_rate = input_wav.getframerate()
                frames = input_wav.getnframes()

                if (min_clip_duration_s > 0) or (not math.isinf(max_clip_duration_s)):
                    sizes = []
                    sizes.extend([os.path.getsize(i) for i in paths_to_clips])

                    # Correct for the wav file header bytes. Assumes all files in the directory have same parameters.
                    header_correction = (
                        os.path.getsize(paths_to_clips[0])
                        - frames * sample_width * channels
                    )

                    durations = []
                    for size in sizes:
                        durations.append(
                            (size - header_correction)
                            / (sample_rate * sample_width * channels)
                        )

                    filtered_paths = [
                        path_to_clip
                        for path_to_clip, duration in zip(paths_to_clips, durations)
                        if (min_clip_duration_s < duration)
                        and (duration < max_clip_duration_s)
                    ]
                else:
                    filtered_paths = paths_to_clips
        else:
            # If not a wave file, use the audio_metadata package to analyze audio file headers for the duration.
            # This is slower!
            filtered_paths = []

            if (min_clip_duration_s > 0) or (not math.isinf(max_clip_duration_s)):
                for audio_file in paths_to_clips:
                    metadata = audio_metadata.load(audio_file)
                    duration = metadata["streaminfo"]["duration"]
                    if (min_clip_duration_s < duration) and (
                        duration < max_clip_duration_s
                    ):
                        filtered_paths.append(audio_file)
            else:
                filtered_paths = paths_to_clips

        # Load all filtered clips
        audio_dataset = datasets.Dataset.from_dict(
            {"audio": [str(i) for i in filtered_paths]}
        ).cast_column("audio", datasets.Audio())

        # Convert all clips to 16 kHz sampling rate when accessed
        audio_dataset = audio_dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=16000)
        )
        self.clips = audio_dataset

    def augment_clip(self, input_audio):
        """Augments the input audio, optionally creating a fixed sized clip first.

        Args:
            input_audio (ndarray): audio data with sample rate 16 kHz and 16-bit samples

        Returns:
            (ndarray): the augmented audio with sample rate 16 kHz and 16-bit samples
        """
        if self.augmented_duration_s is not None:
            input_audio = self.create_fixed_size_clip(input_audio)
        output_audio = self.augment(input_audio, sample_rate=16000)

        return (output_audio * 32767).astype(np.int16)

    def augment_random_clip(self):
        """Augments a random loaded clip.

        Returns:
            (ndarray): a random clip's augmented audio with sample rate 16 kHz and 16-bit samples
        """
        rand_audio = random.choice(self.clips)
        return self.augment_clip(rand_audio["audio"]["array"])

    def save_random_augmented_clip(self, output_file):
        """Saves a random augmented clip.

        Args:
            output_file (str): file name to save augmented clip to with sample rate 16 kHz and 16-bit samples
        """
        augmented_audio = self.augment_random_clip()
        with wave.open(output_file, "wb") as output_wav_file:
            output_wav_file.setframerate(16000)
            output_wav_file.setsampwidth(2)
            output_wav_file.setnchannels(1)
            output_wav_file.writeframes(augmented_audio)

    def generate_augmented_spectrogram(self, input_audio):
        """Generates the spectrogram of the input audio after augmenting.

        Args:
            input_audio (ndarray): audio data with sample rate 16 kHz and 16-bit samples

        Returns:
            (ndarray): the spectrogram of the augmented audio
        """
        augmented_audio = self.augment_clip(input_audio)
        return generate_features_for_clip(augmented_audio)

    def generate_random_augmented_feature(self):
        """Generates the spectrogram of a random audio clip after augmenting.

        Returns:
            (ndarray): the spectrogram of the augmented audio from a random clip
        """
        rand_augmented_clip = self.augment_random_clip()
        return self.generate_augmented_feature(rand_augmented_clip)

    def augmented_features_generator(self):
        """Generator function for augmenting all loaded clips and computing their spectrograms

        Yields:
            (ndarray): the spectrogram of an augmented audio clip
        """
        for clip in self.clips:
            audio = clip["audio"]["array"]

            yield self.generate_augmented_spectrogram(audio)

    def save_augmented_features(self, mmap_output_dir):
        """Saves all augmented features in a RaggedMmap format

        Args:
            mmap_output_dir (str): Path to saved the RaggedMmap data
        """
        RaggedMmap.from_generator(
            out_dir=mmap_output_dir,
            sample_generator=self.augmented_features_generator(),
            batch_size=10,
            verbose=True,
        )

    def create_fixed_size_clip(self, x, sr=16000):
        """Create a fixed-length clip with self.desired_samples samples.

        If self.augmented_duration_s and self.max_start_time_from_right_s are specified, the entire clip is
        inserted randomly up to self.max_start_time_from_right_s duration from the right.

        Based on openWakeWord's data.py create_fixed_size_clip function, accessed on February 23, 2024

        Args:
            x (ndarray): The input audio to pad to a fixed size
            sr (int): The sample rate of the audio

        Returns:
            ndarray: A new array of audio data of the specified length
        """

        dat = np.zeros(self.desired_samples)

        if self.max_start_time_from_right_s is not None:
            max_samples_from_end = int(self.max_start_time_from_right_s * sr)
        else:
            max_samples_from_end = self.desired_samples

        assert max_samples_from_end > len(x)

        samples_from_end = np.random.randint(len(x), max_samples_from_end) + 1

        dat[-samples_from_end : -samples_from_end + len(x)] = x

        return dat
