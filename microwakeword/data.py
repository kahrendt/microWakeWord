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

"""Functions and classes for loading/augmenting spectrograms"""

import copy
import os
import random

import numpy as np

from absl import logging
from pathlib import Path
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.clips import Clips
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.spectrograms import SpectrogramGeneration

from microwakeword.audio.audio_utils import generate_features_for_clip

import torch

torch.set_num_threads(1)
SAMPLING_RATE = 16000
USE_ONNX = True
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)

def silero_vad_probs(audio_samples):
    # speech_probs = []
    window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
    for i in range(0, audio_samples.shape[1], window_size_samples):
        chunk = torch.from_numpy(audio_samples[:,i:i+ window_size_samples])
        if chunk.shape[1] < window_size_samples:
            break
        speech_prob = model(chunk, SAMPLING_RATE)
        # speech_probs.append(speech_prob)
    vad_iterator.reset_states() # reset model states after each audio

    return speech_prob.numpy()

def spec_augment(
    spectrogram: np.ndarray,
    time_mask_max_size: int = 0,
    time_mask_count: int = 0,
    freq_mask_max_size: int = 0,
    freq_mask_count: int = 0,
):
    """Applies SpecAugment to the input spectrogram.
    Based on SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition by D. Park, W. Chan, Y. Zhang, C. Chiu, B. Zoph, E Cubuk, Q Le
    https://arxiv.org/pdf/1904.08779.pdf
    Implementation based on https://github.com/pyyush/SpecAugment/tree/master

    Args:
        spectrogram (numpy.ndarray): The input spectrogram.
        time_mask_max_size (int): The maximum size of time feature masks. Defaults to 0.
        time_mask_count (int): The total number of separate time masks. Defaults to 0.
        freq_mask_max_size (int): The maximum size of frequency feature masks. Defaults to 0.
        time_mask_count (int): The total number of separate feature masks. Defaults to 0.

    Returns:
        numpy.ndarray: The masked spectrogram.
    """

    time_frames = spectrogram.shape[0]
    freq_bins = spectrogram.shape[1]

    # Spectrograms yielded from a generator are read only
    augmented_spectrogram = np.copy(spectrogram)

    for i in range(time_mask_count):
        t = int(np.random.uniform(0, time_mask_max_size))
        t0 = random.randint(0, time_frames - t)
        augmented_spectrogram[t0 : t0 + t, :] = 0

    for i in range(freq_mask_count):
        f = int(np.random.uniform(0, freq_mask_max_size))
        f0 = random.randint(0, freq_bins - f)
        augmented_spectrogram[:, f0 : f0 + f] = 0

    return augmented_spectrogram


def fixed_length_spectrogram(
    spectrogram: np.ndarray, features_length: int, truncation_strategy: str = "random", fixed_right_cutoff: int = 0,
):
    """Returns a spectrogram with specified length. Pads with zeros at the start if too short. Removes feature windows following ``truncation_strategy`` if too long.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to truncate or pad.
        features_length (int): The desired spectrogram length.
        truncation_strategy (str): How to truncate if ``spectrogram`` is longer than ``features_length`` One of:
            random: choose a random portion of the entire spectrogram - useful for long negative samples
            truncate_start: remove the start of the spectrogram
            truncate_end: remove the end of the spectrogram
            none: returns the entire spectrogram regardless of features_length


    Returns:
        numpy.ndarry: The fixed length spectrogram due to padding or truncation.
    """
    data_length = spectrogram.shape[0]
    features_offset = 0
    if data_length > features_length:
        if truncation_strategy == "random":
            features_offset = np.random.randint(0, data_length - features_length)
        elif truncation_strategy == "none":
            # return the entire spectrogram
            features_length = data_length
        elif truncation_strategy == "truncate_start":
            features_offset = data_length - features_length
        elif truncation_strategy == "truncate_end":
            features_offset = 0
        elif truncation_strategy == "fixed_right_cutoff":
            features_offset = data_length - features_length - fixed_right_cutoff
    else:
        pad_slices = features_length - data_length

        spectrogram = np.pad(
            spectrogram, ((pad_slices, 0), (0, 0)), constant_values=(0, 0)
        )
        features_offset = 0

    return spectrogram[features_offset : (features_offset + features_length)]


class MmapFeatureGenerator(object):
    """A class that handles loading spectrograms from Ragged MMaps for training or testing.

    Args:
        path (str): Input directory to the Ragged MMaps. The Ragged MMap folders should be included in the following file structure:
            training/ (spectrograms to use for training the model)
            validation/ (spectrograms used to validate the model while training)
            testing/ (spectrograms used to test the model after training)
            validation_ambient/ (spectrograms of long duration background audio clips that are split and validated while training)
            testing_ambient/ (spectrograms of long duration background audio clips to test the model after training)
        label (bool): The class each spectrogram represents; i.e., wakeword or not.
        sampling_weight (float): The sampling weight for how frequently a spectrogram from this dataset is chosen.
        penalty_weight (float): The penalizing weight for incorrect predictions for each spectrogram.
        truncation_strategy (str): How to truncate if ``spectrogram`` is too long.
        stride (int): The stride in the model's first layer.
        step (float): The window step duration (in seconds).
    """

    def __init__(
        self,
        path: str,
        label: bool,
        sampling_weight: float,
        penalty_weight: float,
        truncation_strategy: str,
        stride: int,
        step: float,
        fixed_right_cutoff: int = 0
    ):
        self.label = float(label)
        self.sampling_weight = sampling_weight
        self.penalty_weight = penalty_weight
        self.truncation_strategy = truncation_strategy
        self.fixed_right_cutoff = fixed_right_cutoff

        self.stride = stride
        self.step = step

        self.stats = {}
        self.feature_sets = {}

        self.feature_sets["testing"] = []
        self.feature_sets["training"] = []
        self.feature_sets["validation"] = []
        self.feature_sets["validation_ambient"] = []
        self.feature_sets["testing_ambient"] = []

        self.loaded_features = []

        dirs = [
            "testing",
            "training",
            "validation",
            "testing_ambient",
            "validation_ambient",
        ]

        for set_index in dirs:
            duration = 0.0
            count = 0

            search_path_directory = os.path.join(path, set_index)
            search_path = [
                str(i)
                for i in Path(os.path.abspath(search_path_directory)).glob("**/*_mmap/")
            ]

            for mmap_path in search_path:
                imported_features = RaggedMmap(mmap_path)

                self.loaded_features.append(imported_features)
                feature_index = len(self.loaded_features) - 1

                for i in range(0, len(imported_features)):
                    self.feature_sets[set_index].append(
                        {
                            "loaded_feature_index": feature_index,
                            "subindex": i,
                        }
                    )

                    duration += step * imported_features[i].shape[0]
                    count += 1

            random.shuffle(self.feature_sets[set_index])

            self.stats[set_index] = {
                "spectrogram_count": count,
                "total_duration": duration,
            }

    def get_mode_duration(self, mode: str):
        """Retrieves the total duration of the spectrograms in the mode set.

        Args:
            mode (str): Specifies the set. One of "training", "validation", "testing", "validation_ambient", "testing_ambient".

        Returns:
            float: The duration in hours.
        """
        return self.stats[mode]["total_duration"]

    def get_mode_size(self, mode):
        """Retrieves the total count of the spectrograms in the mode set.

        Args:
            mode (str): Specifies the set. One of "training", "validation", "testing", "validation_ambient", "testing_ambient".

        Returns:
            int: The spectrogram count.
        """
        return self.stats[mode]["spectrogram_count"]

    def get_random_spectrogram(
        self, mode: str, features_length: int, truncation_strategy: str
    ):
        """Retrieves a random spectrogram from the specified mode with specified length after truncation.

        Args:
            mode (str): Specifies the set. One of "training", "validation", "testing", "validation_ambient", "testing_ambient".
            features_length (int): The length of the spectrogram in feature windows.
            truncation_strategy (str): How to truncate if ``spectrogram`` is too long.

        Returns:
            numpy.ndarray: A random spectrogram of specified length after truncation.
        """

        fixed_right_cutoff = self.fixed_right_cutoff
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy
            fixed_right_cutoff = 0

        feature = random.choice(self.feature_sets[mode])
        spectrogram = self.loaded_features[feature["loaded_feature_index"]][
            feature["subindex"]
        ]

        spectrogram = fixed_length_spectrogram(
            spectrogram,
            features_length,
            truncation_strategy,
            fixed_right_cutoff,
        )

        # Spectrograms with type np.uint16 haven't been scaled
        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = spectrogram.astype(np.float32) * 0.0390625

        return spectrogram

    def get_feature_generator(
        self,
        mode,
        features_length,
        truncation_strategy="default",
        fixed_right_cutoff: int = 0,
    ):
        """A Python generator that yields spectrograms from the specified mode of specified length after truncation.

        Args:
            mode (str): Specifies the set. One of "training", "validation", "testing", "validation_ambient", "testing_ambient".
            features_length (int): The length of the spectrogram in feature windows.
            truncation_strategy (str): How to truncate if ``spectrogram`` is too long.

        Yields:
            numpy.ndarray: A random spectrogram of specified length after truncation.
        """
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy

        for feature in self.feature_sets[mode]:
            spectrogram = self.loaded_features[feature["loaded_feature_index"]][
                feature["subindex"]
            ]

            # Spectrograms with type np.uint16 haven't been scaled
            if np.issubdtype(spectrogram.dtype, np.uint16):
                spectrogram = spectrogram.astype(np.float32) * 0.0390625

            if truncation_strategy == "split":
                for feature_start_index in range(
                    0,
                    spectrogram.shape[0] - features_length,
                    int(1000 * self.step * self.stride),
                ):  # 10*2 features corresponds to 200 ms
                    split_spectrogram = spectrogram[
                        feature_start_index : feature_start_index + features_length
                    ]

                    yield split_spectrogram
            else:
                spectrogram = fixed_length_spectrogram(
                    spectrogram,
                    features_length,
                    truncation_strategy,
                    fixed_right_cutoff,
                )

                yield spectrogram


class ClipsHandlerWrapperGenerator(object):
    """A class that handles loading spectrograms from audio files on the disk to use while training. This generates spectrograms with random augmentations applied during the training process.

    Args:
        spectrogram_generation (SpectrogramGeneration): Object that handles generating spectrograms from audio files.
        label (bool): The class each spectrogram represents; i.e., wakeword or not.
        sampling_weight (float): The sampling weight for how frequently a spectrogram from this dataset is chosen.
        penalty_weight (float): The penalizing weight for incorrect predictions for each spectrogram.
        truncation_strategy (str): How to truncate if ``spectrogram`` is too long.
    """

    def __init__(
        self,
        spectrogram_generation: SpectrogramGeneration,
        label: bool,
        sampling_weight: float,
        penalty_weight: float,
        truncation_strategy: str,
    ):
        self.spectrogram_generation = spectrogram_generation
        self.label = label
        self.sampling_weight = sampling_weight
        self.penalty_weight = penalty_weight
        self.truncation_strategy = truncation_strategy

        self.augmented_generator = self.spectrogram_generation.spectrogram_generator(
            random=True
        )

    def get_mode_duration(self, mode):
        """Function to maintain compatability with the MmapFeatureGenerator class."""
        return 0.0

    def get_mode_size(self, mode):
        """Function to maintain compatability with the MmapFeatureGenerator class. This class is intended only for retrieving spectrograms for training."""
        if mode == "training":
            return len(self.spectrogram_generation.clips.clips)
        else:
            return 0

    def get_random_spectrogram(self, mode, features_length, truncation_strategy, fixed_right_cutoff: int = 0):
        """Retrieves a random spectrogram from the specified mode with specified length after truncation.

        Args:
            mode (str): Specifies the set, but is ignored for this class. It is assumed the spectrograms will be for training.
            features_length (int): The length of the spectrogram in feature windows.
            truncation_strategy (str): How to truncate if ``spectrogram`` is too long.

        Returns:
            numpy.ndarray: A random spectrogram of specified length after truncation.
        """

        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy

        spectrogram = next(self.augmented_generator)

        spectrogram = fixed_length_spectrogram(
            spectrogram,
            features_length,
            truncation_strategy,
            fixed_right_cutoff,
        )

        # Spectrograms with type np.uint16 haven't been scaled
        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = spectrogram.astype(np.float32) * 0.0390625

        return spectrogram

    def get_feature_generator(
        self,
        mode,
        features_length,
        truncation_strategy="default",
    ):
        """Function to maintain compatability with the MmapFeatureGenerator class."""
        for x in []:
            yield x


# class DistilClipsHandlerWrapperGenerator(object):
#     """A class that handles loading spectrograms from audio files on the disk to use while training. This generates spectrograms with random augmentations applied during the training process.

#     Args:
#         spectrogram_generation (SpectrogramGeneration): Object that handles generating spectrograms from audio files.
#         label (bool): The class each spectrogram represents; i.e., wakeword or not.
#         sampling_weight (float): The sampling weight for how frequently a spectrogram from this dataset is chosen.
#         penalty_weight (float): The penalizing weight for incorrect predictions for each spectrogram.
#         truncation_strategy (str): How to truncate if ``spectrogram`` is too long.
#     """

#     def __init__(
#         self,
#         clips_handler: Clips,
#         augmented_generation: Augmentation,
#         sampling_weight: float,
#         penalty_weight: float,
#         truncation_strategy: str,
#     ):
#         self.clips = clips_handler
#         self.augment_generation = augmented_generation
#         self.sampling_weight = sampling_weight
#         self.penalty_weight = penalty_weight
#         self.truncation_strategy = truncation_strategy

#         self.augmented_generator = self.augment_generation.augment_generator(
#             self.clips.random_audio_generator()
#         )

#     def get_mode_duration(self, mode):
#         """Function to maintain compatability with the MmapFeatureGenerator class."""
#         return 0.0

#     def get_mode_size(self, mode):
#         """Function to maintain compatability with the MmapFeatureGenerator class. This class is intended only for retrieving spectrograms for training."""
#         if mode == "training":
#             return len(self.clips.clips)
#         else:
#             return 0

#     def get_random_spectrogram(self, mode, features_length, truncation_strategy):
#         """Retrieves a random spectrogram from the specified mode with specified length after truncation.

#         Args:
#             mode (str): Specifies the set, but is ignored for this class. It is assumed the spectrograms will be for training.
#             features_length (int): The length of the spectrogram in feature windows.
#             truncation_strategy (str): How to truncate if ``spectrogram`` is too long.

#         Returns:
#             numpy.ndarray: A random spectrogram of specified length after truncation.
#         """

#         if truncation_strategy == "default":
#             truncation_strategy = self.truncation_strategy

#         augmented_clip = next(self.augmented_generator)

#         return augmented_clip

#         # mww_features = generate_features_for_clip(augmented_clip)
#         # silero_prob = np.array(silero_vad_probs(augmented_clip)[-1])

#         # spectrogram = fixed_length_spectrogram(
#         #     mww_features,
#         #     features_length,
#         #     truncation_strategy,
#         # )

#         # # Spectrograms with type np.uint16 haven't been scaled
#         # if np.issubdtype(spectrogram.dtype, np.uint16):
#         #     spectrogram = spectrogram.astype(np.float32) * 0.0390625

#         # return spectrogram, silero_prob

#     def get_feature_generator(
#         self,
#         mode,
#         features_length,
#         truncation_strategy="default",
#     ):
#         """Function to maintain compatability with the MmapFeatureGenerator class."""
#         for x in []:
#             yield x, 0.0


class FeatureHandler(object):
    """Class that handles loading spectrogram features and providing them to the training and testing functions.

    Args:
      config: dictionary containing microWakeWord training configuration
    """

    def __init__(
        self,
        config: dict,
    ):
        self.feature_providers = []

        logging.info("Loading and analyzing data sets.")

        for feature_set in config["features"]:
            if feature_set["type"] == "mmap":
                self.feature_providers.append(
                    MmapFeatureGenerator(
                        feature_set["features_dir"],
                        feature_set["truth"],
                        feature_set["sampling_weight"],
                        feature_set["penalty_weight"],
                        feature_set["truncation_strategy"],
                        stride=config["stride"],
                        step=config["window_step_ms"] / 1000.0,
                        fixed_right_cutoff = feature_set.get("fixed_right_cutoff", 0),
                    )
                )
            elif feature_set["type"] == "clips":
                clips_handler = Clips(**feature_set["clips_settings"])
                augmentation_applier = Augmentation(
                    **feature_set["augmentation_settings"]
                )
                spectrogram_generator = SpectrogramGeneration(
                    clips_handler,
                    augmentation_applier,
                    **feature_set["spectrogram_generation_settings"],
                )
                self.feature_providers.append(
                    ClipsHandlerWrapperGenerator(
                        spectrogram_generator,
                        feature_set["truth"],
                        feature_set["sampling_weight"],
                        feature_set["penalty_weight"],
                        feature_set["truncation_strategy"],
                    )
                )
            # elif feature_set["type"] == "distil_clips":
            #     clips_handler = Clips(**feature_set["clips_settings"])
            #     augmentation_applier = Augmentation(
            #         **feature_set["augmentation_settings"]
            #     )
            #     self.feature_providers.append(
            #         DistilClipsHandlerWrapperGenerator(
            #             clips_handler,
            #             augmentation_applier,
            #             feature_set["sampling_weight"],
            #             feature_set["penalty_weight"],
            #             feature_set["truncation_strategy"],
            #         )
            #     )
            set_modes = [
                "training",
                "validation",
                "testing",
                "validation_ambient",
                "testing_ambient",
            ]
            total_spectrograms = 0
            for set in set_modes:
                total_spectrograms += self.feature_providers[-1].get_mode_size(set)

            if total_spectrograms == 0:
                logging.warning("No spectrograms found in a configured feature set:")
                logging.warning(feature_set)

    def get_mode_duration(self, mode: str):
        """Returns the durations of all spectrogram features in the given mode.

        Args:
            mode (str): which training set to compute duration over. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`

        Returns:
            duration, in seconds, of all spectrograms in this mode
        """

        sample_duration = 0
        for provider in self.feature_providers:
            sample_duration += provider.get_mode_duration(mode)
        return sample_duration

    def get_mode_size(self, mode: str):
        """Returns the count of all spectrogram features in the given mode.

        Args:
            mode (str): which training set to count the spectrograms. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`

        Returns:
            count of spectrograms in given mode
        """
        sample_count = 0
        for provider in self.feature_providers:
            sample_count += provider.get_mode_size(mode)
        return sample_count

    def get_data(
        self,
        mode: str,
        batch_size: int,
        features_length: int,
        truncation_strategy: str = "default",
        augmentation_policy: dict = {
            "freq_mix_prob": 0.0,
            "time_mask_max_size": 0,
            "time_mask_count": 0,
            "freq_mask_max_size": 0,
            "freq_mask_count": 0,
        },
    ):
        """Gets spectrograms from the appropriate mode. Ensures spectrograms are the approriate length and optionally applies augmentation.

        Args:
            mode (str): which training set to count the spectrograms. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`
            batch_size (int): number of spectrograms in the sample for training mode
            features_length (int): the length of the spectrograms
            truncation_strategy (str): how to truncate spectrograms longer than `features_length`
            augmentation_policy (dict): dictionary that specifies augmentation settings. It has the following keys:
                freq_mix_prob: probability that FreqMix is applied
                time_mask_max_size: maximum size of time feature masks for SpecAugment
                time_mask_count: the total number of separate time masks applied for SpecAugment
                freq_mask_max_size: maximum size of frequency feature masks for SpecAugment
                time_mask_count: the total number of separate feature masks applied for SpecAugment

        Returns:
            data: spectrograms in a NumPy array (or as a list if in mode is `*_ambient`)
            labels: ground truth for the spectrograms; i.e., whether a positive sample or negative sample
            weights: penalizing weight for incorrect predictions for each spectrogram
        """

        if mode == "training":
            sample_count = batch_size
        elif (mode == "validation") or (mode == "testing"):
            sample_count = self.get_mode_size(mode)

        data = []
        labels = []
        weights = []

        if mode == "training":
            random_feature_providers = random.choices(
                [
                    provider
                    for provider in self.feature_providers
                    if provider.get_mode_size("training")
                ],
                [
                    provider.sampling_weight
                    for provider in self.feature_providers
                    if provider.get_mode_size("training")
                ],
                k=sample_count,
            )

            for provider in random_feature_providers:
                spectrogram = provider.get_random_spectrogram(
                    "training", features_length, truncation_strategy
                )
                spectrogram = spec_augment(
                    spectrogram,
                    augmentation_policy["time_mask_max_size"],
                    augmentation_policy["time_mask_count"],
                    augmentation_policy["freq_mask_max_size"],
                    augmentation_policy["freq_mask_count"],
                )

                data.append(spectrogram)
                labels.append(float(provider.label))
                weights.append(float(provider.penalty_weight))

            # for provider in random_feature_providers:
            #     audio_clip = provider.get_random_spectrogram(
            #         "training", features_length, truncation_strategy
            #     )

            #     mww_features = generate_features_for_clip(audio_clip)
            #     # silero_prob = np.array(silero_vad_probs(augmented_clip)[-1])

            #     if first_clip:
            #         stacked_clip = audio_clip
            #         first_clip = False
            #     else:
            #         stacked_clip = np.vstack((stacked_clip, audio_clip))

            #     spectrogram = fixed_length_spectrogram(
            #         mww_features,
            #         features_length,
            #         truncation_strategy,
            #     )

            #     # Spectrograms with type np.uint16 haven't been scaled
            #     if np.issubdtype(spectrogram.dtype, np.uint16):
            #         spectrogram = spectrogram.astype(np.float32) * 0.0390625

            #     spectrogram = spec_augment(
            #         spectrogram,
            #         augmentation_policy["time_mask_max_size"],
            #         augmentation_policy["time_mask_count"],
            #         augmentation_policy["freq_mask_max_size"],
            #         augmentation_policy["freq_mask_count"],
            #     )

            #     data.append(spectrogram)
            #     # labels.append(label)
            #     weights.append(float(provider.penalty_weight))
            # labels = silero_vad_probs(stacked_clip)

            # for i in range(0, len(labels)):
            #     current_label = labels[i]
            #     if current_label > 0.5:
            #         labels[i] = min(current_label+0.15, 1.0)
            #     else:
            #         labels[i] = max(current_label-0.15, 0.0)
        else:
            for provider in self.feature_providers:
                generator = provider.get_feature_generator(
                    mode, features_length, truncation_strategy
                )

                for spectrogram in generator:
                    data.append(spectrogram)
                    labels.append(provider.label)
                    weights.append(provider.penalty_weight)

        if truncation_strategy != "none":
            # Spectrograms are all the same length, convert to numpy array
            data = np.array(data)
        labels = np.array(labels)
        weights = np.array(weights)

        if truncation_strategy == "none":
            # Spectrograms may be of different length
            return data, np.array(labels), np.array(weights)

        indices = np.arange(labels.shape[0])

        if mode == "testing" or "validation":
            # Randomize the order of the data, weights, and labels
            np.random.shuffle(indices)

        return data[indices], labels[indices], weights[indices]
