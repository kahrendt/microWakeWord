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
from microwakeword.feature_generation import ClipsHandler
# from microwakeword.room_simulation_feature_generation import RoomClipsHandler

def spec_augment(
    spectrogram,
    time_mask_max_size=0,
    time_mask_count=0,
    freq_mask_max_size=0,
    freq_mask_count=0,
):
    """Applies SpecAugment to the input spectrogram.
    Based on SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition by D. Park, W. Chan, Y. Zhang, C. Chiu, B. Zoph, E Cubuk, Q Le
    https://arxiv.org/pdf/1904.08779.pdf
    Implementation based on https://github.com/pyyush/SpecAugment/tree/master

    Args:
        spectrogram: the input spectrogram
        time_mask_max_size: maximum size of time feature masks
        time_mask_count: the total number of separate time masks
        freq_mask_max_size: maximum size of frequency feature masks
        time_mask_count: the total number of separate feature masks

    Returns:
        masked spectrogram
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
    spectrogram, features_length, truncation_strategy="random"
):
    """Returns a spectrogram with specified length. Pads with zeros at the start if too short.

    Args:
        spectrogram: the spectrogram to truncate or pad
        features_length: the desired spectrogram length
        truncation_strategy: how to truncate if ``spectrogram`` is longer than ``features_length`` One of:
            random: choose a random portion of the entire spectrogram - useful for long negative samples
            truncate_start: remove the start of the spectrogram
            truncate_end: remove the end of the spectrogram
            none: returns the entire spectrogram regardless of features_length


    Returns:
        fixed length spectrogram after truncating or padding
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
    else:
        pad_slices = features_length - data_length

        spectrogram = np.pad(
            spectrogram, ((pad_slices, 0), (0, 0)), constant_values=(0, 0)
        )
        features_offset = 0

    return spectrogram[features_offset : (features_offset + features_length)]

class MmapFeatureGenerator(object):
    def __init__(
        self,
        path,
        label,
        sampling_weight,
        penalty_weight,
        truncation_strategy,
    ):
        self.label = float(label)
        self.sampling_weight = sampling_weight
        self.penalty_weight = penalty_weight
        self.truncation_strategy = truncation_strategy
        
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

                    duration += (
                        0.02 * imported_features[i].shape[0]
                    )  # Each feature represents 0.02 seconds of audio
                    count += 1

            random.shuffle(self.feature_sets[set_index])

            self.stats[set_index] = {
                "spectrogram_count": count,
                "total_duration": duration,
            }

    def get_mode_duration(self, mode):
        return self.stats[mode]["total_duration"]                

    def get_mode_size(self, mode):
        return self.stats[mode]["spectrogram_count"]                

    def get_random_spectrogram(self, mode, features_length, truncation_strategy):
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy
            
        feature = random.choice(self.feature_sets[mode])
        spectrogram = self.loaded_features[feature["loaded_feature_index"]][feature["subindex"]]
        
        spectrogram = fixed_length_spectrogram(
            spectrogram,
            features_length,
            truncation_strategy,
        )
        # if spectrogram.shape[-1] == 40:
        #     blank_playback = np.zeros(spectrogram.shape)
        #     spectrogram = np.concatenate([spectrogram, blank_playback], axis=-1)
        return spectrogram

    def get_feature_generator(
        self,
        mode,
        features_length,
        truncation_strategy="default",
    ):
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy
            
        for feature in self.feature_sets[mode]:
            spectrogram = self.loaded_features[feature["loaded_feature_index"]][feature["subindex"]]
            if truncation_strategy == "split":
                for feature_start_index in range(0, spectrogram.shape[0]-features_length, 10):
                    split_spectrogram = spectrogram[feature_start_index : feature_start_index + features_length]
                    
                    # if split_spectrogram.shape[-1] == 40:
                    #     blank_playback = np.zeros(split_spectrogram.shape)
                    #     split_spectrogram = np.concatenate([split_spectrogram, blank_playback], axis=-1)
                        
                    yield split_spectrogram
            else:
                spectrogram = fixed_length_spectrogram(
                    spectrogram,
                    features_length,
                    truncation_strategy,
                )
                
                # if spectrogram.shape[-1] == 40:
                #     blank_playback = np.zeros(spectrogram.shape)
                #     spectrogram = np.concatenate([spectrogram, blank_playback], axis=-1)
                yield spectrogram
    
    # def get_split_feature_generator(
    #     self,
    #     mode,
    #     features_length,
    #     split_stride=10,
    # ):
    #     for feature in self.feature_sets[mode]:
    #         spectrogram = self.loaded_features[feature["loaded_feature_index"]][feature["subindex"]]
    #         for feature_start_index in range(0, spectrogram.shape[0]-features_length, split_stride):
    #             yield spectrogram[feature_start_index : feature_start_index + features_length]

class ClipsHandlerWrapperGenerator(object):
    def __init__(self, 
        clips_handler,
        label,
        sampling_weight,
        penalty_weight,
        truncation_strategy,
        generate=False,
    ):
        self.clips_handler = clips_handler
        self.label = label
        self.sampling_weight = sampling_weight
        self.penalty_weight = penalty_weight
        self.truncation_strategy = truncation_strategy
        self.generate = generate
    
    def get_mode_duration(self, mode):
        return 0.0
    
    def get_mode_size(self, mode):
        if mode == "training":
            return len(self.clips_handler.clips)
        else:
            return 0

    def get_random_spectrogram(self, mode, features_length, truncation_strategy):
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy
        
        if self.generate:
            spectrogram = next(self.clips_handler.generate_clip_and_augmented_spectrogram())
        else:
            spectrogram = self.clips_handler.generate_random_augmented_spectrogram()

        
        spectrogram = fixed_length_spectrogram(
            spectrogram,
            features_length,
            truncation_strategy,
        )
        
        # if spectrogram.shape[-1] == 40:
        #     blank_playback = np.zeros(spectrogram.shape)
        #     spectrogram = np.concatenate([spectrogram, blank_playback], axis=-1)
        
        return spectrogram

    def get_feature_generator(
        self,
        mode,
        features_length,
        truncation_strategy="default",
    ):
        for x in []:
            yield x

    def get_split_feature_generator(
        self,
        mode,
        features_length,
        split_stride=10,
    ):
        for x in []:
            yield x
            
class FeatureHandler(object):
    """Class that handles loading spectrogram features and providing them to the training and testing functions.

    Args:
      config: dictionary containing microWakeWord training configuration
    """

    def __init__(
        self,
        config,
    ):
        self.feature_providers = []
        
        logging.info("Loading and analyzing data sets.")

        for feature_set in config["features"]:
            if feature_set["type"] == "mmap":
                self.feature_providers.append(MmapFeatureGenerator(feature_set['features_dir'], feature_set["truth"], feature_set["sampling_weight"], feature_set["penalty_weight"], feature_set["truncation_strategy"]))
            elif feature_set["type"] == "clips":
                clips_handler = ClipsHandler(**feature_set)
                # clips_handler = ClipsHandler(
                #                 input_path=feature_set['features_dir'],
                #                 input_glob=feature_set["input_glob"],
                #                 impulse_paths=feature_set["impulse_paths"],
                #                 background_paths=feature_set["background_paths"], 
                #                 augmentation_probabilities = feature_set["augmentation_probabilities"],
                #                 augmented_duration_s =feature_set["augmented_duration_s"],
                #                 max_start_time_from_right_s = None,
                #                 max_jitter_s = feature_set["max_jitter_s"],
                #                 min_jitter_s = feature_set["min_jitter_s"],
                #                 max_clip_duration_s = feature_set["max_clip_duration_s"], 
                #                 min_clip_duration_s = feature_set["min_clip_duration_s"],
                #                 remove_silence=False,
                #                 truncate_clip_s=None,
                #                 repeat_clip_min_duration_s=None,
                #                 split_spectrogram_duration_s=None,
                #             )
                self.feature_providers.append(ClipsHandlerWrapperGenerator(clips_handler, feature_set["truth"], feature_set["sampling_weight"], feature_set["penalty_weight"], feature_set["truncation_strategy"], feature_set['generate']))
            elif feature_set["type"] == "room_clips":
                clips_handler = RoomClipsHandler(
                                input_path=feature_set['features_dir'],
                                input_glob=feature_set["input_glob"],
                                impulse_path=feature_set["impulse_path"],
                                impulse_glob=feature_set["impulse_glob"],
                                playback_background_path=feature_set["playback_background_path"], 
                                playback_background_glob=feature_set["playback_background_glob"], 
                                background_path=feature_set["background_path"], 
                                background_glob=feature_set["background_glob"], 
                                augmented_duration_s =feature_set["augmented_duration_s"],
                                max_start_time_from_right_s = None,
                                max_jitter_s = feature_set["max_jitter_s"],
                                min_jitter_s = feature_set["min_jitter_s"],
                                max_clip_duration_s = feature_set["max_clip_duration_s"], 
                                min_clip_duration_s = feature_set["min_clip_duration_s"],
                            )
                self.feature_providers.append(ClipsHandlerWrapperGenerator(clips_handler, feature_set["truth"], feature_set["sampling_weight"], feature_set["penalty_weight"], feature_set["truncation_strategy"], feature_set['generate']))

    def get_mode_duration(self, mode):
        """Returns the durations of all spectrogram features in the given mode.

        Args:
            mode: which training set to compute duration over. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`

        Returns:
            duration, in seconds, of all spectrograms in this mode
        """

        sample_duration = 0
        for provider in self.feature_providers:
            sample_duration += provider.get_mode_duration(mode)
        return sample_duration

    def get_mode_size(self, mode):
        """Returns the count of all spectrogram features in the given mode.

        Args:
            mode: which training set to count the spectrograms. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`

        Returns:
            count of spectrograms in given mode
        """
        sample_count = 0
        for provider in self.feature_providers:
            sample_count += provider.get_mode_size(mode)
        return sample_count

    def get_data(
        self,
        mode,
        batch_size,
        features_length,
        truncation_strategy="default",
        augmentation_policy={
            "mix_up_prob": 0.0,
            "freq_mix_prob": 0.0,
            "time_mask_max_size": 0,
            "time_mask_count": 0,
            "freq_mask_max_size": 0,
            "freq_mask_count": 0,
        },
    ):
        """Gets spectrograms from the appropriate mode. Ensures spectrograms are the approriate length and optionally applies augmentation.

        Args:
            mode: which training set to count the spectrograms. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`
            batch_size: number of spectrograms in the sample for training mode
            features_length: the length of the spectrograms
            truncation_strategy: how to truncate spectrograms longer than `features_length`
            augmentation_policy: dictionary that specifies augmentation settings. It has the following keys:
                mix_up_prob: probability that MixUp is applied
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

        # spectrogram_shape = (features_length, 80)
        # spectrogram_shape = (features_length, 40)

        # data = np.zeros((0,) + spectrogram_shape)
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
                spectrogram = provider.get_random_spectrogram("training", features_length, truncation_strategy)
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
        else:
            for provider in self.feature_providers:
                generator = provider.get_feature_generator(mode, features_length, truncation_strategy)
                
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
