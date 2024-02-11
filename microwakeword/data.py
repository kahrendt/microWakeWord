import copy
import os
import random

import numpy as np

from absl import logging
from pathlib import Path
from mmap_ninja.ragged import RaggedMmap


def mixup_augment(
    spectrogram_1, truth_1, weight_1, spectrogram_2, truth_2, weight_2, mix_ratio
):
    """Applies MixUp augment to the input spectrograms.
    Based on mixup: BEYOND EMPIRICAL RISK MINIMIZATION by H. Zhang, M. Cisse, Y. Dauphin, D. Lopez-Paz
    https://openreview.net/pdf?id=r1Ddp1-Rb
    
    Args:
        spectrogram_1: the first spectrogram.
        truth_1: the ground truth of the first spectrogram.
        weight_1: the penalty weight of the first spectrogram.
        spectrogram_2: the second spectrogram.
        truth_2: the ground truth of the second spectrogram.
        weight_2: the penalty weight of the second spectrogram.

    Returns:
        spectrogram: the blended spectrogram.
        truth: the blended ground truth.
        weight: the blended penalty weight.
    """  
    
    combined_spectrogram = spectrogram_1 * mix_ratio + spectrogram_2 * (1 - mix_ratio)
    combined_truth = float(truth_1) * mix_ratio + float(truth_2) * (1 - mix_ratio)
    combined_weight = weight_1 * mix_ratio + weight_2 * (1 - mix_ratio)

    return combined_spectrogram, combined_truth, combined_weight


def freqmix_augment(
    spectrogram_1, truth_1, weight_1, spectrogram_2, truth_2, weight_2, mix_ratio
):
    """Applies FreqMix augment to the input spectrograms.
    Based on END-TO-END AUDIO STRIKES BACK: BOOSTING AUGMENTATIONS TOWARDS AN EFFICIENT AUDIO CLASSIFICATION NETWORK by A. Gazneli, G. Zimerman, T. Ridnik, G. Sharir, A. Noy
    https://arxiv.org/pdf/2204.11479v5.pdf
    
    Args:
        spectrogram_1: the first spectrogram.
        truth_1: the ground truth of the first spectrogram.
        weight_1: the penalty weight of the first spectrogram.
        spectrogram_2: the second spectrogram.
        truth_2: the ground truth of the second spectrogram.
        weight_2: the penalty weight of the second spectrogram.

    Returns:
        spectrogram: the blended spectrogram.
        truth: the blended ground truth.
        weight: the blended penalty weight.
    """  

    freq_bin_cutoff = int(mix_ratio * 40)

    combined_spectrogram = np.concatenate(
        (spectrogram_1[:, :freq_bin_cutoff], spectrogram_2[:, freq_bin_cutoff:]), axis=1
    )
    combined_truth = float(truth_1) * (freq_bin_cutoff / 40.0) + float(truth_2) * (
        1 - freq_bin_cutoff / 40.0
    )
    combined_weight = weight_1 * (freq_bin_cutoff / 40.0) + weight_2 * (
        1 - freq_bin_cutoff / 40.0
    )

    return combined_spectrogram, combined_truth, combined_weight


def spec_augment(
    spectrogram,
    time_mask_max_size=0,
    time_mask_count=1,
    freq_mask_max_size=0,
    freq_mask_count=1,
):
    """Applies SpecAugment to the input spectrogram.
    Based on SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition by D. Park, W. Chan, Y. Zhang, C. Chiu, B. Zoph, E Cubuk, Q Le
    https://arxiv.org/pdf/1904.08779.pdf
    Implementation based on https://github.com/pyyush/SpecAugment/tree/master
    
    Args:
        spectrogram: the input spectrogram.
        time_mask_max_size: maximum size of time feature masks.
        time_mask_count: the total number of separate time masks.
        freq_mask_max_size: maximum size of frequency feature masks.
        time_mask_count: the total number of separate feature masks.

    Returns:
        The masked spectrogram.
    """

    freq_bins = spectrogram.shape[0]
    time_frames = spectrogram.shape[1]

    for i in range(freq_mask_count):
        f = int(np.random.uniform(0, freq_mask_max_size))
        f0 = random.randint(0, freq_bins - f)
        spectrogram[f0 : f0 + f, :] = 0

    for i in range(time_mask_count):
        t = int(np.random.uniform(0, time_mask_max_size))
        t0 = random.randint(0, time_frames - t)
        spectrogram[:, t0 : t0 + t] = 0

    return spectrogram


class FeatureHandler(object):
    """Class that handles loading spectrogram features and providing them to the training and testing functions.

    Args:
      config: dictionary containing microWakeWord training configuration.
    """
    def __init__(
        self,
        config,
    ):
        self.features = []

        logging.info("Loading and analyzing data sets.")

        features = copy.deepcopy(config["features"])

        for feature_set in features:
            feature_set["testing"] = []
            feature_set["training"] = []
            feature_set["validation"] = []
            feature_set["validation_ambient"] = []
            feature_set["testing_ambient"] = []
            feature_set["loaded_features"] = []
            feature_set["stats"] = {}

            self.prepare_data(feature_set)
            self.features.append(feature_set)

        modes = [
            "training",
            "validation",
            "validation_ambient",
            "testing",
            "testing_ambient",
        ]

        for mode in modes:
            logging.info(
                "%s mode has %d spectrograms representing %.1f hours of audio",
                *(mode, self.get_mode_size(mode), self.get_mode_duration(mode) / 3600.0)
            )

    def prepare_data(self, feature_dict):
        """Loads data from a feature's config entry.
        
        Args:
            feature_dict: dictionary with keys for:
                features_dir: directory containing diffferent mode folders.
                sampling_weight: weight for choosing a spectrogram from this set in the batch.
                penalty_weight: penalizing weight for incorrect predictions from this set.
                truth: boolean representing whether this set has positive samples or negative samples.
                truncation_strategy: if a spectrogram is longer than necessary for training, how is it truncated
        """
        data_dir = feature_dict["features_dir"]

        if not os.path.exists(data_dir):
            print("ERROR:" + str(data_dir) + "directory doesn't exist")

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

            search_path_directory = os.path.join(data_dir, set_index)
            search_path = [
                str(i)
                for i in Path(os.path.abspath(search_path_directory)).glob("**/*_mmap/")
            ]

            for mmap_path in search_path:
                imported_features = RaggedMmap(mmap_path)

                feature_dict["loaded_features"].append(imported_features)
                feature_index = len(feature_dict["loaded_features"]) - 1

                for i in range(0, len(imported_features)):
                    feature_dict[set_index].append(
                        {
                            "loaded_feature_index": feature_index,
                            "subindex": i,
                        }
                    )

                    duration += 0.02 * imported_features[i].shape[0]    # Each feature represents 0.02 seconds of audio
                    count += 1

            random.shuffle(feature_dict[set_index])

            feature_dict["stats"][set_index] = {
                "spectrogram_count": count,
                "total_duration": duration,
            }

    def get_mode_duration(self, mode):
        """Returns the durations of all spectrogram features in the given mode.
        
        Args:
            mode: which training set to compute duration over. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`.

        Returns:
            Duration, in seconds, of all spectrograms in this mode 
        """
        
        sample_duration = 0
        for feature_set in self.features:
            # sample_count += len(feature_set[mode])
            sample_duration += feature_set["stats"][mode]["total_duration"]
        return sample_duration

    def get_mode_size(self, mode):
        """Returns the count of all spectrogram features in the given mode.
        
        Args:
            mode: which training set to count the spectrograms. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`.

        Returns:
            Count of spectrograms in given mode.
        """        
        sample_count = 0
        for feature_set in self.features:
            sample_count += feature_set["stats"][mode]["spectrogram_count"]
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
            "time_mask_max_size": 10,
            "time_mask_count": 1,
            "freq_mask_max_size": 1,
            "freq_mask_count": 3,
        },
    ):
        """Gets spectrograms from the appropriate mode. Ensures spectrograms are the approriate length and optionally applies augmentation.
        
        Args:
            mode: which training set to count the spectrograms. One of `training`, `testing`, `testing_ambient`, `validation`, or `validation_ambient`.
            batch_size: number of spectrograms in the sample for training mode.
            features_length: the length of the spectrograms.
            truncation_strategy: how to truncate spectrograms longer than `features_length`
            augmentation_policy: dictionary that specifies augmentation settings. It has the following keys:
                mix_up_prob: probability that MixUp is applied.
                freq_mix_prob: probability that FreqMix is applied.
                time_mask_max_size: maximum size of time feature masks for SpecAugment.
                time_mask_count: the total number of separate time masks applied for SpecAugment.
                freq_mask_max_size: maximum size of frequency feature masks for SpecAugment.
                time_mask_count: the total number of separate feature masks applied for SpecAugment.

        Returns:
            data: spectrograms in a NumPy array (or as a list if in mode is `*_ambient`).
            labels: ground truth for the spectrograms; i.e., whether a positive sample or negative sample.
            weights: penalizing weight for incorrect predictions for each spectrogram.
        """     
        if mode == "training":
            sample_count = batch_size
        elif (mode == "validation") or (mode == "testing"):
            sample_count = self.get_mode_size(mode)
        else:
            sample_count = 0
            for feature_set in self.features:
                features_in_this_mode = feature_set[mode]

                if len(features_in_this_mode) > 0:
                    if truncation_strategy != "none":
                        for features in features_in_this_mode:
                            spectrogram = feature_set["loaded_features"][
                                features["loaded_feature_index"]
                            ][features["subindex"]]
                            sample_count += (
                                spectrogram.shape[0] - features_length
                            ) // 10 + 1
                    else:
                        sample_count += len(
                            features_in_this_mode
                        )

        spectrogram_shape = (features_length, 40)

        data = np.zeros((sample_count,) + spectrogram_shape)
        labels = np.full(sample_count, 0.0)
        weights = np.ones(sample_count)

        if mode.endswith("ambient") and truncation_strategy == "none":
            # Use a list instead of a numpy array; ambient set's spectrograms may have different lengths
            data = []

        if mode == "training":
            random_feature_sets = random.choices(
                [
                    feature_set
                    for feature_set in self.features
                    if len(feature_set["training"])
                ],
                [
                    feature_set["sampling_weight"]
                    for feature_set in self.features
                    if len(feature_set["training"])
                ],
                k=sample_count,
            )
            random_feature_sets2 = random.choices(
                [
                    feature_set
                    for feature_set in self.features
                    if len(feature_set["training"])
                ],
                [
                    feature_set["sampling_weight"]
                    for feature_set in self.features
                    if len(feature_set["training"])
                ],
                k=sample_count,
            )

            for i in range(sample_count):
                feature_set_1 = random_feature_sets[i]
                feature_set_2 = random_feature_sets2[i]

                feature_1 = random.choice(feature_set_1["training"])
                feature_2 = random.choice(feature_set_2["training"])

                spectrogram_1 = feature_set_1["loaded_features"][
                    feature_1["loaded_feature_index"]
                ][feature_1["subindex"]]
                spectrogram_2 = feature_set_2["loaded_features"][
                    feature_2["loaded_feature_index"]
                ][feature_2["subindex"]]

                if truncation_strategy == "default":
                    truncation_strategy_1 = feature_set_1["truncation_strategy"]
                    truncation_strategy_2 = feature_set_2["truncation_strategy"]
                else:
                    truncation_strategy_1 = truncation_strategy
                    truncation_strategy_2 = truncation_strategy

                spectrogram_1 = self.fixed_length_spectrogram(
                    spectrogram_1,
                    features_length,
                    truncation_strategy=truncation_strategy_1,
                )
                spectrogram_2 = self.fixed_length_spectrogram(
                    spectrogram_2,
                    features_length,
                    truncation_strategy=truncation_strategy_2,
                )

                data[i] = spectrogram_1
                labels[i] = float(feature_set_1["truth"])
                weights[i] = float(feature_set_1["penalty_weight"])

                if (
                    np.random.rand()
                    < augmentation_policy["mix_up_prob"]
                    + augmentation_policy["freq_mix_prob"]
                ):
                    mix_ratio = np.random.rand()

                    which_augment = random.choices(
                        [0, 1],
                        [
                            augmentation_policy["mix_up_prob"],
                            augmentation_policy["freq_mix_prob"],
                        ],
                        k=1,
                    )

                    if which_augment[0] == 0:
                        data[i], labels[i], weights[i] = mixup_augment(
                            spectrogram_1,
                            feature_set_1["truth"],
                            feature_set_1["penalty_weight"],
                            spectrogram_2,
                            feature_set_2["truth"],
                            feature_set_2["penalty_weight"],
                            mix_ratio,
                        )
                    else:
                        data[i], labels[i], weights[i] = freqmix_augment(
                            spectrogram_1,
                            feature_set_1["truth"],
                            feature_set_1["penalty_weight"],
                            spectrogram_2,
                            feature_set_2["truth"],
                            feature_set_2["penalty_weight"],
                            mix_ratio,
                        )

                data[i] = spec_augment(
                    data[i],
                    augmentation_policy["time_mask_max_size"],
                    augmentation_policy["time_mask_count"],
                    augmentation_policy["freq_mask_max_size"],
                    augmentation_policy["freq_mask_count"],
                )
        elif (mode == "validation") or (mode == "testing"):
            index = 0
            for feature_set in self.features:
                for feature_index in feature_set[mode]:
                    spectrogram = feature_set["loaded_features"][
                        feature_index["loaded_feature_index"]
                    ][feature_index["subindex"]]

                    if truncation_strategy == "default":
                        truncation_strategy = feature_set["truncation_strategy"]

                    data[index] = self.fixed_length_spectrogram(
                        spectrogram,
                        features_length,
                        truncation_strategy=truncation_strategy,
                    )
                    labels[index] = feature_set["truth"]
                    weights[index] = feature_set["penalty_weight"]

                    index += 1

            # Randomize the order of the testing and validation sets
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)

            data = data[indices]
            labels = labels[indices]
            weights = weights[indices]
        else:
            # ambient testing, split the long spectrograms into overlapping chunks
            index = 0
            for feature_set in self.features:
                features_in_this_mode = feature_set[mode]

                if len(features_in_this_mode) > 0:
                    for features in features_in_this_mode:
                        spectrogram = feature_set["loaded_features"][
                            features["loaded_feature_index"]
                        ][features["subindex"]]

                        if truncation_strategy == "default":
                            truncation_strategy = feature_set["truncation_strategy"]

                        if truncation_strategy == "split":
                            for subset_feature_index in range(
                                0, spectrogram.shape[0], 10
                            ):
                                if (
                                    subset_feature_index + features_length
                                    < spectrogram.shape[0]
                                ):
                                    data[index] = spectrogram[
                                        subset_feature_index : subset_feature_index
                                        + features_length
                                    ]
                                    labels[index] = 0.0
                                    weights[index] = 1.0
                                    index += 1
                        else:
                            data.append(spectrogram)
                            labels[index] = 0.0
                            weights[index] = 1.0

        return data, labels, weights

    def fixed_length_spectrogram(
        self, spectrogram, features_length, truncation_strategy="random"
    ):
        """Returns a spectrogram with specified length. Pads with zeros at the start if too short.
        
        Args:
            spectrogram: the spectrogram to truncate or pad.
            features_length: the desired spectrogram length.
            truncation_strategy: how to truncate if ``spectrogram`` is longer than ``features_length``. One of:
                random: choose a random portion of the entire spectrogram - useful for long negative samples
                truncate_start: remove the start of the spectrogram
                truncate_end: remove the end of the spectrogram
                none: returns the entire spectrogram regardless of features_length


        Returns:
            The fixed length spectrogram after truncating or padding.
        """  
        data_length = spectrogram.shape[0]
        if data_length > features_length:
            if truncation_strategy == "random":
                features_offset = np.random.randint(0, data_length - features_length)
            elif truncation_strategy == "none":
                # return the entire spectrogram
                features_offset = 0
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
