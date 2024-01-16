import os
import random

import numpy as np

from pathlib import Path
from mmap_ninja.ragged import RaggedMmap


class FeatureHandler(object):
    def __init__(
        self,
        config,
    ):
        self.features = []

        for feature_set in config["features"]:
            feature_set["testing"] = []
            feature_set["training"] = []
            feature_set["validation"] = []
            feature_set["loaded_features"] = []

            self.prepare_data(feature_set)
            self.features.append(feature_set)

    def prepare_data(self, feature_dict):
        data_dir = feature_dict["features_dir"]

        if not os.path.exists(data_dir):
            print("ERROR:" + str(data_dir) + "directory doesn't exist")

        dirs = ["testing", "training", "validation"]

        for set_index in dirs:
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

            random.shuffle(feature_dict[set_index])

    def get_data(self, mode, batch_size, features_length, truncation_strategy="random"):
        if mode == "training":
            sample_count = batch_size
        else:
            sample_count = 0
            for feature_set in self.features:
                sample_count += len(feature_set[mode])

        spectrogram_shape = (features_length, 40)
        data = np.zeros((sample_count,) + spectrogram_shape)
        labels = np.full(sample_count, False)
        weights = np.ones(sample_count)

        if mode == "training":
            random_feature_sets = random.choices(
                self.features,
                [feature_set["sampling_weight"] for feature_set in self.features],
                k=sample_count,
            )
            for i in range(sample_count):
                feature_set = random_feature_sets[i]
                random_feature = random.choice(feature_set["training"])

                spectrogram = feature_set["loaded_features"][
                    random_feature["loaded_feature_index"]
                ][random_feature["subindex"]]

                data[i] = self.truncate_spectrogram(
                    spectrogram,
                    features_length,
                    truncation_strategy=truncation_strategy,
                )
                labels[i] = feature_set["truth"]
                weights[i] = feature_set["penalty_weight"]
        else:
            index = 0
            for feature_set in self.features:
                for feature_index in feature_set[mode]:
                    spectrogram = feature_set["loaded_features"][
                        feature_index["loaded_feature_index"]
                    ][feature_index["subindex"]]

                    data[index] = self.truncate_spectrogram(
                        spectrogram,
                        features_length,
                        truncation_strategy=truncation_strategy,
                    )
                    labels[index] = feature_set["truth"]
                    weights[index] = feature_set["penalty_weight"]

                    index += 1

        return data, labels, weights

    def truncate_spectrogram(
        self, spectrogram, features_length, truncation_strategy="random"
    ):
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
            features_offset = 0

        return spectrogram[features_offset : (features_offset + features_length)]
