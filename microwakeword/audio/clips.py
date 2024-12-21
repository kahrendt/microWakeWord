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

from __future__ import annotations

import audio_metadata
import datasets
import math
import os
import random
import wave

import numpy as np

from pathlib import Path

from microwakeword.audio.audio_utils import remove_silence_webrtc


class Clips:
    """Class for loading audio clips from the specified directory. The clips can first be filtered by their duration using the `min_clip_duration_s` and `max_clip_duration_s` parameters. Clips are retrieved as numpy float arrays via the `get_random_clip` method or via the `audio_generator` or `random_audio_generator` generators. Before retrieval, the audio clip can trim non-voice activiity. Before retrieval, the audio clip can be repeated until it is longer than a specified minimum duration.

    Args:
        input_directory (str): Path to audio clip files.
        file_pattern (str | list[str]): File glob pattern(s) for selecting audio clip files.
        min_clip_duration_s (float | None, optional): The minimum clip duration (in seconds). Set to None to disable filtering by minimum clip duration. Defaults to None.
        max_clip_duration_s (float | None, optional): The maximum clip duration (in seconds). Set to None to disable filtering by maximum clip duration. Defaults to None.
        repeat_clip_min_duration_s (float | None, optional): If a clip is shorter than this duration, then it is repeated until it is longer than this duration. Set to None to disable repeating the clip. Defaults to None.
        remove_silence (bool, optional): Use webrtcvad to trim non-voice activity in the clip. Defaults to False.
        random_splits (dict, optional): Specifies how the clips are split into different sets. Only takes effect if `random_split_seed` is set.
        random_split_seed (int | None, optional): The random seed used to split the clips into different sets. Set to None to disable splitting the clips. Defaults to None.
        trimmed_clip_duration_s: (float | None, optional): The duration of the clips to trim the end of long clips. Set to None to disable trimming. Defaults to None.
        trim_zerios: (bool, optional): If true, any leading and trailling zeros are removed. Defaults to false.
    """

    def __init__(
        self,
        input_directory: str,
        file_pattern: str | list[str],
        min_clip_duration_s: float | None = None,
        max_clip_duration_s: float | None = None,
        repeat_clip_min_duration_s: float | None = None,
        remove_silence: bool = False,
        random_splits: dict[str, float] = {
            "training": 0.8,
            "testing": 0.1,
            "validation": 0.1,
            "testing_ambient": 0,
            "validation_ambient": 0,
        },
        random_split_seed: int | None = None,
        trimmed_clip_duration_s: float | None = None,
        trim_zeros: bool = False,
    ):
        self.trim_zeros = trim_zeros
        self.trimmed_clip_duration_s = trimmed_clip_duration_s

        if min_clip_duration_s is not None:
            self.min_clip_duration_s = min_clip_duration_s
        else:
            self.min_clip_duration_s = 0.0

        if max_clip_duration_s is not None:
            self.max_clip_duration_s = max_clip_duration_s
        else:
            self.max_clip_duration_s = math.inf

        if repeat_clip_min_duration_s is not None:
            self.repeat_clip_min_duration_s = repeat_clip_min_duration_s
        else:
            self.repeat_clip_min_duration_s = 0.0

        self.remove_silence = remove_silence
        self.remove_silence_function = remove_silence_webrtc

        self.input_directory = input_directory

        if isinstance(file_pattern, str):
            file_pattern = [file_pattern]

        paths_to_clips = []

        for pattern in file_pattern:
            paths_to_clips.extend([str(i) for i in Path(input_directory).glob(pattern)])

        if (self.min_clip_duration_s == 0) and (math.isinf(self.max_clip_duration_s)):
            # No durations specified, so do not filter by length
            filtered_paths = paths_to_clips
        else:
            # Filter audio clips by length
            if file_pattern[0].endswith("wav"):
                # If it is a wave file, assume all wave files have the same parameters and filter by file size.
                # Based on openWakeWord's estimate_clip_duration and filter_audio_paths in data.py, accessed March 2, 2024.
                with wave.open(paths_to_clips[0], "rb") as input_wav:
                    channels = input_wav.getnchannels()
                    sample_width = input_wav.getsampwidth()
                    sample_rate = input_wav.getframerate()
                    frames = input_wav.getnframes()

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
                    if (self.min_clip_duration_s < duration)
                    and (duration < self.max_clip_duration_s)
                ]
            else:
                # If not a wave file, use the audio_metadata package to analyze audio file headers for the duration.
                # This is slower!
                filtered_paths = []

                if (self.min_clip_duration_s > 0) or (
                    not math.isinf(self.max_clip_duration_s)
                ):
                    for audio_file in paths_to_clips:
                        metadata = audio_metadata.load(audio_file)
                        duration = metadata["streaminfo"]["duration"]
                        if (self.min_clip_duration_s < duration) and (
                            duration < self.max_clip_duration_s
                        ):
                            filtered_paths.append(audio_file)

        # Load all filtered clips
        audio_dataset = datasets.Dataset.from_dict(
            {"audio": [str(i) for i in filtered_paths]}
        ).cast_column("audio", datasets.Audio())

        # Convert all clips to 16 kHz sampling rate when accessed
        audio_dataset = audio_dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=16000)
        )

        dataset_splits = {
            "training": [],
            "testing": [],
            "validation": [],
            "testing_ambient": [],
            "validation_ambient": [],
        }

        assigned_splits = [(k, v) for k, v in random_splits.items() if v > 0]
        assert abs(sum(dict(assigned_splits).values()) - 1.0) < 1e-6

        self.single_split = None

        if len(assigned_splits) == 1:
            # With a single class, we don't split
            self.single_split = assigned_splits[0]
            dataset_splits[self.single_split] = audio_dataset
        elif random_split_seed is None:
            raise ValueError("Random split seed must be set to split the dataset")

        if len(assigned_splits) == 2:
            # With two classes, it's simple
            split1, split2 = assigned_splits
            split_dataset = audio_dataset.train_test_split(
                train_size=split1[1], seed=random_split_seed
            )
            dataset_splits[split1[0]] = split_dataset["train"]
            dataset_splits[split2[0]] = split_dataset["test"]
        elif len(assigned_splits) == 3:
            # Three classes requires two splits
            split1, split2, split3 = assigned_splits
            split_dataset1 = audio_dataset.train_test_split(
                train_size=split1[1] + split2[1], seed=random_split_seed
            )
            split_dataset2 = split_dataset1["train"].train_test_split(
                train_size=split1[1], seed=random_split_seed
            )
            dataset_splits[split3[0]] = split_dataset1["test"]
            dataset_splits[split1[0]] = split_dataset2["train"]
            dataset_splits[split2[0]] = split_dataset2["test"]
        else:
            raise ValueError(f"Only up to three dataset splits are supported: {assigned_splits}")

        self.split_clips = datasets.DatasetDict(dataset_splits)

    def _process_clip(self, clip_audio):
        if self.remove_silence:
            clip_audio = self.remove_silence_function(clip_audio)

        if self.trim_zeros:
            clip_audio = np.trim_zeros(clip_audio)

        if self.trimmed_clip_duration_s:
            total_samples = int(self.trimmed_clip_duration_s * 16000)
            clip_audio = clip_audio[:total_samples]

        return self.repeat_clip(clip_audio)

    def _get_clips_from_split(self, split: str | None = None):
        if split is None:
            if self.single_split is None:
                raise ValueError("`split` must be provided for multi-class Clips")

            split = self.single_split

        return self.split_clips[split]

    def audio_generator(self, split: str | None = None, repeat: int = 1):
        """A Python generator that retrieves all loaded audio clips.

        Args:
            split (str | None, optional): Specifies which set the clips are retrieved from. If None, all clips are retrieved. Otherwise, it can be set to `train`, `test`, or `validation`. Defaults to None.
            repeat (int, optional): The number of times each audio clip will be yielded. Defaults to 1.

        Yields:
            numpy.ndarray: Array with the audio clip's samples.
        """
        clip_list = self._get_clips_from_split(split)

        for _ in range(repeat):
            for clip in clip_list:
                yield self._process_clip(clip["audio"]["array"])

    def get_random_clip(self, split: str | None = None):
        """Retrieves a random audio clip.

        Returns:
            numpy.ndarray: Array with the audio clip's samples.
        """
        clip_list = self._get_clips_from_split(split)
        rand_audio_entry = random.choice(clip_list)

        return self._process_clip(rand_audio_entry["audio"]["array"])

    def random_audio_generator(self, split: str | None = None, max_clips: int = math.inf):
        """A Python generator that retrieves random audio clips.

        Args:
            max_clips (int, optional): The total number of clips the generator will yield before the StopIteration. Defaults to math.inf.

        Yields:
            numpy.ndarray: Array with the random audio clip's samples.
        """
        clip_list = self._get_clips_from_split(split)

        while max_clips > 0:
            max_clips -= 1

            # TODO: Sampling with replacement isn't good for small datasets
            yield self.get_random_clip(split=split)

    def repeat_clip(self, audio_samples: np.array):
        """Repeats the audio clip until its duration exceeds the minimum specified in the class.

        Args:
            audio_samples numpy.ndarray: Original audio clip's samples.

        Returns:
            numpy.ndarray: Array with duration exceeding self.repeat_clip_min_duration_s.
        """
        original_clip = audio_samples
        desired_samples = int(self.repeat_clip_min_duration_s * 16000)
        while audio_samples.shape[0] < desired_samples:
            audio_samples = np.append(audio_samples, original_clip)
        return audio_samples
