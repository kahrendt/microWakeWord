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

import numpy as np

from typing import Optional

from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips


class SpectrogramGeneration:
    """A class that handles generating spectrogram features for audio clips. Spectrograms can optionally be split into nonoverlapping segments for faster file loading or they can optionally be strided by dropping the last feature windows to simulate a streaming model's sequential inputs.

    Args:
        clips (Clips): Object that retrieves audio clips.
        augmenter (Augmentation | None, optional): Object that augments audio clips. If None, no augmentations are applied. Defaults to None.
        step_ms (int, optional): The window step size in ms for the spectrogram features. Defaults to 20.
        split_spectrogram_duration_s (float | None, optional): Splits generated spectrograms to yield nonoverlapping spectrograms with this duration. If None, the entire spectrogram is yielded. Defaults to None.
        slide_frames (int | None, optional): Strides the generated spectrograms to yield `slide_frames` overlapping spectrogram by removing features at the end of the spectrogram. If None, the entire spectrogram is yielded. Defaults to None.
    """

    def __init__(
        self,
        clips: Clips,
        augmenter: Augmentation | None = None,
        step_ms: int = 20,
        split_spectrogram_duration_s: float | None = None,
        slide_frames: int | None = None,
        **kwargs,
    ):

        self.clips = clips
        self.augmenter = augmenter
        self.step_ms = step_ms
        self.split_spectrogram_duration_s = split_spectrogram_duration_s
        self.slide_frames = slide_frames

    def get_random_spectrogram(self):
        """Retrieves a random audio clip's spectrogram that is optionally augmented.

        Returns:
            numpy.ndarry: 2D spectrogram array for the random (augmented) audio clip.
        """
        clip = self.clips.get_random_clip()
        if self.augmenter is not None:
            clip = self.augmenter.augment_clip(clip)

        return generate_features_for_clip(clip, self.step_ms)

    def spectrogram_generator(self, random=False, **kwargs):
        """A Python generator that retrieves (augmented) spectrograms.

        Args:
            random (bool, optional): Specifies if the source audio clips should be chosen randomly. Defaults to False.
            kwargs: Parameters to pass to the clips audio generator.

        Yields:
            numpy.ndarry: 2D spectrogram array for the random (augmented) audio clip.
        """
        if random:
            clip_generator = self.clips.random_audio_generator()
        else:
            clip_generator = self.clips.audio_generator(**kwargs)

        if self.augmenter is not None:
            augmented_generator = self.augmenter.augment_generator(clip_generator)
        else:
            augmented_generator = clip_generator

        for augmented_clip in augmented_generator:
            spectrogram = generate_features_for_clip(augmented_clip, self.step_ms)

            if self.split_spectrogram_duration_s is not None:
                # Splits the resulting spectrogram into non-overlapping spectrograms. The features from the first 20 feature windows are dropped.
                desired_spectrogram_length = int(
                    self.split_spectrogram_duration_s / (self.step_ms / 1000)
                )

                if spectrogram.shape[0] > desired_spectrogram_length + 20:
                    slided_spectrograms = np.lib.stride_tricks.sliding_window_view(
                        spectrogram,
                        window_shape=(desired_spectrogram_length, spectrogram.shape[1]),
                    )[20::desired_spectrogram_length, ...]

                    for i in range(slided_spectrograms.shape[0]):
                        yield np.squeeze(slided_spectrograms[i])
                else:
                    yield spectrogram
            elif self.slide_frames is not None:
                # Generates self.slide_frames spectrograms by shifting over the already generated spectrogram
                spectrogram_length = spectrogram.shape[0] - self.slide_frames + 1

                slided_spectrograms = np.lib.stride_tricks.sliding_window_view(
                    spectrogram, window_shape=(spectrogram_length, spectrogram.shape[1])
                )
                for i in range(self.slide_frames):
                    yield np.squeeze(slided_spectrograms[i])
            else:
                yield spectrogram
