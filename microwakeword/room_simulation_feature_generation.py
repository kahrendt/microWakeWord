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

from scipy.signal import convolve

import numpy as np
import tensorflow as tf

import torchaudio



from mmap_ninja.ragged import RaggedMmap
from pathlib import Path
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)

import webrtcvad

import warnings

from microwakeword.feature_generation import generate_features_for_clip

class RoomClipsHandler:
    """RoomClipsHandler 
    """

    def __init__(
        self,
        input_path,
        input_glob,
        impulses_path=None,
        impulses_glob=None,
        playback_background_path=None,
        playback_background_glob=None,
        background_path=None,
        background_glob=None,
        min_clip_duration_s=None,
        max_clip_duration_s=None,
        augmented_duration_s=None,
        max_start_time_from_right_s=None,
        max_jitter_s=None,
        min_jitter_s=None,
        background_probability = 0.75,
        playback_background_probability = 0.75,
    ):
        #######################
        # Setup augmentations #
        #######################

        # If either the the background_paths or impulse_paths are not specified, use an identity transform instead

        self.impulses = [str(i) for i in Path(impulses_path).glob(impulses_glob)]
        self.playback_background_clips = [str(i) for i in Path(playback_background_path).glob(playback_background_glob)]
        self.background_clips = [str(i) for i in Path(background_path).glob(background_glob)]

        self.background_probability = background_probability
        self.playback_background_probability = playback_background_probability

        
        # self.impulse_paths = impulse_paths
        # self.playback_background_paths = playback_background_paths
        # self.background_paths = background_paths

        
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

        self.max_jitter_s = max_jitter_s
        self.min_jitter_s = min_jitter_s
        self.max_start_time_from_right_s = max_start_time_from_right_s
        
        if self.max_jitter_s is not None and self.max_start_time_from_right_s is not None:
            raise ValueError(
                "max_start_time_from_s and max_jitter_s cannot both be configured."
            )
        
        self.augmented_duration_s = augmented_duration_s
        self.repeat_clip_min_duration_s = augmented_duration_s

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
        
        sounds = []
        gains = []
        
        if self.augmented_duration_s is not None:
            sounds.append(self.create_fixed_size_clip(input_audio))
            gains.append(np.random.randint(5,15))
            

        augmentation_probabilities = np.random.rand(2)
        
        if augmentation_probabilities[0] < self.background_probability:
            clip_path = random.choice(self.background_clips)
            
            clip,_ = torchaudio.load(clip_path)
            
            clip = np.squeeze(clip).numpy()
            
            if clip.shape[0] > self.desired_samples:
                fixed_length_clip = self.truncate_clip(clip,16000,self.augmented_duration_s)
            elif clip.shape[0] < self.desired_samples:
                fixed_length_clip = self.repeat_clip(clip)
                fixed_length_clip = fixed_length_clip[:self.desired_samples]
            else:
                fixed_length_clip = clip
                
            sounds.append(fixed_length_clip)
            gains.append(np.random.randint(-5,15))
            
        if augmentation_probabilities[1] < self.playback_background_probability:
            clip_path = random.choice(self.playback_background_clips)
            
            clip,_ = torchaudio.load(clip_path)
            
            clip = np.squeeze(clip).numpy()
            
            if clip.shape[0] > self.desired_samples:
                fixed_length_clip = self.truncate_clip(clip,16000,self.augmented_duration_s)
            elif clip.shape[0] < self.desired_samples:
                fixed_length_clip = self.repeat_clip(clip)
                fixed_length_clip = fixed_length_clip[:self.desired_samples]
            else:
                fixed_length_clip = clip
        
            sounds.append(fixed_length_clip)
            gains.append(np.random.randint(-5,15))
            
        
        room_impulse_path = random.choice(self.impulses)
        rir,_ = torchaudio.load(room_impulse_path)
        reverbed, original_samples = self.apply_impulse(sounds, rir, self.augmented_duration_s, gain=gains)

        mic_0 = (reverbed[:,0]*32767).astype(np.int16)
        if augmentation_probabilities[1] < self.playback_background_probability:
            playback_background = (original_samples[:,-1]*32767).astype(np.int16)
        else:
            playback_background = None

        return mic_0, playback_background

    def apply_impulse(self, sounds, rir, desired_duration,gain=[10,5]):
        desired_samples = int(desired_duration*16000)
        ys = np.zeros((desired_samples,2), dtype=np.float32)
        
        input_samples = np.zeros((desired_samples, len(sounds)), dtype=np.float32)
        
        for source_index in range(0,len(sounds)):
            h0 = rir[source_index*2,:].numpy()
            h1 = rir[source_index*2+1,:].numpy()
            
            audio_samples = np.squeeze(sounds[source_index])
            
            if audio_samples.shape[0] > desired_samples:
                trimmed_start = np.random.randint(0,audio_samples.shape[0]-desired_samples)
                fixed_length_samples = audio_samples[trimmed_start:trimmed_start+desired_samples]
            elif audio_samples.shape[0] < desired_samples:
                fixed_length_samples = np.zeros(desired_samples)
                fixed_length_samples[-audio_samples.shape[0]:] = audio_samples
            else:
                fixed_length_samples = audio_samples

            input_samples[:, source_index] = fixed_length_samples

            mic_0 = convolve(fixed_length_samples, h0)
            mic_1 = convolve(fixed_length_samples, h1)
                
            # Trim the end samples as a result of the convolution
            mic_0 = mic_0[:desired_samples]
            mic_1 = mic_1[:desired_samples]

            E1 = np.sum(mic_0 ** 2)
            E2 = np.sum(mic_1 ** 2)
            E = 0.5 * (E1 + E2)

            mic_0 /= (E ** 0.5 + 1E-10)
            mic_1 /= (E ** 0.5 + 1E-10)
            
            g = 10 ** (gain[source_index]/10.0)
            ys[:,0] += g*mic_0
            ys[:,1] += g*mic_1
            
        return ys, input_samples

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
        augmented_audio, background = self.augment_random_clip()
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
        augmented_audio, background = self.augment_clip(input_audio)
        
        main_clip_features = generate_features_for_clip(augmented_audio)
        if background is not None:
            background_clip_features = generate_features_for_clip(background)
        else:
            background_clip_features = np.zeros(main_clip_features.shape, dtype=np.float32)
            
        concat_features = np.concatenate(main_clip_features, background_clip_features, axis=-1)
        
        return concat_features

    def generate_random_augmented_spectrogram(self):
        """Generates the spectrogram of a random audio clip after augmenting.

        Returns:
            (ndarray): the spectrogram of the augmented audio from a random clip
        """
        augmented_audio, background = self.augment_random_clip()
        
        main_clip_features = generate_features_for_clip(augmented_audio)
        if background is not None:
            background_clip_features = generate_features_for_clip(background)
        else:
            background_clip_features = np.zeros(main_clip_features.shape, dtype=np.float32)
        print(main_clip_features.shape)
        concat_features = np.concatenate([main_clip_features, background_clip_features], axis=-1)
        
        return concat_features

    def augmented_features_generator(self, split=None, repeat=1):
        """Generator function for augmenting all loaded clips and computing their spectrograms

        Args:
            split (string): which data set split to generate features for. One of "train", "test", or "validation", so long as random_split_seed is not None. Set to None in this situation.
            repeat (int): number of times to augment the data set.

        Yields:
            (ndarray): the spectrogram of an augmented audio clip
        """
        if split is None:
            clip_list = self.clips
        else:
            clip_list = self.split_clips[split]
        for _ in range(repeat):
            for clip in clip_list:
                audio = clip["audio"]["array"]

                spectrogram = self.generate_augmented_spectrogram(audio)

                if self.split_spectrogram_duration_s is not None:
                    desired_spectrogram_length = int(self.split_spectrogram_duration_s/0.02) # each window is 20 ms long
                    if spectrogram.shape[0] > desired_spectrogram_length+20:
                        for start_index in range(20, spectrogram.shape[0]-desired_spectrogram_length, desired_spectrogram_length):
                            yield spectrogram[start_index:start_index+desired_spectrogram_length,:]
                    else:
                        yield spectrogram
                elif self.truncate_spectrogram_duration_s is not None:
                    desired_spectrogram_length = int(self.truncate_spectrogram_duration_s/0.02)
                    yield spectrogram[-desired_spectrogram_length:]
                else:
                    yield spectrogram


    def save_augmented_features(self, mmap_output_dir, split=None, repeat=1):
        """Saves all augmented features in a RaggedMmap format

        Args:
            mmap_output_dir (str): Path to saved the RaggedMmap data
        """
        RaggedMmap.from_generator(
            out_dir=mmap_output_dir,
            sample_generator=self.augmented_features_generator(split, repeat),
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

        if len(x) < self.desired_samples:
            min_samples_from_end = len(x)
            if self.max_start_time_from_right_s is not None:
                max_samples_from_end = int(self.max_start_time_from_right_s * sr)
            elif self.max_jitter_s is not None:
                if self.min_jitter_s is not None:
                    min_samples_from_end = len(x) + int(self.min_jitter_s)*sr
                max_samples_from_end = len(x) + int(self.max_jitter_s * sr)
            else:
                max_samples_from_end = self.desired_samples

            assert max_samples_from_end > len(x)

            samples_from_end = np.random.randint(min_samples_from_end, max_samples_from_end) + 1
            # samples_from_end = np.random.randint(len(x), max_samples_from_end) + 1

            dat[-samples_from_end : -samples_from_end + len(x)] = x
        elif len(x) > self.desired_samples:
            samples_from_start = np.random.randint(0, len(x)-self.desired_samples)
            dat = x[samples_from_start:samples_from_start+self.desired_samples]
        else:
            dat = x

        return dat

    def truncate_clip(self, x, sr=16000, duration_s=None):
        desired_samples = int(duration_s * sr)
        if len(x) > desired_samples:
            rn = np.random.randint(0, x.shape[0] - desired_samples)
            x = x[rn:rn + desired_samples]
        return x

    def repeat_clip(self, x, sr=16000):
        original_clip = x
        desired_samples = int(self.repeat_clip_min_duration_s * sr)
        while x.shape[0] < desired_samples:
            x=np.append(x,original_clip)
        return x