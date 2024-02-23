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

"""Functions to augment audio clips"""

import audiomentations
import numpy as np
import os
import wave

from mmap_ninja.ragged import RaggedMmap
from pathlib import Path

from microwakeword.feature_generation import generate_features_for_clip


def create_fixed_size_clip(x, n_samples, sr=16000, start=None, end_jitter=0.100):
    """
    Create a fixed-length clip of the specified size by padding an input clip with zeros
    Optionally specify the start/end position of the input clip, or let it be chosen randomly.

    Borrowed from openWakeWord's data.py, accessed on February 23, 2024

    Args:
        x (ndarray): The input audio to pad to a fixed size
        n_samples (int): The total number of samples for the fixed length clip
        sr (int): The sample rate of the audio
        start (int): The start position of the clip in the fixed length output, in samples (default: None)
        end_jitter (float): The time (in seconds) from the end of the fixed length output
                            that the input clip should end, if `start` is None.

    Returns:
        ndarray: A new array of audio data of the specified length
    """
    dat = np.zeros(n_samples)
    end_jitter = int(np.random.uniform(0, end_jitter) * sr)
    if start is None:
        start = max(0, n_samples - (int(len(x)) + end_jitter))

    if len(x) > n_samples:
        if np.random.random() >= 0.5:
            dat = x[0:n_samples].numpy()
        else:
            dat = x[-n_samples:].numpy()
    else:
        dat[start : start + len(x)] = x

    return dat


def augment_clips_generator(
    input_path,
    impulses_path,
    background_path,
    augmentation_probabilities: dict = {
        "SevenBandParametricEQ": 0.25,
        "TanhDistortion": 0.25,
        "PitchShift": 0.25,
        "BandStopFilter": 0.25,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    max_end_jitter_s=0.1,
    augmented_duration_s=None,
    max_clip_duration_s=None,
):
    """
    Generator function that augments audio data (16 khz, 16-bit PCM audio data).

    Default augmentation settings and probabilities are borrowed from
    openWakeWord's data.py, accessed on February 23, 2024.

    Args:
        input_path (string): The path to audio files to be augmented (glob is **/*.wav)
        impulses_path (string): The path to room impulse response files (glob is **/*.wav)
        background_path (string): The path to background audio files (glob is **/*.wav)
        augmentation_probabilities (dict): The individual probabilities of each augmentation. If all probabilities
                                           are zero, the input audio files will simply be padded with silence. THe
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
        max_end_jitter_s (float): The maximum time (in seconds) to pad files on the right.
        augmented_duration_s (float): The final duration (in seconds) of the augmented file.
                                      It will be padded on the right randomly up to max_end_jitter_s,
                                      and the remaining time padded on the left.
        max_cli_duration_s (float): The maximum clip duration (in seconds) of the input audio clips.


    Yields:
        ndarray: An array containing the augmented audio
    """

    input_path = Path(input_path)
    impulses = list((Path(impulses_path)).glob("**/*.wav"))
    backgrounds = list((Path(background_path)).glob("**/*.wav"))

    # Augmentation settings are borrow from openWakeWord's data.py, accessed on February 23, 2024
    augment = audiomentations.Compose(
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
            audiomentations.AddBackgroundNoise(
                p=augmentation_probabilities["AddBackgroundNoise"],
                sounds_path=backgrounds,
                min_snr_in_db=-10,
                max_snr_in_db=15,
            ),
            audiomentations.Gain(
                p=augmentation_probabilities["Gain"],
                min_gain_in_db=-12,
                max_gain_in_db=0,
            ),
            audiomentations.ApplyImpulseResponse(
                p=augmentation_probabilities["RIR"],
                ir_path=impulses,
            ),
        ]
    )

    for input_wav in input_path.glob("**/*"):
        # wav file reading is borrowed from Piper Sample Generator's augment.py accessed on February 23, 2024
        with wave.open(str(input_wav), "rb") as input_wav_file:
            assert input_wav_file.getsampwidth() == 2
            assert input_wav_file.getnchannels() == 1

            input_audio = (
                np.frombuffer(
                    input_wav_file.readframes(input_wav_file.getnframes()),
                    dtype=np.int16,
                ).astype(np.float32)
                / 32767.0
            )

            if max_clip_duration_s is not None:
                # Skip augmenting clip if longer than the specified max clip duration
                max_samples = int(max_clip_duration_s * 16000)
                if input_audio.shape[0] > max_samples:
                    continue

            if augmented_duration_s is not None:
                # Make augmented clip have fixed duration by padding at the end by a random length up to max_end_jitter_s and the rest padded at the start
                desired_samples = int(
                    augmented_duration_s * 16000
                )  # Assumes 16000 Hz audio
                input_audio = create_fixed_size_clip(
                    input_audio, desired_samples, end_jitter=max_end_jitter_s
                )

            output_audio = augment(
                input_audio, sample_rate=input_wav_file.getframerate()
            )

            yield (output_audio * 32767).astype(np.int16)


def generate_augmented_clips(
    clips_output_dir,
    input_path,
    impulses_path,
    background_path,
    augmentation_probabilities: dict = {
        "SevenBandParametricEQ": 0.25,
        "TanhDistortion": 0.25,
        "PitchShift": 0.25,
        "BandStopFilter": 0.25,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    augmented_duration_s=5,
    max_end_jitter_s=0.1,
    max_clip_duration_s=1.39,
):
    """
    Augments input audio data (16 khz, 16-bit PCM audio data) and saves as wave files.

    Default augmentation settings and probabilities are borrowed from
    openWakeWord's data.py, accessed on February 23, 2024.

    Args:
        clips_output_dir (string): The path to save the augmented audio files.
        See `augment_clips_generator` function for description of other options.
    """
    audio_generator = augment_clips_generator(
        input_path=input_path,
        impulses_path=impulses_path,
        background_path=background_path,
        augmentation_probabilities=augmentation_probabilities,
        augmented_duration_s=augmented_duration_s,
        max_end_jitter_s=max_end_jitter_s,
        max_clip_duration_s=max_clip_duration_s,
    )

    for counter, augmented_audio in enumerate(audio_generator):
        output_path = os.path.join(clips_output_dir, str(counter) + ".wav")
        # wav file saving is borrowed from piper sample generator's augment.py accessed on February 23, 2024
        with wave.open(output_path, "wb") as output_wav_file:
            output_wav_file.setframerate(16000)
            output_wav_file.setsampwidth(2)
            output_wav_file.setnchannels(1)
            output_wav_file.writeframes(augmented_audio)


def generate_augmented_features(
    mmap_output_dir,
    input_path,
    impulses_path,
    background_path,
    augmentation_probabilities: dict = {
        "SevenBandParametricEQ": 0.25,
        "TanhDistortion": 0.25,
        "PitchShift": 0.25,
        "BandStopFilter": 0.25,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    augmented_duration_s=5,
    max_end_jitter_s=0.1,
    max_clip_duration_s=1.39,
):
    """
    Augments input audio data (16 khz, 16-bit PCM audio data) and computes TFLM
    spectrogram features saved as a ragged mmap.

    Args:
        mmap_output_dir (string): The directory save the ragged mmap containing features.
        See `augment_clips_generator` function for description of other options.
    """

    def features_generator():
        for audio_data in augment_clips_generator(
            input_path=input_path,
            impulses_path=impulses_path,
            background_path=background_path,
            augmentation_probabilities=augmentation_probabilities,
            augmented_duration_s=augmented_duration_s,
            max_end_jitter_s=max_end_jitter_s,
            max_clip_duration_s=max_clip_duration_s,
        ):
            yield generate_features_for_clip(audio_data)

    RaggedMmap.from_generator(
        out_dir=mmap_output_dir,
        sample_generator=features_generator(),
        batch_size=1024,
        verbose=True,
    )
