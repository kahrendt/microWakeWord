import audiomentations
import audio_metadata
import datasets
import math
import os
import random
import warnings
import wave

import numpy as np

import tensorflow as tf

from pathlib import Path
from scipy.io import wavfile
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)

def save_clip(audio, output_file):
    wavfile.write(output_file, 16000, audio)

def generate_features_for_clip(audio, desired_spectrogram_length=None):
    """Generates spectrogram features for the given audio data.

    Args:
        clip (ndarray): audio data with sample rate 16 kHz and 16-bit samples
        desired_spectrogram_length (int, optional): Number of time features to include in the spectrogram.
                                                    Truncates earlier time features. Set to None to disable.

    Returns:
        (ndarray): spectrogram audio features
    """
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)
    
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


class Augmentation:
    def __init__(
        self,
        augmentation_probabilities: dict = {
            "SevenBandParametricEQ": 0.0,
            "TanhDistortion": 0.0,
            "PitchShift": 0.0,
            "BandStopFilter": 0.0,
            "AddColorNoise": 0.25,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },        
        impulse_paths=None,
        background_paths=None,
        background_min_snr_db=-10,
        background_max_snr_db=10,
        min_jitter_s=None,
        max_jitter_s=None,
        max_start_time_from_right_s=None,
        augmentation_duration_s=None,     
    ):
        ###
        # Configure audio duration and positioning #
        ###
        
        self.min_jitter_samples = 0
        self.max_jitter_samples = 0
        if min_jitter_s is not None:
            self.min_jitter_samples = int(min_jitter_s*16000)
        if max_jitter_s is not None:
            self.max_jitter_samples = int(max_jitter_s*16000)
            
        self.augmented_samples = None
        if augmentation_duration_s is not None:
            self.augmented_samples = int(augmentation_duration_s*16000)

        self.max_start_time_from_right_s=max_start_time_from_right_s
        if max_start_time_from_right_s is not None and (min_jitter_s is not None or max_jitter_s is not None):
            raise ValueError(
                "max_start_time_from_s and max_jitter_s/min_jitter_s cannot both be configured."
            )
        
        if (max_start_time_from_right_s is not None) and (
            augmentation_duration_s is None
        ):
            raise ValueError(
                "max_start_time_from_right_s cannot be specified if augmentation_duration_s is not configured."
            )

        if (
            (max_start_time_from_right_s is not None)
            and (max_start_time_from_right_s > augmentation_duration_s)
        ):
            raise ValueError(
                "max_start_time_from_right_s cannot be greater than augmentation_duration_s."
            )

        assert self.min_jitter_samples <= self.max_jitter_samples, "Minimum jitter must be less than or equal to maximum jitter."

        #######################
        # Setup augmentations #
        #######################

        # If either the background_paths or impulse_paths are not specified, use an identity transform instead
        def identity_transform(samples, sample_rate):
            return samples

        background_noise_augment = audiomentations.Lambda(
            transform=identity_transform, p=0.0
        )
        reverb_augment = audiomentations.Lambda(transform=identity_transform, p=0.0)

        if background_paths is not None:
            background_noise_augment = audiomentations.AddBackgroundNoise(
                p=augmentation_probabilities.get("AddBackgroundNoise", 0.0),
                sounds_path=background_paths,
                min_snr_db=background_min_snr_db,
                max_snr_db=background_max_snr_db,
            )

        if impulse_paths is not None:
            reverb_augment = audiomentations.ApplyImpulseResponse(
                p=augmentation_probabilities.get("RIR", 0.0),
                ir_path=impulse_paths,
            )

        # Based on openWakeWord's augmentations, accessed on February 23, 2024.
        self.augment = audiomentations.Compose(
            transforms=[
                audiomentations.SevenBandParametricEQ(
                    p=augmentation_probabilities.get("SevenBandParametricEQ", 0.0),
                    min_gain_db=-6,
                    max_gain_db=6,
                ),
                audiomentations.TanhDistortion(
                    p=augmentation_probabilities.get("TanhDistortion", 0.0),
                    min_distortion=0.0001,
                    max_distortion=0.10,
                ),
                audiomentations.PitchShift(
                    p=augmentation_probabilities.get("PitchShift", 0.0),
                    min_semitones=-3,
                    max_semitones=3,
                ),
                audiomentations.BandStopFilter(
                    p=augmentation_probabilities.get("BandStopFilter", 0.0),
                ),
                audiomentations.AddColorNoise(
                    p=augmentation_probabilities.get("AddColorNoise", 0.0),
                    min_snr_db=10,
                    max_snr_db=30,
                ),
                background_noise_augment,
                audiomentations.GainTransition(
                    p=augmentation_probabilities.get("Gain", 0.0),
                    min_gain_db=-12,
                    max_gain_in_db=0,
                ),
                reverb_augment,
            ],
            shuffle=False,
        )        

    def add_jitter(self, input_audio):
        if self.min_jitter_samples < self.max_jitter_samples:
            jitter_samples = np.random.randint(self.min_jitter_samples, self.max_jitter_samples)
        elif self.max_start_time_from_right_s is not None:
            max_start_from_right_samples = int(self.max_start_time_from_right_s*16000)
            max_padding_samples = max_start_from_right_samples - input_audio.shape[0]
            jitter_samples = np.random.randint(0, max_padding_samples)
        else:
            jitter_samples = self.min_jitter_samples
            
        # Pad audio on the right by jitter samples
        return np.pad(input_audio, (0,jitter_samples))
        
    def create_fixed_size_clip(self, input_audio):
        if self.augmented_samples is not None:
            if self.augmented_samples < input_audio.shape[0]:
                # Truncate the too long audio by removing the start of the clip
                input_audio = input_audio[-self.augmented_samples:]
            else:
                # Pad with zeros at start of too short audio clip
                left_padding_samples = self.augmented_samples - input_audio.shape[0]
                
                input_audio = np.pad(input_audio, (left_padding_samples,0))
            
        return input_audio
                
    def augment_clip(self, input_audio):
        """Augments the input audio after creating a fixed size clip

        Args:
            input_audio (ndarray): audio data with sample rate 16 kHz and 16-bit samples

        Returns:
            (ndarray): the augmented audio with sample rate 16 kHz and 16-bit samples
        """
        input_audio = self.add_jitter(input_audio)
        # input_audio = self.offset_from_right(input_audio)
        input_audio = self.create_fixed_size_clip(input_audio)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Suppresses warning about background clip being too quiet... TODO: find better approach!
            output_audio = self.augment(input_audio, sample_rate=16000)
            
        return output_audio
    
    def augment_generator(self, audio_generator):
        for audio in audio_generator:
            yield self.augment_clip(audio)
            
class Clips:
    def __init__(
        self,
        input_path,
        input_glob,
        min_clip_duration_s=0,
        max_clip_duration_s=math.inf,
        # repeat_clip_min_duration_s=0,        
        # remove_silence=False,
        random_split_seed=None,
        split_count=200,  
        **kwargs,
    ):
        self.min_clip_duration_s = min_clip_duration_s
        if min_clip_duration_s is None:
            self.min_clip_duration_s = 0
            
        self.max_clip_duration_s = max_clip_duration_s
        if max_clip_duration_s is None:
            self.max_clip_duration_s = math.inf
        
        # self.repeat_clip_min_duration_s = repeat_clip_min_duration_s
        # if repeat_clip_min_duration_s is None:
        #     self.repeat_clip_min_duration_s = repeat_clip_min_duration_s
        
        # self.remove_silence=remove_silence
        
        paths_to_clips = [str(i) for i in Path(input_path).glob(input_glob)]        
        
        if (min_clip_duration_s == 0) and (math.isinf(max_clip_duration_s)):
            # No durations specified, so do not filter by length
            filtered_paths = paths_to_clips
        else:
            # Filter audio clips by length
            if input_glob.endswith("wav"):
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
                    if (min_clip_duration_s < duration)
                    and (duration < max_clip_duration_s)
                ]
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

        # Load all filtered clips
        audio_dataset = datasets.Dataset.from_dict(
            {"audio": [str(i) for i in filtered_paths]}
        ).cast_column("audio", datasets.Audio())

        # Convert all clips to 16 kHz sampling rate when accessed
        audio_dataset = audio_dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=16000)
        )
        
        if random_split_seed is not None:
            train_testvalid = audio_dataset.train_test_split(test_size=2*split_count, seed=random_split_seed)
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
            split_dataset = datasets.DatasetDict({
                'train': train_testvalid['train'],
                'test': test_valid['test'],
                'validation': test_valid['train'],
            })
            self.split_clips = split_dataset
                
        
        self.clips = audio_dataset
    
    def audio_generator(self, split=None, repeat=1, **kwargs):
        if split is None:
            clip_list = self.clips
        else:
            clip_list = self.split_clips[split]
        for _ in range(repeat):
            for clip in clip_list:
                yield clip["audio"]["array"]
    
    def get_random_clip(self):
        rand_audio = random.choice(self.clips)
        return rand_audio["audio"]["array"]
    
    def random_audio_generator(self, max_clips=math.inf):
        while max_clips > 0:
            max_clips -= 1
            
            yield self.get_random_clip()
                
    # def repeat_clip(self, x, sr=16000):
    #     original_clip = x
    #     desired_samples = int(self.repeat_clip_min_duration_s * sr)
    #     while x.shape[0] < desired_samples:
    #         x=np.append(x,original_clip)
    #     return x
            
class SpectrogramGeneration:
    def __init__(
        self,
        clips,
        augmenter=None,
    ):
        self.clips = clips
        self.augmenter = augmenter
        
    def get_random_spectrogram(self):
        clip = self.clips.get_random_clip()
        if self.augmenter is not None:
            clip = self.augmenter.augment_clip(clip)
        
        return generate_features_for_clip(clip)
    
    def spectrogram_generator(self, **kwargs):
        for augmented_clip in self.augmenter.augment_generator(self.clips.audio_generator(**kwargs)):
            yield generate_features_for_clip(augmented_clip)
            
    def random_spectrogram_generator(self):
        for augmented_clip in self.augmenter.augment_generator(self.clips.random_audio_generator()):
            yield generate_features_for_clip(augmented_clip)