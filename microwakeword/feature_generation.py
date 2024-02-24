import datasets
import os

import numpy as np
import tensorflow as tf

from mmap_ninja.ragged import RaggedMmap
from pathlib import Path
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)

def generate_features_for_folder(
    path_to_audio, features_output_dir, set_name, audio_files_glob="*.wav"
):
    # Generates audio features and save as a RaggedMmap
    
    audio_dataset = datasets.Dataset.from_dict(
        {"audio": [str(i) for i in Path(path_to_audio).glob(audio_files_glob)]}
    )
    audio_dataset = audio_dataset.cast_column(
        "audio", datasets.Audio(sampling_rate=16000)
    )

    def features_generator():
        for row in audio_dataset:
            yield generate_features_for_clip(
                (row["audio"]["array"] * 32767).astype(np.int16)
            )


    if not os.path.exists(features_output_dir):
        os.makedirs(features_output_dir)
        
    out_dir = os.path.join(features_output_dir, set_name + "_mmap")


    RaggedMmap.from_generator(
        out_dir=out_dir,
        sample_generator=features_generator(),
        batch_size=1024,
        verbose=True,
    )

def generate_features_for_clip(clip, desired_spectrogram_length=0):
    with tf.device('/cpu:0'):
        micro_frontend = frontend_op.audio_microfrontend(
            tf.convert_to_tensor(clip),
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
        if desired_spectrogram_length > 0:
            return spectrogram[
                -desired_spectrogram_length:
            ]  # truncate to match desired spectrogram size
        return spectrogram

def features_generator(generator, desired_spectrogram_length=0):
    for data in generator:
        for clip in data:
            yield generate_features_for_clip(clip, desired_spectrogram_length)
