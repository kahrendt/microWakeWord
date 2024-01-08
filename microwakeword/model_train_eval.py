# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

import argparse
import json
import os
from absl import logging

import microwakeword.inception as inception
import microwakeword.train as train
import microwakeword.test as test
import microwakeword.utils as utils

from microwakeword.layers import modes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--background_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
        Where background negative features are stored.
        """)
    parser.add_argument(
        '--generated_negative_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
        Where generated negative features are stored.
        """)
    parser.add_argument(
        '--generated_positive_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
        Where generated positive features are stored.
      """)
    parser.add_argument(
        '--background_weight',
        type=str,
        default=5,
        help="""\
        How much to weigh the background negative samples in each batch.
        """)
    parser.add_argument(
        '--generated_negative_weight',
        type=str,
        default=5,
        help="""\
        How much to weigh the generated negative samples in each batch.
        """)
    parser.add_argument(
        '--generated_positive_weight',
        type=str,
        default=5,
        help="""\
        How much to weigh the generated positive samples in each batch.
        """)
    parser.add_argument(
        '--background_probability',
        type=str,
        default=5,
        help="""\
        How much to weigh the generated positive samples in each batch.
        """)
    parser.add_argument(
        '--positive_probability',
        type=str,
        default=5,
        help="""\
        How much to weigh the generated positive samples in each batch.
        """)
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=500,
        help='How often to evaluate the training results.'
    )    
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='10000,10000,10000',
        help='How many training loops to run',
    )
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.0005,0.0001,0.00002',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/speech_commands_train',
        help='Directory to write event logs and checkpoint.',
    )    
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the input wavs',
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',
    )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.',
    )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=20.0,
        help='How far to move in time between spectrogram timeslices.',
    )    
    parser.add_argument(
        '--train',
        type=int,
        default=1,
        help='If 1 run train and test, else run only test',
    )
    parser.add_argument(
        '--mel_num_bins',
        type=int,
        default=40,
        help='How many bands in the resulting mel spectrum.',
    )
    parser.add_argument(
        '--restore_checkpoint',
        type=int,
        default=0,
        help='If 1 it will restore a checkpoint and resume the training '
        'by initializing model weights and optimizer with checkpoint values. '
        'It will use learning rate and number of training iterations from '
        '--learning_rate and --how_many_training_steps accordinlgy. '
        'This option is useful in cases when training was interrupted. '
        'With it you should adjust learning_rate and how_many_training_steps.',
    )
    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
        value: A member of tf.logging.

        Returns:
        TF logging mode

        Raises:
        ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == 'INFO':
            return logging.INFO
        elif value == 'DEBUG':
            return logging.DEBUG
        elif value == 'ERROR':
            return logging.ERROR
        elif value == 'FATAL':
            return logging.FATAL
        elif value == 'WARN':
            return logging.WARN
        else:
            raise argparse.ArgumentTypeError('Not an expected value')

    parser.add_argument(
        '--verbosity',
        type=verbosity_arg,
        default=logging.INFO,
        help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"'
    )
    
    # sub parser for model settings
    subparsers = parser.add_subparsers(dest='model_name', help='NN model name')
    
    # inception model settings
    parser_inception = subparsers.add_parser('inception')
    inception.model_parameters(parser_inception)

    flags, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError('Unknown argument: {}'.format(unparsed))        
    
    flags.summaries_dir = os.path.join(flags.train_dir, 'logs/')
    
    desired_samples = int(flags.sample_rate * flags.clip_duration_ms /
                            1000)
    window_size_samples = int(flags.sample_rate * flags.window_size_ms /
                                1000)
    window_stride_samples = int(flags.sample_rate * flags.window_stride_ms /
                                1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        flags.spectrogram_length = 0
    else:
        flags.spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        
    flags.preprocess = 'micro'
    
    if flags.train:
        try:
            os.makedirs(flags.train_dir)
            os.makedirs(os.path.join(flags.train_dir, 'restore')) # DO I USE THIS?
            os.mkdir(flags.summaries_dir)
        except OSError as e:
            if flags.restore_checkpoint:
                pass
            else:
                raise ValueError('model already exists in folder %s' %
                                flags.train_dir) from None
        
        train.train(flags)
    else:
        if not os.path.isdir(flags.train_dir):
            raise ValueError('model is not trained set "--train 1" and retrain it')

    # write all flags settings into json
    with open(os.path.join(flags.train_dir, 'flags.json'), 'wt') as f:
        json.dump(flags.__dict__, f)

    utils.convert_model_saved(flags, 'non_stream', modes.Modes.NON_STREAM_INFERENCE)
    utils.convert_model_saved(flags, 'stream_state_internal', modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)

    folder_name = 'tflite_non_stream' 
    file_name = 'non_stream.tflite'
    utils.convert_saved_model_to_tflite(flags, os.path.join(flags.train_dir, 'non_stream'), os.path.join(flags.train_dir,folder_name),file_name)
    test.tflite_model_accuracy(flags, folder_name, file_name)
    
    folder_name = 'tflite_stream_state_internal'
    file_name = 'stream_state_internal.tflite'
    utils.convert_saved_model_to_tflite(flags, os.path.join(flags.train_dir, 'stream_state_internal'), os.path.join(flags.train_dir,folder_name),file_name)
    # test.streaming_model_false_accept_rate(flags, folder_name, file_name, 'dipco_features.npy')
    test.tflite_model_accuracy(flags, folder_name, file_name)

    # quantize the internal streaming model here and then test it
    folder_name = 'tflite_stream_state_internal_quant'
    file_name = 'stream_state_internal_quantize.tflite'
    utils.convert_saved_model_to_tflite(flags, os.path.join(flags.train_dir, 'stream_state_internal'), os.path.join(flags.train_dir,folder_name),file_name, quantize=True)
    # test.streaming_model_false_accept_rate(flags, folder_name, file_name, 'dipco_features.npy')
    test.tflite_model_accuracy(flags, folder_name, file_name)
