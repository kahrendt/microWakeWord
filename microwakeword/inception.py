# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
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

"""Inception - reduced version of keras/applications/inception_v3.py ."""
import ast
import tensorflow as tf

from microwakeword.layers import modes
from microwakeword.layers import stream
from microwakeword.layers import delay
from microwakeword.layers import sub_spectral_normalization


class StridedDrop(tf.keras.layers.Layer):
    """StridedDrop

    Drops the specified audio feature slices in nonstreaming mode only.
    Used for matching the dimensions of convolutions with valid padding.

    Attributes:
        time_sclices_to_drop: number of audio feature slices to drop
        mode: inference mode; e.g., non-streaming, internal streaming
    """

    def __init__(
        self, time_slices_to_drop, mode=modes.Modes.NON_STREAM_INFERENCE, **kwargs
    ):
        super(StridedDrop, self).__init__(**kwargs)
        self.time_slices_to_drop = time_slices_to_drop
        self.mode = mode

    def call(self, inputs):
        if self.mode == modes.Modes.NON_STREAM_INFERENCE:
            return inputs[:, self.time_slices_to_drop :, :, :]

        return inputs

    def get_config(self):
        config = {
            "time_slices_to_drop": self.time_slices_to_drop,
            "mode": self.mode,
        }
        base_config = super(StridedDrop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def parse(text):
    """Parse model parameters.

    Args:
      text: string with layer parameters: '128,128' or "'relu','relu'".

    Returns:
      list of parsed parameters
    """
    if not text:
        return []
    res = ast.literal_eval(text)
    if isinstance(res, tuple):
        return res
    else:
        return [res]


def conv2d_bn(
    x,
    filters,
    kernel_size,
    dilation=(1, 1),
    padding="same",
    strides=(1, 1),
    activation="relu",
    use_bias=False,
    subgroups=1,
):
    """Utility function to apply conv + BN.

    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: size of convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        activation: activation function applied in the end.
        use_bias: use bias for convolution.
        scale: scale batch normalization.

    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        dilation_rate=dilation,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
    )(x)

    sub_spectral_normalization_layer = (
        sub_spectral_normalization.SubSpectralNormalization(subgroups)
    )
    x = sub_spectral_normalization_layer(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x


def conv2d_bn_delay(
    x,
    filters,
    kernel_size,
    dilation,
    padding="same",
    strides=(1, 1),
    activation="relu",
    use_bias=False,
    delay_val=1,
    subgroups=1,
):
    """Utility function to apply conv + BN while managing the streaming wrapper.

    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: size of convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        activation: activation function applied in the end.
        use_bias: use bias for convolution.
        scale: scale batch normalization.
        use_one_steP : use one_step mode for streaming wrapper

    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    if padding == "same":
        x = delay.Delay(delay=delay_val)(x)

    x = stream.Stream(
        cell=tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            dilation_rate=dilation,
            strides=strides,
            padding="valid",
            use_bias=use_bias,
        ),
        use_one_step=False,
        pad_time_dim=padding,
        pad_freq_dim="same",
    )(x)
    sub_spectral_normalization_layer = (
        sub_spectral_normalization.SubSpectralNormalization(subgroups)
    )
    x = sub_spectral_normalization_layer(x)

    x = tf.keras.layers.Activation(activation)(x)
    return x


def model_parameters(parser_nn):
    """Inception model parameters.

    Args:
      parser_nn: global command line args parser
    Returns: parser with updated arguments
    """
    parser_nn.add_argument(
        "--cnn1_filters",
        type=str,
        default="24",
        help="Number of filters in the first conv blocks",
    )
    parser_nn.add_argument(
        "--cnn1_kernel_sizes",
        type=str,
        default="5",
        help="Kernel size in time dim of conv blocks",
    )
    parser_nn.add_argument(
        "--cnn1_subspectral_groups",
        type=str,
        default="4",
        help="The number of subspectral groups for normalization",
    )
    parser_nn.add_argument(
        "--cnn2_filters1",
        type=str,
        default="10,10,16",
        help="Number of filters inside of inception block "
        "will be multipled by 4 because of concatenation of 4 branches",
    )
    parser_nn.add_argument(
        "--cnn2_filters2",
        type=str,
        default="10,10,16",
        help="Number of filters inside of inception block "
        "it is used to reduce the dim of cnn2_filters1*4",
    )
    parser_nn.add_argument(
        "--cnn2_kernel_sizes",
        type=str,
        default="5,5,5",
        help="Kernel sizes of conv layers in the inception block",
    )
    parser_nn.add_argument(
        "--cnn2_subspectral_groups",
        type=str,
        default="1,1,1",
        help="The number of subspectral groups for normalization",
    )
    parser_nn.add_argument(
        "--cnn2_dilation",
        type=str,
        default="1,1,1",
        help="Dilation rate",
    )
    parser_nn.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Percentage of data dropped",
    )


def model(flags, config):
    """Inception model.

    It is based on paper:
    Rethinking the Inception Architecture for Computer Vision
        http://arxiv.org/abs/1512.00567
    Args:
      flags: data/model parameters

    Returns:
      Keras model for training
    """
    input_audio = tf.keras.layers.Input(
        shape=modes.get_input_data_shape(config, modes.Modes.TRAINING),
        batch_size=config["batch_size"],
    )
    net = input_audio

    # [batch, time, feature]
    net = tf.keras.backend.expand_dims(net, axis=2)
    # [batch, time, 1, feature]

    for filters, kernel_size, subgroups in zip(
        parse(flags.cnn1_filters),
        parse(flags.cnn1_kernel_sizes),
        parse(flags.cnn1_subspectral_groups),
    ):
        # Streaming Conv2D with 'valid' padding
        net = stream.Stream(
            cell=tf.keras.layers.Conv2D(
                filters, (kernel_size, 1), padding="valid", use_bias=False
            ),
            use_one_step=True,
            pad_time_dim=None,
            pad_freq_dim="same",
        )(net)
        sub_spectral_normalization_layer = (
            sub_spectral_normalization.SubSpectralNormalization(subgroups)
        )
        net = sub_spectral_normalization_layer(net)
        net = tf.keras.layers.Activation("relu")(net)

    for filters1, filters2, kernel_size, subgroups, dilation in zip(
        parse(flags.cnn2_filters1),
        parse(flags.cnn2_filters2),
        parse(flags.cnn2_kernel_sizes),
        parse(flags.cnn2_subspectral_groups),
        parse(flags.cnn2_dilation),
    ):
        time_buffer_size = dilation * (kernel_size - 1)

        branch1 = conv2d_bn(net, filters1, (1, 1), dilation=(1, 1), subgroups=subgroups)

        branch2 = conv2d_bn(net, filters1, (1, 1), subgroups=subgroups)
        branch2 = conv2d_bn_delay(
            branch2,
            filters1,
            (kernel_size, 1),
            (dilation, 1),
            padding="None",
            delay_val=time_buffer_size // 2,
            subgroups=subgroups,
        )

        branch3 = conv2d_bn(net, filters1, (1, 1), subgroups=subgroups)
        branch3 = conv2d_bn_delay(
            branch3,
            filters1,
            (kernel_size, 1),
            (dilation, 1),
            padding="None",
            delay_val=time_buffer_size // 2,
            subgroups=subgroups,
        )
        branch3 = conv2d_bn_delay(
            branch3,
            filters1,
            (kernel_size, 1),
            (dilation, 1),
            padding="None",
            delay_val=time_buffer_size // 2,
            subgroups=subgroups,
        )

        branch1_drop_layer = StridedDrop(branch1.shape[1] - branch3.shape[1])
        branch1 = branch1_drop_layer(branch1)

        branch2_drop_layer = StridedDrop(branch2.shape[1] - branch3.shape[1])
        branch2 = branch2_drop_layer(branch2)

        net = tf.keras.layers.concatenate([branch1, branch2, branch3])
        # [batch, time, 1, filters*4]
        net = conv2d_bn(net, filters2, (1, 1))
        # [batch, time, 1, filters2]

    net = stream.Stream(cell=tf.keras.layers.Flatten())(net)
    # [batch, filters*4]
    net = tf.keras.layers.Dropout(flags.dropout)(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid")(net)

    return tf.keras.Model(input_audio, net)
