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


from microwakeword.layers import delay
from microwakeword.layers import stream
from microwakeword.layers import strided_drop
from microwakeword.layers import sub_spectral_normalization


def parse(text):
    """Parse model parameters.

    Args:
      text: string with layer parameters: '128,128' or "'relu','relu'"

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
        x: input tensor
        filters: filters in `Conv2D`
        kernel_size: size of convolution kernel
        dilation: dilation rate
        padding: padding mode in `Conv2D`
        strides: strides in `Conv2D`
        activation: activation function applied in the end
        use_bias: use bias for convolution
        subgroups: the number of subgroups used for sub-spectral normaliation

    Returns:
        output tensor after applying `Conv2D` and `SubSpectralNormalization`
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
    """Utility function to apply conv + BN.

    Arguments:
        x: input tensor
        filters: filters in `Conv2D`
        kernel_size: size of convolution kernel
        dilation: dilation rate
        padding: padding mode in `Conv2D`
        strides: strides in `Conv2D`
        activation: activation function applied in the end
        use_bias: use bias for convolution
        delay_val: number of features for delay layer when using `same` padding
        subgroups: the number of subgroups used for sub-spectral normaliation

    Returns:
        output tensor after applying `Conv2D` and `SubSpectralNormalization`.
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
    Returns:
        parser with updated arguments
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


def spectrogram_slices_dropped(flags):
    """Computes the number of spectrogram slices dropped due to valid padding.

    Args:
        flags: data/model parameters

    Returns:
        int: number of spectrogram slices dropped
    """
    spectrogram_slices_dropped = 0

    for kernel_size in parse(flags.cnn1_kernel_sizes):
        spectrogram_slices_dropped += kernel_size - 1
    for kernel_size, dilation in zip(
        parse(flags.cnn2_kernel_sizes), parse(flags.cnn2_dilation)
    ):
        spectrogram_slices_dropped += 2 * dilation * (kernel_size - 1)

    return spectrogram_slices_dropped


def model(flags, shape, batch_size):
    """Inception model.

    It is based on paper:
    Rethinking the Inception Architecture for Computer Vision
        http://arxiv.org/abs/1512.00567
    Args:
      flags: data/model parameters
      config: dictionary containing microWakeWord training configuration

    Returns:
      Keras model for training
    """
    input_audio = tf.keras.layers.Input(
        shape=shape,
        batch_size=batch_size,
    )
    net = input_audio

    # [batch, time, feature]
    net = tf.keras.ops.expand_dims(net, axis=2)
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

        branch1_drop_layer = strided_drop.StridedDrop(
            branch1.shape[1] - branch3.shape[1]
        )
        branch1 = branch1_drop_layer(branch1)

        branch2_drop_layer = strided_drop.StridedDrop(
            branch2.shape[1] - branch3.shape[1]
        )
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
