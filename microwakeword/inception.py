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
    padding="same",
    strides=(1, 1),
    activation="relu",
    use_bias=False,
    scale=False,
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
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    x = tf.keras.layers.BatchNormalization(scale=scale)(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x


def conv2d_bn_delay(
    x,
    filters,
    kernel_size,
    padding="same",
    strides=(1, 1),
    activation="relu",
    use_bias=False,
    scale=False,
    delay_val=1,
    use_one_step=True,
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

    if padding == "same":
        x = delay.Delay(delay=delay_val)(x)

    x = stream.Stream(
        cell=tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="valid", use_bias=use_bias
        ),
        use_one_step=False,
        pad_time_dim=padding,
        pad_freq_dim="same",
    )(x)
    x = tf.keras.layers.BatchNormalization(scale=scale)(x)
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
        "--cnn1_strides",
        type=str,
        default="1",
        help="Strides applied in pooling layer in the first conv block",
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
        "--cnn2_strides",
        type=str,
        default="2,2,1",
        help="Stride parameter of pooling layer in the inception block",
    )
    parser_nn.add_argument(
        "--cnn2_use_one_step_stream",
        type=bool,
        default=True,
        help="Should the streaming convolution layer use one_step mode",
    )
    parser_nn.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Percentage of data dropped",
    )
    parser_nn.add_argument(
        "--bn_scale",
        type=int,
        default=0,
        help="If True, multiply by gamma. If False, gamma is not used. "
        "When the next layer is linear (also e.g. nn.relu), this can be disabled"
        "since the scaling will be done by the next layer.",
    )


def model(flags):
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
        shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
        batch_size=flags.batch_size,
    )
    net = input_audio

    # [batch, time, feature]
    net = tf.keras.backend.expand_dims(net, axis=2)
    # [batch, time, 1, feature]

    for stride, filters, kernel_size in zip(
        parse(flags.cnn1_strides),
        parse(flags.cnn1_filters),
        parse(flags.cnn1_kernel_sizes),
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
        net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
        net = tf.keras.layers.Activation("relu")(net)

        if stride > 1:
            net = tf.keras.layers.MaxPooling2D((3, 1), strides=(stride, 1))(net)

    for stride, filters1, filters2, kernel_size in zip(
        parse(flags.cnn2_strides),
        parse(flags.cnn2_filters1),
        parse(flags.cnn2_filters2),
        parse(flags.cnn2_kernel_sizes),
    ):
        time_buffer_size = kernel_size - 1

        branch1 = conv2d_bn(net, filters1, (1, 1), scale=flags.bn_scale)

        branch2 = conv2d_bn(net, filters1, (1, 1), scale=flags.bn_scale)
        branch2 = conv2d_bn_delay(
            branch2,
            filters1,
            (kernel_size, 1),
            padding="causal",
            scale=flags.bn_scale,
            delay_val=time_buffer_size // 2,
            use_one_step=flags.cnn2_use_one_step_stream,
        )

        branch3 = conv2d_bn(net, filters1, (1, 1), scale=flags.bn_scale)
        branch3 = conv2d_bn_delay(
            branch3,
            filters1,
            (kernel_size, 1),
            padding="causal",
            scale=flags.bn_scale,
            delay_val=time_buffer_size // 2,
            use_one_step=flags.cnn2_use_one_step_stream,
        )
        branch3 = conv2d_bn_delay(
            branch3,
            filters1,
            (kernel_size, 1),
            padding="causal",
            scale=flags.bn_scale,
            delay_val=time_buffer_size // 2,
            use_one_step=flags.cnn2_use_one_step_stream,
        )

        net = tf.keras.layers.concatenate([branch1, branch2, branch3])
        # [batch, time, 1, filters*4]
        net = conv2d_bn(net, filters2, (1, 1), scale=flags.bn_scale)
        # [batch, time, 1, filters2]

        if stride > 1:
            net = tf.keras.layers.MaxPooling2D((3, 1), strides=(stride, 1))(net)

    net = stream.Stream(cell=tf.keras.layers.GlobalAveragePooling2D())(net)
    # [batch, filters*4]
    net = tf.keras.layers.Dropout(flags.dropout)(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid")(net)

    return tf.keras.Model(input_audio, net)
