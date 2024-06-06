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

"""Model based on 1D depthwise MixedConvs and 1x1 convolutions in time + residual."""

from microwakeword.layers import stream
from microwakeword.layers import strided_drop

import ast
import tensorflow as tf

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


def model_parameters(parser_nn):
    """MatchboxNet model parameters."""

    parser_nn.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Percentage of data dropped",
    )
    parser_nn.add_argument(
        "--pointwise_filters",
        type=str,
        default="64, 32, 48, 64, 64",
        help="Number of filters in every residual block's branch pointwise convolutions",
    )
    parser_nn.add_argument(
        "--residual_connection",
        type=str,
        default="0,0,0,0,0",
    )
    parser_nn.add_argument(
        "--repeat_in_block",
        type=str,
        default="1,1,1,1,1",
        help="Number of repeating conv blocks inside of residual block",
    )
    parser_nn.add_argument(
        "--mixconv_kernel_sizes",
        type=str,
        default="[5], [5,9], [9,13], [13,17], [9,13]",
        help="Kernel size of DepthwiseConv1D in time dim for every residual block",
    )
    parser_nn.add_argument(
        "--max_pool",
        type=int,
        default=0,
        help="apply max pool instead of average pool before final convolution and sigmoid activation",
    )
    parser_nn.add_argument(
        "--first_conv_filters",
        type=int,
        default=0,
        help="Number of filters on initial convolution layer. Set to 0 to disable",
    )
    parser_nn.add_argument(
        "--spatial_attention",
        type=int,
        default=0,
        help="add a spatial attention layer before the final pooling layer",
    )


def spectrogram_slices_dropped(flags):
    """Computes the number of spectrogram slices dropped due to valid padding.

    Args:
        flags: data/model parameters

    Returns:
        int: number of spectrogram slices dropped
    """
    spectrogram_slices_dropped = 0

    # initial 3x1 convolution drops 2
    if flags.first_conv_filters > 0:
        # spectrogram_slices_dropped += 4
        spectrogram_slices_dropped += 2

    for repeat, ksize in zip(
        parse(flags.repeat_in_block),
        parse(flags.mixconv_kernel_sizes),
    ):
        spectrogram_slices_dropped += repeat * (max(ksize) - 1)

    return spectrogram_slices_dropped


def _split_channels(total_filters, num_groups):
    """ Helper for MixConv
    """
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


def _get_shape_value(maybe_v2_shape):
    """ Helper for MixConv
    """
    if maybe_v2_shape is None:
        return None
    elif isinstance(maybe_v2_shape, int):
        return maybe_v2_shape
    else:
        return maybe_v2_shape.value


class MixConv(object):
    """MixConv with mixed depthwise convolutional kernels.

    MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
    3x1, 5x1, etc). Right now, we use an naive implementation that split channels
    into multiple groups and perform different kernels for each group.

    See Mixnet paper for more details.
    """

    def __init__(self, kernel_size, **kwargs):
        """Initialize the layer.

        Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
        an extra parameter "dilated" to indicate whether to use dilated conv to
        simulate large kernel size. If dilated=True, then dilation_rate is ignored.

        Args:
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
            then we split the channels and perform different kernel for each group.
          strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width.
          dilated: Bool. indicate whether to use dilated conv to simulate large
            kernel size.
          **kwargs: other parameters passed to the original depthwise_conv layer.
        """
        self._channel_axis = -1

        self.ring_buffer_length = max(kernel_size) - 1

        self.kernel_sizes = kernel_size

    def __call__(self, inputs):
        # We manually handle the streaming ring buffer for each layer
        #   - There is some latency overhead on the esp devices for loading each ring buffer's data
        #   - This avoids variable's holding redundant information
        #   - Reduces the necessary size of the tensor arena
        net = stream.Stream(
            cell=tf.identity,
            ring_buffer_size_in_time_dim=self.ring_buffer_length,
            use_one_step=False,
        )(inputs)

        if len(self.kernel_sizes) == 1:
            return tf.keras.layers.DepthwiseConv2D(
                (self.kernel_sizes[0], 1), strides=1, padding="valid"
            )(net)

        filters = _get_shape_value(net.shape[self._channel_axis])
        splits = _split_channels(filters, len(self.kernel_sizes))
        x_splits = tf.split(net, splits, self._channel_axis)

        x_outputs = []
        for x, ks in zip(x_splits, self.kernel_sizes):
            fit = strided_drop.StridedKeep(ks)(x)
            x_outputs.append(
                tf.keras.layers.DepthwiseConv2D((ks, 1), strides=1, padding="valid")(
                    fit
                )
            )

        for i, output in enumerate(x_outputs):
            features_drop = output.shape[1] - x_outputs[-1].shape[1]
            x_outputs[i] = strided_drop.StridedDrop(features_drop)(output)

        x = tf.concat(x_outputs, self._channel_axis)
        return x

class SpatialAttention(object):
    """Spatial Attention Layer based on CBAM: Convolutional Block Attention Module
    https://arxiv.org/pdf/1807.06521v2

    Args:
        object (_type_): _description_
    """
    def __init__(self, kernel_size, ring_buffer_size, **kwargs):
        self.kernel_size = kernel_size
        self.ring_buffer_size= ring_buffer_size
        
    def __call__(self, inputs):
        tranposed = tf.transpose(inputs,perm=[0,1,3,2])
        channel_avg = tf.keras.layers.AveragePooling2D(pool_size=(1, tranposed.shape[2]), strides=(1,tranposed.shape[2]))(tranposed)
        channel_max = tf.keras.layers.MaxPooling2D(pool_size=(1, tranposed.shape[2]), strides=(1,tranposed.shape[2]))(tranposed)
        pooled = tf.keras.layers.Concatenate(axis=-1)([channel_avg, channel_max])
        attention = stream.Stream(
                cell=tf.keras.layers.Conv2D(
                    1,
                    (self.kernel_size, 1),
                    strides=(1, 1),
                    padding="valid",
                    use_bias=False,
                    activation="sigmoid",
                ),
            use_one_step=False)(pooled)

        net = stream.Stream(
            cell=tf.identity,
            ring_buffer_size_in_time_dim=self.ring_buffer_size,
            use_one_step=False,
        )(inputs)
        net = net[:,-attention.shape[1]:,:,:]

        return net*attention
        
def model(flags, shape, batch_size):
    """MixedNet model.

    It is based on the paper
    MixConv: Mixed Depthwise Convolutional Kernels
    MatchboxNet model.
    https://arxiv.org/abs/1907.09595
    Args:
      flags: data/model parameters
      shape: shape of the input vector
      config: dictionary containing microWakeWord training configuration

    Returns:
      Keras model for training
    """

    pointwise_filters = parse(flags.pointwise_filters)
    repeat_in_block = parse(flags.repeat_in_block)
    mixconv_kernel_sizes = parse(flags.mixconv_kernel_sizes)
    residual_connections = parse(flags.residual_connection)

    for l in (
        pointwise_filters,
        repeat_in_block,
        mixconv_kernel_sizes,
        residual_connections,
    ):
        if len(pointwise_filters) != len(l):
            raise ValueError("all input lists have to be the same length")

    input_audio = tf.keras.layers.Input(
        shape=shape,
        batch_size=batch_size,
    )
    net = input_audio

    # make it [batch, time, 1, feature]
    net = tf.keras.backend.expand_dims(net, axis=2)

    # Streaming Conv2D with 'valid' padding
    if flags.first_conv_filters > 0:
        net = stream.Stream(
            cell=tf.keras.layers.Conv2D(
                flags.first_conv_filters,
                # (5, 1),
                (3, 1),
                strides=(1, 1),
                padding="valid",
                use_bias=False,
            ),
            use_one_step=True,
            pad_time_dim=None,
            pad_freq_dim="valid",
        )(net)

        net = tf.keras.layers.Activation("relu")(net)
        
        # ###
        # # Squeeze and Excitation block
        # # Based on Depthwise Separable Convolutional ResNet with Squeeze-and-Excitation Blocks for Small-footprint Keyword Spotting
        # # https://arxiv.org/pdf/2004.12200
        # ###
        # x = stream.Stream(
        #     cell=tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1), strides=(net.shape[1], 1))
        # )(net)
        # x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(flags.first_conv_filters//16, activation='relu')(x)
        # x = tf.keras.layers.Dense(flags.first_conv_filters, activation='sigmoid')(x)
        # net = tf.keras.layers.Multiply()([net,x])
        
        
    # encoder
    for filters, repeat, ksize, res in zip(
        pointwise_filters,
        repeat_in_block,
        mixconv_kernel_sizes,
        residual_connections,
    ):
        if res:
            residual = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(net)
            residual = tf.keras.layers.BatchNormalization()(residual)
            # residual = tf.keras.layers.Activation("relu")(residual)

        for _ in range(repeat):
            if max(ksize) > 1:
                net = MixConv(kernel_size=ksize)(net)
            net = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(net)
            net = tf.keras.layers.BatchNormalization()(net)
            # net = tf.keras.layers.Activation("relu")(net)

            if res:
                residual = strided_drop.StridedDrop(residual.shape[1] - net.shape[1])(
                    residual
                )
                net = net + residual

            net = tf.keras.layers.Activation("relu")(net)

    if net.shape[1] > 1:
        if flags.spatial_attention:
            net = SpatialAttention(5,net.shape[1]-1)(net)
        else:
            net = stream.Stream(
                    cell=tf.identity,
                    ring_buffer_size_in_time_dim=net.shape[1]-1,
                    use_one_step=False,
                )(net)
        # We want to use either Global Max Pooling or Global Average Pooling, but the esp-nn operator optimizations only benefit regular pooling operations

        if flags.max_pool:
            # net = stream.Stream(
            #     cell=tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))
            # )(net)
            net = tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))(net)
        else:
            net = tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))(net)
            # net = stream.Stream(
            #     cell=tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))
            # )(net)
    
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid")(net)
    
    # net = stream.Stream(cell=tf.keras.layers.Flatten())(net)
    # net = tf.keras.layers.Dense(1, activation="sigmoid")(net)
    
    return tf.keras.Model(input_audio, net)
