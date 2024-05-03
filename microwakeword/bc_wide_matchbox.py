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

"""Model based on 1D depthwise and 1x1 convolutions in time + residual."""
from microwakeword.layers import modes
from microwakeword.layers import stream
from microwakeword.layers import strided_drop
from microwakeword.layers import sub_spectral_normalization

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
        "--activation", type=str, default="relu", help="activation function"
    )
    parser_nn.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Percentage of data dropped",
    )
    parser_nn.add_argument(
        "--dropout_final_layer",
        type=float,
        default=0.0,
        help="Percentage of data dropped before final convolution layer",
    )
    parser_nn.add_argument(
        "--ds_filters",
        type=str,
        default="128, 64, 64, 64, 128, 128",
        help="Number of filters in every residual block's branch pointwise convolutions",
    )
    parser_nn.add_argument(
        "--ds_filters2",
        type=str,
        default="128, 64, 64, 64, 128, 128",
        help="Number of filters in every residual block's final pointwise convolution",
    )
    parser_nn.add_argument(
        "--ds_repeat",
        type=str,
        default="1, 1, 1, 1, 1, 1",
        help="Number of repeating conv blocks inside of residual block",
    )
    parser_nn.add_argument(
        "--ds_residual",
        type=str,
        default="0, 1, 1, 1, 0, 0",
        help="Apply/not apply branching in the residual block residual block",
    )
    parser_nn.add_argument(
        "--ds_padding",
        type=str,
        default="'valid', 'valid', 'valid', 'valid', 'valid', 'valid'",
        help="padding can be same or causal, causal should be used for streaming",
    )
    parser_nn.add_argument(
        "--ds_kernel_size",
        type=str,
        default="11, 13, 15, 17, 29, 1",
        help="Kernel size of DepthwiseConv1D in time dim for every residual block",
    )
    parser_nn.add_argument(
        "--ds_stride",
        type=str,
        default="1, 1, 1, 1, 1, 1",
        help="stride value in time dim of DepthwiseConv1D for residual block",
    )
    parser_nn.add_argument(
        "--ds_dilation",
        type=str,
        default="1, 1, 1, 1, 2, 1",
        help="dilation value of DepthwiseConv1D for every residual block",
    )
    parser_nn.add_argument(
        "--ds_pool",
        type=str,
        default="1, 1, 1, 1, 1, 1",
        help="Apply pooling after every residual block: pooling size",
    )
    parser_nn.add_argument(
        "--ds_max_pool",
        type=int,
        default=0,
        help="Pooling type: 0 - average pooling; 1 - max pooling",
    )
    parser_nn.add_argument(
        "--ds_scale",
        type=int,
        default=1,
        help="apply scaling in batch normalization layer",
    )
    parser_nn.add_argument(
        "--max_pool",
        type=int,
        default=0,
        help="apply max pool instead of aver4age pool before final convolution and sigmoid activation",
    )
    parser_nn.add_argument(
        "--freq_stride",
        type=str,
        default="1,1,1,1,1,1",
        help="Stride to apply in frequency dimensions",
    )
    parser_nn.add_argument(
        "--first_conv_filters",
        type=int,
        default=16,
        help="Stride to apply in frequency dimensions",
    )


def spectrogram_slices_dropped(flags):
    """Computes the number of spectrogram slices dropped due to valid padding.

    Args:
        flags: data/model parameters

    Returns:
        int: number of spectrogram slices dropped
    """
    spectrogram_slices_dropped = 0

    for kernel_size, dilation, residual, repeat in zip(
        parse(flags.ds_kernel_size),
        parse(flags.ds_dilation),
        parse(flags.ds_residual),
        parse(flags.ds_repeat),
    ):
        if residual:
            spectrogram_slices_dropped += (repeat - 1) * dilation * (kernel_size - 1)
        else:
            spectrogram_slices_dropped += repeat * dilation * (kernel_size - 1)

    return spectrogram_slices_dropped #+ 4   # plus 4 for the initial 5x5 convolution


def depthwise_branch_block(
    inputs,
    repeat,
    kernel_size,
    filters,
    dilation,
    stride,
    flags,
    dropout,
    padding="valid",
):
    activation = flags.activation
    scale = flags.ds_scale

    branch = inputs

    if repeat > 0:
        for _ in range(repeat):
            branch = stream.Stream(
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(kernel_size, 1),
                    strides=(stride, stride),
                    padding="valid",
                    dilation_rate=(dilation, 1),
                    use_bias=False,
                ),
                use_one_step=False,
                pad_time_dim=padding,
            )(branch)
            branch = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(branch)
            branch = tf.keras.layers.BatchNormalization(scale=scale)(branch)
            branch = tf.keras.layers.Activation(activation)(branch)
            branch = tf.keras.layers.SpatialDropout2D(rate=dropout)(branch)
    else:
        branch = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, use_bias=False, padding="same"
        )(branch)
        branch = tf.keras.layers.BatchNormalization(scale=scale)(branch)
        branch = tf.keras.layers.Activation(activation)(branch)
        branch = tf.keras.layers.SpatialDropout2D(rate=dropout)(branch)

    return branch


def resnet_block(
    inputs,
    repeat,
    kernel_size,
    filters,
    filters2,
    dilation,
    stride,
    flags,
    freq_stride,
    branch=False,
    padding="valid",
):
    """Residual block.

    It is based on paper
    Jasper: An End-to-End Convolutional Neural Acoustic Model
    https://arxiv.org/pdf/1904.03288.pdf

    Args:
      inputs: input tensor
      repeat: number of repeating DepthwiseConv1D and Conv1D block
      kernel_size: kernel size of DepthwiseConv1D in time dim
      filters: number of filters in DepthwiseConv1D and Conv1D
      dilation: dilation in time dim for DepthwiseConv1D
      stride: stride in time dim for DepthwiseConv1D
      filter_separable: use separable conv or standard conv
      flags: model parameters
      residual: if True residual connection is added
      padding: can be 'same', 'causal', or 'valid'

    Returns:
      output tensor

    Raises:
      ValueError: if padding has invalid value
    """
    if branch and (padding not in ("same", "causal", "valid")):
        raise ValueError("padding should be same, causal, or valid")

    dropout = flags.dropout
    activation = flags.activation
    scale = flags.ds_scale  # apply scaling in batchnormalization layer

    net = inputs
    residual = tf.zeros(net.shape)
    identity = tf.zeros(net.shape)            
    
    # if filters2 != net.shape[3]:
    #     # print("applying transition block")
    #     net, residual, identity = depthwise_broadcast(net, residual, identity, freq_stride, filters2)

    #     # identity = tf.zeros(net.shape)
    # else:
    net, residual, identity = depthwise_broadcast(net, residual, identity, freq_stride)    

    if branch:
        branches = []
        for branch_repeat in range(1,repeat):
            branch = net
            pointwise_filters = filters
            branch = depthwise_branch_block(
                branch,
                branch_repeat,
                kernel_size,
                pointwise_filters,
                dilation,
                stride,
                flags,
                dropout,
                padding,
            )
            branches.append(branch)

        dropped_branches = []
        for branch in branches:
            features_drop = branch.shape[1] - branches[-1].shape[1]
            dropped_branches.append(strided_drop.StridedDrop(features_drop)(branch))

        net = tf.keras.layers.concatenate(dropped_branches)
        net = tf.keras.layers.Conv2D(
            filters=filters2, kernel_size=1, use_bias=False, padding="same"
        )(net)
        net = tf.keras.layers.BatchNormalization(scale=scale)(net)
        net = tf.keras.layers.Activation(activation)(net)
    else:
        net = depthwise_branch_block(
            net, repeat, kernel_size, filters, dilation, stride, flags, dropout, padding
        )

    if (repeat > 0):
        features_drop = residual.shape[1] - net.shape[1]
        residual = strided_drop.StridedDrop(features_drop)(residual)
        identity = strided_drop.StridedDrop(identity.shape[1] - net.shape[1])(identity)
        print("before addition net,residual,identity=", net.shape, residual.shape, identity.shape)
        
        if net.shape[3] != residual.shape[3]:
            net = tf.keras.layers.Conv2D(
                kernel_size=(1, 1),
                padding="same",
                strides=(1, 1),
                # strides=(1, freq_stride),
                filters=residual.shape[3],
            )(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Activation("relu")(net)
            identity = tf.zeros(net.shape)                   

        # net = tf.keras.layers.Add()([net,residual,identity])
        # net = net + residual + identity
        # net = tf.keras.layers.Add()([net,residual])
        # broadcasted_net =  tf.broadcast_to(net, residual.shape)
        # new_net = tf.zeros(residual.shape)
        # new_net = []
        # for i in range(residual.shape[2]):
        #     new_net.append(net[:,:,0,:])
        # #     new_net.append(net[:,:,0,:] + residual[:,:,i,:])
        # net =tf.stack(new_net, axis=2)
        
        # net = tf.tile(net, [1, 1, residual.shape[2], 1])
            
        net = net + residual #+ identity

    return net


def depthwise_broadcast(input, old_residual, old_identity, freq_stride, filters=None):
    net = input
    identity = net

    if freq_stride > 1:
        net = tf.keras.layers.AveragePooling2D(pool_size=(1,freq_stride), padding="valid",strides=(1,freq_stride))(net)

    # if filters is not None:
    # net = net + old_residual
    net = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1, 3), strides=1, padding="same"
    )(net)
    # net = stream.Stream(
    #     tf.keras.layers.DepthwiseConv2D(
    #         kernel_size=3,
    #         strides=1,
    #         padding="valid",
    #         dilation_rate=1,
    #         use_bias=False,
    #     ),
    #     use_one_step=False,
    #     pad_freq_dim="same",
    # )(net)    
    net = sub_spectral_normalization.SubSpectralNormalization(1)(net)
    # net = sub_spectral_normalization.SubSpectralNormalization(5)(net)
    residual = net

    # Combines all features into 1
    net = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1,net.shape[2]), padding="valid"
    )(net)
    
    return net, residual, identity


def model(flags, shape, batch_size):
    """MatchboxNet model.

    It is based on paper
    MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network
    Architecture for Speech Commands Recognition
    https://arxiv.org/pdf/2004.08531.pdf
    Args:
      flags: data/model parameters
      config: dictionary containing microWakeWord training configuration

    Returns:
      Keras model for training
    """

    ds_filters = parse(flags.ds_filters)
    ds_filters2 = parse(flags.ds_filters2)
    ds_repeat = parse(flags.ds_repeat)
    ds_kernel_size = parse(flags.ds_kernel_size)
    ds_stride = parse(flags.ds_stride)
    ds_dilation = parse(flags.ds_dilation)
    ds_residual = parse(flags.ds_residual)
    ds_pool = parse(flags.ds_pool)
    ds_padding = parse(flags.ds_padding)
    freq_stride = parse(flags.freq_stride)

    for l in (
        ds_filters,
        ds_filters2,
        ds_repeat,
        ds_kernel_size,
        ds_stride,
        ds_dilation,
        ds_residual,
        ds_pool,
        ds_padding,
        freq_stride,
    ):
        if len(ds_filters) != len(l):
            raise ValueError("all input lists have to be the same length")

    input_audio = tf.keras.layers.Input(
        shape=shape,
        batch_size=batch_size,
    )
    net = input_audio

    # make it [batch, time, feature, 1]
    net = tf.keras.backend.expand_dims(net, axis=3)
    net = stream.Stream(
        tf.keras.layers.Conv2D(
            kernel_size=3,
            strides=(1,2),
            padding="valid",
            dilation_rate=(1, 1),
            use_bias=False,
            filters=flags.first_conv_filters,
        ),
        use_one_step=False,
        pad_time_dim="valid",
        pad_freq_dim="same",
    )(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.ds_scale)(net)
    net = tf.keras.layers.Activation("relu")(net)

    # encoder
    for (
        filters,
        filters2,
        repeat,
        ksize,
        stride,
        dilation,
        res,
        pool,
        pad,
        freq_stride,
    ) in zip(
        ds_filters,
        ds_filters2,
        ds_repeat,
        ds_kernel_size,
        ds_stride,
        ds_dilation,
        ds_residual,
        ds_pool,
        ds_padding,
        freq_stride,
    ):

        # features_drop = residual.shape[1] - net.shape[1]
        # residual = strided_drop.StridedDrop(features_drop)(residual)
        # # identity = strided_drop.StridedDrop(features_drop)(identity)
        # # net = stream.Stream(
        # #     cell=tf.identity,
        # #     ring_buffer_size_in_time_dim=net.shape[1],
        # #     use_one_step=False,
        # #     pad_time_dim=None)(net)              
        # # residual = stream.Stream(
        # #     cell=tf.identity,
        # #     ring_buffer_size_in_time_dim=net.shape[1],
        # #     use_one_step=False,
        # #     pad_time_dim=None)(residual)              
        # # identity = stream.Stream(
        # #     cell=tf.identity,
        # #     ring_buffer_size_in_time_dim=net.shape[1],
        # #     use_one_step=False,
        # #     pad_time_dim=None)(identity)        
        # print("before addition net,residual,identity=", net.shape, residual.shape, identity.shape)                  

        # # net = tf.keras.layers.Add()([net,residual,identity])
        # # net = net + residual + identity
        # # net = tf.keras.layers.Add()([net,residual])        
        # net = net + residual
        # if filters2 != net.shape[3]:
        #     # print("applying transition block")
        #     net, residual, identity = depthwise_broadcast(net, freq_stride, filters2)

        #     # identity = tf.zeros(net.shape)
        # else:
        #     net, residual, identity = depthwise_broadcast(net, freq_stride)
      

        net = resnet_block(
            net, repeat, ksize, filters, filters2, dilation, stride, flags, freq_stride, res, pad
        )

        if pool > 1:
            if flags.ds_max_pool:
                net = tf.keras.layers.MaxPooling2D(
                    pool_size=(pool, 1), strides=(pool, 1)
                )(net)
            else:
                net = tf.keras.layers.AveragePooling2D(
                    pool_size=(pool, 1), strides=(pool, 1)
                )(net)

    # We want to use either Global Max Pooling or Global Average Pooling, but the esp-nn operator optimizations only benefit regulr pooling operations
    if net.shape[1] > 1:
        if flags.max_pool:
            net = stream.Stream(
                cell=tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))
            )(net)
        else:
            net = stream.Stream(
                cell=tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))
            )(net)

    net = tf.keras.layers.Dropout(rate=flags.dropout_final_layer)(net)
    net = tf.keras.layers.Conv2D(filters=1, kernel_size=1, use_bias=False)(net)

    net = tf.squeeze(net, [1, 2])
    net = tf.keras.layers.Activation("sigmoid")(net)

    return tf.keras.Model(input_audio, net)
