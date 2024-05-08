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

import tensorflow as tf

from microwakeword.layers import modes

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
        self.state_shape = []

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
    def get_input_state(self):
        return []

    def get_output_state(self):
        return []    
class StridedKeep(tf.keras.layers.Layer):
    """StridedDrop

    Drops the specified audio feature slices in nonstreaming mode only.
    Used for matching the dimensions of convolutions with valid padding.

    Attributes:
        time_sclices_to_drop: number of audio feature slices to drop
        mode: inference mode; e.g., non-streaming, internal streaming
    """

    def __init__(
        self, time_slices_to_keep, mode=modes.Modes.NON_STREAM_INFERENCE, **kwargs
    ):
        super(StridedKeep, self).__init__(**kwargs)
        self.time_slices_to_keep = max(time_slices_to_keep,1)
        self.mode = mode
        self.state_shape = []

    def call(self, inputs):
        if self.mode != modes.Modes.NON_STREAM_INFERENCE:
            return inputs[:, -self.time_slices_to_keep :, :, :]

        return inputs

    def get_config(self):
        config = {
            "time_slices_to_keep": self.time_slices_to_keep,
            "mode": self.mode,
        }
        base_config = super(StridedKeep, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def get_input_state(self):
        return []

    def get_output_state(self):
        return []    
