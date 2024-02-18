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
