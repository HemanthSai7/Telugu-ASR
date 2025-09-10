from src.utils import shape_util

import tensorflow as tf


class FreqMasking:
    def __init__(
        self,
        num_masks: int = 1,
        mask_factor: int = 27,
    ):
        self.num_masks = num_masks
        self.mask_factor = mask_factor

    @tf.function
    def augment(self, spectrogram: tf.Tensor):
        with tf.name_scope("freq_masking"):
            T, F, V = shape_util.shape_list(spectrogram)
            for _ in range(self.num_masks):
                f = tf.random.uniform([], minval=0, maxval = self.mask_factor, dtype=tf.int32)
                f = tf.minimum(f, F)
                f0 = tf.random.uniform([], minval=0, maxval=(F - f), dtype=tf.int32)
                mask = tf.concat(
                    [
                        tf.ones([T, f0, V], dtype=spectrogram.dtype),
                        tf.zeros([T, f, V], dtype=spectrogram.dtype),
                        tf.ones([T, F - f - f0, V], dtype=spectrogram.dtype),
                    ],
                    axis=1,
                )
                spectrogram = spectrogram * mask
            return spectrogram

class TimeMasking:
    def __init__(self, num_masks: int = 1, mask_factor: float = 100, p_upperbound: float = 1.0):
        self.num_masks = num_masks
        self.mask_factor = mask_factor
        self.p_upperbound = p_upperbound

    @tf.function
    def augment(self, spectrogram: tf.Tensor):
        with tf.name_scope("time_masking"):
            T, F, V = shape_util.shape_list(spectrogram, out_type=tf.int32) # Get the shape of the spectrogram
            for _ in range(self.num_masks):
                # Generate a random number from a uniform distribution
                t = tf.random.uniform([], minval=0, maxval=self.mask_factor, dtype=tf.int32) 
                # Multiply `t` by the p_upperbound. This is used to control the maximum masking length.
                t = tf.minimum(t, tf.cast(tf.cast(T, dtype=tf.float32) * self.p_upperbound, dtype=tf.int32))
                # Generate a random number from a uniform distribution between 0 and remaining time steps
                t0 = tf.random.uniform([], minval=0, maxval=(T - t), dtype=tf.int32)
                # Create a mask
                mask = tf.concat(
                    [
                        tf.ones([t0, F, V], dtype=spectrogram.dtype),
                        tf.zeros([t, F, V], dtype=spectrogram.dtype),
                        tf.ones([T - t0 - t, F, V], dtype=spectrogram.dtype),
                    ],
                    axis=0,
                )
                # Apply the mask to the spectrogram
                spectrogram = spectrogram * mask
            return spectrogram