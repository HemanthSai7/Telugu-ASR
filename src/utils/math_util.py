from src.utils import shape_util

import math
import tensorflow as tf

def get_num_batches(
    nsamples,
    batch_size,
    drop_remainders=True,
):
    if nsamples is None or batch_size is None:
        return None
    if drop_remainders:
        return math.floor(float(nsamples) / float(batch_size))
    return math.ceil(float(nsamples) / float(batch_size))

def log10(x):
    return tf.math.log(x) / tf.math.log(10.0)

def get_conv_length(input_length, kernel_size, padding, strides):
    length = input_length

    length = tf.cast(length, tf.float32)
    kernel_size = tf.cast(kernel_size, tf.float32)
    strides = tf.cast(strides, tf.float32)
    
    if padding == "same":
        length = tf.math.ceil(length / strides)
    elif padding == "valid":
        length = (length - kernel_size ) / strides + 1.0
            
    return tf.cast(length, tf.int32)

def merge_two_last_dims(x):
    """Reshape x by merging the two last dimensions

    Args:
        x (tf.Tensor): Input tensor

    Returns:
        tf.Tensor: Reshaped tensor
    """
    batch, _, features, channels = shape_util.shape_list(x)
    return tf.reshape(x, shape=[batch, -1, features * channels])

def find_max_length_prediction_tfarray(
    tfarray: tf.TensorArray,
) -> tf.Tensor:
    with tf.name_scope("find_max_length_prediction_tfarray"):
        index = tf.constant(0, dtype=tf.int32)
        total = tfarray.size()
        max_length = tf.constant(0, dtype=tf.int32)

        def condition(index, _):
            return tf.less(index, total)

        def body(index, max_length):
            prediction = tfarray.read(index)
            length = tf.shape(prediction)[0]
            max_length = tf.where(tf.greater(length, max_length), length, max_length)
            return index + 1, max_length

        index, max_length = tf.while_loop(condition, body, loop_vars=[index, max_length], swap_memory=False)
        return max_length


def pad_prediction_tfarray(
    tfarray: tf.TensorArray,
    blank: int or tf.Tensor,
) -> tf.TensorArray:
    with tf.name_scope("pad_prediction_tfarray"):
        index = tf.constant(0, dtype=tf.int32)
        total = tfarray.size()
        max_length = find_max_length_prediction_tfarray(tfarray) + 1

        def condition(index, _):
            return tf.less(index, total)

        def body(index, tfarray):
            prediction = tfarray.read(index)
            prediction = tf.pad(
                prediction,
                paddings=[[0, max_length - tf.shape(prediction)[0]]],
                mode="CONSTANT",
                constant_values=blank,
            )
            tfarray = tfarray.write(index, prediction)
            return index + 1, tfarray

        index, tfarray = tf.while_loop(condition, body, loop_vars=[index, tfarray], swap_memory=False)
        return tfarray