from typing import Tuple
import jiwer
import tensorflow as tf

def execute_wer(decode, target):
    """Execute WER calculation using jiwer."""
    total_wer = 0.0
    total_length = 0.0
    
    for dec, tar in zip(decode, target):
        wer_score = jiwer.wer(tar, dec)
        word_count = len(tar.split())
        total_wer += wer_score * word_count
        total_length += word_count
    
    return tf.convert_to_tensor(total_wer, tf.float32), tf.convert_to_tensor(total_length, tf.float32)


def wer(
    decode: tf.Tensor,
    target: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Word Error Rate

    Args:
        decode (tf.Tensor): array of prediction texts
        target (tf.Tensor): array of groundtruth texts

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of words) of each text
    """
    return tf.numpy_function(execute_wer, inp=[decode, target], Tout=[tf.float32, tf.float32])


def execute_cer(decode, target):
    """Execute CER calculation using jiwer."""
    total_cer = 0.0
    total_length = 0.0
    
    for dec, tar in zip(decode, target):
        cer_score = jiwer.cer(tar, dec)
        char_count = len(tar)
        total_cer += cer_score * char_count
        total_length += char_count
    
    return tf.convert_to_tensor(total_cer, tf.float32), tf.convert_to_tensor(total_length, tf.float32)


def cer(
    decode: tf.Tensor,
    target: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Character Error Rate

    Args:
        decode (tf.Tensor): array of prediction texts
        target (tf.Tensor): array of groundtruth texts

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of characters) of each text
    """
    return tf.numpy_function(execute_cer, inp=[decode, target], Tout=[tf.float32, tf.float32])


def tf_cer(
    decode: tf.Tensor,
    target: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Tensorflow Character Error Rate

    Args:
        decode (tf.Tensor): tensor shape [B]
        target (tf.Tensor): tensor shape [B]

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of characters) of each text
    """
    # Convert strings to character sequences (handles Unicode properly)
    decode_chars = tf.strings.unicode_split(decode, 'UTF-8')
    target_chars = tf.strings.unicode_split(target, 'UTF-8') 
    
    distances = tf.edit_distance(decode_chars.to_sparse(), target_chars.to_sparse(), normalize=False)
    lengths = tf.cast(target_chars.row_lengths(), dtype=tf.float32)
    
    return tf.reduce_sum(distances), tf.reduce_sum(lengths)