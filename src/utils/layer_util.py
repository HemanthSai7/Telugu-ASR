import tensorflow as tf


def get_rnn(
        rnn_type: str,
):
    assert rnn_type in ["lstm", "gru", "rnn"]
    rnn_layers = {
        "lstm": tf.keras.layers.LSTM,
        "gru": tf.keras.layers.GRU,
        "rnn": tf.keras.layers.SimpleRNN,
    }
    return rnn_layers[rnn_type]

def get_conv(
        conv_type: str,
):
    assert conv_type in ["conv1d", "conv2d"]
    conv_layers = {
        "conv1d": tf.keras.layers.Conv1D,
        "conv2d": tf.keras.layers.Conv2D,
    }
    return conv_layers[conv_type]
    