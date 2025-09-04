import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package=__name__)
class SwiGLU(tf.keras.layers.Layer):
    def __init__(
            self,
            axis = -1,
            name = "swiglu_activation",
            **kwargs,
    ):
        super(SwiGLU, self).__init__(name=name, **kwargs)
        self.axis = axis

    def call(self,inputs,**kwargs):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.silu(b)
        return tf.multiply(a, b)
    
    def get_config(self):
        config = super(SwiGLU, self).get_config()
        config.update({
            "axis": self.axis,
        })
        return config

def get_activation(name: str):
    activations = {
        "gelu": tf.keras.activations.gelu,
        "swiglu": SwiGLU(),
        "relu": tf.keras.activations.relu,
        "sigmoid": tf.keras.activations.sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Activation {name} not supported. Supported activations are: {list(activations.keys())}")
    return activations[name]