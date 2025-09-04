from src.model.layers import get_activation
from typing import Union, Optional

import tensorflow as tf

__all__ = ["FFNModule"]

@tf.keras.utils.register_keras_serializable(package=__name__)
class FFNModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.0,
        fc_factor: int = 4,
        activation: str = "gelu",
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "ffn_module",
        **kwargs,
    ):
        super(FFNModule, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.fc_factor = fc_factor
        self.dropout = dropout
        self.activation = get_activation(activation)

        self.ln = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            name=f"{name}_ln",
        )
        self.dense1 = tf.keras.layers.Dense(
            units=self.input_dim * fc_factor,
            activation=self.activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_dense1"
        )
        self.do = tf.keras.layers.Dropout(rate=dropout, name=f"{name}_dropout")
        self.dense2 = tf.keras.layers.Dense(
            units=self.input_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_dense2"
        )
        self.res_add = tf.keras.layers.Add(name=f"{name}_residual_add")

    def call(self, inputs, training=False):
        outputs = self.dense1(inputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.dense2(outputs)
        outputs = self.res_add([outputs, inputs])
        outputs = self.ln(outputs, training=training)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "fc_factor": self.fc_factor,
            "dropout": self.dropout,
            "activation": self.activation.__name__,
            "kernel_initializer": self.dense1.kernel_initializer,
            "bias_initializer": self.dense1.bias_initializer,
            "kernel_regularizer": self.dense1.kernel_regularizer,
            "bias_regularizer": self.dense1.bias_regularizer,
        })
        return config