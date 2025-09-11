from src.models.layers import get_activation
from typing import Union, Optional

import tensorflow as tf

__all__ = ["FFNModule", "ConformerFFModule"]

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


@tf.keras.utils.register_keras_serializable(package=__name__)
class ConformerFFModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
        fc_factor: float = 0.5,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "conformer_ff_module",
        **kwargs,
    ):
        super(ConformerFFModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            name = f"{name}_ln",
            gamma_regularizer = kernel_regularizer,
            beta_regularizer = kernel_regularizer,
        )
        self.ffn1 = tf.keras.layers.Dense(
            units = 4 * input_dim,
            name = f"{name}_ffn1",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.swish1 = tf.keras.layers.Activation(tf.nn.swish, name = f"{name}_swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout, name = f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim,
            name = f"{name}_dense2",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name = f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name = f"{name}_add")

    def call(
            self,
            inputs,
            training = False,
    ):
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish1(outputs)
        outputs = self.do1(outputs, training = training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training = training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs
    
    def get_config(self):
        conf = super(ConformerFFModule, self).get_config()
        conf.update({"fc_factor": self.fc_factor})
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        conf.update(self.res_add.get_config())
        return 