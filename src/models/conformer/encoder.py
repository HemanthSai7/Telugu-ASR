from src.utils import shape_util, math_util
from typing import Union, Optional
from src.models.layers import ConformerFFModule, get_activation, PositionalEncoding, MHSAModule

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class Conv2dSubsampling(tf.keras.layers.Layer):
    """Convolutional 2D subsampling (to 1/4 length) layer"""
    def __init__(
        self,
        subsampling_config: dict,
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_initializer=None,
        bias_initializer=None,
        name="Conv2dSubsampling",
        **kwargs,
    ):
        super(Conv2dSubsampling, self).__init__(name=name, **kwargs)
        self.filter = subsampling_config.get("filters", 128)
        self.kernel_size = subsampling_config.get("kernel_size", 3)
        self.stride = subsampling_config.get("strides", 2)
        self.padding = subsampling_config.get("padding", "same")
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filter,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filter,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="same",
            name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )

    def call(
            self,
            inputs,
            training=False,
            **kwargs,
    ):
        outputs, outputs_length = inputs
        outputs = self.conv1(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs_length = math_util.get_conv_length(
            outputs_length,
            kernel_size=self.conv1.kernel_size[0],
            padding="same",
            strides=self.conv1.strides[0],
        )
        outputs = math_util.merge_two_last_dims(outputs)
        return outputs, outputs_length
    
    def get_config(self):
        conf = super(Conv2dSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        return conf


@tf.keras.utils.register_keras_serializable(package=__name__)
class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        kernel_size: int = 31,
        dropout: float = 0.0,
        scale_factor: int = 2,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "conv_module",
        **kwargs,
    ):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.pw_conv_1 = tf.keras.layers.Conv1D(
            filters = scale_factor * input_dim,
            kernel_size = 1,
            strides = 1,
            padding = "valid",
            name = f"{name}_pw_conv_1",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.glu = get_activation("glu")
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size = kernel_size,
            strides = 1,
            padding = "same",
            name = f"{name}_dw_conv",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name = f"{name}_bn",
            gamma_regularizer = kernel_regularizer,
            beta_regularizer = kernel_regularizer,
        )
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name = f"{name}_swish_activation")
        self.pw_conv_2 = tf.keras.layers.Conv1D(
            filters = input_dim,
            kernel_size = 1,
            strides = 1,
            padding = "valid",
            name = f"{name}_pw_conv_2",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(rate=dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_residual_add")

    def call(
            self,
            inputs,
            training = False,
            **kwargs
    ):
        outputs = self.ln(inputs, training=training)
        B, T, E = shape_util.shape_list(outputs)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = tf.reshape(outputs, [B, T, E])
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs
    
    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.0,
        fc_factor: float = 0.5,
        head_dim: int = 36,
        num_heads: int = 4,
        attention_type: str = "relmha",
        kernel_size: int = 31,
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        name: str = "conformer_block",
        **kwargs,
    ):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = ConformerFFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name=f"{name}_ffm1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.mhsam = MHSAModule(
            attention_type=attention_type,
            head_dim=head_dim,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{name}_mhsa_module",
        )
        self.convm = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{name}_conv_module",
        )
        self.ffm2 = ConformerFFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name=f"{name}_ff_module2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer,
        )

    def call(
            self,
            inputs,
            training = False,
            mask = None,
            **kwargs,
    ):
        inputs, pos = inputs
        outputs = self.ffm1(inputs, training=training, **kwargs)
        outputs = self.mhsam([outputs, pos], training=training, mask=mask, **kwargs)
        outputs = self.convm(outputs, training=training, **kwargs)
        outputs = self.ffm2(outputs, training=training, **kwargs)
        outputs = self.ln(outputs, training=training)
        return outputs
    
    def get_config(self):
        conf = super(ConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf
    
class ConformerEncoder(tf.keras.Model):
    def __init__(
        self,
        subsampling_config: Optional[dict],
        d_model: int = 256,
        num_blocks: int = 16,
        head_dim: int = 36,
        num_heads: int = 4,
        attention_type: str = "relmha",
        kernel_size: int = 31,
        fc_factor: float = 0.5,
        dropout: float = 0.0,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "conformer_encoder",
        **kwargs,
    ):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)

        self.conv_subsampling = Conv2dSubsampling(
            subsampling_config=subsampling_config,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.positional_encoding = PositionalEncoding(name=f"{name}_pe")
        self.linear = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_linear"
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=d_model,
                dropout=dropout,
                fc_factor=fc_factor,
                head_dim=head_dim,
                num_heads=num_heads,
                attention_type=attention_type,
                kernel_size=kernel_size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f"{name}_block_{i}"
            )
            self.conformer_blocks.append(conformer_block)

    def call(
            self,
            inputs,
            training = False,
            mask = None,
            **kwargs,
    ):
        # inputs with shape [BS, T, V1, V2]
        outputs, outputs_length = inputs
        outputs, outputs_length = self.conv_subsampling([outputs, outputs_length], training=training)
        outputs = self.linear(outputs, training=training)
        pe = self.positional_encoding(outputs)
        outputs = self.do(outputs, training=training)
        for cblock in self.conformer_blocks:
            outputs = cblock([outputs, pe], training=training, mask=mask, **kwargs)
        return outputs, outputs_length
    
    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        conf.update(self.pe.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf
            