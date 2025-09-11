from src.utils import math_util
from src.models.layers import FFNModule, MHSAModule
from typing import Optional, Union

import tensorflow as tf

__all__ = ["Encoder"]

@tf.keras.utils.register_keras_serializable(package=__name__)
class Conv1DSubsamplingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        model_dim: int = 288,
        subsampling_config: dict = None,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        name: str = "conv1d_subsampling",
        **kwargs,
    ):
        super(Conv1DSubsamplingLayer, self).__init__(name=name, **kwargs)
        self.filters = [model_dim, 2 * model_dim, model_dim]
        self.kernel_size = subsampling_config.get("kernel_size", [9, 9, 9])
        self.strides = subsampling_config.get("strides", [2, 2, 2])
        self.padding = subsampling_config.get("padding", ["same", "same", "same"])
        self.activations = subsampling_config.get("activations", ["tanh", "gelu", "gelu"])
        if len(self.kernel_size) != len(self.strides) or len(self.kernel_size) != len(self.padding) or len(self.kernel_size) != len(self.activations):
            raise ValueError("kernel_size, strides, padding, and activation must have the same length.")
        
        self.conv = []
        for i in range(len(self.kernel_size)):
            conv = tf.keras.layers.SeparableConv1D(
                filters=self.filters[i],
                kernel_size=self.kernel_size[i],
                strides=self.strides[i],
                padding=self.padding[i],
                activation=self.activations[i],
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_conv_{i+1}"
            )
            self.conv.append({"conv": conv})

    @staticmethod
    def lengths_to_padding_mask(lengths: int):
        _, max_len = tf.shape(lengths[0]), tf.reduce_max(lengths)
        range_row = tf.expand_dims(tf.range(max_len), axis=0)
        mask = range_row < tf.expand_dims(lengths, axis=-1)
        return tf.cast(mask, tf.float32)
    
    def call(self, inputs, training=False, mask=None):
        inputs = tf.squeeze(inputs, axis=-1)

        if mask is not None:
            mask = tf.cast(mask, tf.int32)
            mask = tf.reduce_max(mask, axis=-1)
            lengths = tf.reduce_sum(mask, axis=1)
        else:
            lengths = None

        for layer in self.conv:
            inputs = layer["conv"](inputs, training=training)
            if lengths is not None:
                lengths = math_util.get_conv_length(
                    lengths,
                    kernel_size=layer["conv"].kernel_size,
                    padding=layer["conv"].padding,
                    strides=layer["conv"].strides,
                )

        padding_mask = self.lengths_to_padding_mask(lengths) if lengths is not None else None
        return inputs, padding_mask
    
    def compute_output_shape(self, input_shape):
        # (BS, num_frames, num_bins, 1)
        input_shape = tf.TensorShape(input_shape).as_list()[:-1]
        bsz = input_shape[0]
        sequence_length = input_shape[1]
        
        current_sequence_dim = sequence_length
        for i in tf.range(len(self.kernel_size)):
            if current_sequence_dim is not None:
                current_sequence_dim = math_util.get_conv_length(
                    input_length = current_sequence_dim,
                    kernel_size = self.kernel_size[i],
                    padding = self.padding[i],
                    strides = self.strides[i]
                )
            else:
                current_sequence_dim = None
            
        output_feature_dim = self.filters[-1]
        return (bsz, current_sequence_dim, output_feature_dim)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dim": self.filters[-1],
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "strides": self.strides,
            "padding": self.padding,
            "name": self.name,
            "activations": self.activations,
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int = 288,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_heads: int = 8,
        head_dim: int = 36,
        fc_factor: int = 1,
        kernel_initializer: Union[str, callable] = None,
        bias_initializer: Union[str, callable] = None,
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "odv_encoder_block",
        **kwargs,
    ):
        super(EncoderBlock, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.mhsa = MHSAModule(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            output_shape=input_dim,
            return_attn_scores=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_mhsa_module",
        )
        self.ffn = FFNModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_ffn",
        )

    def call(self, inputs, training=False, use_causal_mask=False, mask=None):
        outputs = self.mhsa(inputs, training=training, use_causal_mask=use_causal_mask, mask=mask)
        outputs = self.ffn(outputs, training=training)
        return outputs
    
    def compute_output_shape(self, input_shape):
        # print(f"Input Decoder block shape: {input_shape}")
        outputs_shape = tf.TensorShape(input_shape)
        output_feature_dim = self.input_dim 
        # print(f"RETURN SHAPE: {outputs_shape[:-1].concatenate([output_feature_dim])}")
        return outputs_shape[:-1].concatenate([output_feature_dim])

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "dropout": self.mhsa.mha.dropout,
            "activation": self.ffn.activation,
            "num_heads": self.mhsa.mha.num_heads,
            "head_dim": self.mhsa.mha.key_dim,
            "fc_factor": self.ffn.fc_factor,
            "kernel_initializer": self.mhsa.mha.kernel_initializer,
            "bias_initializer": self.mhsa.mha.bias_initializer,
            "kernel_regularizer": self.mhsa.mha.kernel_regularizer,
            "bias_regularizer": self.mhsa.mha.bias_regularizer,
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class Encoder(tf.keras.Model):
    def __init__(
        self,
        input_dim: int = 288,
        subsampling_config: dict = None,
        activation: str = "gelu",
        num_blocks: int = 6,
        num_heads: int = 8,
        head_dim: int = 36,
        dropout: float = 0.1,
        fc_factor: int = 1,
        kernel_initializer: Union[str, callable] = None,
        bias_initializer: Union[str, callable] = None,
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "asr_encoder",
        **kwargs,
    ):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.activation = activation
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.fc_factor = fc_factor

        assert self.head_dim * self.num_heads == self.input_dim, "head_dim * num_heads must equal input_dim"

        self.conv_subsampling = Conv1DSubsamplingLayer(
            model_dim=input_dim,
            subsampling_config=subsampling_config,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_conv_subsampling",
            **kwargs,
        )

        self.encoder_blocks = [
            EncoderBlock(
                input_dim=input_dim,
                dropout=dropout,
                activation=activation,
                num_heads=num_heads,
                head_dim=head_dim,
                fc_factor=fc_factor,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_block_{i+1}"
            ) for i in range(num_blocks)
        ]

        self.encoder_input_padding_mask = None

    def call(self, inputs, training=False, use_causal_mask=False, mask=None):
        outputs, padding_mask = self.conv_subsampling(inputs, training=training, mask=mask)
        self.encoder_input_padding_mask = tf.identity(padding_mask)
        for block in self.encoder_blocks:
            outputs = block(
                outputs, 
                use_causal_mask=use_causal_mask, 
                mask=padding_mask,
                training=training,
            )
        return outputs
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        sequence_length = self.conv_subsampling.compute_output_shape(input_shape)[1]
        return (batch_size, sequence_length, self.input_dim)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "num_layers": self.num_blocks,
            "dropout": self.dropout,
            "activation": self.activation,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "fc_factor": self.fc_factor,
            "kernel_initializer": self.encoder_blocks[0].mhsa.mha.kernel_initializer,
            "bias_initializer": self.encoder_blocks[0].mhsa.mha.bias_initializer,
            "kernel_regularizer": self.encoder_blocks[0].mhsa.mha.kernel_regularizer,
            "bias_regularizer": self.encoder_blocks[0].mhsa.mha.bias_regularizer,
        })
        return config