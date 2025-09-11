from src.models.layers.attention import MHSAModule, CrossAttentionModule
from src.models.layers.mlp import FFNModule
from typing import Optional, Union

import tensorflow as tf

__all__ = ["Decoder"]

@tf.keras.utils.register_keras_serializable(package=__name__)
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int = 288,
        dropout: float = 0.1,
        activation: str = "swiglu",
        num_heads: int = 8,
        head_dim: int = 36,
        fc_factor: int = 1,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = None,
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = None,
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "asr_decoder_block",
        **kwargs,
    ):
        super(DecoderBlock, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.masked_mhsa = MHSAModule(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            output_shape=input_dim,
            return_attn_scores=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_masked_mhsa",
        )
        self.cross_attention = CrossAttentionModule(
            d_model=input_dim, 
            num_heads=num_heads, 
            head_dim=head_dim, 
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_cross_attention"
        )
        self.ffn = FFNModule(
            input_dim=input_dim, 
            activation=activation, 
            fc_factor=fc_factor, 
            dropout=dropout,
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer, 
            name=f"{name}_ffn"
        )

    def call(self, inputs, training=False, use_causal_mask=True, mask=None):
        shifted_right_text_inputs, encoder_outputs = inputs
        attention_mask, cross_attention_mask = mask

        outputs = self.masked_mhsa(shifted_right_text_inputs, training=training, use_causal_mask=use_causal_mask, mask=attention_mask)
        outputs = self.cross_attention([outputs, encoder_outputs], training=training, mask=cross_attention_mask)
        outputs = self.ffn(outputs, training=training)
        return outputs
    
    def compute_output_shape(self, input_shape):
        print(f"Decoder Block Input Shape: {input_shape}")
        decoder_query_input_shape, _ = input_shape

        return (
            decoder_query_input_shape[0],  # batch size
            decoder_query_input_shape[1],  # sequence length
            self.input_dim  # output dimension
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "dropout": self.masked_mhsa.dropout,
            "activation": self.ffn.activation,
            "num_heads": self.masked_mhsa.num_heads,
            "head_dim": self.masked_mhsa.head_dim,
            "fc_factor": self.ffn.fc_factor,
            "kernel_initializer": self.masked_mhsa.kernel_initializer,
            "bias_initializer": self.masked_mhsa.bias_initializer,
            "kernel_regularizer": self.masked_mhsa.kernel_regularizer,
            "bias_regularizer": self.masked_mhsa.bias_regularizer,
        })
        return config

class Decoder(tf.keras.Model):
    def __init__(
        self,
        input_dim: int = 288,
        activation: str = "swiglu",
        num_blocks: int = 6,
        num_heads: int = 8,
        head_dim: int = 36,
        dropout: float = 0.1,
        fc_factor: int = 1,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = None,
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = None,
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "asr_decoder",
        **kwargs,
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.activation = activation
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.fc_factor = fc_factor

        assert self.head_dim * self.num_heads == self.input_dim, "head_dim * num_heads must equal input_dim"

        self.decoder_blocks = [
            DecoderBlock(
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
    
    def call(self, inputs, training=False, use_causal_mask=True, mask=None):
        shifted_right_text_inputs, encoder_outputs = inputs
        decoder_self_attention_mask, cross_attention_encoder_mask = mask

        outputs = shifted_right_text_inputs
        for block in self.decoder_blocks:
            outputs = block(
                [outputs, encoder_outputs],
                training=training,
                use_causal_mask=use_causal_mask,
                mask=[decoder_self_attention_mask, cross_attention_encoder_mask]
            )
        return outputs
            

    def compute_output_shape(self, input_shape):
        # tf.print("ASR DECODER INPUT SHAPE", input_shape)
        decoder_query_input_shape, _ = input_shape

        return (decoder_query_input_shape[0], decoder_query_input_shape[1], self.input_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "activation": self.activation,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "fc_factor": self.fc_factor,
            "kernel_initializer": self.decoder_blocks[0].masked_mhsa.kernel_initializer,
            "bias_initializer": self.decoder_blocks[0].masked_mhsa.bias_initializer,
            "kernel_regularizer": self.decoder_blocks[0].masked_mhsa.kernel_regularizer,
            "bias_regularizer": self.decoder_blocks[0].masked_mhsa.bias_regularizer,
        })
        return config