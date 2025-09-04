from src.model.layers.positional_encoding import RoPEPositionalEncoding
from typing import Optional, Union

import tensorflow as tf

__all__ = ["MHSAModule"]

@tf.keras.utils.register_keras_serializable(package=__name__)
class MultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        dropout: float = 0.0,
        use_bias: bool = False,
        output_shape: int = None,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "multi_head_attention",
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            use_bias=use_bias,
            output_shape=output_shape,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name,
            **kwargs,
        )
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.rope_pos_emb = RoPEPositionalEncoding(name="rope_positional_encoding", head_dim=key_dim)
        
    def build(self, input_shape):
        d_model = input_shape[-1] 

        self.query_kernel = self.add_weight(
            shape = [d_model, self._num_heads * self._key_dim],
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            name = "query_kernel",
            trainable = True,
        )
        self.key_kernel = self.add_weight(
            shape = [d_model, self._num_heads * self._key_dim],
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            name = "key_kernel",
            trainable = True,
        )
        self.value_kernel = self.add_weight(
            shape = [d_model, self._num_heads * self._key_dim],
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            name = "value_kernel",
            trainable = True,
        )
        self.projection_kernel = self.add_weight(
            shape = [self._num_heads * self._key_dim, self._output_shape],
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            name = "projection_kernel",
            trainable = True,
        )
        self._build_attention(rank=4)
        super().build(input_shape)

    def call_qkv(self, query, value, key):
        if key.shape[-1] != value.shape[-1]:
            raise ValueError("Key and value must have the same last dimension.")
        
        batch_size = tf.shape(query)[0]
        seq_len_q = tf.shape(query)[1]
        seq_len_v = tf.shape(value)[1]
        seq_len_k = tf.shape(key)[1]
        
        query = tf.matmul(query, self.query_kernel)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads * head_dim)
        value = tf.matmul(value, self.value_kernel)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads * head_dim)
        key = tf.matmul(key, self.key_kernel) if key is not None else value  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads * head_dim)

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        query = tf.reshape(query, (batch_size, seq_len_q, self._num_heads, self._key_dim)) # (batch_size, seq_len, num_heads * head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        value = tf.reshape(value, (batch_size, seq_len_v, self._num_heads, self._key_dim)) # (batch_size, seq_len, num_heads * head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        key = tf.reshape(key, (batch_size, seq_len_k, self._num_heads, self._key_dim)) # (batch_size, seq_len, num_heads * head_dim) -> (batch_size, seq_len, num_heads, head_dim)

        return query, value, key
    
    def compute_attention(self, query, value, key, attention_mask=None, training=False):
        # query shape: (batch_size, num_heads, S_q, head_dim)
        # key shape: (batch_size, num_heads, S_k, head_dim)
        query = tf.multiply(query, 1.0 / tf.math.sqrt(tf.cast(self._key_dim, query.dtype)))  # Scale query
        attention_scores = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, S_q, head_dim) @ (batch_size, num_heads, head_dim, S_k) -> (batch_size, num_heads, S_q, S_k)
        attention_scores = self._masked_softmax(
            attention_scores,
            attention_mask=attention_mask,
        )  # Apply softmax to attention scores
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_scores_dropout = self._dropout_layer(
        #     attention_scores, training=training
        # )
        # attention_scores = attention_scores * tf.expand_dims(tf.cast(attention_mask, tf.float32), axis=1)
        # attention_scores = tf.where(tf.expand_dims(tf.cast(attention_mask, tf.float32), axis=1) > 0, attention_scores, 0.0)
        attention_output = tf.matmul(attention_scores, value) # (batch_size, num_heads, S_q, S_k) @ (batch_size, num_heads, S_k, head_dim) -> (batch_size, num_heads, S_q, head_dim)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3]) # (batch_size, num_heads, S_q, head_dim) -> (batch_size, S_q, num_heads, head_dim)
        attention_output = tf.reshape(attention_output, [tf.shape(attention_output)[0], -1, self._num_heads * self._key_dim]) # (batch_size, S_q, num_heads, head_dim) -> (batch_size, S_q, num_heads * head_dim)
        attention_output = tf.matmul(attention_output, self.projection_kernel) # (batch_size, S_q, num_heads * head_dim) -> (batch_size, S_q, output_shape)
        return attention_output, attention_scores
    
    def compute_attention_mask(
        self,
        query,
        value,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_causal_mask=False,
    ):
        auto_mask = None
        if query_mask is not None:
            query_mask = tf.cast(query_mask, "bool")  # defensive casting
            # B = batch size, T = max query length
            auto_mask = tf.expand_dims(query_mask, -1)  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = tf.cast(value_mask, "bool")  # defensive casting
            # B = batch size, S == max value length
            mask = tf.expand_dims(value_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = tf.cast(key_mask, "bool")  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = tf.expand_dims(key_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self.compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, "bool")
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else attention_mask & auto_mask
            )
        return attention_mask
    
    def compute_causal_mask(self, query, value):
        q_seq_length = tf.shape(query)[2]
        v_seq_length = tf.shape(value)[2]
        
        # More memory efficient for large sequences
        if tf.equal(q_seq_length, v_seq_length):
            # Use more efficient triangular mask for square matrices
            mask = tf.linalg.band_part(tf.ones((q_seq_length, v_seq_length), dtype=tf.bool), -1, 0)
            return tf.expand_dims(mask, 0)  # Add batch dimension
        else:
            return tf.linalg.band_part(
                tf.ones((1, q_seq_length, v_seq_length), dtype=tf.bool), -1, 0
            )

    def call(
        self,
        query,
        value,
        key,
        attention_mask=None,
        key_value_mask=None,
        use_causal_mask=False,
        return_attention_scores=False,
        training=False,
        **kwargs,
    ):
        # query,value,key shape: (batch_size, seq_len, d_model)
        query, value, key = self.call_qkv(query, value, key)

        query = self.rope_pos_emb(query, training=training) # (batch_size, seq_len, num_heads, head_dim)
        key = self.rope_pos_emb(key, training=training) # (batch_size, seq_len, num_heads, head_dim)

        query = tf.transpose(query, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        value = tf.transpose(value, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        key = tf.transpose(key, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)

        if key_value_mask is not None: # cross attention case
            computed_attention_mask = self.compute_attention_mask(
                query=query,
                value=value,
                query_mask=None,
                value_mask=key_value_mask,
                key_mask=key_value_mask,
                use_causal_mask=use_causal_mask,
            )
        elif attention_mask is not None:  # Causal mask case
            computed_attention_mask = self.compute_attention_mask(
                query=query,
                value=value,
                query_mask=attention_mask,
                value_mask=attention_mask,
                key_mask=attention_mask,
                use_causal_mask=use_causal_mask,
            )
        else:
            computed_attention_mask = None


        attention_output, attention_scores = self.compute_attention(
            query=query,
            value=value,
            key=key, 
            attention_mask=computed_attention_mask,
            training=training,
        )
        if return_attention_scores:
            return attention_output, attention_scores
        
        return attention_output
    
    def compute_output_shape(self, input_shape):
        return self.compute_output_shape(query_shape=input_shape, value_shape=input_shape, key_shape=input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "dropout": self._dropout,
            "use_bias": self._use_bias,
            "output_shape": self._output_shape,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

@tf.keras.utils.register_keras_serializable(package=__name__)
class MHSAModule(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.0,
        output_shape: int = None,
        return_attn_scores: bool = False,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        name: str = "mhsa_module",
        **kwargs,
    ):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        self.return_attn_scores = return_attn_scores
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            dropout=dropout,
            output_shape=output_shape,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(
            rate=dropout,
            name=f"{name}_dropout",
        )
        self.res_add = tf.keras.layers.Add(name=f"{name}_residual_add")

    def call(self, inputs, training=False, use_causal_mask=False, mask=None):
        outputs = self.mha(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
            use_causal_mask=use_causal_mask, 
            attention_mask=mask,
            return_attention_scores=self.return_attn_scores
        )
        outputs = self.res_add([inputs, outputs])
        return self.ln(outputs)        
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        feature_dim = input_shape[2]
        return (batch_size, sequence_length, feature_dim)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.mha.num_heads,
            "dropout": self.mha.dropout,
            "output_shape": self.mha.output_shape,
            "return_attn_scores": self.mha.return_attn_scores,
            "kernel_initializer": self.mha.kernel_initializer,
            "bias_initializer": self.mha.bias_initializer,
            "kernel_regularizer": self.mha.kernel_regularizer,
            "bias_regularizer": self.mha.bias_regularizer,
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class CrossAttentionModule(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        name: str = "cross_attention_module",
        **kwargs,
    ):
        super(CrossAttentionModule, self).__init__(name=name, **kwargs)
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            dropout=dropout,
            output_shape=d_model,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_mha",
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(
            rate=dropout,
            name=f"{name}_dropout",
        )
        self.res_add = tf.keras.layers.Add(name=f"{name}_residual_add")

    def call(self, inputs, training=False, mask=None):
        decoder_query, encoder_key_value = inputs
        
        outputs = self.mha(
            decoder_query,
            encoder_key_value,
            encoder_key_value,
            training=training,
            key_value_mask=mask,
        )
        outputs = self.res_add([decoder_query, outputs])
        return self.ln(outputs)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        feature_dim = input_shape[2]
        return (batch_size, sequence_length, feature_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.mha.output_shape,
            "num_heads": self.mha.num_heads,
            "dropout": self.mha.dropout,
            "kernel_initializer": self.mha.kernel_initializer,
            "bias_initializer": self.mha.bias_initializer,
            "kernel_regularizer": self.mha.kernel_regularizer,
            "bias_regularizer": self.mha.bias_regularizer,
        })
        return config