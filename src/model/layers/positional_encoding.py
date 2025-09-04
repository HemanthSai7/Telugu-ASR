from src.utils.shape_util import shape_list

import tensorflow as tf

__all__ = [
    "RoPEPositionalEncoding",
]


class InvFreqInitializer(tf.keras.initializers.Initializer):
    def __init__(self, head_dim: int, base: float = 10000.0):
        self.head_dim = head_dim
        self.base = base

    def __call__(self, shape, dtype=None):
        index = tf.range(0, self.head_dim, 2, dtype=tf.float32)
        return 1.0 / tf.pow(self.base, index / tf.cast(self.head_dim, dtype=tf.float32))

@tf.keras.utils.register_keras_serializable(package=__name__)
class RoPEPositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        name: str = "rope_positional_encoding",
        **kwargs,
    ):
        super(RoPEPositionalEncoding, self).__init__(name=name, **kwargs)
        self.base = base
        self.head_dim = head_dim
        self.rot_dim = max(head_dim // 2, 32)

    
    def build(self, input_shape):
        head_dim = input_shape[-1]
        assert head_dim % 2 == 0, "head_dim (Dimension of each head) must be even for RoPE"
        self.dim = head_dim
        self.inv_freq = self.add_weight(
            shape=(self.rot_dim // 2,),
            initializer=InvFreqInitializer(self.rot_dim, self.base),
            trainable=False,
            name="inv_freq",
        )

    def encode(self, seq_len: int):
        positions = tf.expand_dims(tf.range(0, seq_len, dtype=tf.float32), axis=-1)  # Shape: (seq_len, 1)
        inv_freq = tf.expand_dims(self.inv_freq, axis=0)  # Shape: (1, head_dim // 2)
        freq = positions * inv_freq  # Shape: (seq_len, head_dim // 2)
        # print(f"Frequency shape: {freq.shape}, Inv Frequency shape: {self.inv_freq.shape}")
        freq = tf.stack([freq, freq], axis=-1)  # Shape: (seq_len, head_dim // 2, 2)
        # print(f"Frequency shape after stacking: {freq.shape}")
        freq = tf.reshape(freq, [seq_len, self.rot_dim])  # Shape: (seq_len, head_dim)
        # print(f"Frequency shape after reshape: {freq.shape}")
        return freq

    def rotate_half(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch_size, seq_len, num_heads, head_dim / 2)
        original_shape = shape_list(x)
        leading_dims = original_shape[:-1]
        intermediate_shape = tf.concat([leading_dims, [self.rot_dim // 2, 2]], axis=0) # (batch_size, seq_len, d_model/2, 2)
        x = tf.reshape(x, intermediate_shape)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x_rotated = tf.stack([-x2, x1], axis=-1)
        # print(f"Rotated shape: {x_rotated.shape}, Original shape: {original_shape}")
        return tf.reshape(x_rotated, original_shape)

    def call(self, inputs, training=False):
        # Inpute shape: (batch_size, seq_len, num_heads, head_dim)
        
        seq_len = tf.shape(inputs)[1]

        freq = self.encode(seq_len)
        rot_dim = tf.shape(freq)[-1]
        freq = tf.expand_dims(tf.expand_dims(freq, axis=0), axis=2)  # Shape: (seq_len, head_dim) -> (1, seq_len, 1, head_dim)

        t_rotated = inputs[..., :self.rot_dim]  # Shape: (batch_size, seq_len, num_heads, head_dim // 2)
        # print(f"t_rotated shape: {t_rotated.shape}, freq shape: {freq.shape}")
        t_unrotated = inputs[..., self.rot_dim:]  # Shape: (batch_size, seq_len, num_heads, head_dim // 2)
        # print(f"t_unrotated shape: {t_unrotated.shape}")

        cos = tf.cos(freq)
        sin = tf.sin(freq)

        x_rotated = t_rotated * cos + self.rotate_half(t_rotated) * sin  # Shape: (batch_size, seq_len, num_heads, head_dim // 2)
        return tf.concat([t_unrotated, x_rotated], axis=-1) # Shape: (batch_size, seq_len, num_heads, head_dim)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        return config