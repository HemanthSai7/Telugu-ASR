import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            regularizer = None,
            initializer = None,
            name = "embedding",
            **kwargs,
    ):
        super(Embedding, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            dtype=tf.float32,
            shape=[self.vocab_size, self.embed_dim],
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        outputs = tf.cast(inputs, dtype=tf.int32)
        return tf.nn.embedding_lookup(self.embeddings, outputs)
    
    def get_config(self):
        conf = super(Embedding, self).get_config()
        conf.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "regularizer": self.regularizer,
                "initializer": self.initializer,
            }
        )
        return conf
