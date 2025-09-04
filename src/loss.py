import tensorflow as tf


class MaskedCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(
            self, 
            global_batch_size=None,
            ignore_class = 0,
            from_logits=False,
            name="masked_cross_entropy_loss"
        ):
        """
        Cross Entropy loss is a measurement of the dissimilarity b/w two probability distribution.
        """
        super(MaskedCrossEntropyLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.global_batch_size = global_batch_size
        self.ignore_class =ignore_class
        self.from_logits = from_logits
        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, 
            ignore_class=ignore_class,
            reduction=tf.keras.losses.Reduction.NONE, 
            name=name
        )

    def call(self, y_true, y_pred):
        # print(y_true)
        # print(sum(y_pred[0][1]))
        # print(self.global_batch_size)
        per_element_loss = self._loss_fn(y_true, y_pred)
        per_example_loss = tf.reduce_sum(per_element_loss, axis=-1)

        if self.global_batch_size is not None:
            return tf.nn.compute_average_loss(
                per_example_loss, 
                global_batch_size=self.global_batch_size
            )
        return per_example_loss
    
    def get_config(self):
        config = super(MaskedCrossEntropyLoss, self).get_config()
        config.update({
            "global_batch_size": self.global_batch_size,
            "ignore_class": self.ignore_class,
            "from_logits": self.from_logits,
        })
        return config