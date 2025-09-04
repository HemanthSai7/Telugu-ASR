import tensorflow as tf


class ErrorRate(tf.keras.metrics.Metric):
    """Metric for WER or CER"""

    def __init__(
            self,
            func,
            name="error_rates",
            **kwargs,
    ):
        super(ErrorRate, self).__init__(name=name, **kwargs)
        self.numerator = self.add_weight(name="numerator", initializer="zeros")
        self.denominator = self.add_weight(name="denominator", initializer="zeros")
        self.func = func

    def update_state(
            self,
            decode: tf.Tensor,
            target: tf.Tensor,
    ):
        n, d = self.func(decode, target)
        self.numerator.assign_add(n)
        self.denominator.assign_add(d)

    def result(self):
        return tf.math.divide_no_nan(self.numerator, self.denominator)