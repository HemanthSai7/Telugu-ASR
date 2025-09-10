from src.augmentations import specaugment

import tensorflow as tf

AUGMENTATIONS = {
    "freq_masking": specaugment.FreqMasking,
    "time_masking": specaugment.TimeMasking,
}


class Augmentation:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.prob = float(config.get("prob", 0.5))
        self.signal_augmentations = self.parse(config.get("signal_augment", {}))
        self.feature_augmentations = self.parse(config.get("feature_augment", {}))

    def _augment(self, inputs, augmentations):
        """Apply augmentations to inputs

        Args:
            inputs: tf.Tensor: inputs
            augmentations: dict: augmentations to apply

        Returns:
            augmented inputs
        """
        outputs = inputs
        for au in augmentations:
            # Generate a random number from a uniform distribution
            p = tf.random.uniform([])
             # Apply the augmentation if the random number is less than the probability
            outputs = tf.where(tf.less(p, self.prob), au.augment(outputs), outputs)
        return outputs
    
    @tf.function
    def signal_augment(self, inputs):
        """Apply signal augmentations to inputs

        Args:
            inputs: tf.Tensor: inputs

        Returns:
            augmented inputs
        """
        return self._augment(inputs, self.signal_augmentations)
    
    @tf.function
    def feature_augment(self, inputs):
        """Apply feature augmentations to inputs

        Args:
            inputs: tf.Tensor: inputs

        Returns:
            augmented inputs
        """
        return self._augment(inputs, self.feature_augmentations)
    
    @staticmethod
    def parse(config: dict) -> list:
        """Parse augmentations from config

        Args:
            config: dict: configuration

        Returns:
            list of augmentations
        """
        augmentations = []
        for key, value in config.items():
            au = AUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(f"No tf augmentation named: {key}\n" 
                               f"Available tf augmentations: {AUGMENTATIONS.keys()}")
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return augmentations
