from typing import List, Union

import random
import logging
import warnings
import numpy as np
import tensorflow as tf

logger = tf.get_logger()

__all__ = [
    "setup_environment",
    "setup_strategy",
]

def setup_environment():
    warnings.simplefilter("ignore")
    logger.setLevel(logging.INFO)
    return logger

def setup_devices(devices: List[int], cpu: bool = False):
    if cpu:
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_visible_devices(cpus, "CPU")
        tf.config.set_visible_devices([], "GPU")
        logger.info(f"Run on {cpus}")
        return tf.config.list_logical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if devices is not None:
            gpus = [gpus[i] for i in devices]
            tf.config.set_visible_devices(gpus, "GPU")
    logger.info(f"Run on {gpus}")
    return tf.config.list_logical_devices("GPU")

def setup_strategy(devices: List[int]):
    available_devices = setup_devices(devices)
    if len(available_devices) == 1:
        return tf.distribute.get_strategy()
    return tf.distribute.MultiWorkerMirroredStrategy()

def has_devices(
    devices: Union[List[str], str],
):
    if isinstance(devices, list):
        return all((len(tf.config.list_logical_devices(d)) > 0 for d in devices))
    return len(tf.config.list_logical_devices(devices)) > 0

def setup_seed(
    seed: int = 42,
):
    """
    The seed is given an integer value to ensure that the results of pseudo-random generation are reproducible
    Why 42?
    "It was a joke. It had to be a number, an ordinary, smallish number, and I chose that one.
    I sat at my desk, stared into the garden and thought 42 will do!"
    - Douglas Adams's popular 1979 science-fiction novel The Hitchhiker's Guide to the Galaxy

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.experimental.enable_tf_random_generator()
    tf.keras.utils.set_random_seed(seed)