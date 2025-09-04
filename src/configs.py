from src.utils import file_util

from typing import Union

import json
import tensorflow as tf

logger = tf.get_logger()

__all__ = [
    "Config",
]

class SpeechConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.sample_rate: str = config.get("sample_rate", 16000)
        self.frame_ms: int = config.get("frame_ms", 25)
        self.stride_ms: int = config.get("stride_ms", 10)
        self.num_feature_bins: int = config.get("num_feature_bins", 80)
        self.feature_type: str = config.get("feature_type", "mfcc")
        self.preemphasis: float = config.get("preemphasis", 0.97)
        self.pad_end: bool = config.get("pad_end", False)
        self.lower_edge_hertz: float = config.get("lower_edge_hertz", 0.0)
        self.upper_edge_hertz: float = config.get("upper_edge_hertz", 8000.0)
        self.output_floor: float = config.get("output_floor", 1e-9)
        self.log_base: str = config.get("log_base", "10")
        self.normalize_signal: bool = config.get("normalize_signal", True)
        self.normalize_zscore: bool = config.get("normalize_zscore",False)
        self.normalize_min_max: bool = config.get("normalize_min_max", False)
        self.padding: float = config.get("padding", 0.0)
        for k, v in config.items():
            setattr(self, k, v)

class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.enabled: bool = config.get("enabled", True)
        self.stage: str = config.get("stage", None)
        self.data_paths = config.get("data_paths", None)
        self.shuffle: bool = config.get("shuffle", False)
        self.drop_remainder: bool = config.get("drop_remainder", True)
        self.buffer_size: int = config.get("buffer_size", 1000)
        self.metadata: str = config.get("metadata", None)
        self.indefinite: bool = config.get("indefinite", True)
        for k, v in config.items():
            setattr(self, k, v)


class DataConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.train_dataset_config = DatasetConfig(config.get("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.get("eval_dataset_config", {}))
        self.test_dataset_configs = DatasetConfig(config.get("test_dataset_configs", {}))

class RunningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.batch_size: int = config.get("batch_size", 32)
        self.num_epochs: int = config.get("num_epochs", 10)
        for k, v in config.items():
            setattr(self, k, v)

class LearningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.pretrained = file_util.preprocess_paths(config.get("pretrained", None))
        self.optimizer_config: dict = config.get("optimizer_config", {})
        self.running_config = config.get("running_config", {})
        for k, v in config.items():
            setattr(self, k, v)

class Config:
    """User config class for training, testing or infering"""

    def __init__(self, config: Union[str, dict], training=True, **kwargs):
        self.speech_config = SpeechConfig(config.get("speech_config", {}))
        self.model_config: dict = config.get("model_config", {})
        self.data_config = DataConfig(config.get("data_config", {}))
        self.learning_config = LearningConfig(config.get("learning_config", {})) if training else None
        for k, v in config.items():
            setattr(self, k, v)
        # logger.info(str(self))
        
    def __str__(self) -> str:
        def default(x):
            try:
                return {k: v for k, v in vars(x).items() if not str(k).startswith("_")}
            except:
                return str(x)

        return json.dumps(vars(self), indent=2, default=default)