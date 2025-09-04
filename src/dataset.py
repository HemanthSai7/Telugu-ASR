from src.schemas import TrainInput, TargetLabels
from src.configs import DatasetConfig
from src.speech_featurizer import SpeechFeaturizer
from src.tokenizer import CharacterTokenizer
from src.utils import (
    data_util,
    file_util,
    math_util,
)

import os
import numpy as np
import tensorflow as tf

logger = tf.get_logger()

def get(
    tokenizer: CharacterTokenizer,
    speech_featurizer: SpeechFeaturizer,
    dataset_config: DatasetConfig,
):
    print(f"Creating dataset for stage: {dataset_config}")
    return ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        stage=dataset_config["stage"],
        data_paths=list(dataset_config["data_paths"])
    )

def get_shape(
    config,
    *datasets,
):
    max_input_length, max_shifted_right_input_length_shape, max_text_target_length_shape = None, None, None

    audio_input_shape = [max_input_length]
    shifted_right_text_input_shape = [max_shifted_right_input_length_shape]
    text_target_shape = [max_text_target_length_shape]

    audio_input_shape.extend([config.speech_config["num_feature_bins"], 1])

    padded_shapes = (
        TrainInput(
            audio_inputs = tf.TensorShape(audio_input_shape),
            shifted_right_text_inputs = tf.TensorShape(shifted_right_text_input_shape),
        ),
        TargetLabels(
            text_targets=tf.TensorShape(text_target_shape),
        )
    )

    return dict(
        audio_input_shape=audio_input_shape,
        shifted_right_text_input_shape=shifted_right_text_input_shape,
        text_target_shape=text_target_shape,
        padded_shapes=padded_shapes,
    )

BUFFER_SIZE = 100
AUTOTUNE = int(os.environ.get("AUTOTUNE", tf.data.AUTOTUNE))

class BaseDataset:
    def __init__(
        self,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        indefinite: bool = False,
        drop_remainder: bool = True,
        enabled: bool = True,
        metadata: str = None,
        sample_rate: int = 16000,
        stage: str = "train",
        name: str = "base_dataset",
        **kwargs,
    ):
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError("data_paths must be a list of string paths")
        self.cache = cache
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.stage = stage
        self.enabled = enabled
        self.drop_remainder = drop_remainder
        self.indefinite = indefinite
        self.total_steps = None
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.name = name

    def parse(self, *args, **kwargs):
        raise NotImplementedError()
    
    def create(self, *args, **kwargs):
        raise NotImplementedError()
    
class ASRDataset(BaseDataset):
    def __init__(
        self,
        stage: str,
        tokenizer: CharacterTokenizer,
        speech_featurizer: SpeechFeaturizer,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        indefinite: bool = True,
        drop_remainder: bool = True,
        enabled: bool = True,
        metadata: str = None,
        buffer_size: int = BUFFER_SIZE,
        sample_rate: int = 16000,
        training=False,
        name: str = "asr_dataset",
        **kwargs,
    ):
        super(ASRDataset, self).__init__(
            data_paths=data_paths,
            cache=cache,
            shuffle=shuffle,
            buffer_size=buffer_size,
            indefinite=indefinite,
            drop_remainder=drop_remainder,
            enabled=enabled,
            metadata=metadata,
            sample_rate=sample_rate,
            stage=stage,
            name=name,
        )
        self.entries = []
        self.tokenizer = tokenizer
        self.speech_featurizer = speech_featurizer
        self.training = training

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0:
            return
        self.data_paths = file_util.preprocess_paths(self.data_paths, enabled=self.enabled, check_exists=True)
        for file_path in self.data_paths:
            logger.info(f"Reading entries from {file_path}")
            with tf.io.gfile.GFile(file_path, "r") as f:
                for line in f.read().splitlines()[1:]:
                    self.entries.append(line.split("\t", 2))
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)
        self.total_steps = len(self.entries)
        self.num_entries = self.total_steps
        logger.info(f"Total entries: {self.num_entries}")

    def _process_item(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        with tf.device("/CPU:0"):
            audio_inputs = data_util.read_raw_audio(audio, sample_rate=self.sample_rate)
            audio_inputs = self.speech_featurizer(audio_inputs, training=self.training)
            audio_inputs = tf.expand_dims(audio_inputs, axis=-1)

            transcript_str = tf.strings.as_string(transcript)
            transcript_str = tf.ensure_shape(transcript_str, [])

            def tokenize_transcript(text):
                text_str = text.numpy().decode("utf-8")
                # print(f"Tokenizing transcript: {text_str}")
                # print("@@@@@@@@@@@@@@@", np.array(self.tokenizer.encode(text_str, add_special_tokens=True), dtype=np.int32))
                return np.array(self.tokenizer.encode(text_str, add_special_tokens=True), dtype=np.int32)
            
            tokens = tf.py_function(
                func=tokenize_transcript,
                inp=[transcript_str],
                Tout=tf.int32,
            )

            # Transformer Training
            # 1. Audio input: Waveform/ MFCC
            # 2. Shifted right text input: [BOS] + transcript
            # 3. Text target: transcript + [EOS]

            # Decoder inputs: [BOS, token1, token2, ..., tokenN]
            shifted_right_text_inputs = tokens[:-1]

            # Target labels: [token1, token2, ..., tokenN, EOS]
            text_target = tokens[1:]

        return path, audio_inputs, shifted_right_text_inputs, text_target

    def parse(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        (
            _, 
            audio_inputs, 
            shifted_right_text_inputs, 
            text_target
        ) = self._process_item(path=path, audio=audio, transcript=transcript)

        return (
            TrainInput(
                audio_inputs=audio_inputs, 
                shifted_right_text_inputs=shifted_right_text_inputs
            ), 
            TargetLabels(
                text_targets=text_target
            )
        )

    def process(self, dataset: tf.data.Dataset, batch_size: int, shapes=None):
        if self.cache:
            dataset = dataset.cache()

        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE, deterministic=False)
        self.total_steps = math_util.get_num_batches(self.num_entries, batch_size, drop_remainders=self.drop_remainder)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)

        if self.indefinite and self.total_steps:
            dataset = dataset.repeat()

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=shapes["padded_shapes"],
            padding_values = (
                TrainInput(
                    audio_inputs=0.0,
                    shifted_right_text_inputs=tf.constant(self.tokenizer.pad_token_id, dtype=tf.int32),
                ),
                TargetLabels(
                    text_targets=tf.constant(self.tokenizer.pad_token_id, dtype=tf.int32),
                ),
            ),
            drop_remainder=self.drop_remainder,
        )

        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

class ASRSliceDataset(ASRDataset):

    @staticmethod
    def load(record: tf.Tensor):
        audio = tf.py_function(
            lambda path: data_util.load_and_convert_to_wav(path.numpy().decode("utf-8")).numpy(),
            inp = [record[0]],
            Tout=tf.string,
        )
        return record[0], audio, record[2]
    
    def create(self, batch_size: int, padded_shapes=None):
        if not self.enabled:
            return None
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        # options = tf.data.Options()
        # options.deterministic = False
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # dataset = dataset.with_options(options)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE, deterministic=False)

        return self.process(dataset, batch_size=batch_size, shapes=padded_shapes)