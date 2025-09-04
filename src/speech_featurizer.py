from src.utils import math_util
from dataclasses import dataclass, asdict

import tensorflow as tf

__all__ = [
    "SpeechFeaturizer",
]

@dataclass
class FeaturizerConfig:
    waveform: str = "waveform"
    spectrogram: str = "spectrogram"
    log_mel_spectrogram: str = "log_mel_spectrogram"
    mfcc: str = "mfcc"

@tf.keras.utils.register_keras_serializable(package=__name__)
class SpeechFeaturizer(tf.keras.layers.Layer):
    def __init__(
            self,
            sample_rate: int = 16000,
            frame_ms: int = 25,
            stride_ms: int = 10,
            num_feature_bins: int = 80,
            feature_type: str = "log_mel_spectrogram",
            preemphasis: float = 0.97,
            pad_end: bool = False,
            lower_edge_hertz: int = 0.0,
            upper_edge_hertz: int = 8000.0,
            output_floor: float = 1e-9,
            log_base: str = "10",
            nfft: int = 512,
            normalize_signal: bool = False,
            normalize_zscore: bool = False,
            normalize_min_max: bool = False,
            padding: float = 0.0,
            augmentation_config: dict = {},
            **kwargs,
    ):
        assert feature_type in asdict(FeaturizerConfig()).values(), f"Unsupported feature type: {feature_type}. Supported types: {asdict(FeaturizerConfig()).values()}"

        super().__init__(name=feature_type, **kwargs)
        self.sample_rate = sample_rate

        self.frame_ms = frame_ms
        self.frame_length = int(round(self.sample_rate * self.frame_ms / 1000.0))

        self.stride_ms = stride_ms
        self.frame_step = int(round(self.sample_rate * self.stride_ms / 1000.0))

        self.num_feature_bins = num_feature_bins
        self.feature_type = feature_type
        self.preemphasis = preemphasis
        self.pad_end = pad_end
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.output_floor = output_floor
        self.log_base = log_base
        assert self.log_base in ("10", "e"), "log_base must be '10' or 'e'"

        self._normalize_signal = normalize_signal
        self._normalize_zscore = normalize_zscore
        self._normalize_min_max = normalize_min_max
        self.padding = padding
        self.nfft = self.frame_length if nfft is None else nfft
        # self.augmentation = Augmentation(augmentation_config)

    def normalize_signal(self, signal: tf.Tensor) -> tf.Tensor:
        if self._normalize_signal:
            gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
            return signal * gain
        return signal
    
    def preemphasis_signal(self, signal):
        if not self.preemphasis or self.preemphasis <= 0.0:
            return signal
        s0 = tf.expand_dims(signal[0], axis=-1)
        s1 = signal[1:] - self.preemphasis * signal[:-1]
        return tf.concat([s0, s1], -1)
    
    def normalize_audio_feature(self, audio_feature):
        if self._normalize_zscore:
            mean = tf.reduce_mean(audio_feature, axis=1, keepdims=True)
            stddev = tf.sqrt(tf.math.reduce_variance(audio_feature, axis=1, keepdims=True) + 1e-9)
            return tf.divide(tf.subtract(audio_feature, mean), stddev)
        if self._normalize_min_max:
            if self.feature_type == FeaturizerConfig.spectrogram:
                min_value = self.logarithm(self.output_floor)
            else:
                min_value = tf.reduce_min(audio_feature, axis=1, keepdims=True)
            return (audio_feature - min_value ) / (tf.reduce_max(audio_feature, axis=1, keepdims=True) - min_value)
        
        return audio_feature
    
    def stft(self, signal):
        fft_features = tf.signal.stft(
            signal,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            pad_end=self.pad_end,
        )
        fft_features = tf.abs(fft_features)
        fft_features = tf.square(fft_features)
        fft_features = tf.cast(fft_features, self.dtype)
        return fft_features
    
    def logarithm(self, S):
        if self.log_base == "10":
            return math_util.log10(tf.maximum(S, self.output_floor))
        return tf.math.log(tf.maximum(S, self.output_floor))

    def log_mel_spectrogram(self, signal):
        S = self.stft(signal)
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_feature_bins,
            num_spectrogram_bins=tf.shape(S)[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz,
        )
        mel_spectrogram = tf.matmul(S, linear_to_mel_weight_matrix)
        return self.logarithm(mel_spectrogram)
    
    def spectrogram(self, signal):
        spectrogram = self.logarithm(self.stft(signal))
        return spectrogram[:, :, :self.num_feature_bins]
    
    def mfcc(self, signal):
        log_mel_spectrogram = self.log_mel_spectrogram(signal)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    
    def waveform(self, signal):
        return signal

    
    def call(self, inputs, training=False):
        signals = inputs

        if self.padding > 0:
            signals = tf.pad(signals, [[0, 0], [0, self.padding]], mode="CONSTANT", constant_values=0.0)

        signals = self.normalize_signal(signals)
        signals = self.preemphasis_signal(signals)

        feature_compute_methods = {
            "waveform": self.waveform,
            "mfcc": self.mfcc,
            "log_mel_spectrogram": self.log_mel_spectrogram,
            "spectrogram": self.spectrogram,
        }
        compute_method = feature_compute_methods.get(self.feature_type)

        assert compute_method is not None, f"Unsupported feature type: {self.feature_type}. Supported types: {asdict(FeaturizerConfig()).values()}"

        features = compute_method(signals)
        features = self.normalize_audio_feature(features)

        if training:
            features = self.augmentation.signal_augment(features)

        return features
    
    def get_nframes(self, nsamples):
        if self.pad_end:
            return -(-nsamples // self.frame_step)
        return 1 + (nsamples - self.frame_length) // self.frame_step
    
    def compute_output_shape(self, input_shape):
        signal_shape = input_shape
        B, nsamples = signal_shape
        if nsamples is None:
            output_shape = [B, None, self.num_feature_bins, 1]
        elif self.feature_type == FeaturizerConfig.waveform:
            output_shape = [B, None, 1]
        else:
            output_shape = [B, self.get_nframes(nsamples + self.padding), self.num_feature_bins, 1]

        return tf.TensorShape(output_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "feature_type": self.feature_type,
            "normalize_signal": self._normalize_signal,
            "preemphasis": self.preemphasis,
            "padding": self.padding,
            "augmentation_config": self.augmentation_config,
        })
        return config


