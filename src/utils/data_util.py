from typing import Union

import os
import io
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

def read_raw_audio(
    audio: Union[str, bytes, np.ndarray, tf.Tensor],
    sample_rate: int = 16000,
) -> Union[np.ndarray, tf.Tensor]:
    
    if isinstance(audio, str):
        return librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)[0]
        
    if isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if wave.ndim > 1:
            wave = np.mean(wave, axis=-1)
        wave = np.asfortranarray(wave)
        return librosa.resample(wave, orig_sr=sr, target_sr=sample_rate) if sr != sample_rate else wave
        
    if isinstance(audio, np.ndarray):
        if audio.ndim > 1:
            raise ValueError("Input audio must be single channel")
        return audio

    if isinstance(audio, tf.Tensor):
        wave, _ = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=-1)
        return tf.reshape(wave, shape=[-1])
        
    raise ValueError(f"Input audio must be either a path, bytes, numpy array, or tensor, got {type(audio)}")

def load_and_convert_to_wav(path: str) -> tf.Tensor:
    wave, rate = librosa.load(os.path.expanduser(path), sr=None, mono=True)
    return tf.audio.encode_wav(tf.expand_dims(wave, axis=-1), sample_rate=rate)