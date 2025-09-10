from src.model.moonshine import Encoder, Decoder, BaseModel
from typing import Union, Optional, List

import tensorflow as tf

__all__ = ["ASRModel"]

@tf.keras.utils.register_keras_serializable(package=__name__)
class ASRModel(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        kernel_initializer: dict,
        bias_initializer: dict,
        kernel_regularizer: dict,
        bias_regularizer: dict,
        d_model: int = 288,
        subsampling_config: Optional[dict] = None,
        encoder_config: Optional[dict] = None,
        decoder_config: Optional[dict] = None,
        name: str = "asrmodel",
        **kwargs,
    ):
        super(ASRModel, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder = Encoder(
            input_dim=d_model,
            subsampling_config=subsampling_config,
            num_blocks=encoder_config.get("num_blocks", 6),
            num_heads=encoder_config.get("num_heads", 4),
            head_dim=encoder_config.get("head_dim", 32),
            activation=encoder_config.get("activation", "relu"),
            fc_factor=encoder_config.get("fc_factor", 1),
            dropout=encoder_config.get("dropout", 0.1),
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(dict(kernel_regularizer)),
            bias_regularizer=tf.keras.regularizers.get(dict(bias_regularizer)),
        )
        self.decoder = Decoder(
            input_dim=d_model,
            num_blocks=decoder_config.get("num_blocks", 6),
            num_heads=decoder_config.get("num_heads", 4),
            head_dim=decoder_config.get("head_dim", 32),
            activation=decoder_config.get("activation", "swiglu"),
            fc_factor=decoder_config.get("fc_factor", 1),
            dropout=decoder_config.get("dropout", 0.1),
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(dict(kernel_regularizer)),
            bias_regularizer=tf.keras.regularizers.get(dict(bias_regularizer)),
        )
        self.text_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            name="text_embedding",
        )
        self.final_dense = tf.keras.layers.Dense(
            units=vocab_size,
            activation="softmax",
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(dict(kernel_regularizer)),
            bias_regularizer=tf.keras.regularizers.get(dict(bias_regularizer)),
            name="classifier",
        )

    def add_featurizers(self, speech_featurizer, tokenizer):
        self.speech_featurizer = speech_featurizer
        self.tokenizer = tokenizer

    @staticmethod
    def create_masks(
        audio_inputs: tf.Tensor, 
        shifted_right_text_inputs: tf.Tensor,
        audio_pad_value: float,
        text_pad_value: int
    ):
        audio_mask = tf.cast(tf.reduce_any(tf.not_equal(audio_inputs, audio_pad_value), axis=-1), tf.float32) # [BS, seq_len]
        text_mask = tf.cast(tf.not_equal(shifted_right_text_inputs, text_pad_value), tf.float32) # [BS, seq_len]
        return audio_mask, text_mask
    
    def call(self, inputs, training=False):
        audio_inputs, shifted_right_text_inputs = inputs["audio_inputs"], inputs["shifted_right_text_inputs"]

        embedded_text_inputs = self.text_embedding(shifted_right_text_inputs)
        audio_mask, text_mask = self.create_masks(audio_inputs, shifted_right_text_inputs, audio_pad_value=0.0, text_pad_value=2)

        encoder_outputs = self.encoder(inputs=audio_inputs, training=training, mask=audio_mask)
        decoder_outputs = self.decoder(
            inputs=[embedded_text_inputs, encoder_outputs], 
            mask=[text_mask, self.encoder.encoder_input_padding_mask],
            training=training, 
        )
        decoder_outputs = self.final_dense(decoder_outputs, training=training)

        return decoder_outputs
    
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the model based on the input shape.
        """
        audio_input_shape, shifted_right_text_inputs_shape = input_shape["audio_inputs"], input_shape["shifted_right_text_inputs"]
        batch_size = audio_input_shape[0]
        seq_length = self.decoder.compute_output_shape((shifted_right_text_inputs_shape, audio_input_shape))[1]
        return (batch_size, seq_length, self.vocab_size) 
    
    def get_config(self):
        config = super(ASRModel, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "encoder": self.encoder.get_config(),
        })
        return config
    
    # ----------------------------------------- GREEDY SEARCH -------------------------------------------------------

    def _perform_greedy_batch(
        self,
        encoder_outputs: tf.Tensor,
        max_length: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        batch_size = tf.shape(encoder_outputs)[0]
        decoded = tf.TensorArray(
            dtype=tf.int32, 
            size=batch_size, 
            dynamic_size=False, 
            clear_after_read=False
        )
        
        def condition(batch, decoded):
            return tf.less(batch, batch_size)
        
        def body(batch, decoded):
            output = self._perform_greedy(
                encoder_outputs[batch],
                max_length,
                bos_token_id,
                eos_token_id,
                pad_token_id,
            )
            # Pad output to max_length
            current_length = tf.shape(output)[0]
            padded_output = tf.pad(
                output,
                [[0, max_length - current_length]],
                constant_values=pad_token_id
            )
            decoded = decoded.write(batch, padded_output)
            return batch + 1, decoded
        
        batch, decoded = tf.while_loop(
            condition,
            body,
            [tf.constant(0, dtype=tf.int32), decoded],
        )
        
        return decoded.stack()  # [batch_size, max_length]
    
    def _perform_greedy(
        self,
        encoder_outputs: tf.Tensor,
        max_length: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        
        step = tf.constant(0, dtype=tf.int32)
        decoder_input = tf.expand_dims([bos_token_id], 0) # [1, 1]
        finished = tf.constant(False)
        generated = tf.TensorArray(
            tf.int32,
            size=max_length,
            dynamic_size=False,
            clear_after_read=False
        )

        def condition(step, decoder_input, finished, generated):
            return tf.logical_and(tf.less(step, max_length), tf.logical_not(finished))
        
        def body(step, decoder_input, finished, generated):
            embedded = self.text_embedding(decoder_input)
            text_mask = tf.cast(tf.not_equal(decoder_input, pad_token_id), tf.float32)
            decoder_out = self.decoder(
                inputs=[embedded, tf.expand_dims(encoder_outputs, 0)],
                mask=[text_mask, None],
                training=False,
            )
            logits = self.final_dense(decoder_out, training=False)
            next_token = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)  # [1]

            generated = generated.write(step, next_token[0])

            # is_eos = tf.equal(next_token[0], eos_token_id)
            # is_bos_late = tf.logical_and(tf.equal(next_token[0], bos_token_id), step > 0)

            finished = tf.equal(next_token[0], eos_token_id)

            decoder_input = tf.cond(
                finished,
                lambda: decoder_input,
                lambda: tf.concat([decoder_input, tf.expand_dims(next_token, 1)], axis=1)
            )

            return step + 1, decoder_input, finished, generated
        
        step, decoder_input, finished, generated = tf.while_loop(
            condition,
            body,
            [step, decoder_input, finished, generated],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([1, None]),
                tf.TensorShape([]),
                tf.TensorShape(None)
            ]
        )

        actual_length = tf.minimum(step, max_length)
        generated_tokens = generated.gather(tf.range(actual_length))
        # tf.print(f"Generated tokens shape: {tf.shape(generated_tokens)}")
        # tf.print(f"Generated tokens: {generated_tokens}")
        return generated_tokens
    
    def recognize(self, signal: tf.Tensor, model_max_length: int = None):
        if model_max_length is None:
            duration = (((tf.shape(signal)[1] - 1) * 160) + 400) / 16000
            model_max_length = tf.cast(duration * 24, tf.int32) # small hack to generate only 24 tokens per second of audio. Reason being the model was sometimes not able to generate the [EOS] token.

        audio_mask = tf.cast(tf.reduce_any(tf.not_equal(signal, 0.0), axis=-1), tf.float32)
        encoder_outputs = self.encoder(
            inputs=signal,
            training=False,
            mask=audio_mask,
        )

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        decoded = self._perform_greedy_batch(
            encoder_outputs, model_max_length, bos_token_id, eos_token_id, pad_token_id,
        )

        return decoded