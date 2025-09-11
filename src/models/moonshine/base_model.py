from src.utils import file_util, data_util
from src.schemas import TrainInput

import tensorflow as tf
import unicodedata

logger = tf.get_logger()


class BaseModel(tf.keras.Model):
    def __init__(self, tokenizer=None, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self._tfasr_metrics = {}
        self.loss_metric = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        self._tfasr_metrics["loss"] = self.loss_metric
        self.tokenizer = tokenizer

    @property
    def metrics(self):
        return list(self._tfasr_metrics.values())
    
    def save(
        self,
        filepath: str,
        overwrite: bool = True,
        include_optimizer: bool = True,
        save_format: str = None,
        signatures: dict = None,
        options: tf.saved_model.SaveOptions = None,
        save_traces: bool = True,
    ):
        with file_util.save_file(filepath) as path:
            super(BaseModel, self).save(
                filepath=filepath,
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces,
            )

    def save_weights(
        self,
        filepath: str,
        overwrite: bool = True,
        save_format: str = None,
        options: tf.saved_model.SaveOptions = None,
    ):
        with file_util.save_file(filepath) as path:
            super(BaseModel, self).save_weights(filepath=path, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(
            self,
            filepath,
            by_name=False,
            skip_mismatch=False,
            options=None,
    ):
        with file_util.read_file(filepath) as path:
            super().load_weights(filepath=path, by_name=by_name, skip_mismatch=skip_mismatch, options=options)


    def make(
            self,
            audio_input_shape = [None],
            shifted_right_text_input_shape = [None],
            batch_size = None,
            **kwargs
    ):
        audio_inputs = tf.keras.Input(shape=audio_input_shape, batch_size=batch_size, dtype=tf.float32)
        shifted_right_text_inputs = tf.keras.Input(shape=shifted_right_text_input_shape, batch_size=batch_size, dtype=tf.int32)

        outputs = self(
            TrainInput(
                audio_inputs=audio_inputs,
                shifted_right_text_inputs=shifted_right_text_inputs,
            ),
            training=False,
        )
        return outputs
    
    def compile(
        self,
        loss,
        optimizer,
        run_eagerly=None,
        **kwargs,
    ):
        optimizer = tf.keras.optimizers.get(optimizer)
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    def call(self, inputs, training=False, mask=None):
        raise NotImplementedError("The call method is not implemented in the base model.")

    def _train_step(self, data):
        x = data[0]
        y = data[1]["text_targets"]

        with tf.GradientTape() as tape:
            tape.watch(x["audio_inputs"])
            outputs = self(x, training=True)
            tape.watch(outputs)
            y_pred = outputs
            loss = self.compute_loss(x, y, y_pred)
            gradients = tape.gradient(loss, self.trainable_variables)

        self.loss_metric.update_state(loss)
        
        return gradients
    
    def train_step(self, data):
        gradients  = self._train_step(data)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
    
    def _test_step(self, data):
        x = data[0]
        y = data[1]["text_targets"]
        outputs = self(x, training=False)
        y_pred = outputs
        tokens = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        # print(f"Predictions: {self.tokenizer.batch_decode(tokens.numpy().tolist(), skip_special_tokens=True)}")  # Debug: Print predictions
        loss = self.compute_loss(x, y, y_pred)

        self.loss_metric.update_state(loss)
        return loss
    
    def test_step(self, data):
        self._test_step(data)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        """Fixed predict_step with proper Telugu handling."""
        x = data[0]
        y = data[1]["text_targets"] if len(data) > 1 and "text_targets" in data[1] else None
        
        audio_inputs = x["audio_inputs"]
        
        # Run recognition
        predicted_ids = self.recognize(audio_inputs, model_max_length=None)
        # print("Predicted IDs:", predicted_ids)
        
        batch_size = tf.shape(predicted_ids)[0]
        
        # Decode predictions
        def decode_predictions(sequences):
            # Use the tokenizer's batch_decode which now handles Telugu properly
            decoded = self.tokenizer.batch_decode(
                sequences.numpy().tolist(), 
                skip_special_tokens=True
            )
            # print("Decoded Predictions:", decoded)
            return tf.constant(decoded, dtype=tf.string)
        
        greedy_decoding = tf.py_function(
            func=decode_predictions,
            inp=[predicted_ids],
            Tout=tf.string
        )
        greedy_decoding.set_shape([None])
        
        # Decode ground truth if available
        if y is not None:
            def decode_labels(labels_tensor):
                decoded = self.tokenizer.batch_decode(
                    labels_tensor.numpy().tolist(), 
                    skip_special_tokens=True
                )
                return tf.constant(decoded, dtype=tf.string)
            
            labels = tf.py_function(
                func=decode_labels,
                inp=[y],
                Tout=tf.string
            )
            labels.set_shape([None])
        else:
            labels = tf.fill([batch_size], "")
        
        return tf.stack([labels, greedy_decoding], axis=-1)
    
    # def predict_step(self, data):
    #     x = data[0]
    #     y = data[1]["text_targets"] if len(data) > 1 and "text_targets" in data[1] else None

    #     y_pred = self(x, training=False)
    #     tokens = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    #     def decode_tokens(toks):
    #         decoded = self.tokenizer.batch_decode(toks.numpy().tolist(), skip_special_tokens=True)
    #         normalized = [unicodedata.normalize('NFC', text) for text in decoded]
    #         return tf.constant(normalized, dtype=tf.string)

    #     decoded = tf.py_function(func=decode_tokens, inp=[tokens], Tout=tf.string)
    #     decoded.set_shape([None])

    #     if y is not None:
    #         def decode_labels(lbls):
    #             decoded = self.tokenizer.batch_decode(lbls.numpy().tolist(), skip_special_tokens=True)
    #             normalized = [unicodedata.normalize('NFC', text) for text in decoded]
    #             return tf.constant(normalized, dtype=tf.string)
    #         labels = tf.py_function(func=decode_labels, inp=[y], Tout=tf.string)
    #         labels.set_shape([None])
    #     else:
    #         labels = tf.fill([tf.shape(decoded)[0]], "")

    #     return tf.stack([labels, decoded], axis=-1)
        
    # --------------------------------------------- TFLITE -----------------------------------

    def recognize(self, signal: tf.Tensor, model_max_length: int = None):
        raise NotImplementedError("The recognize method is not implemented in the base model.")