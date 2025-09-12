from src.utils import env_util, data_util, file_util
from src.schemas import TrainInput

import tensorflow as tf


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
            audio_input_shape=[None],
            audio_input_length_shape=[],
            prediction_shape=[None],
            prediction_length_shape=[],
            batch_size = None,
    ):
        audio_inputs = tf.keras.Input(shape=audio_input_shape, batch_size=batch_size, dtype=tf.float32)
        audio_inputs_length = tf.keras.Input(shape=audio_input_length_shape, batch_size=batch_size, dtype=tf.int32)
        predictions = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = tf.keras.Input(shape=prediction_length_shape, batch_size=batch_size, dtype=tf.int32)

        self(
            TrainInput(
                audio_inputs=audio_inputs,
                audio_inputs_length=audio_inputs_length,
                prediction=predictions,
                prediction_length=predictions_length,
            ),
            training=False,
        )

    def compile(
        self,
        loss,
        optimizer,
        run_eagerly=None,
        **kwargs,
    ):
        optimizer = tf.keras.optimizers.get(optimizer)
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

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
    
    def _test_step(self, data):
        x = data[0]
        y = data[1]["text_targets"]
        outputs = self(x, training=False)
        y_pred = outputs
        # tokens = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        # print(f"Predictions: {self.tokenizer.batch_decode(tokens.numpy().tolist(), skip_special_tokens=True)}")  # Debug: Print predictions
        loss = self.compute_loss(x, y, y_pred)

        self.loss_metric.update_state(loss)
        return loss
    
    def test_step(self, data):
        self._test_step(data)
        return {m.name: m.result() for m in self.metrics}