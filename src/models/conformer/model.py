from src.models.conformer import (
    BaseModel, 
    ConformerEncoder, 
    TransducerPrediction, 
    TransducerJoint
)
from typing import Optional, List, Dict
from src.utils import data_util, math_util
from src.schemas import OutputLogits

import collections
import tensorflow as tf

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))

__all__ = ["Conformer"]

@tf.keras.utils.register_keras_serializable(package=__name__)
class Conformer(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        subsampling_config: dict,
        encoder_config: dict,
        decoder_config: dict,
        kernel_initializer: dict,
        bias_initializer: dict,
        kernel_regularizer: dict,
        bias_regularizer: dict,
        name: str = "conformer_transducer",
        **kwargs,
    ):
        super(Conformer, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder = ConformerEncoder(
            subsampling_config=subsampling_config,
            d_model=d_model,
            num_blocks=encoder_config.get("num_blocks", 16),
            head_dim=encoder_config.get("head_dim", 16),
            num_heads=encoder_config.get("num_heads", 16),
            attention_type=encoder_config.get("attention_type", "relmha"),
            kernel_size=encoder_config.get("kernel_size", 16),
            fc_factor=encoder_config.get("fc_factor", 16),
            dropout=encoder_config.get("dropout", 0.1),
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(dict(kernel_regularizer)),
            bias_regularizer=tf.keras.regularizers.get(dict(bias_regularizer)),
        )
        self.predict_net = TransducerPrediction(
            vocab_size=vocab_size,
            embed_dim=decoder_config.get("embed_dim", 256),
            embed_dropout=decoder_config.get("embed_dropout", 256),
            num_rnns=decoder_config.get("num_rnns", 1),
            rnn_units=decoder_config.get("rnn_units", 128),
            rnn_type=decoder_config.get("rnn_type", "lstm"),
            rnn_implementation=decoder_config.get("rnn_implementation", 2),
            layer_norm=decoder_config.get("layer_norm", True),
            projection_units=decoder_config.get("projection_units", 256),
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(dict(kernel_regularizer)),
            bias_regularizer=tf.keras.regularizers.get(dict(bias_regularizer)),
            name=f"{name}_prediction",
        )
        self.joint_net = TransducerJoint(
            vocab_size=vocab_size,
            joint_dim=decoder_config.get("joint_dim", 256),
            activation=decoder_config.get("activation", "relu"),
            prejoint_linear=decoder_config.get("prejoint_linear", True),
            postjoint_linear=decoder_config.get("postjoint_linear", True),
            joint_mode=decoder_config.get("joint_mode", "add"),
            kernel_regularizer=tf.keras.regularizers.get(dict(kernel_regularizer)),
            bias_regularizer=tf.keras.regularizers.get(dict(bias_regularizer)),
            name=f"{name}_joint",
        )

    def call(
            self,
            inputs,
            training=False,
            **kwargs,
    ):
        logits, logits_length = self.encoder((inputs["audio_inputs"], inputs["audio_inputs_length"]), training=training, **kwargs)
        pred = self.predict_net((inputs["prediction"], inputs["prediction_length"]), training=training, **kwargs)
        logits = self.joint_net([logits, pred], training=training, **kwargs)
        return OutputLogits(
            logits=logits,
            logits_length = logits_length,
        )
    
    def encoder_inference(
            self,
            features: tf.Tensor,
    ):
        with tf.name_scope(f"{self.name}_encoder"):
            outputs = tf.expand_dims(features, axis=0)
            outputs, _ = self.encoder((outputs, tf.shape(outputs)[0]), training=False)
            return tf.squeeze(outputs, axis=0)
        
    def decoder_inference(
            self,
            encoded: tf.Tensor,
            predicted: tf.Tensor,
            states: tf.Tensor,
            tflite: bool = False,
    ):
        with tf.name_scope(f"{self.name}_decoder"):
            encoded = tf.reshape(encoded, [1, 1, -1])
            predicted = tf.reshape(predicted, [1, -1])
            y, new_states = self.predict_net.recognize(predicted, states, tflite=tflite)
            ytu = tf.nn.log_softmax(self.joint_net([encoded, y], training=False))
            ytu = tf.reshape(ytu, shape=[-1])
            return ytu, new_states
        
    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf
    
    ############################## INFERENCE #####################################
    
    @tf.function
    def recognize(
        self,
        inputs: Dict[str, tf.Tensor],
    ):
        """RNN Transducer Greedy decoding

        Args:
            features (tf.Tensor): a batch of extracted features
            input_length (tf.Tensor): a batch of extracted features length

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded, encoded_length = self.encoder((inputs["inputs"], inputs["inputs_length"]), training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        return self._perform_greedy_batch(encoded, encoded_length)
    
    def _perform_greedy_batch(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor, 
            parallel_iterations: int = 10,
            swap_memory: bool = True,
    ):
        with tf.name_scope(f"{self.name}_perform_greedy_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            decoded = tf.TensorArray(
                dtype=tf.int32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([None]),
            )

            def condition(batch, _):
                return tf.less(batch, total_batch)
            
            def body(batch, decoded):
                hypothesis = self._perform_greedy(
                    encoded=encoded[batch],
                    encoded_length=encoded_length[batch],
                    predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                    states=self.predict_net.get_initial_state(),
                    parallel_iterations=parallel_iterations,
                    swap_memory = swap_memory,
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded
            
            batch, decoded = tf.while_loop(
                condition,
                body,
                loop_vars = [batch, decoded],
                parallel_iterations=parallel_iterations,
                swap_memory=True,
            )

            decoded = math_util.pad_prediction_tfarray(decoded, blank = self.text_featurizer.blank)
            return self.text_featurizer.iextract(decoded.stack())
        
    def _perform_greedy(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            predicted: tf.Tensor,
            states: tf.Tensor,
            parallel_iterations: int = 10,
            swap_memory: bool = False,
            tflite: bool = False
    ):
        with tf.name_scope(f"{self.name}_greedy"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length
            
            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=total,
                    dynamic_size=False,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)
            
            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states,
                    tflite=tflite,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32) # argmax over vocabulary []

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)

                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

                return _time + 1, _hypothesis
            
            time, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            return Hypothesis(
                index=hypothesis.index,
                prediction=hypothesis.prediction.stack(),
                states=hypothesis.states,
            )