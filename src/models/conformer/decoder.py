from src.models.layers import Embedding
from src.utils import layer_util
from typing import Union, Optional

import collections
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class TransducerPrediction(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        embed_dropout: float = 0.0,
        num_rnns: int = 1,
        rnn_units: int = 512,
        rnn_type: str = "lstm",
        rnn_implementation: int = 2,
        layer_norm: bool = True,
        projection_units: int = 0,
        kernel_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
        kernel_initializer: Optional[Union[str, tf.keras.initializers.Initializer]] = None,
        bias_initializer: Optional[Union[str, tf.keras.initializers.Initializer]] = None,
        name: str = "transducer_prediction",
        **kwargs,
    ):
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embedding = Embedding(vocab_size, embed_dim, initializer=kernel_initializer, name=f"{name}_embedding")
        self.do = tf.keras.layers.Dropout(embed_dropout, name=f"{name}_dropout")
        RNN = layer_util.get_rnn(rnn_type)
        self.rnns = []
        for i in range(num_rnns):
            rnn = RNN(
                units = rnn_units,
                return_sequences = True,
                return_state = True,
                implementation=rnn_implementation,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name = f"{name}_{rnn_type}_{i}",
            )
            ln = (
                tf.keras.layers.LayerNormalization(
                    name=f"ln_{i}", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=tf.float32
                )
                if layer_norm
                else None
            )

            projection = (
                tf.keras.layers.Dense(
                    projection_units,
                    name=f"projection_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    dtype=tf.float32,
                )
                if projection_units > 0
                else None
            )
            self.rnns.append({"rnn": rnn, "ln": ln, "projection": projection})

    def get_initial_state(self):
        states = []
        for rnn in self.rnns:
            states.append(tf.stack(rnn["rnn"].get_initial_state(tf.zeros([1, 1, 1], dtype=tf.float32)), axis=0))
        return tf.stack(states, axis=0)
    
    def call(self, inputs, training=False, **kwargs):
        outputs, prediction_length = inputs
        outputs = self.embedding(outputs, training=training)
        outputs = self.do(outputs, training=training)
        for rnn in self.rnns:
            mask = tf.sequence_mask(prediction_length, maxlen=tf.shape(outputs)[1])
            outputs = rnn["rnn"](outputs, training=training, mask=mask)
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=training)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=training)
            return outputs
        
    def recognize(self, inputs, states):
        outputs = self.embedding(inputs, training=False)
        outputs = self.do(outputs, training=False)

        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn["rnn"](outputs, initial_state=tf.unstack(states[i], axis=0), training=False)
            new_states.append(outputs[1:])
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=False)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=False)
            return outputs, tf.stack(new_states, axis=0)
        
    def get_config(self):
        conf = self.embedding.get_config()
        conf.update(self.do.get_config())
        for rnn in self.rnns:
            conf.update(rnn["rnn"].get_config())
            if rnn["ln"] is not None:
                conf.update(rnn["ln"].get_config())
            if rnn["projection"] is not None:
                conf.update(rnn["projection"].get_config())
        return conf
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class TransducerJointMerge(tf.keras.layers.Layer):
    def __init__(self, joint_mode: str = "add", name="transducer_joint_merge", **kwargs):
        super(TransducerJointMerge, self).__init__(name=name, **kwargs)
        self.joint_mode = joint_mode

    def call(self, inputs):
        enc_out, pred_out = inputs
        enc_out = tf.expand_dims(enc_out, axis=2)
        pred_out = tf.expand_dims(pred_out, axis=1)
        if self.joint_mode == "add":
            outputs = tf.add(enc_out, pred_out)
        elif self.joint_mode == "concat":
            outputs = tf.concat([enc_out, pred_out], axis=-1)
        elif self.joint_mode == "mul":
            outputs = tf.multiply(enc_out, pred_out)
        else:
            raise ValueError("joint_mode must be either 'add', 'concat' or 'mul'")
        return outputs
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class TransducerJoint(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        joint_dim: int = 1024,
        activation: str = "tanh",
        prejoint_linear: bool = True,
        postjoint_linear: bool = False,
        joint_mode: str = "add",
        kernel_regularizer = None,
        bias_regularizer = None,
        name: str = "transducer_joint",
        **kwargs,
    ):
        super(TransducerJoint, self).__init__(name=name, **kwargs)

        self.prejoint_linear = prejoint_linear
        self.postjoint_linear = postjoint_linear

        if self.prejoint_linear:
            self.ffn_enc = tf.keras.layers.Dense(
                joint_dim,
                name=f"{name}_enc",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
            self.ffn_pred = tf.keras.layers.Dense(
                joint_dim, 
                name = f"{name}_pred", 
                kernel_regularizer=kernel_regularizer, 
                bias_regularizer=bias_regularizer
            )

        self.joint = TransducerJointMerge(joint_mode=joint_mode, name=f"{name}_merge")

        activation = activation.lower()
        self.activation = tf.keras.layers.Activation(activation, name=f"{name}_{activation}")

        if self.postjoint_linear:
            self.ffn = tf.keras.layers.Dense(
                joint_dim, 
                name=f"{name}_ffn", 
                kernel_regularizer=kernel_regularizer, 
                bias_regularizer=bias_regularizer
            )
        
        self.ffn_out = tf.keras.layers.Dense(
            vocab_size, 
            name=f"{name}_vocab", 
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer
        )

    def call(self, inputs, training = False, **kwargs):
        enc_out, pred_out = inputs
        if self.prejoint_linear:
            enc_out = self.ffn_enc(enc_out, training=training)
            pred_out = self.ffn_pred(pred_out, training=training)
        outputs = self.joint((enc_out, pred_out), training=training)
        if self.postjoint_linear:
            outputs = self.ffn(outputs, training=training)
        outputs = self.activation(outputs, training=training)
        outputs = self.ffn_out(outputs, training=training)
        return outputs
    
    def get_config(self):
        conf = self.ffn_enc.get_config()
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        conf.update(self.activation.get_config())
        conf.update(self.joint.get_config())
        conf.update({"prejoint_linear": self.prejoint_linear, "postjoint_linear": self.postjoint_linear})
        return conf