from src.utils import env_util
from tensorflow.python.ops.gen_array_ops import matrix_diag_part_v2

import tensorflow as tf

logger = tf.get_logger()

LOG_0 = float("-inf")

class RnntLoss(tf.keras.losses.Loss):
    """Computes the RNN-T loss between `y_true` and `y_pred`.
    
    ## References
    - [Sequence transduction with recurrent neural networks](https://arxiv.org/abs/1211.3711)

    ## The use of RNN-T loss
    For a general-purpose sequence transducer, where the output length is unknown in advance,
    we would prefer a distribution over sequences of all lengths.
    """
    def __init__(
            self,
            blank=0,
            global_batch_size=None,
            name=None,
    ):
        super(RnntLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.blank = blank
        self.global_batch_size = global_batch_size

    def call(self, y_true, y_pred):
        loss = rnnt_loss(
            logits=y_pred["logits"],
            logit_length=y_pred["logits_length"],
            labels=y_true["labels"],
            label_length=y_true["labels_length"],
            blank=self.blank,
            name=self.name,
        )
        return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)
    
def nan_to_zero(input_tensor):
    """Replace NaN with zero"""
    return tf.where(tf.math.is_nan(input_tensor), tf.zeros_like(input_tensor), input_tensor)

def reduce_logsumexp(
        input_tensor,
        axis,
):
    """Compute log(sum(exp(input_tensor), axis)) in a numerically stable way.
    
    ## Reason and Proof
    - The log-sum-exp trick is a useful mathematical identity for numerical stability.
    To prevent overflow and underflow in the computation, we can 
    use the following trick:
    - [LogSumExp](https://raw.org/math/the-log-sum-exp-trick-in-machine-learning/)
    """
    maximum = tf.reduce_max(input_tensor, axis=axis)
    input_tensor = nan_to_zero(input_tensor - maximum)
    return tf.math.log(tf.reduce_sum(tf.exp(input_tensor), axis=axis)) + maximum

def extract_diagonals(
    log_probs,
):
    """Extract diagonals from a 3D tensor."""
    time_steps = tf.shape(log_probs)[1]  # T
    output_steps = tf.shape(log_probs)[2]  # U + 1
    reverse_log_probs = tf.reverse(log_probs, axis=[-1])
    paddings = [[0, 0], [0, 0], [time_steps - 1, 0]]
    padded_reverse_log_probs = tf.pad(reverse_log_probs, paddings, "CONSTANT", constant_values=LOG_0)
    diagonals = matrix_diag_part_v2(
        padded_reverse_log_probs,
        k=(0, time_steps + output_steps - 2),
        padding_value=LOG_0,
    )

    return tf.transpose(diagonals, perm=[1, 0, 2])

def transition_probs(
        one_hot_labels,
        log_probs,
):
    """Compute the transition probabilities.
    
    Args:
        one_hot_labels (tf.Tensor): One-hot encoded labels with shape (batch_size, max_label_length, vocab_size).
        log_probs (tf.Tensor): Log probabilities with shape (batch_size, max_input_length, max_label_length + 1).

    Returns:
        tf.Tensor: Transition probabilities with shape (batch_size, max_input_length, max_label_length + 1).

    ## Equations
    - y(t, u) = Pr(y(u+1) | t, u)
    - Phi(t, u) = Pr(Phi | t, u)

    """
    blank_probs = log_probs[:, :, :, 0]
    truth_probs = tf.reduce_sum(tf.multiply(log_probs[:, :, :-1, :], one_hot_labels), axis=-1)

    return blank_probs, truth_probs

def forward_dp(
        bp_diags,
        tp_diags,
        batch_size,
        input_max_len,
        target_max_len,
):
    """Compute the forward pass of the RNN-T loss.
    
    Args:
        bp_diags (tf.Tensor): Blank probabilities diagonals with shape (batch_size, input_max_len, target_max_len).
        tp_diags (tf.Tensor): Truth probabilities diagonals with shape (batch_size, input_max_len, target_max_len).
        batch_size (int): The batch size.
        input_max_len (int): The maximum input length.
        target_max_len (int): The maximum target length.
        
    Returns:
        tf.Tensor: The forward probabilities with shape (batch_size, input_max_len, target_max_len).

    ## Equations
    - alpha(t, u) = alpha(t-1, u)*Phi(t-1, u) + alpha(t, u-1)*y(t, u-1) - (16)
    """
    def next_state(x, trans_probs):
        blank_probs = trans_probs[0]
        truth_probs = trans_probs[1]

        x_b = tf.concat([LOG_0 * tf.ones(shape=[batch_size, 1]), x[:, :-1] + blank_probs], axis=1)
        x_t = x + truth_probs

        x = tf.math.reduce_logsumexp(tf.stack([x_b, x_t], axis=0), axis=0)
        return x

    initial_alpha = tf.concat(
        [
            tf.zeros(shape=[batch_size, 1]),
            tf.ones(shape=[batch_size, input_max_len - 1]) * LOG_0,
        ],
        axis=1,
    )

    fwd = tf.scan(next_state, (bp_diags[:-1, :, :-1], tp_diags), initializer=initial_alpha)

    alpha = tf.transpose(tf.concat([tf.expand_dims(initial_alpha, axis=0), fwd], axis=0), perm=[1, 2, 0])
    alpha = matrix_diag_part_v2(alpha, k=(0, target_max_len - 1), padding_value=LOG_0)
    alpha = tf.transpose(tf.reverse(alpha, axis=[1]), perm=[0, 2, 1])

    return alpha

def backward_dp(
        bp_diags,
        tp_diags,
        batch_size,
        input_max_len,
        target_max_len,
        label_length,
        logit_length,
        blank_sl,
):
    """Compute the backward pass of the RNN-T loss.
    
    Args:
        bp_diags (tf.Tensor): Blank probabilities diagonals with shape (batch_size, input_max_len, target_max_len).
        tp_diags (tf.Tensor): Truth probabilities diagonals with shape (batch_size, input_max_len, target_max_len).
        batch_size (int): The batch size.
        input_max_len (int): The maximum input length.
        target_max_len (int): The maximum target length.
        label_length (tf.Tensor): The length of the labels with shape (batch_size,).
        logit_length (tf.Tensor): The length of the logits with shape (batch_size,).
        blank_sl (tf.Tensor): The blank sequence length with shape (batch_size,).

    Returns:
        tf.Tensor: The backward probabilities with shape (batch_size, input_max_len, target_max_len).

    ## Equations
    - beta(t, u) = beta(t+1, u)*Phi(t, u) + beta(t, u+1)*y(t, u) - (18)
    """
    def next_state(x, mask_and_trans_probs):
        mask_s, blank_probs_s, truth_probs = mask_and_trans_probs

        beta_b = tf.concat([x[:, 1:] + blank_probs_s, LOG_0 * tf.ones(shape=[batch_size, 1])], axis=1)
        beta_t = tf.concat([x[:, :-1] + truth_probs, LOG_0 * tf.ones(shape=[batch_size, 1])], axis=1)

        beta_next = reduce_logsumexp(tf.stack([beta_b, beta_t], axis=0), axis=0)
        masked_beta_next = nan_to_zero(beta_next * tf.expand_dims(mask_s, axis=1)) + nan_to_zero(
            x * tf.expand_dims((1.0 - mask_s), axis=1)
        )
        return tf.reshape(masked_beta_next, shape=tf.shape(x))

    # Initial beta for batches.
    initial_beta_mask = tf.one_hot(logit_length - 1, depth=input_max_len + 1)
    initial_beta = tf.expand_dims(blank_sl, axis=1) * initial_beta_mask + nan_to_zero(LOG_0 * (1.0 - initial_beta_mask))

    # Mask for scan iterations.
    mask = tf.sequence_mask(
        logit_length + label_length - 1,
        input_max_len + target_max_len - 2,
        dtype=tf.dtypes.float32,
    )
    mask = tf.transpose(mask, perm=[1, 0])

    bwd = tf.scan(
        next_state,
        (mask, bp_diags[:-1, :, :], tp_diags),
        initializer=initial_beta,
        reverse=True,
    )

    beta = tf.transpose(tf.concat([bwd, tf.expand_dims(initial_beta, axis=0)], axis=0), perm=[1, 2, 0])[:, :-1, :]
    beta = matrix_diag_part_v2(beta, k=(0, target_max_len - 1), padding_value=LOG_0)
    beta = tf.transpose(tf.reverse(beta, axis=[1]), perm=[0, 2, 1])

    return beta

def compute_rnnt_loss_and_grad_helper(logits, labels, label_length, logit_length):
    """Compute the RNN-T loss and gradients.
    
    Args:
        logits (tf.Tensor): The logits with shape (batch_size, input_max_len, target_max_len, vocab_size).
        labels (tf.Tensor): The labels with shape (batch_size, target_max_len).
        label_length (tf.Tensor): The length of the labels with shape (batch_size,).
        logit_length (tf.Tensor): The length of the logits with shape (batch_size,).
        
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple of the loss and gradients with respect to the logits.
        
    ## Loss functions for RNN-T
    - log-loss L= -ln(Pr(y* | x))
    - Pr(y* | x) = sum(alpha(t, u)beta(t, u))
    """
    batch_size = tf.shape(logits)[0]
    input_max_len = tf.shape(logits)[1]
    target_max_len = tf.shape(logits)[2]
    vocab_size = tf.shape(logits)[3]

    one_hot_labels = tf.one_hot(
        tf.tile(tf.expand_dims(labels, axis=1), multiples=[1, input_max_len, 1]),
        depth = vocab_size,
    )

    log_probs = tf.nn.log_softmax(logits)
    blank_probs, truth_probs = transition_probs(one_hot_labels, log_probs)
    bp_diags = extract_diagonals(blank_probs) # blank probs diagonals
    tp_diags = extract_diagonals(truth_probs) # truth probs diagonals

    # It masks positions beyond each target sequence’s actual length plus one \
    # (for the blank symbol at the end). This mask ensures that no unnecessary \
    # computations are performed on labels beyond the target sequence length.
    label_mask = tf.expand_dims(
        tf.sequence_mask(label_length + 1, maxlen=target_max_len, dtype=tf.float32),
        axis=1,
    )
    # This mask excludes blank transitions at the end of each output sequence, \
    # focusing on valid label emissions only.
    small_label_mask = tf.expand_dims(tf.sequence_mask(label_length, maxlen=target_max_len, dtype=tf.float32), axis=1)
    # This mask excludes positions beyond each input sequence’s actual length, \
    # ensuring alignment stays within bounds for input steps.
    input_mask = tf.expand_dims(tf.sequence_mask(logit_length, maxlen=input_max_len, dtype=tf.float32), axis=2)
    # Excludes the last input position, used to control blank transitions and \
    # ensure valid forward-backward alignment steps only.
    small_input_mask = tf.expand_dims(
        tf.sequence_mask(logit_length - 1, maxlen=input_max_len, dtype=tf.float32),
        axis=2,
    )
    mask = label_mask * input_mask
    grad_blank_mask = (label_mask * small_input_mask)[:, :-1, :] # applies to blank emissions
    grad_truth_mask = (small_label_mask * input_mask)[:, :, :-1] # applies to label emissions

    alpha = forward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len) * mask

    indices = tf.stack([logit_length - 1, label_length], axis=1)
    blank_sl = tf.gather_nd(blank_probs, indices, batch_dims=1)

    # Compute the final state probabilities.
    beta = (
        backward_dp(
            bp_diags,
            tp_diags,
            batch_size,
            input_max_len,
            target_max_len,
            label_length,
            logit_length,
            blank_sl,
        )
        * mask
    )
    beta = tf.where(tf.math.is_nan(beta), tf.zeros_like(beta), beta)
    final_state_probs = beta[:, 0, 0]

    # Compute gradients of loss w.r.t. blank log-probabilities.
    grads_blank = (
        -tf.exp(
            (
                alpha[:, :-1, :]
                + beta[:, 1:, :]
                - tf.reshape(final_state_probs, shape=[batch_size, 1, 1])
                + blank_probs[:, :-1, :]
            )
            * grad_blank_mask
        )
        * grad_blank_mask
    )
    grads_blank = tf.concat([grads_blank, tf.zeros(shape=(batch_size, 1, target_max_len))], axis=1)
    last_grads_blank = -1 * tf.scatter_nd(
        tf.concat(
            [
                tf.reshape(tf.range(batch_size, dtype=tf.int64), shape=[batch_size, 1]),
                tf.cast(indices, dtype=tf.int64),
            ],
            axis=1,
        ),
        tf.ones(batch_size, dtype=tf.float32),
        [batch_size, input_max_len, target_max_len],
    )
    grads_blank = grads_blank + last_grads_blank

    # Compute gradients of loss w.r.t. truth log-probabilities.
    grads_truth = (
        -tf.exp(
            (alpha[:, :, :-1] + beta[:, :, 1:] - tf.reshape(final_state_probs, shape=[batch_size, 1, 1]) + truth_probs)
            * grad_truth_mask
        )
        * grad_truth_mask
    )

    # Compute gradients of loss w.r.t. activations.
    a = tf.tile(
        tf.reshape(
            tf.range(target_max_len - 1, dtype=tf.int64),
            shape=(1, 1, target_max_len - 1, 1),
        ),
        multiples=[batch_size, 1, 1, 1],
    )
    b = tf.cast(
        tf.reshape(labels - 1, shape=(batch_size, 1, target_max_len - 1, 1)),
        dtype=tf.int64,
    )
    if not env_util.has_devices(["GPU", "TPU"]):
        b = tf.where(tf.equal(b, -1), tf.zeros_like(b), b)  # for cpu testing (index -1 on cpu will raise errors)
    c = tf.concat([a, b], axis=3)
    d = tf.tile(c, multiples=(1, input_max_len, 1, 1))
    e = tf.tile(
        tf.reshape(tf.range(input_max_len, dtype=tf.int64), shape=(1, input_max_len, 1, 1)),
        multiples=(batch_size, 1, target_max_len - 1, 1),
    )
    f = tf.concat([e, d], axis=3)
    g = tf.tile(
        tf.reshape(tf.range(batch_size, dtype=tf.int64), shape=(batch_size, 1, 1, 1)),
        multiples=[1, input_max_len, target_max_len - 1, 1],
    )
    scatter_idx = tf.concat([g, f], axis=3)
    # TODO - improve the part of code for scatter_idx computation.
    probs = tf.exp(log_probs)
    grads_truth_scatter = tf.scatter_nd(
        scatter_idx,
        grads_truth,
        [batch_size, input_max_len, target_max_len, vocab_size - 1],
    )
    grads = tf.concat(
        [
            tf.reshape(grads_blank, shape=(batch_size, input_max_len, target_max_len, -1)),
            grads_truth_scatter,
        ],
        axis=3,
    )
    grads_logits = grads - probs * (tf.reduce_sum(grads, axis=3, keepdims=True))

    loss = -final_state_probs
    return loss, grads_logits

@tf.function
def rnnt_loss(
    logits,
    labels,
    label_length,
    logit_length,
    blank=0,
    name=None,
):
    name = "rnnt_loss" if name is None else name
    with tf.name_scope(name):
        logits = tf.convert_to_tensor(logits, name="logits")
        labels = tf.convert_to_tensor(labels, name="labels")
        label_length = tf.convert_to_tensor(label_length, name="label_length")
        logit_length = tf.convert_to_tensor(logit_length, name="logit_length")

        args = [logits, labels, label_length, logit_length]

        @tf.custom_gradient
        def compute_rnnt_loss_and_grad(logits_t, labels_t, label_length_t, logit_length_t):
            """Compute RNN-T loss and gradients."""
            logits_t.set_shape(logits.shape)
            labels_t.set_shape(labels.shape)
            label_length_t.set_shape(label_length.shape)
            logit_length_t.set_shape(logit_length.shape)
            kwargs = dict(
                logits=logits_t,
                labels=labels_t,
                label_length=label_length_t,
                logit_length=logit_length_t,
            )
            result = compute_rnnt_loss_and_grad_helper(**kwargs)

            def grad(grad_loss):
                """Computes the gradients for the custom RNN-T loss with respect to `logits`.
                
                This function receives the incoming gradient `grad_loss` (usually from 
                further layers in backpropagation) and calculates the gradients for the 
                RNN-T loss with respect to the `logits`. The gradients are scaled by 
                `grad_loss`, and placeholders are set to `None` for non-differentiable 
                inputs (`labels`, `label_length`, `logit_length`).

                Args:
                    grad_loss (tf.Tensor): The upstream gradient of the loss, typically 
                        with shape (batch_size,). Used to scale the computed RNN-T gradients 
                        for backpropagation.

                Returns:
                    List[tf.Tensor or None]: A list of gradients for each argument in 
                    `compute_rnnt_loss_and_grad`:
                    
                    - The first element contains the scaled gradient tensor of shape 
                    (batch_size, input_max_len, target_max_len, vocab_size) for `logits`.
                    - Remaining elements are set to `None` since `labels`, `label_length`, 
                    and `logit_length` are non-differentiable. TensorFlow uses `None` values 
                    in the gradient output to indicate to the optimizer that these arguments 
                    should be ignored during backpropagation.
                """
                
                # [BS] -> [BS, 1, 1, 1] for broadcasting with logits ie result[1]
                grads = [tf.reshape(grad_loss, [-1, 1, 1, 1]) * result[1]]
                # Pad with None for the gradients of the other arguments. They are not needed.
                # [None] * 0 = [], [None] * 1 = [None], [None] * 2 = [None, None], ... In this case,
                # the other not needed gradients are for labels, label_length, and logit_length.
                # Hence grads += [None] * (len(args) - len(grads)) is equivalent to grads += [None, None, None]
                grads += [None] * (len(args) - len(grads))
                return grads

            return result[0], grad

        return compute_rnnt_loss_and_grad(*args)
