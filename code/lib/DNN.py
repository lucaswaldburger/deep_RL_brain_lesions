import logging
import os

import numpy as np
import tensorflow as tf

from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Agent actions: 0 (right), 1 (down), 2 (scale up), 3 (aspect ratio up),
# 4 (left), 5 (up), 6 (scale down), 7 (aspect ratio down),
# 8 (split horizontal), 9 (split vertical), 10 (termination)
VALID_ACTIONS = list(range(11))


class StateProcessor:
    """Converts raw RGB images to grayscale and resizes to (84, 84)."""

    def __init__(self):
        with tf.variable_scope("state_processor"):
            dim = DEFAULT_CONFIG.dimension
            self.input_state = tf.placeholder(shape=[dim, dim, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.resize_images(
                self.output, [dim, dim],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A TensorFlow session object.
            state: An [H, W, 3] RGB image.

        Returns:
            A processed [84, 84] grayscale array.
        """
        return sess.run(self.output, {self.input_state: state})


class Estimator:
    """Q-Value Estimator neural network.

    Used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """Builds the TensorFlow computation graph."""
        cfg = DEFAULT_CONFIG
        dim = cfg.dimension

        self.X_pl = tf.placeholder(shape=[None, dim, dim, 4], dtype=tf.uint8, name="X")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.keep_prob = tf.placeholder(tf.float32)

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

        flattened = tf.contrib.layers.flatten(conv3)
        flattened = tf.nn.dropout(flattened, self.keep_prob)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(
            cfg.learning_rate, cfg.rms_decay, 0.0, cfg.rms_epsilon,
        )
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step(),
        )

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions)),
        ])

    def predict(self, sess, s, keep_prob=1):
        """
        Predicts action values.

        Args:
            sess: TensorFlow session.
            s: State input of shape [batch_size, 84, 84, 4].

        Returns:
            Array of shape [batch_size, NUM_VALID_ACTIONS].
        """
        return sess.run(self.predictions, {self.X_pl: s, self.keep_prob: keep_prob})

    def visualize_layers(self, sess, s, layer):
        """
        Returns convolutional layer activations for visualization.

        Args:
            sess: TensorFlow session.
            s: State input of shape [batch_size, 84, 84, 4].
            layer: Layer number ('1', '2', or '3').

        Returns:
            Activations tensor of the requested layer.
        """
        conv1, conv2, conv3 = sess.run(
            [self.conv1, self.conv2, self.conv3], {self.X_pl: s},
        )
        layer_map = {'1': conv1, '2': conv2, '3': conv3}
        return layer_map.get(layer, conv3)

    def update(self, sess, s, a, y, keep_prob=None):
        """
        Updates the estimator towards the given targets.

        Args:
            sess: TensorFlow session object.
            s: State input of shape [batch_size, 84, 84, 4].
            a: Chosen actions of shape [batch_size].
            y: Targets of shape [batch_size].
            keep_prob: Dropout keep probability (defaults to config value).

        Returns:
            The calculated loss on the batch.
        """
        if keep_prob is None:
            keep_prob = DEFAULT_CONFIG.dropout_keep_prob
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a, self.keep_prob: keep_prob}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict,
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


class ModelParametersCopier:
    """Copies model parameters from one estimator to another."""

    def __init__(self, estimator1, estimator2):
        """
        Args:
            estimator1: Estimator to copy parameters from.
            estimator2: Estimator to copy parameters to.
        """
        e1_params = sorted(
            [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)],
            key=lambda v: v.name,
        )
        e2_params = sorted(
            [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)],
            key=lambda v: v.name,
        )
        self.update_ops = [e2_v.assign(e1_v) for e1_v, e2_v in zip(e1_params, e2_params)]

    def make(self, sess):
        """Executes the copy operation.

        Args:
            sess: TensorFlow session instance.
        """
        sess.run(self.update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator.

    Args:
        estimator: An estimator that returns Q-values for a given state.
        nA: Number of actions in the environment.

    Returns:
        A policy function.
    """
    def policy_fn(sess, observation, epsilon):
        """
        Predicts Q-values and returns a probability distribution over actions.

        Args:
            sess: TensorFlow session object.
            observation: State input of shape [84, 84, 4].
            epsilon: Probability of taking a random action.

        Returns:
            Tuple of (action probabilities, Q-values).
        """
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A, q_values
    return policy_fn
