import logging
import os

import numpy as np
import tensorflow as tf

from config import DEFAULT_CONFIG
from lib.DNN import Estimator, StateProcessor, VALID_ACTIONS, make_epsilon_greedy_policy

logger = logging.getLogger(__name__)


def setup_model(model_name, with_summaries=True):
    """Reset the TF graph and create estimators, state processor, and policy.

    Args:
        model_name: Name of the model (used to locate experiment directory).
        with_summaries: Whether to attach a summary writer to the Q-estimator.

    Returns:
        Tuple of (experiment_dir, q_estimator, state_processor, policy).
    """
    tf.reset_default_graph()
    tf.Variable(0, name='global_step', trainable=False)

    experiment_dir = os.path.abspath(
        os.path.join(DEFAULT_CONFIG.experiments_dir, model_name),
    )

    summaries_dir = experiment_dir if with_summaries else None
    q_estimator = Estimator(scope="q_estimator", summaries_dir=summaries_dir)
    state_processor = StateProcessor()
    policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))

    return experiment_dir, q_estimator, state_processor, policy


def load_checkpoint(sess, experiment_dir, subdir="bestModel"):
    """Load a TF checkpoint from the experiment directory.

    Args:
        sess: TensorFlow session.
        experiment_dir: Path to the experiment directory.
        subdir: Subdirectory containing the checkpoint.

    Returns:
        The Saver instance.
    """
    checkpoint_dir = os.path.join(experiment_dir, subdir)
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        logger.info("Loading model checkpoint %s...", latest_checkpoint)
        saver.restore(sess, latest_checkpoint)
    return saver


def run_episode(env, im2, state_processor, policy, sess, epsilon=None):
    """Run a single episode of agent interaction with an image.

    The agent acts until termination (action 10) or reaching the
    max actions threshold.

    Args:
        env: An ObjLocaliser instance.
        im2: The original image as a numpy array.
        state_processor: A StateProcessor instance.
        policy: An epsilon-greedy policy function.
        sess: TensorFlow session.
        epsilon: Exploration rate (defaults to config eval_epsilon).

    Returns:
        Tuple of (success, num_steps) where success is True if the agent
        received the positive termination reward (3).
    """
    cfg = DEFAULT_CONFIG
    if epsilon is None:
        epsilon = cfg.eval_epsilon

    env.Reset(np.array(im2))
    state = env.wrapping()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)

    t = 0
    action = 0
    success = False

    while action != 10 and t < cfg.max_actions_per_episode:
        action_probs, _ = policy(sess, state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        reward = env.takingActions(VALID_ACTIONS[action])
        if reward == 3:
            success = True

        next_state = env.wrapping()
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        state = next_state
        t += 1

    return success, t
