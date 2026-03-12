import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import DEFAULT_CONFIG
from lib.Agent import ObjLocaliser
from lib.DNN import VALID_ACTIONS
from lib.session_utils import setup_model, load_checkpoint

logger = logging.getLogger(__name__)


def visualizing_seq_act(model_name, add, ground_truth, output_name):
    """Visualize the sequence of actions taken by the agent as a video.

    Args:
        model_name: Model to load for visualization.
        add: Path to the input image.
        ground_truth: Target coordinates [xmin, ymin, xmax, ymax].
        output_name: Name for the output video file.
    """
    cfg = DEFAULT_CONFIG

    experiment_dir, q_estimator, state_processor, policy = setup_model(model_name)

    target = {
        'xmin': [ground_truth[0]],
        'xmax': [ground_truth[2]],
        'ymin': [ground_truth[1]],
        'ymax': [ground_truth[3]],
    }
    im2 = np.array(Image.open(add))
    env = ObjLocaliser(np.array(im2), target)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        load_checkpoint(sess, experiment_dir, subdir="bestModel")

        fig = plt.figure()
        ims = []
        final_reward = 0

        while final_reward != 3:
            plt.close()
            fig = plt.figure()
            ims = []

            env.Reset(np.array(im2))
            state = env.wrapping()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)

            t = 0
            action = 0

            while (action != 10) and (t < cfg.max_actions_per_episode):
                action_probs, qs = policy(sess, state, cfg.eval_epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                reward = env.takingActions(VALID_ACTIONS[action])
                next_state = env.wrapping()
                if reward == 3:
                    final_reward = 3

                imgplot = plt.imshow(env.my_draw())
                ims.append([imgplot])

                next_state = state_processor.process(sess, next_state)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                state = next_state
                t += 1

            logger.debug("Unsuccessful attempt, retrying...")

        anim_dir = os.path.join(experiment_dir, "anim")
        if not os.path.exists(anim_dir):
            os.makedirs(anim_dir)

        ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
        output_path = os.path.join(anim_dir, "{}.mp4".format(output_name))
        ani.save(output_path)
        logger.info("Video saved to %s", output_path)
