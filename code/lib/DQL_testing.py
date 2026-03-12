import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from config import DEFAULT_CONFIG
from lib.Agent import ObjLocaliser
from lib.DNN import VALID_ACTIONS
from lib.ReadData import extractData
from lib.session_utils import setup_model, load_checkpoint, run_episode

logger = logging.getLogger(__name__)


def DQL_testing(num_episodes, category, model_name, type):
    """Evaluate a model on the testing set.

    Args:
        num_episodes: Number of episodes per image.
        category: Category to evaluate.
        model_name: Name of the model to load.
        type: Tumor type ('HGG' or 'LGG').

    Returns:
        Mean precision for the given category over the test set.
    """
    cfg = DEFAULT_CONFIG

    data_dir = cfg.data_dir
    if not (os.path.isfile(os.path.join(data_dir, "test_input.npz"))
            or os.path.isfile(os.path.join(data_dir, "test_target.npz"))):
        logger.warning("Test data files not found in %s", data_dir)
        return 0
    else:
        logger.info("Test data files found.")

    experiment_dir, q_estimator, state_processor, policy = setup_model(model_name)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        load_checkpoint(sess, experiment_dir, subdir="bestModel")

        precisions = []

        for indx, tmp in enumerate(extractData([category], type, "test", cfg.batch_size)):
            img = tmp[0]
            target = tmp[1]
            succ = 0

            im2 = Image.frombytes("RGB", (img['image_width'], img['image_height']), img['image'])
            env = ObjLocaliser(np.array(im2), target)
            logger.info("Image %d is being loaded: %s", indx, img['image_filename'])

            for i_episode in range(num_episodes):
                success, t = run_episode(env, np.array(im2), state_processor, policy, sess)
                if success:
                    succ += 1
                logger.debug("Number of actions for step %d: %d", i_episode, t)

            precisions.append(float(succ) / num_episodes)
            logger.info("Image %s precision: %s", img['image_filename'], precisions[-1])

    logger.info("Number of images: %d", len(precisions))
    logger.info("Mean precision: %s", np.mean(precisions))

    return np.mean(precisions)
