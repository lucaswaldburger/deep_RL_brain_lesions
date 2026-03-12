import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from config import DEFAULT_CONFIG
from lib.Agent import ObjLocaliser
from lib.session_utils import setup_model, load_checkpoint

logger = logging.getLogger(__name__)


def plotNNFilter(units, model_name, layer_num):
    """Save visualizations of convolutional layer filters.

    Args:
        units: Convolutional layer activations.
        model_name: Model name (used for output path).
        layer_num: Layer number being visualized.
    """
    filters = units.shape[3]
    experiment_dir = os.path.join(
        DEFAULT_CONFIG.experiments_dir, model_name, "visu", "layer_{}".format(layer_num),
    )
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    for i in range(filters):
        fig = plt.figure(1, figsize=(10, 10))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
        fig.suptitle('layer{} filter{}'.format(layer_num, i + 1), fontsize=60)
        fig.savefig(os.path.join(
            experiment_dir, 'layer{}filter{}.png'.format(layer_num, i + 1),
        ))
        plt.close()
        logger.info("Filter %d plotted.", i)

    logger.info("Plots saved to %s", experiment_dir)


def visualize_layers(model_name, add, layer_num):
    """Visualize CNN layer activations for a given image.

    Args:
        model_name: Model to load for visualization.
        add: Path to the input image.
        layer_num: Layer number to visualize ('1', '2', or '3').
    """
    experiment_dir, q_estimator, state_processor, policy = setup_model(model_name)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        load_checkpoint(sess, experiment_dir, subdir="bestModel")

        im2 = np.array(Image.open(add))
        dummy_target = {'xmin': [0], 'xmax': [1], 'ymin': [0], 'ymax': [1]}
        env = ObjLocaliser(np.array(im2), dummy_target)

        env.Reset(np.array(im2))
        state = env.wrapping()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)

        dim = DEFAULT_CONFIG.dimension
        layer = q_estimator.visualize_layers(sess, state.reshape((-1, dim, dim, 4)), layer_num)
        plotNNFilter(layer, model_name, layer_num)
