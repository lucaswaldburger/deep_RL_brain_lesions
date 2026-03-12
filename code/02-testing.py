import argparse
import logging
import os

import numpy as np

from config import DEFAULT_CONFIG
from lib.DQL_testing import DQL_testing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model on test set')

    parser.add_argument('-n', '--num_episodes', type=int, default=15,
                        help="Number of episodes per image. Default: 15")
    parser.add_argument('-c', '--category', type=str, nargs='+', default=None,
                        help="Categories for testing (e.g. -c T1ce)")
    parser.add_argument('-m', '--model_name', type=str, default='default_model',
                        help="Model name to load. Default: default_model")
    parser.add_argument('-t', '--type', type=str, default=None,
                        help="Tumor type: HGG or LGG")

    args = parser.parse_args()

    report_dir = os.path.join(
        DEFAULT_CONFIG.experiments_dir,
        "{}_{}_experiments".format(args.category[0], args.type),
        args.model_name, "report",
    )
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_path = os.path.join(
        report_dir,
        "evaluate_{}_{}.txt".format(args.category[0], args.type),
    )
    f = open(report_path, 'w')

    MAP = []
    for category in args.category:
        logger.info("%s images are being evaluated...", category)
        MAP.append(DQL_testing(args.num_episodes, category, args.model_name, args.type))
        f.write("Precision for {} category: {}\n".format(category, MAP[-1]))

    mean_map = np.mean(MAP)
    f.write("MAP over the given category(s): {}\n".format(mean_map))
    logger.info("MAP over the given category(s): %s", mean_map)
    f.close()
