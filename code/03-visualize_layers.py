import argparse
import logging

from lib.DQL_visualization_layers import visualize_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CNN layers')

    parser.add_argument('-m', '--model_name', type=str, default='default_model',
                        help="Model name to load. Default: default_model")
    parser.add_argument('-i', '--image_path', type=str, default=None,
                        help="Path to an image.")
    parser.add_argument('-ln', '--layer_num', type=str, default="1",
                        help="Layer number to visualize (1, 2, or 3). Default: 1")

    args = parser.parse_args()

    visualize_layers(args.model_name, args.image_path, args.layer_num)
