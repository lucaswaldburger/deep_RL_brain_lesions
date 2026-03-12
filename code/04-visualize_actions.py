import argparse
import logging

from lib.DQL_visualization_actions import visualizing_seq_act

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize sequence of actions')

    parser.add_argument('-m', '--model_name', type=str, default='default_model',
                        help="Model name to load. Default: default_model")
    parser.add_argument('-i', '--image_path', type=str, default=None,
                        help="Path to an image.")
    parser.add_argument('-g', '--ground_truth', type=int, nargs='+', default=[0, 0, 1, 1],
                        help="Target coordinates: xmin ymin xmax ymax. Default: 0 0 1 1")
    parser.add_argument('-n', '--name', type=str, default="anim",
                        help="Output file name. Default: anim")

    args = parser.parse_args()

    visualizing_seq_act(
        args.model_name, args.image_path, args.ground_truth, args.name,
    )
