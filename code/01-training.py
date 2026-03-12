import argparse
import logging

from lib.DQL import DQL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an object localizer')

    parser.add_argument('-n', '--num_episodes', type=int, default=5,
                        help="Number of episodes per image. Default: 5")
    parser.add_argument('-rms', '--replay_memory_size', type=int, default=500000,
                        help="Replay memory capacity. Default: 500000")
    parser.add_argument('-rmis', '--replay_memory_init_size', type=int, default=500,
                        help="Initial replay memory size. Default: 500")
    parser.add_argument('-u', '--update_target_estimator_every', type=int, default=10000,
                        help="Steps between target network updates. Default: 10000")
    parser.add_argument('-d', '--discount_factor', type=float, default=0.99,
                        help="Discount factor. Default: 0.99")
    parser.add_argument('-es', '--epsilon_start', type=float, default=1.0,
                        help="Epsilon decay schedule start. Default: 1.0")
    parser.add_argument('-ee', '--epsilon_end', type=float, default=0.2,
                        help="Epsilon decay schedule end. Default: 0.2")
    parser.add_argument('-ed', '--epsilon_decay_steps', type=int, default=500,
                        help="Epsilon decay steps. Default: 500")
    parser.add_argument('-c', '--category', type=str, nargs='+', default=None,
                        help="Image categories for training (e.g. -c T1ce)")
    parser.add_argument('-m', '--model_name', type=str, default='default_model',
                        help="Model save name. Default: default_model")
    parser.add_argument('-t', '--type', type=str, default=None,
                        help="Tumor type: HGG or LGG")

    args = parser.parse_args()

    DQL(args.num_episodes,
        args.replay_memory_size,
        args.replay_memory_init_size,
        args.update_target_estimator_every,
        args.discount_factor,
        args.epsilon_start,
        args.epsilon_end,
        args.epsilon_decay_steps,
        args.category,
        args.model_name,
        args.type)
