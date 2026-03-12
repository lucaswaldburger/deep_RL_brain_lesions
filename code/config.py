from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration for the DQL brain lesion localizer."""

    # Image / agent window
    dimension: int = 84
    step_factor: float = 0.2
    max_aspect_ratio: float = 6.0
    min_aspect_ratio: float = 0.15
    min_box_side: int = 10
    iou_threshold: float = 0.5

    # Episode control
    max_actions_per_episode: int = 50
    eval_epsilon: float = 0.2

    # Training
    batch_size: int = 32
    eval_set_size: int = 100
    eval_every_n_images: int = 20

    # Paths
    data_dir: str = "../new_data"
    experiments_dir: str = "../experiments"

    # CNN / optimizer hyper-parameters
    learning_rate: float = 0.00025
    rms_decay: float = 0.99
    rms_epsilon: float = 1e-6
    dropout_keep_prob: float = 0.5


DEFAULT_CONFIG = Config()
