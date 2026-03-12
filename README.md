# Deep Q-Learning for Brain Lesion Localization

A Deep Q-Learning (DQL) agent that localizes brain lesions in MRI images. The agent uses a sliding window with 11 possible actions (move, scale, change aspect ratio, split, and terminate) to iteratively refine a bounding box prediction for tumor regions.

Supports multiple MRI modalities (T1, T1ce, T2, Flair) and tumor types (HGG, LGG) from the BraTS dataset.

## Requirements

- Python 3.6+
- TensorFlow 1.8
- See `requirements.txt` for full dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── requirements.txt                # Python dependencies
├── code/
│   ├── config.py                   # Central configuration (hyperparameters, paths)
│   ├── 00-import_data_py3.py       # Data preparation: MRI images -> .npz files
│   ├── 01-training.py              # Training entry point
│   ├── 02-testing.py               # Evaluation entry point
│   ├── 03-visualize_layers.py      # Visualize CNN layers (single image)
│   ├── 03-vis_all_layers.py        # Visualize CNN layers (batch)
│   ├── 04-visualize_actions.py     # Visualize action sequence (single image)
│   ├── 04-vis_all_actions.py       # Visualize action sequences (batch)
│   ├── 05-view_inputs.py           # Quick data inspection
│   └── lib/
│       ├── Agent.py                # ObjLocaliser: RL environment (actions, rewards, IoU)
│       ├── DNN.py                  # Q-network, state processor, epsilon-greedy policy
│       ├── DQL.py                  # Deep Q-Learning training loop
│       ├── DQL_testing.py          # Model evaluation
│       ├── DQL_visualization_layers.py   # CNN filter visualization
│       ├── DQL_visualization_actions.py  # Action sequence animation
│       ├── DataProvider.py         # Batched data loading from .npz files
│       ├── ReadData.py             # Data extraction by category and tumor type
│       └── session_utils.py        # Shared TensorFlow session setup helpers
└── experiments/                    # Trained models, checkpoints, reports, visualizations
```

## Usage

All scripts should be run from the `code/` directory.

### 1. Data Preparation

Place BraTS data under `brain_tumor_data/dataset/` and create a `test_idx.txt` file listing test set indices (comma-separated). Then run:

```bash
python 00-import_data_py3.py
```

This creates `.npz` files in `new_data/` containing the training and test sets.

### 2. Training

```bash
python 01-training.py \
    -c T1ce \
    -t HGG \
    -n 5 \
    -d 0.99 \
    -es 1.0 \
    -ee 0.2 \
    -ed 500 \
    -m my_model
```

Key arguments:
- `-c`: MRI modality (T1, T1ce, T2, Flair)
- `-t`: Tumor type (HGG or LGG)
- `-n`: Episodes per image (default: 5)
- `-d`: Discount factor (default: 0.99)
- `-m`: Model name for saving checkpoints

Checkpoints and TensorBoard logs are saved under `experiments/`.

### 3. Evaluation

```bash
python 02-testing.py -c T1ce -t HGG -m my_model -n 15
```

### 4. Visualization

Visualize CNN layer activations:

```bash
python 03-visualize_layers.py -m my_model -i path/to/image.jpeg -ln 1
```

Visualize the agent's action sequence as a video:

```bash
python 04-visualize_actions.py -m my_model -i path/to/image.jpeg -g 10 20 50 60
```

## Configuration

All hyperparameters and paths are centralized in `code/config.py`. Modify the `Config` dataclass to change defaults such as image dimension, learning rate, batch size, or data directories.
