import glob
import logging
import os

import numpy as np
from PIL import Image
from scipy import ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MIN_MASK_SIZE = 1000
MODALITIES = ['T1', 'T1ce', 'T2', 'Flair']
DATASET_DIR = 'brain_tumor_data/dataset'
OUTPUT_DIR = 'new_data'


def find_bounding_box(file_path):
    """Find the bounding box of the largest connected component in a segmentation mask.

    Args:
        file_path: Path to the segmentation mask image.

    Returns:
        Tuple of (xmin, xmax, ymin, ymax).
    """
    target = np.array(Image.open(file_path))
    mask = target > 0
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < MIN_MASK_SIZE
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)
    slice_x, slice_y = ndimage.find_objects(label_im)[0]
    return slice_x.start, slice_x.stop, slice_y.start, slice_y.stop


def create_example(file_path, xmin, xmax, ymin, ymax):
    """Create a data example dict from an image and bounding box coordinates.

    Args:
        file_path: Path to the MRI image.
        xmin, xmax, ymin, ymax: Bounding box coordinates.

    Returns:
        Dictionary with image data and metadata.
    """
    def find_class(filename, classes=MODALITIES):
        for c in classes:
            if c in filename:
                return c

    def find_type(filename, types=['HGG', 'LGG']):
        for t in types:
            if t in filename:
                return t
        return 'None'

    def find_year(filename):
        return '20' + filename.split('_')[2][-2:]

    img = Image.open(file_path)
    width, height = img.size
    filename = file_path
    class_text = find_class(filename)

    return {
        'image_height': height,
        'image_width': width,
        'image_depth': len(img.getbands()),
        'image_filename': filename,
        'image': np.array(img).tobytes(),
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'classes': class_text,
        'type': find_type(filename),
        'year': find_year(filename),
    }


def build_dataset(test_idxs, is_test=False):
    """Build and save a dataset (train or test) as .npz files.

    Args:
        test_idxs: List of directory indices reserved for testing.
        is_test: If True, build test set; otherwise build training set.

    Returns:
        Tuple of (inputs_list, targets_list).
    """
    inputs = []
    targets = []
    prefix = "test" if is_test else "train1"

    for c in MODALITIES:
        for idx, dir_path in enumerate(sorted(glob.glob(os.path.join(DATASET_DIR, '*')))):
            if is_test and idx not in test_idxs:
                continue
            if not is_test and idx in test_idxs:
                continue

            seg_path = os.path.join(dir_path, 'Lesion_Seg.jpeg')
            try:
                xmin, xmax, ymin, ymax = find_bounding_box(seg_path)
            except Exception:
                logger.warning("Could not process %s, skipping.", seg_path)
                continue

            example = create_example(
                os.path.join(dir_path, c + '.jpeg'), xmin, xmax, ymin, ymax,
            )

            inputs.append({
                'image_height': example['image_height'],
                'image_width': example['image_width'],
                'image_depth': example['image_depth'],
                'image': example['image'],
                'image_filename': example['image_filename'],
            })

            targets.append({
                'xmin': example['xmin'],
                'xmax': example['xmax'],
                'ymin': example['ymin'],
                'ymax': example['ymax'],
                'objName': example['classes'],
                'type': example['type'],
            })

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    np.savez_compressed(os.path.join(OUTPUT_DIR, prefix + '_input.npz'), inputs)
    np.savez_compressed(os.path.join(OUTPUT_DIR, prefix + '_target.npz'), targets)
    logger.info("Saved %s set: %d examples.", "test" if is_test else "train", len(inputs))

    return inputs, targets


if __name__ == "__main__":
    with open('test_idx.txt', 'r') as f:
        arr = f.read()
    test_idxs = [int(a) for a in arr.split(',')]

    logger.info("Building training set...")
    build_dataset(test_idxs, is_test=False)

    logger.info("Building testing set...")
    build_dataset(test_idxs, is_test=True)
