import gc
import logging

from lib.DataProvider import PascalDataProvider

logger = logging.getLogger(__name__)


def giveData(which_set, batch_size):
    """Load .npz data files.

    Args:
        which_set: 'train' or 'test'.
        batch_size: Batch size.

    Yields:
        A PascalDataProvider instance.
    """
    if which_set == 'train':
        logger.info("Loading training input file...")
        yield PascalDataProvider(1, which_set=which_set, batch_size=batch_size)
    if which_set == 'test':
        logger.info("Loading testing input file...")
        yield PascalDataProvider("", which_set=which_set, batch_size=batch_size)


def extractData(objClassName, type, which_set, batch_size):
    """Extract images and ground truths for a given category and tumor type.

    Args:
        objClassName: List of object category names (e.g. ['T1ce']).
        type: Tumor type ('HGG' or 'LGG').
        which_set: 'train' or 'test'.
        batch_size: Batch size.

    Yields:
        Tuple of (image_dict, groundtruth_dict).
    """
    for fileInp in giveData(which_set, batch_size):
        for img_batch, targ_batch in fileInp:
            for batch_index, _ in enumerate(img_batch):
                found = False
                if (targ_batch[batch_index]['objName'] == objClassName[0]
                        and targ_batch[batch_index]['type'] == type):
                    found = True
                    groundtruth = {
                        'xmin': targ_batch[batch_index]['xmin'],
                        'ymin': targ_batch[batch_index]['ymin'],
                        'xmax': targ_batch[batch_index]['xmax'],
                        'ymax': targ_batch[batch_index]['ymax'],
                        'objName': targ_batch[batch_index]['objName'],
                        'type': targ_batch[batch_index]['type'],
                    }

                if found:
                    yield img_batch[batch_index], groundtruth

            del img_batch
            del targ_batch

        del fileInp
        gc.collect()
