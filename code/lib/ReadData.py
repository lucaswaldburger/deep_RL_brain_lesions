import gc
import DataProvider as DataProvider

def giveData(which_set, batch_size):
    """
    Reads .npz files
    Args:
       which_set: This indicates whether training or testing set needs to be loaded
       batch_size: Batch size
    Returns:
       A batch of images
    """
    if which_set == 'train':
        print("train input file is loading...".format(1))
        yield DataProvider.PascalDataProvider(1, which_set = which_set, batch_size = batch_size)
    if which_set == 'test':
        print("test input file is loading...")
        yield DataProvider.PascalDataProvider("", which_set = which_set, batch_size = batch_size)

def extractData(objClassName, type, which_set, batch_size):
    """
    Reads dataset 
    Args:
      objClassName: Object category that is needed
      which_set: This indicates whether training or testing set needs to be loaded
      batch_size: Batch size
    Returns:
      Image and its ground truth from the given category
    """
    
    # Loading .npz files
    for fileInp in giveData(which_set, batch_size):
        # Getting images batch in the current file
        for img_batch, targ_batch in fileInp:
            # Iterating over the current batch
            for batch_index, _ in enumerate(img_batch):
                xmin = []
                xmax = []
                ymin = []
                ymax = []
                objectName = ''
                found = False
                # Iterates over objects in the current image
                if targ_batch[batch_index]['objName'] == objClassName[0] and targ_batch[batch_index]['type'] == type:
                    found = True
		            # Checks whether the desired object exist in the loaded image                
                    groundtruth = {'xmin':targ_batch[batch_index]['xmin'], 
                    'ymin':targ_batch[batch_index]['ymin'], 
                    'xmax':targ_batch[batch_index]['xmax'], 
                    'ymax':targ_batch[batch_index]['ymax'], 
                    'objName':targ_batch[batch_index]['objName'],
                    'type':targ_batch[batch_index]['type']}

                if found:
                    yield img_batch[batch_index], groundtruth
                else:
                    pass
                
            del img_batch
            del targ_batch

        del fileInp
        gc.collect()