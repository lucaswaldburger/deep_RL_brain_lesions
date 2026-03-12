import numpy as np
from PIL import Image
import glob
import gc
from scipy import ndimage

def find_bounding_box(file_path, show_box=False):
    target = Image.open(file_path)
    target = np.array(target)
    mask = target > 0
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < 1000
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)
    slice_x, slice_y = ndimage.find_objects(label_im)[0]
    xmin, xmax = slice_x.start, slice_x.stop
    ymin, ymax = slice_y.start, slice_y.stop
    return xmin, xmax, ymin, ymax

def create_example(file_path, xmin, xmax, ymin, ymax):
    def find_class(filename,classes=['T1','T1ce','T2','Flair']):
        for c in classes:
            if c in filename:
                return c

    def find_type(filename,types=['HGG','LGG']):
        for t in types:
            if t in filename:
                return t
        return 'None'

    def find_year(filename):
        return '20' + filename.split('_')[2][-2:]

    # import image
    test = Image.open(file_path)
    width, height = test.__dict__['_size']
    filename = test.__dict__['filename']
    class_text = find_class(filename)

    example = {'image_height': height,
                'image_width': width,
                'image_depth': test.__dict__['layers'],
                'image_filename': filename,
                'image': np.array(test).tobytes(),
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'classes': class_text,
                'type': find_type(filename),
                'year': find_year(filename)
                }
    return example

f = open('test_idx.txt','r')
arr = f.read()
test_idxs = [int(a) for a in arr.split(',')]

# make training set
dest_dir = 'new_data/'
inputs = []
targets = []
for c in ['T1','T1ce','T2','Flair']:
    for idx, dir in enumerate(glob.glob('brain_tumor_data/dataset/*')):
        if idx in test_idxs:
            continue
        try:
            xmin, xmax, ymin, ymax = find_bounding_box(dir+'/Lesion_Seg.jpeg',show_box=False)
        except:
            continue
        example = create_example(dir+'/'+c+'.jpeg', xmin, xmax, ymin, ymax)

        tmp = {'image_height':example['image_height'],
        'image_width':example['image_width'], 
        'image_depth':example['image_depth'], 
        'image':example['image'], 
        'image_filename':example['image_filename']}

        inputs.append(tmp)

        tmp = {'xmin':example['xmin'],
        'xmax':example['xmax'],
        'ymin':example['ymin'],
        'ymax':example['ymax'],
        'objName':example['classes']}
        
        targets.append(tmp)
np.savez_compressed(dest_dir+'train1_input.npz', inputs)
np.savez_compressed(dest_dir+'train1_target.npz', targets)

# make testing set
dest_dir = 'new_data/'
inputs = []
targets = []
for c in ['T1','T1ce','T2','Flair']:
    for idx, dir in enumerate(glob.glob('brain_tumor_data/dataset/*')):
        if idx not in test_idxs:
            continue
        try:
            xmin, xmax, ymin, ymax = find_bounding_box(dir+'/Lesion_Seg.jpeg',show_box=False)
        except:
            continue

        example = create_example(dir+'/'+c+'.jpeg', xmin, xmax, ymin, ymax)

        tmp = {'image_height':example['image_height'],
        'image_width':example['image_width'], 
        'image_depth':example['image_depth'], 
        'image':example['image'], 
        'image_filename':example['image_filename']}

        inputs.append(tmp)

        tmp = {'xmin':example['xmin'],
        'xmax':example['xmax'],
        'ymin':example['ymin'],
        'ymax':example['ymax'],
        'objName':example['classes']}
        
        targets.append(tmp)
np.savez_compressed(dest_dir+'test1_input.npz', inputs)
np.savez_compressed(dest_dir+'test1_target.npz', targets)