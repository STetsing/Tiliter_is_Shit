from PIL import Image
import numpy as np
import os

flowers_dir = './flowers/'
new_img_size = (50, 50)
classes = 5

# the flowers are classed according to the sorting

def one_hot_converter(data, n_class):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_class)[targets]

def loadImages(dir, label):
    img_files = sorted(os.listdir(flowers_dir + dir))
    images = []
    labels = []
    for im in img_files:
        img = Image.open(flowers_dir + dir + str('/') + im)
        
        # One should consider removing some channels or convert the image to grey scale for faster computations
        # as these feature migh not bring that much of information
        
        #if(img.mode != 'L'):
        #    img = img.convert('L')
        
        # reshape image as vector
        img = img.resize(new_img_size, Image.ANTIALIAS)
        #print(img.size)

        in_img = np.asarray(img).reshape(np.prod(np.asarray(img).shape))/255
        images.append(np.asarray(in_img))
        labels.append(np.asarray(label ))
    return [images, np.squeeze(labels)]

def shuffle(arr1, arr2, seed):
    '''Shuffle 2 array in the same order'''
    rds = np.random.RandomState(seed)
    rds.shuffle(arr1)
    rds.seed(seed)
    rds.shuffle(arr2)

def loadTrainingImages():
    '''Load the training mnist images. Return the images and their coresponding labels'''
    print('loading the training images ...')
    dirs = [d for d in sorted(os.listdir(flowers_dir)) if os.path.isdir(os.path.join(flowers_dir, d))]
    
    train_imgs = np.array([], dtype=np.int64).reshape(0, np.prod(new_img_size) * 3)
    train_lbls = np.array([], dtype=np.int64).reshape(0, 1)
    
    # parallel execution might be implemented
    for i,d in enumerate(dirs):
        print('processing:', i, '/', len(dirs))
        print(d, 'has the label', i +1 )
        
        imgs, lbs = loadImages(d, [i+1] )
        #lbs = np.expand_dims(lbs, axis=1)
        lbs = np.expand_dims(lbs, axis=1)
 
        train_imgs = np.vstack(([train_imgs, imgs]))
        train_lbls = np.vstack(([train_lbls, lbs ]))
        
    
    #Shuffle the data
    shuffle(train_imgs, train_lbls, 1)
    print('Done loading the training images')
          
    return [train_imgs, train_lbls]

