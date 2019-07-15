from PIL import Image
import numpy as np
import os

train_dir = './data/mnist/training/'
test_dir = './data/mnist/testing/'

classes = 10

def one_hot_converter(data, n_class):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_class)[targets]

def loadImages(dir, label):
    '''Load the images from a given directory'''
    img_files = sorted(os.listdir(train_dir + dir))
    images = []
    labels = []
    for im in img_files:
        img = Image.open(train_dir + dir + str('/') + im)
        
        # convert to greyscale if needed
        if(img.mode != 'L'):
            img = img.convert('L')
        
        # reshape image as vector
        in_img = np.asarray(img).reshape(784)/255
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
    dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    train_imgs = np.array([], dtype=np.int64).reshape(0, 784)
    train_lbls = np.array([], dtype=np.int64).reshape(0, 10)
    
    # parallel execution might be implemented
    for i,d in enumerate(dirs):
        print('processing:', i, '/', len(dirs))
        imgs, lbs = loadImages(d,one_hot_converter(int(d), classes))
        train_imgs = np.vstack(([train_imgs, imgs]))
        train_lbls = np.vstack(([train_lbls, lbs ]))
    
    #Shuffle the data
    #shuffle(train_imgs, train_lbls, len(train_lbls))
    print('Done loading the training images')
          
    return [train_imgs, train_lbls]

def loadTestImages():
    '''Load the testing mnist images. Return the images and their coresponding labels'''
    dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    test_imgs = np.array([], dtype=np.int64).reshape(0, 784)
    test_lbls = np.array([], dtype=np.int64).reshape(0, 10)
    
    # parallel execution might be implemented
    for i,d in enumerate(dirs):
        print('processing:', d)
        print(one_hot_converter(int(d), classes))
        imgs, lbs = loadImages(d,one_hot_converter(int(d), classes))
        test_imgs = np.vstack(([test_imgs, imgs]))
        test_lbls = np.vstack(([test_lbls, lbs ]))
    
    print(len(test_imgs))
    print(len(test_lbls))

    #Testing does not require shuffling the data
    return [test_imgs, test_lbls]


