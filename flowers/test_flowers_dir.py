import pickle
from PIL import Image
import numpy as np
import argparse
import sys
import os
import loadData


def loadModel():
    filename = 'svm.dat'
    classifier = pickle.load(open(filename, 'rb'))
    return classifier

def loadPCA():
    filename = 'pca_params.dat'
    pca = pickle.load(open(filename, 'rb'))
    return pca

def getClassName(cls:int):
    classes = {1: 'daisy', 2: 'dandelion', 3: 'rose', 4: 'sunflower', 5: 'tulip'}
    return classes[cls]

def one_hot_converter(data, n_class):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_class)[targets]

def loadImage(filename, label):
    img_files = sorted(os.listdir(filename))
    images = []
    labels = []
    for im in img_files:
        img = Image.open(filename + str('/') + im)
        
        # One should consider removing some channels or convert the image to grey scale for faster computations
        # as these feature might not bring that much of information
        
        #if(img.mode != 'L'):
        #    img = img.convert('L')
        
        # reshape image as vector
        img = img.resize(loadData.new_img_size, Image.ANTIALIAS)
        #print(img.size)
        
        in_img = np.asarray(img).reshape(np.prod(np.asarray(img).shape))/255
        images.append(np.asarray(in_img))
        labels.append(np.asarray(label))
    return [images, labels]

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--image_directory', type=str, help='the image directory you want to predict')
    return parser.parse_args(argv)

def main(args):
    if args.image_directory == None:
        print('!!!!!! No image submitted for classification !!!!!!')
        return
    
    #load image to be predicted
    pca = loadPCA()
    dir_name = args.image_directory.split('/')[1]
    for i in range(5):
        print(getClassName(i + 1), dir_name)
        if getClassName(i + 1) == dir_name:
            label = [i + 1]
            break

    images, labels =  loadImage(args.image_directory, label)
    labels = np.squeeze(labels)
    print('images shape', np.shape(images))
    to_predict =  pca.transform( images )
    print(np.shape(to_predict))
 
    # read the serialize tree
    svm = loadModel()
    predicted = svm.predict(to_predict)
    predicted_score = svm.score(to_predict, labels)
    for p in predicted:
        print(p, getClassName(p))
    print('----------------------------------------------------------------')
    print('The prediction accuracy on the directory is:', predicted_score)




if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

