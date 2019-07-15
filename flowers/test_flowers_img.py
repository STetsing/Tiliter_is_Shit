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

def getLabel(filename):
    '''Return the class label'''
    if 'daisy' in filename:
        return 1
    elif 'dandelion' in filename:
        return 2
    elif 'rose' in filename:
        return 3
    elif 'sunflower' in filename:
        return 4

    elif 'tulip' in filename:
        return 5
    else:
        return 1


def one_hot_converter(data, n_class):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_class)[targets]

def loadImage(filename, label):
    images = []
    labels = []
 
    img = Image.open(filename )
    img = img.resize(loadData.new_img_size, Image.ANTIALIAS)

        
    in_img = np.asarray(img).reshape(np.prod(np.asarray(img).shape))/255
    images.append(np.asarray(in_img))
    labels.append(np.asarray(label))
    return [images, labels]

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--image', type=str, help='the image you want to predict')
    return parser.parse_args(argv)

def main(args):
    if args.image == None:
        print('!!!!!! No image submitted for classification !!!!!!')
        return
    
    #load image to be predicted
    pca = loadPCA()
    label = getLabel

    images, labels =  loadImage(args.image, label)
    labels = np.squeeze(labels)

    to_predict =  pca.transform( images )

 
    # read the serialize tree
    svm = loadModel()
    predicted = svm.predict(to_predict)
    #predicted_score = svm.score(to_predict, labels)
    #for p in predicted:
    #    print(p, getClassName(p))
    print('----------------------------------------------------------------')
    #print('The prediction accuracy on the directory is:', predicted_score)
    print('The classifier predicted the submitted image to be:', getClassName(predicted[0]) )




if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

