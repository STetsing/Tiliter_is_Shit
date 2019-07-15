###                This aims to train a neural network for recognizing hand writen digits.                     #####
####################################################################################################################
###  This is a very simple neural network architecture using the relu activation an the dropout regularisation #####
###  Due to the size of the original image, thge 2x2 pooling can only be carried out twice without much coding #####
###  efforts. The cross entropy is used as we deal with a binary problem. For expriming the answer, i apply at #####
###  the network output an softmax activation. The following implementation is 'old school'. Existing libraries#####
###  achieve this in much lesser code. I use purposely tensorflow with basics operations.                      #####
####################################################################################################################

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import os
import loadData
import time
from sklearn.model_selection import train_test_split
import argparse
import sys

# enable log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# to prevent running out of memory all the time
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

batch_size = 100

# load the training data and split it into train an test set
train, labels = loadData.loadTrainingImages()
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=1)


def weight_variable(shape):
    '''Defines the weight variables for a layer'''
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.01, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    '''Initialize the bias variable with 0.1 for a given layer'''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''Carries out the convolution operation'''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''2x2 Pooling'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



def train(e):
    ''' Train the network on the images read'''
    x = tf.placeholder(tf.float32, shape=[None, 784], name='images')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    #for dropout regulatization
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    # Layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) #14x14 feature maps
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    # Layer 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # 7x7feature maps
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(e).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction_reduce = tf.count_nonzero(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    
    # save the net as training. No recall functionality implemented
    saver = tf.train.Saver(tf.trainable_variables() , max_to_keep = 1)

    
    print('configuring the session ...')
    with tf.Session(config=config) as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        
        for i in range(200): # epochs
            correct = 0
            samples = 0
            accuracy_logged = False
            
            for c in range(0, len(X_train), batch_size):
                batch = X_train[c:c + batch_size]
                batch_l = y_train [c:c + batch_size]
            
                # train
                train_step.run(feed_dict={x: batch, y_: batch_l, keep_prob: 0.5})
                
                # track the accuracy
                if i % 10== 0 and accuracy_logged == False:
                    correct = correct + sess.run(correct_prediction_reduce,
                                                 feed_dict={x: batch,
                                                 y_: batch_l,
                                                 keep_prob: 1.0})
                    samples = samples + batch.shape[0]
                    acc_test = sess.run(accuracy, feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})
                    
                    print(str(i)+ ": train accuracy = " + str(float(correct) / samples) +
                          "  | test accuracy = " + str(acc_test))
                        
                        #print('test accuracy %g' % accuracy.eval(feed_dict={
                        #                                     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
                    print("executed in {0:.2f} s!".format(time.time() - start_time))
                    saver.save(sess,'./mnist_model.ckpt',global_step = i, write_meta_graph = True)
                    accuracy_logged = True


def parse_arguments(argv):
    '''Parse sytem arguments'''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-e', '--epsilon', type=float, help='learning rate')
    return parser.parse_args(argv)

def main(args):
    '''Main '''
    
    # Setup default config
    eps = 1e-4 if args.epsilon == None else args.epsilon
    print('Learning rate is:', eps)
    print('')
    # train the network
    train(eps)

    print('########################')
    print('Training done')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




