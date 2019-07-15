# This aims to train a neural network for recognizing hand writen digits.
# I therefore use purposely tensorflow with rather basics operations.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import time

# enable GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# to prevent running out of memory all the time
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

# download data if not already done
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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


####################################################################################################################
###  This is a very simple neural network architecture using the relu activation an the dropout regularisation #####
###  Due to the size of the original image, thge 2x2 pooling can only be carried out twice without much coding #####
###  efforts. The cross entropy is used as we deal with a binary problem. For expriming the answer, i apply at
###  the network output an softmax activation
####################################################################################################################
def train():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='images')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    x = tf.layers.conv2d(x_image, filters=32, kernel_size=(3,3), strides=1, padding='same', activation=tf.nn.relu, name='conv1')
    maxpool1 = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2),name='pool1')#14x14
    x = tf.nn.dropout(maxpool1, keep_prob)
    
    x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), strides=1, padding='same', activation=tf.nn.relu, name='conv12')#
    x = tf.nn.dropout(x, keep_prob)
    
    x = tf.layers.conv2d(x, filters=128, kernel_size=(3,3), strides=1, padding='same', activation=tf.nn.relu, name='conv2')
        
    maxpool2 = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2),name='pool2')#7x7
    x = tf.nn.dropout(maxpool2, keep_prob)

    x = tf.layers.conv2d(x, filters=128, kernel_size=(3,3), strides=1, padding='same', activation=tf.nn.relu, name='conv22')
    
    x_flat = tf.reshape(x, [-1, 7 * 7 * 64])
    xd = tf.layers.dense(x_flat, 512,activation=tf.nn.relu, name='dense1')
    xdo = tf.nn.dropout(xd, keep_prob)
    x = tf.layers.dense(xdo, units=10, name='dense2')

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=x))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(x, 1), tf.argmax(y_, 1))
    correct_prediction_reduce = tf.count_nonzero(tf.equal(tf.argmax(x, 1), tf.argmax(y_, 1)))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    saver = tf.train.Saver(tf.trainable_variables() , max_to_keep = 1)

    
    print('configuring the session ...')
    with tf.Session(config=config) as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if i % 100 == 0:
                # need to calculate training accuracy in batches because it doesn't fit into memory
                total_batches = int(mnist.train.num_examples/5000) # 55,000 / 5,000 = 11
                correct = 0
                samples = 0
                for j in range(total_batches):
                    batch = mnist.train.next_batch(5000)
                    correct = correct + sess.run(correct_prediction_reduce,
                                                 feed_dict={x: batch[0],
                                                 y_: batch[1],
                                                 keep_prob: 1.0})
                    samples = samples + batch[0].shape[0]
        
            if i % 1000 == 0:
                print(str(i)+ ": train accuracy = " + str(float(correct) / samples) +
                      "  | test accuracy = " + str(acc_test))
                    
                    #print('test accuracy %g' % accuracy.eval(feed_dict={
                    #                                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
                print("executed in {0:.2f} s!".format(time.time() - start_time))
                saver.save(sess,'./mnist_model.ckpt',global_step = i, write_meta_graph = True)
train()
