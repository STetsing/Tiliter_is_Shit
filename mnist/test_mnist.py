from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow as tf
import loadData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# download data if not already done
x_test, y_test = loadData.loadTestImages()

def main(args):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    print(" ")
    print("started the test process ----------------")
    print(" ")
    with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
        print('restoring the model')
        saver = tf.train.import_meta_graph('./mnist_model.ckpt-19000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        print('restored the model')

        test_images = graph.get_tensor_by_name('images:0')
        labls = graph.get_tensor_by_name('labels:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        acc = graph.get_tensor_by_name('accuracy:0')

        #get the test accuracy
        test_acc= sess.run(acc, feed_dict = {test_images:x_test,
                           labls:y_test,
                           keep_prob:1} )
        print('The test accuracy is:', test_acc)

# no arguments
if __name__ == '__main__':
    main(1)
