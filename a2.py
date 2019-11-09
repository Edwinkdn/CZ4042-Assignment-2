#
# Project 2, starter code Part a
#

import math
from functools import reduce

import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
from mpl_toolkits import mplot3d


NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 500
batch_size = 128


seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)


    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_




def cnn(images, map1, map2):
    with tf.device('/gpu:0'):
        images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

        #Conv 1 maps RGB image to map1 feature maps of 24x24, pooled to 12x12
        W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, map1], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
        b1 = tf.Variable(tf.zeros([map1]), name='biases_1')

        conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
        pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

        # Conv 2 maps map1 feature maps of 12x12 to map2 feature maps of 8x8, pooled to 4x4
        W2 = tf.Variable(tf.truncated_normal([5, 5, map1, map2], stddev=1.0 / np.sqrt(map1 * 5 * 5)),
                         name='weights_2')
        b2 = tf.Variable(tf.zeros([map2]), name='biases_2')

        conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')

        # Fully connected layer -- after 2 round of downsampling, our 32x32 image
        # is down to 4x4xmap2 feature maps -- maps this to 300 features.
        W_fc1 = tf.Variable(tf.truncated_normal([4 * 4 * map2, 300], stddev=1.0 / np.sqrt(4 * 4 * map2)), name='weights_fc1')
        b_fc1 = tf.Variable(tf.zeros([300]), name='biases_fc1')

        #flatten the 3 channels and matmul to the weights_fc1 for input to the softmax output layer
        dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
        h_pool2_flat = tf.reshape(pool_2, [-1, dim])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #Softmax
        W3 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_3')
        b3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_3')
        logits = tf.matmul(h_fc1, W3) + b3

    return conv_1, pool_1, conv_2, pool_2, logits


def main():

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)

    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    max_acc = 0
    max_m1= 0
    max_m2= 0
    test_acc_mean =np.array([])

    for map1 in range(50, 101, 10):
        for map2 in range(50, 101, 10):
            print("Testing with", map1, "feature maps at convo 1 and ", map2, "feature maps at convo2")
            conv_1, pool_1, conv_2, pool_2, logits = cnn(x, map1, map2)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
            loss = tf.reduce_mean(cross_entropy)

            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

            N = len(trainX)
            idx = np.arange(N)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                test_acc = []
                for e in range(epochs):
                    np.random.shuffle(idx)
                    trainX, trainY = trainX[idx], trainY[idx]
                    for start, end in zip(range(0,N, batch_size), range(batch_size, N, batch_size)):
                        _, loss_ = sess.run([train_step, loss], feed_dict={x: trainX[start:end], y_: trainY[start:end]})
                    test_acc.append(accuracy.eval(feed_dict={x:testX,y_:testY}))
                acc = reduce(lambda x, y: x + y, test_acc) / len(test_acc)
                test_acc_mean = np.append(test_acc_mean,acc)
                print("Test accuracy (", map1,",",map2,") is ", acc)
                if(acc>max_acc):
                    max_acc, max_m1, max_m2 = acc, map1, map2
    print("Best accuracy of ", max_acc, " after grid search is with ", max_m1, "&", max_m2)

    # plot 3d graph to show the mean test accuracy of each model
    x = range(10, 101, 10)
    y = range(10, 101, 10)
    X, Y = np.meshgrid(x, y)
    test_acc_mean=test_acc_mean.reshape(6,6)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, test_acc_mean, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Convolutional layer 2')
    ax.set_ylabel('Convolutional layer 1')
    ax.set_zlabel('Test accuracy')
    plt.savefig('./figures/Grid_Search_Accuracy')

if __name__ == '__main__':
  main()
