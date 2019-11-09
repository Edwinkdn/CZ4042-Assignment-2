#
# Project 2, starter code Part a
#

import math
from functools import reduce

import tensorflow as tf
import numpy as np
import pylab as plt
import pickle



NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 2500
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




def cnn(images):
    with tf.device('/gpu:0'):
        images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

        #Conv 1 maps RGB image to 50 feature maps of 24x24, pooled to 12x12
        W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
        b1 = tf.Variable(tf.zeros([50]), name='biases_1')

        conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
        pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

        # Conv 2 maps 50 feature maps of 12x12 to 60 feature maps of 8x8, pooled to 4x4
        W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0 / np.sqrt(50 * 5 * 5)),
                         name='weights_2')
        b2 = tf.Variable(tf.zeros([60]), name='biases_2')

        conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')

        # Fully connected layer -- after 2 round of downsampling, our 32x32 image
        # is down to 4x4x60 feature maps -- maps this to 300 features.
        W_fc1 = tf.Variable(tf.truncated_normal([4 * 4 * 60, 300], stddev=1.0 / np.sqrt(4 * 4 * 60)), name='weights_fc1')
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


    conv_1, pool_1, conv_2, pool_2, logits = cnn(x)

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
        acc_train_loss = []
        train_loss =[]
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            for start, end in zip(range(0,N, batch_size), range(batch_size, N, batch_size)):
                _, loss_ = sess.run([train_step, loss], feed_dict={x: trainX[start:end], y_: trainY[start:end]})
                acc_train_loss.append(loss_)
            train_loss.append(reduce(lambda x, y: x + y, acc_train_loss) / len(acc_train_loss))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            print('epoch', e, 'entropy', loss_)

        plt.figure(1)
        plt.plot(np.arange(epochs), test_acc, label='gradient descent')
        plt.xlabel('epochs')
        plt.ylabel('test accuracy')
        plt.legend(loc='lower right')
        plt.savefig('./figures/Test_Accuracy_Graph_2500.png')

        plt.figure(2)
        plt.plot(np.arange(epochs), train_loss, label='gradient descent')
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.legend(loc='lower right')
        plt.savefig('./figures/Cost_Graph_2500.png')

        index = np.random.randint(low=0, high=1999)
        X1 = testX[index,:]
        index = np.random.randint(low=0, high=1999)
        X2 = testX[index, :]

        plt.figure(3)
        plt.axis('off'); plt.imshow(X1.reshape(32,32,3).astype('uint32'))
        plt.savefig('./figures/T1_original_image.png')

        conv_1_, pool_1_, conv_2_, pool_2_ = sess.run([conv_1, pool_1, conv_2, pool_2],
                                                  {x: X1.reshape(1,3072)})
        plt.figure(4)
        h_conv_1 = np.array(conv_1_)
        for i in range(50):
            plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(h_conv_1[0,:,:,i])
        plt.savefig('./figures/T1_convo1_feature_maps.png')

        plt.figure(5)
        h_pool_1 = np.array(pool_1_)
        for i in range(50):
            plt.subplot(10, 5, i + 1); plt.axis('off'); plt.imshow(h_pool_1[0, :, :, i])
        plt.savefig('./figures/T1_pool1_feature_maps.png')

        plt.figure(6)
        h_conv_2 = np.array(conv_2_)
        for i in range(50):
            plt.subplot(10, 5, i + 1); plt.axis('off'); plt.imshow(h_conv_2[0, :, :, i])
        plt.savefig('./figures/T1_convo2_feature_maps.png')

        plt.figure(7)
        h_pool_2 = np.array(pool_2_)
        for i in range(50):
            plt.subplot(10, 5, i + 1); plt.axis('off'); plt.imshow(h_pool_2[0, :, :, i])
        plt.savefig('./figures/T1_pool2_feature_maps.png')

        plt.figure(8)
        plt.axis('off'); plt.imshow(X2.reshape(32,32,3).astype('uint32'))
        plt.savefig('./figures/T2_original_image.png')

        conv_1_, pool_1_, conv_2_, pool_2_ = sess.run([conv_1, pool_1, conv_2, pool_2],
                                                  {x: X2.reshape(1,3072)})
        plt.figure(9)
        h_conv_1 = np.array(conv_1_)
        for i in range(50):
            plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(h_conv_1[0,:,:,i])
        plt.savefig('./figures/T2_convo1_feature_maps.png')

        plt.figure(10)
        h_pool_1 = np.array(pool_1_)
        for i in range(50):
            plt.subplot(10, 5, i + 1); plt.axis('off'); plt.imshow(h_pool_1[0, :, :, i])
        plt.savefig('./figures/T2_pool1_feature_maps.png')

        plt.figure(11)
        h_conv_2 = np.array(conv_2_)
        for i in range(50):
            plt.subplot(10, 5, i + 1); plt.axis('off'); plt.imshow(h_conv_2[0, :, :, i])
        plt.savefig('./figures/T2_convo2_feature_maps.png')

        plt.figure(12)
        h_pool_2 = np.array(pool_2_)
        for i in range(50):
            plt.subplot(10, 5, i + 1); plt.axis('off'); plt.imshow(h_pool_2[0, :, :, i])
        plt.savefig('./figures/T2_pool2_feature_maps.png')


if __name__ == '__main__':
  main()
