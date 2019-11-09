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
OPTIMAL_MAP1 = 80
OPTIMAL_MAP2 = 90
DROPOUT = 0.9
momentum =0.1
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




def cnn(images, dropout):
    with tf.device('/gpu:0'):
        images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

        #Conv 1 maps RGB image to OPTIMAL_MAP1 feature maps of 24x24, pooled to 12x12
        W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, OPTIMAL_MAP1], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
        b1 = tf.Variable(tf.zeros([OPTIMAL_MAP1]), name='biases_1')

        conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
        if(dropout):
            conv_1 = tf.nn.dropout(conv_1, DROPOUT)
        pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

        # Conv 2 maps OPTIMAL_MAP1 feature maps of 12x12 to OPTIMAL_MAP2 feature maps of 8x8, pooled to 4x4
        W2 = tf.Variable(tf.truncated_normal([5, 5, OPTIMAL_MAP1, OPTIMAL_MAP2], stddev=1.0 / np.sqrt(OPTIMAL_MAP1 * 5 * 5)),
                         name='weights_2')
        b2 = tf.Variable(tf.zeros([OPTIMAL_MAP2]), name='biases_2')

        conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
        if(dropout):
            conv_2 = tf.nn.dropout(conv_2, DROPOUT)
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')

        # Fully connected layer -- after 2 round of downsampling, our 32x32 image
        # is down to 4x4xOPTIMAL_MAP2 feature maps -- maps this to 300 features.
        W_fc1 = tf.Variable(tf.truncated_normal([4 * 4 * OPTIMAL_MAP2, 300], stddev=1.0 / np.sqrt(4 * 4 * OPTIMAL_MAP2)), name='weights_fc1')
        b_fc1 = tf.Variable(tf.zeros([300]), name='biases_fc1')

        #flatten the 3 channels and matmul to the weights_fc1 for input to the softmax output layer
        dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
        h_pool2_flat = tf.reshape(pool_2, [-1, dim])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        if(dropout):
            h_fc1 = tf.nn.dropout(h_fc1, DROPOUT)

        #Softmax
        W3 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_3')
        b3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_3')
        logits = tf.matmul(h_fc1, W3) + b3

    return conv_1, pool_1, conv_2, pool_2, logits


def main():
    for mode in range(0,5,1):
        trainX, trainY = load_data('data_batch_1')
        print(trainX.shape, trainY.shape)

        testX, testY = load_data('test_batch_trim')
        print(testX.shape, testY.shape)

        trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

        # Create the model
        x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

        if(mode==4):
            conv_1, pool_1, conv_2, pool_2, logits = cnn(x, True)
        else:
            conv_1, pool_1, conv_2, pool_2, logits = cnn(x, False)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        if(mode==1):
            train_step =tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)
        elif(mode==2):
            train_step =tf.train.RMSPropOptimizer(learning_rate,momentum).minimize(loss)
        elif(mode==3):
            train_step =tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else:
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
            if (mode==0):
                plt.figure(1)
                plt.plot(np.arange(epochs), test_acc, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('test accuracy')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Test_Accuracy_Graph_2500.png')

                plt.figure(2)
                plt.plot(np.arange(epochs), train_loss, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('cost')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Cost_Graph_2500.png')
            elif (mode==1):
                plt.figure(3)
                plt.plot(np.arange(epochs), test_acc, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('test accuracy')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Test_Accuracy_Graph_2500_momentum.png')

                plt.figure(4)
                plt.plot(np.arange(epochs), train_loss, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('cost')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Cost_Graph_2500_momentum.png')
            elif (mode==2):
                plt.figure(5)
                plt.plot(np.arange(epochs), test_acc, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('test accuracy')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Test_Accuracy_Graph_2500_RMSPROP.png')

                plt.figure(6)
                plt.plot(np.arange(epochs), train_loss, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('cost')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Cost_Graph_2500_RMSPROP.png')
            elif(mode==3):
                plt.figure(7)
                plt.plot(np.arange(epochs), test_acc, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('test accuracy')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Test_Accuracy_Graph_2500_Adam.png')

                plt.figure(8)
                plt.plot(np.arange(epochs), train_loss, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('cost')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Cost_Graph_2500_Adam.png')
            elif (mode==4):
                plt.figure(3)
                plt.plot(np.arange(epochs), test_acc, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('test accuracy')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Test_Accuracy_Graph_2500_dropout.png')

                plt.figure(4)
                plt.plot(np.arange(epochs), train_loss, label='gradient descent')
                plt.xlabel('epochs')
                plt.ylabel('cost')
                plt.legend(loc='lower right')
                plt.savefig('./figures/Optimal_Cost_Graph_2500_dropout.png')



if __name__ == '__main__':
  main()
