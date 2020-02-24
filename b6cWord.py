# from google.colab import files
# uploaded = files.upload()
import datetime
import time
from functools import reduce

import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 50
batch_size = 128
DROPOUT = False
dropout = 0.9

no_epochs = 2
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def word_rnn_model(x, DROPOUT):
    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    def create_cell():
        cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
        return cell
    cell = create_cell()
    if DROPOUT:
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout, output_keep_prob=dropout)

    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
    # LSTM Function
    if isinstance(encoding, tf.nn.rnn_cell.LSTMStateTuple) or isinstance(encoding, tuple):
        encoding = encoding[-1]

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits, word_list


def data_read_words():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open('test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)
    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)

    return x_train, y_train, x_test, y_test, no_words


def main():
    global n_words

    x_train, y_train, x_test, y_test, n_words = data_read_words()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    logits, word_list = word_rnn_model(x, DROPOUT)

    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    gvs = optimizer.compute_gradients(entropy)
    capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # training
    N = len(x_train)
    idx = np.arange(N)
    timer = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_acc = []
        acc_train_loss = []
        train_loss = []
        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                word_list_, _, loss_ = sess.run([word_list, train_op, entropy],
                                                {x: x_train[start:end], y_: y_train[start:end]})
            train_loss.append(loss_)
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
            if e % 10 == 0:
                print('epoch: %d, entropy: %g' % (e, loss_))
    sess.close()
    elapsed_time = (time.time() - timer)
    elapsed_time_str = 'Elapsed time: ' + str(datetime.timedelta(seconds=elapsed_time)).split(".")[0]
    # Store Variable for each type
    # 0 for GRU
    # 1 for RNN
    # 2 for LSTM
    print("Storing gradient clipping accuracy, training loss and elasped time")
    gradient_clipping_acc = test_acc
    gradient_clipping_train_loss = train_loss
    gradient_clipping_elapsed_time = elapsed_time


    # new
    plt.title("Training Loss W/O Dropout Vs Epochs")
    # plt.plot(range(no_epochs), GRU_train_loss, label='GRU Training Loss')
    # plt.plot(range(no_epochs), RNN_train_loss, label='Vanilla RNN Training Loss')
    # plt.plot(range(no_epochs), LSTM_train_loss, label='LSTM Training Loss')
    # plt.plot(range(no_epochs), two_layer_train_loss, label='2 layer Training Loss')
    plt.plot(range(no_epochs), gradient_clipping_train_loss, label='gradient clipping Training Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.show()

    plt.title("Testing Accuracy Vs Epochs")
    # plt.plot(range(no_epochs), GRU_acc, label='GRU Testing Accuracy')
    # plt.plot(range(no_epochs), RNN_acc, label='RNN Testing Accuracy')
    # plt.plot(range(no_epochs), LSTM_acc, label='LSTM Testing Accuracy')
    # plt.plot(range(no_epochs), two_layer_acc, label='2 layer Training Loss')
    plt.plot(range(no_epochs), gradient_clipping_acc, label='gradient clipping Training Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Testing Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
