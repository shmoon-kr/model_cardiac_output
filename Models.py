from math import ceil, sqrt
from scipy.stats import pearsonr, trim_mean
from sklearn.metrics import mean_squared_error
from matplotlib.dates import DateFormatter
import tensorflow as tf
import numpy as np
import CommonFunction as cf
import datetime
import MySQLdb
import os
import re
import csv
import random
import logging
import matplotlib.pyplot as plt

batch_size = 256
learning_rate = 1e-4

model_dir = '/home/shmoon/model/production'
validation_dir = '/home/shmoon/Result/validation'

def derivative_matrix(input_size):
    mat = np.zeros((input_size, input_size), dtype=np.float32)
    mat[1, 0] = 1
    mat[0, 0] = -1
    for i in range(1, input_size):
        mat[i, i] = 1
        mat[i-1, i] = -1
    return tf.constant(mat)


class ModelSingleBase:

    def __init__(self, sess, name, f=(8, 16, 32, 64), k=(12, 6, 4, 3), d=0, max_to_keep=5):
        self.sess = sess
        self.name = name
        self.max_to_keep = max_to_keep
        self._build_net(f, k, d)

    def _build_net(self, f, k, d):

        assert len(f) == 4 and len(k) == 4, 'Wrong filter vector or kernel vector.'

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):

            self.I = tf.placeholder(tf.int32, shape=[None, 1], name='Ind')

        with tf.variable_scope(self.name+'/cnn', reuse=tf.AUTO_REUSE):

            # input place holders
            self.X = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Target')

            self.training = tf.placeholder(dtype=tf.bool)

            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y, self.I)).batch(batch_size).repeat()
            self.iterator = dataset.make_initializable_iterator()
            self.it_x_cnn, self.it_y_cnn, self.it_i = self.iterator.get_next()

            # Input
            AVG0 = tf.reshape(self.it_x_cnn, [-1, 1024, 1])
            DRV0 = tf.reshape(tf.matmul(tf.reshape(AVG0, [-1, 1024]), derivative_matrix(1024)), [-1, 1024, 1])
            F0 = tf.concat([AVG0, DRV0], axis=2)

            # L1 SigIn shape = (?, 1024, 1)
            L1 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation=tf.nn.relu, name='C1')(F0)
            M1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L1)
            AVG1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0)
            DRV1 = tf.reshape(tf.matmul(tf.reshape(AVG1, [-1, 512]), derivative_matrix(512)), [-1, 512, 1])
            MAX1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0)
            MIN1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-AVG0)
            F1 = tf.concat([M1, MAX1, AVG1, DRV1, MIN1], axis=2)
            # Conv -> (?, 512, 8)

            L2 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation=tf.nn.relu, name='C2')(F1)
            M2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L2)
            AVG2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1)
            DRV2 = tf.reshape(tf.matmul(tf.reshape(AVG2, [-1, 256]), derivative_matrix(256)), [-1, 256, 1])
            MAX2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX1)
            MIN2 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN1)
            F2 = tf.concat([M2, MAX2, AVG2, DRV2, MIN2], axis=2)
            # Conv -> (?, 256, 8)

            L3 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation=tf.nn.relu, name='C3')(F2)
            M3 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L3)
            AVG3 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2)
            DRV3 = tf.reshape(tf.matmul(tf.reshape(AVG3, [-1, 128]), derivative_matrix(128)), [-1, 128, 1])
            MAX3 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX2)
            MIN3 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN2)
            F3 = tf.concat([M3, MAX3, AVG3, DRV3, MIN3], axis=2)
            # Conv -> (?, 128, 16)

            L4 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation=tf.nn.relu, name='C4')(F3)
            M4 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L4)
            AVG4 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3)
            DRV4 = tf.reshape(tf.matmul(tf.reshape(AVG4, [-1, 64]), derivative_matrix(64)), [-1, 64, 1])
            MAX4 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX3)
            MIN4 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN3)
            F4 = tf.concat([M4, MAX4, AVG4, DRV4, MIN4], axis=2)
            # Conv -> (?, 64, 16)

            L5 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation=tf.nn.relu, name='C5')(F4)
            M5 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L5)
            AVG5 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4)
            DRV5 = tf.reshape(tf.matmul(tf.reshape(AVG5, [-1, 32]), derivative_matrix(32)), [-1, 32, 1])
            MAX5 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX4)
            MIN5 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN4)
            F5 = tf.concat([M5, MAX5, AVG5, DRV5, MIN5], axis=2)

            # Conv -> (?, 32, 32)
            L6 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation=tf.nn.relu, name='C6')(F5)
            M6 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L6)
            AVG6 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5)
            DRV6 = tf.reshape(tf.matmul(tf.reshape(AVG6, [-1, 16]), derivative_matrix(16)), [-1, 16, 1])
            MAX6 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX5)
            MIN6 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN5)
            F6 = tf.concat([M6, MAX6, AVG6, DRV6, MIN6], axis=2)
            # Conv -> (?, 16, 32)

            L7 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation=tf.nn.relu, name='C7')(F6)
            M7 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L7)
            AVG7 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6)
            DRV7 = tf.reshape(tf.matmul(tf.reshape(AVG7, [-1, 8]), derivative_matrix(8)), [-1, 8, 1])
            MAX7 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX6)
            MIN7 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN6)
            F7 = tf.concat([M7, MAX7, AVG7, DRV7, MIN7], axis=2)
            # Conv -> (?, 8, 64)

            L8 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation=tf.nn.relu, name='C8')(F7)
            M8 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L8)
            AVG8 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7)
            MAX8 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX7)
            MIN8 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN7)
            F8 = tf.concat([M8, MAX8, AVG8, MIN8], axis=2)
            # Conv -> (?, 4, 64)

            flat = tf.layers.Flatten()(F8)

            if d in (0, 1):
                self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat), 0, name='Output')
            else:
                FC1 = tf.layers.Dense(d, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
                FC2 = tf.layers.Dense(d, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())(FC1)
                self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC2), 0, name='Output')

            # Simplified cost/loss function
            self.cost_cnn = tf.reduce_mean(tf.square(self.Output_cnn - self.it_y_cnn))  # - hypothesis_std

            # Minimize
            self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_cnn)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'), max_to_keep=self.max_to_keep)

    def build_isc(self, init_isc, bias=True):

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):
            init_w = tf.constant_initializer(init_isc[:, 0])
            init_b = tf.constant_initializer(init_isc[:, 1])
            SELECTOR = tf.one_hot(name="individual_selector", indices=tf.reshape(self.it_i, [-1]),
                                  depth=len(init_isc))
            WI = tf.get_variable(name="weight_individual", shape=(len(init_isc), 1), initializer=init_w)
            if bias:
                BI = tf.get_variable(name="bias_individual", shape=(len(init_isc), 1), initializer=init_b)
                self.Output_scaled = self.Output_cnn * tf.matmul(SELECTOR, WI) + tf.matmul(SELECTOR, BI)
            else:
                self.Output_scaled = self.Output_cnn * tf.matmul(SELECTOR, WI)

            # Simplified cost/loss function
            self.cost_isc = tf.reduce_mean(tf.square(self.Output_scaled - self.it_y_cnn))  # - hypothesis_std

        var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        var_cnn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn')

        # Minimize
        self.train_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_isc, var_list=var_all)
        self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_isc, var_list=var_cnn)
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'), max_to_keep=len(init_isc))

    def predict(self, x_test):
        len_d = x_test.shape[0]
        feed_dict = {self.X: x_test, self.Y: np.zeros((len_d, 1)), self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x_test.shape[0] / batch_size)

        print('Prediction Started!')
        avg_cost = 0
        result = list()
        for _ in range(num_batches):
            cost_val, hy_val = self.sess.run([self.cost_cnn, self.Output_cnn], feed_dict={self.training: False})
            avg_cost += cost_val / num_batches
            result.extend(hy_val)

        return result

    def train(self, x_data, y_data, epochs):
        len_d = x_data.shape[0]
        feed_dict = {self.X: x_data, self.Y: y_data, self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x_data.shape[0] / batch_size)

        unchanged = last_cost = 0
        effective = True
        print('Learning Started!')
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_cnn, self.Output_cnn, self.train_cnn],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch+1, avg_cost))
            if last_cost == avg_cost:
                unchanged += 1
            else:
                unchanged = 0
            if unchanged > 10:
                effective = False
                break
            last_cost = avg_cost

        return effective

    def train_isc(self, x_data, y_data, i_data, epochs, train='all'):
        feed_dict = {self.X: x_data, self.Y: y_data, self.I: i_data}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x_data.shape[0] / batch_size)

        print('Learning Started!')
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                if train == 'all':
                    cost_val, hy_val, _ = self.sess.run([self.cost_isc, self.Output_scaled, self.train_all],
                                                        feed_dict={self.training: True})
                elif train == 'cnn':
                    cost_val, hy_val, _ = self.sess.run([self.cost_isc, self.Output_scaled, self.train_cnn],
                                                        feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))

        return

    def save_model(self, file_name):
        save_path = self.saver.save(self.sess, file_name)
        return save_path

    def restore_model(self, file_name):
        self.saver.restore(self.sess, file_name)
        return


class ModelSingle2(ModelSingleBase):

    def _build_net(self, f, k, d):

        assert len(f) == 4 and len(k) == 4, 'Wrong filter vector or kernel vector.'

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):

            self.I = tf.placeholder(tf.int32, shape=[None, 1], name='Ind')

        with tf.variable_scope(self.name+'/cnn', reuse=tf.AUTO_REUSE):

            # input place holders
            self.X = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Target')

            self.training = tf.placeholder(dtype=tf.bool)

            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y, self.I)).batch(batch_size).repeat()
            self.iterator = dataset.make_initializable_iterator()
            self.it_x_cnn, self.it_y_cnn, self.it_i = self.iterator.get_next()

            # Input
            AVG0 = tf.reshape(self.it_x_cnn, [-1, 1024, 1])
            DRV0 = tf.reshape(tf.matmul(tf.reshape(AVG0, [-1, 1024]), derivative_matrix(1024)), [-1, 1024, 1])
            F0 = tf.concat([AVG0, DRV0], axis=2)

            # L1 SigIn shape = (?, 1024, 1)
            C11 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C11')(F0)
            C12 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C12')(C11)
            M1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C12)
            AVG1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0)
            DRV1 = tf.reshape(tf.matmul(tf.reshape(AVG1, [-1, 512]), derivative_matrix(512)), [-1, 512, 1])
            MAX1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0)
            MIN1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-AVG0)
            F1 = tf.concat([M1, MAX1, AVG1, DRV1, MIN1], axis=2)
            # Conv -> (?, 512, 8)

            C21 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C21')(F1)
            C22 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C22')(C21)
            M2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C22)
            AVG2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1)
            DRV2 = tf.reshape(tf.matmul(tf.reshape(AVG2, [-1, 256]), derivative_matrix(256)), [-1, 256, 1])
            MAX2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX1)
            MIN2 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN1)
            F2 = tf.concat([M2, MAX2, AVG2, DRV2, MIN2], axis=2)
            # Conv -> (?, 256, 8)

            C31 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C31')(F2)
            C31 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C32')(C31)
            M3 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C31)
            AVG3 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2)
            DRV3 = tf.reshape(tf.matmul(tf.reshape(AVG3, [-1, 128]), derivative_matrix(128)), [-1, 128, 1])
            MAX3 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX2)
            MIN3 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN2)
            F3 = tf.concat([M3, MAX3, AVG3, DRV3, MIN3], axis=2)
            # Conv -> (?, 128, 16)

            C41 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C41')(F3)
            C42 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C42')(C41)
            M4 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C42)
            AVG4 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3)
            DRV4 = tf.reshape(tf.matmul(tf.reshape(AVG4, [-1, 64]), derivative_matrix(64)), [-1, 64, 1])
            MAX4 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX3)
            MIN4 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN3)
            F4 = tf.concat([M4, MAX4, AVG4, DRV4, MIN4], axis=2)
            # Conv -> (?, 64, 16)

            C51 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C51')(F4)
            C52 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C52')(C51)
            M5 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C52)
            AVG5 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4)
            DRV5 = tf.reshape(tf.matmul(tf.reshape(AVG5, [-1, 32]), derivative_matrix(32)), [-1, 32, 1])
            MAX5 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX4)
            MIN5 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN4)
            F5 = tf.concat([M5, MAX5, AVG5, DRV5, MIN5], axis=2)

            # Conv -> (?, 32, 32)
            C61 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C61')(F5)
            C62 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C62')(C61)
            M6 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C62)
            AVG6 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5)
            DRV6 = tf.reshape(tf.matmul(tf.reshape(AVG6, [-1, 16]), derivative_matrix(16)), [-1, 16, 1])
            MAX6 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX5)
            MIN6 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN5)
            F6 = tf.concat([M6, MAX6, AVG6, DRV6, MIN6], axis=2)
            # Conv -> (?, 16, 32)

            C71 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C71')(F6)
            C72 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C72')(C71)
            M7 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C72)
            AVG7 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6)
            DRV7 = tf.reshape(tf.matmul(tf.reshape(AVG7, [-1, 8]), derivative_matrix(8)), [-1, 8, 1])
            MAX7 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX6)
            MIN7 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN6)
            F7 = tf.concat([M7, MAX7, AVG7, DRV7, MIN7], axis=2)
            # Conv -> (?, 8, 64)

            C81 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C81')(F7)
            C82 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C82')(C81)
            M8 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C82)
            AVG8 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7)
            MAX8 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX7)
            MIN8 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN7)
            F8 = tf.concat([M8, MAX8, AVG8, MIN8], axis=2)
            # Conv -> (?, 4, 64)

            flat = tf.layers.Flatten()(F8)

            if d in (0, 1):
                self.Output_cnn = tf.add(
                    tf.layers.Dense(1, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())(flat), 0, name='Output')
            else:
                FC1 = tf.layers.Dense(d, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
                FC2 = tf.layers.Dense(d, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC1)
                self.Output_cnn = tf.add(
                    tf.layers.Dense(1, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())(FC2), 0, name='Output')

            # Simplified cost/loss function
            self.cost_cnn = tf.reduce_mean(tf.square(self.Output_cnn - self.it_y_cnn))  # - hypothesis_std

            # Minimize
            self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_cnn)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'))


class Model_CNN_2:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):

            self.I = tf.placeholder(tf.int32, shape=[None, 1], name='Ind')

        with tf.variable_scope(self.name+'/cnn', reuse=tf.AUTO_REUSE):

            # input place holders
            self.X1 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave1')
            self.X2 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave2')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Target')

            self.training = tf.placeholder(dtype=tf.bool)

            dataset = tf.data.Dataset.from_tensor_slices((self.X1, self.X2, self.Y, self.I)).batch(batch_size).repeat()
            self.iterator = dataset.make_initializable_iterator()
            self.it_x1_cnn, self.it_x2_cnn, self.it_y_cnn, self.it_i = self.iterator.get_next()

            DM_1024 = derivative_matrix(1024)

            # Input
            AVG0_X1 = tf.reshape(self.it_x1_cnn, [-1, 1024, 1])
            DRV0_X1 = tf.reshape(tf.matmul(tf.reshape(AVG0_X1, [-1, 1024]), DM_1024), [-1, 1024, 1])
            AVG0_X2 = tf.reshape(self.it_x2_cnn, [-1, 1024, 1])
            DRV0_X2 = tf.reshape(tf.matmul(tf.reshape(AVG0_X2, [-1, 1024]), DM_1024), [-1, 1024, 1])
            F0 = tf.concat([AVG0_X1, DRV0_X1, AVG0_X2, DRV0_X2], axis=2)

            DM_512 = derivative_matrix(512)

            # L1 SigIn shape = (?, 1024, 1)
            L1 = tf.layers.Conv1D(filters=8, kernel_size=12, strides=1, padding='same', activation=tf.nn.relu, name='C1')(F0)
            M1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L1)
            AVG1_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            DRV1_X1 = tf.reshape(tf.matmul(tf.reshape(AVG1_X1, [-1, 512]), DM_512), [-1, 512, 1])
            MAX1_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            MIN1_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            AVG1_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X2)
            DRV1_X2 = tf.reshape(tf.matmul(tf.reshape(AVG1_X2, [-1, 512]), DM_512), [-1, 512, 1])
            F1 = tf.concat([M1, MAX1_X1, AVG1_X1, DRV1_X1, MIN1_X1, AVG1_X2, DRV1_X2], axis=2)
            # Conv -> (?, 512, 8)

            DM_256 = derivative_matrix(256)

            L2 = tf.layers.Conv1D(filters=8, kernel_size=12, strides=1, padding='same', activation=tf.nn.relu, name='C2')(F1)
            M2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L2)
            AVG2_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X1)
            DRV2_X1 = tf.reshape(tf.matmul(tf.reshape(AVG2_X1, [-1, 256]), DM_256), [-1, 256, 1])
            MAX2_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX1_X1)
            MIN2_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN1_X1)
            AVG2_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X2)
            DRV2_X2 = tf.reshape(tf.matmul(tf.reshape(AVG2_X2, [-1, 256]), DM_256), [-1, 256, 1])
            F2 = tf.concat([M2, MAX2_X1, AVG2_X1, DRV2_X1, MIN2_X1, AVG2_X2, DRV2_X2], axis=2)
            # Conv -> (?, 256, 8)

            DM_128 = derivative_matrix(128)

            L3 = tf.layers.Conv1D(filters=16, kernel_size=6, strides=1, padding='same', activation=tf.nn.relu, name='C3')(F2)
            M3 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L3)
            AVG3_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X1)
            DRV3_X1 = tf.reshape(tf.matmul(tf.reshape(AVG3_X1, [-1, 128]), DM_128), [-1, 128, 1])
            MAX3_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX2_X1)
            MIN3_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN2_X1)
            AVG3_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X2)
            DRV3_X2 = tf.reshape(tf.matmul(tf.reshape(AVG3_X2, [-1, 128]), DM_128), [-1, 128, 1])
            F3 = tf.concat([M3, MAX3_X1, AVG3_X1, DRV3_X1, MIN3_X1, AVG3_X2, DRV3_X2], axis=2)
            # Conv -> (?, 128, 16)

            DM_64 = derivative_matrix(64)

            L4 = tf.layers.Conv1D(filters=16, kernel_size=6, strides=1, padding='same', activation=tf.nn.relu, name='C4')(F3)
            M4 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L4)
            AVG4_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X1)
            DRV4_X1 = tf.reshape(tf.matmul(tf.reshape(AVG4_X1, [-1, 64]), DM_64), [-1, 64, 1])
            MAX4_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX3_X1)
            MIN4_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN3_X1)
            AVG4_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X2)
            DRV4_X2 = tf.reshape(tf.matmul(tf.reshape(AVG4_X2, [-1, 64]), DM_64), [-1, 64, 1])
            F4 = tf.concat([M4, MAX4_X1, AVG4_X1, DRV4_X1, MIN4_X1, AVG4_X2, DRV4_X2], axis=2)
            # Conv -> (?, 64, 16)

            DM_32 = derivative_matrix(32)

            L5 = tf.layers.Conv1D(filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name='C5')(F4)
            M5 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L5)
            AVG5_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X1)
            DRV5_X1 = tf.reshape(tf.matmul(tf.reshape(AVG5_X1, [-1, 32]), DM_32), [-1, 32, 1])
            MAX5_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX4_X1)
            MIN5_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN4_X1)
            AVG5_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X2)
            DRV5_X2 = tf.reshape(tf.matmul(tf.reshape(AVG5_X2, [-1, 32]), DM_32), [-1, 32, 1])
            F5 = tf.concat([M5, MAX5_X1, AVG5_X1, DRV5_X1, MIN5_X1, AVG5_X2, DRV5_X2], axis=2)

            DM_16 = derivative_matrix(16)

            # Conv -> (?, 32, 32)
            L6 = tf.layers.Conv1D(filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name='C6')(F5)
            M6 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L6)
            AVG6_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X1)
            DRV6_X1 = tf.reshape(tf.matmul(tf.reshape(AVG6_X1, [-1, 16]), DM_16), [-1, 16, 1])
            MAX6_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX5_X1)
            MIN6_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN5_X1)
            AVG6_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X2)
            DRV6_X2 = tf.reshape(tf.matmul(tf.reshape(AVG6_X2, [-1, 16]), DM_16), [-1, 16, 1])
            F6 = tf.concat([M6, MAX6_X1, AVG6_X1, DRV6_X1, MIN6_X1, AVG6_X2, DRV6_X2], axis=2)
            # Conv -> (?, 16, 32)

            DM_8 = derivative_matrix(8)

            L7 = tf.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='C7')(F6)
            M7 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L7)
            AVG7_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X1)
            DRV7_X1 = tf.reshape(tf.matmul(tf.reshape(AVG7_X1, [-1, 8]), DM_8), [-1, 8, 1])
            MAX7_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX6_X1)
            MIN7_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN6_X1)
            AVG7_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X2)
            DRV7_X2 = tf.reshape(tf.matmul(tf.reshape(AVG7_X2, [-1, 8]), DM_8), [-1, 8, 1])
            F7 = tf.concat([M7, MAX7_X1, AVG7_X1, DRV7_X1, MIN7_X1, AVG7_X2, DRV7_X2], axis=2)
            # Conv -> (?, 8, 64)

            L8 = tf.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='C8')(F7)
            M8 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(L8)
            AVG8_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X1)
            MAX8_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX7_X1)
            MIN8_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN7_X1)
            AVG8_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X2)
            F8 = tf.concat([M8, MAX8_X1, AVG8_X1, MIN8_X1, AVG8_X2], axis=2)
            # Conv -> (?, 4, 64)

            flat = tf.layers.Flatten()(F8)
            self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat), 0, name='Output')

            # Simplified cost/loss function
            self.cost_cnn = tf.reduce_mean(tf.square(self.Output_cnn - self.it_y_cnn))  # - hypothesis_std

            # Minimize
            self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_cnn)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'))

    def build_isc(self, init_isc):

        print('init isc length : %d' % len(init_isc))

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):
            init_w = tf.constant_initializer(init_isc[:, 0])
            init_b = tf.constant_initializer(init_isc[:, 1])
            SELECTOR = tf.one_hot(name="individual_selector", indices=tf.reshape(self.it_i, [-1]),
                                  depth=len(init_isc))
            WI = tf.get_variable(name="weight_individual", shape=(len(init_isc), 1), initializer=init_w)
            BI = tf.get_variable(name="bias_individual", shape=(len(init_isc), 1), initializer=init_b)
            self.Output_scaled = self.Output_cnn * tf.matmul(SELECTOR, WI) + tf.matmul(SELECTOR, BI)

            # Simplified cost/loss function
            self.cost_isc = tf.reduce_mean(tf.square(self.Output_scaled - self.it_y_cnn))  # - hypothesis_std

        var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        # Minimize
        self.train_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_isc, var_list=var_all)
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'), max_to_keep=len(init_isc))

    def predict(self, x1_test, x2_test):
        len_d = x1_test.shape[0]
        feed_dict = {self.X1: x1_test, self.X2: x2_test, self.Y: np.zeros((len_d, 1)), self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_test.shape[0] / batch_size)

        print('Prediction Started!')
        avg_cost = 0
        result = list()
        for _ in range(num_batches):
            cost_val, hy_val = self.sess.run([self.cost_cnn, self.Output_cnn], feed_dict={self.training: False})
            avg_cost += cost_val / num_batches
            result.extend(hy_val)

        return result

    def train(self, x1_data, x2_data, y_data, epochs):
        len_d = x1_data.shape[0]
        feed_dict = {self.X1: x1_data, self.X2: x2_data, self.Y: y_data, self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_data.shape[0] / batch_size)

        print('Learning Started!')
        unchanged = last_cost = 0
        effective = True
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_cnn, self.Output_cnn, self.train_cnn],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))
            if last_cost == avg_cost:
                unchanged += 1
            else:
                unchanged = 0
            if unchanged > 10:
                effective = False
                break
            last_cost = avg_cost

        return effective

    def train_isc(self, x1_data, x2_data, y_data, i_data, epochs):
        feed_dict = {self.X1: x1_data, self.X2: x2_data, self.Y: y_data, self.I: i_data}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_data.shape[0] / batch_size)

        print('Learning Started!')
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_isc, self.Output_scaled, self.train_all],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))

        return

    def save_model(self, file_name):
        save_path = self.saver.save(self.sess, file_name)
        return save_path

    def restore_model(self, file_name):
        self.saver.restore(self.sess, file_name)
        return


class ModelDualBase:

    def __init__(self, sess, name, f=(8, 16, 32, 64), k=(12, 6, 4, 3), d=0):
        self.sess = sess
        self.name = name
        self._build_net(f, k, d)

    def _build_net(self, f, k, d):

        assert len(f) == 4 and len(k) == 4, 'Wrong filter vector or kernel vector.'

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):

            self.I = tf.placeholder(tf.int32, shape=[None, 1], name='Ind')

        with tf.variable_scope(self.name+'/cnn', reuse=tf.AUTO_REUSE):

            # input place holders
            self.X1 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave1')
            self.X2 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave2')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Target')

            self.training = tf.placeholder(dtype=tf.bool)

            dataset = tf.data.Dataset.from_tensor_slices((self.X1, self.X2, self.Y, self.I)).batch(batch_size).repeat()
            self.iterator = dataset.make_initializable_iterator()
            self.it_x1_cnn, self.it_x2_cnn, self.it_y_cnn, self.it_i = self.iterator.get_next()

            DM_1024 = derivative_matrix(1024)

            # Input
            AVG0_X1 = tf.reshape(self.it_x1_cnn, [-1, 1024, 1])
            DRV0_X1 = tf.reshape(tf.matmul(tf.reshape(AVG0_X1, [-1, 1024]), DM_1024), [-1, 1024, 1])
            AVG0_X2 = tf.reshape(self.it_x2_cnn, [-1, 1024, 1])
            DRV0_X2 = tf.reshape(tf.matmul(tf.reshape(AVG0_X2, [-1, 1024]), DM_1024), [-1, 1024, 1])
            F0 = tf.concat([AVG0_X1, DRV0_X1, AVG0_X2, DRV0_X2], axis=2)

            DM_512 = derivative_matrix(512)

            # L1 SigIn shape = (?, 1024, 1)
            C1 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C1')(F0)
            M1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C1)
            AVG1_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            DRV1_X1 = tf.reshape(tf.matmul(tf.reshape(AVG1_X1, [-1, 512]), DM_512), [-1, 512, 1])
            MAX1_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            MIN1_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            AVG1_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X2)
            DRV1_X2 = tf.reshape(tf.matmul(tf.reshape(AVG1_X2, [-1, 512]), DM_512), [-1, 512, 1])
            F1 = tf.concat([M1, MAX1_X1, AVG1_X1, DRV1_X1, MIN1_X1, AVG1_X2, DRV1_X2], axis=2)
            # Conv -> (?, 512, 8)

            DM_256 = derivative_matrix(256)

            C2 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C2')(F1)
            M2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C2)
            AVG2_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X1)
            DRV2_X1 = tf.reshape(tf.matmul(tf.reshape(AVG2_X1, [-1, 256]), DM_256), [-1, 256, 1])
            MAX2_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX1_X1)
            MIN2_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN1_X1)
            AVG2_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X2)
            DRV2_X2 = tf.reshape(tf.matmul(tf.reshape(AVG2_X2, [-1, 256]), DM_256), [-1, 256, 1])
            F2 = tf.concat([M2, MAX2_X1, AVG2_X1, DRV2_X1, MIN2_X1, AVG2_X2, DRV2_X2], axis=2)
            # Conv -> (?, 256, 8)

            DM_128 = derivative_matrix(128)

            C3 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C3')(F2)
            M3 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C3)
            AVG3_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X1)
            DRV3_X1 = tf.reshape(tf.matmul(tf.reshape(AVG3_X1, [-1, 128]), DM_128), [-1, 128, 1])
            MAX3_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX2_X1)
            MIN3_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN2_X1)
            AVG3_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X2)
            DRV3_X2 = tf.reshape(tf.matmul(tf.reshape(AVG3_X2, [-1, 128]), DM_128), [-1, 128, 1])
            F3 = tf.concat([M3, MAX3_X1, AVG3_X1, DRV3_X1, MIN3_X1, AVG3_X2, DRV3_X2], axis=2)
            # Conv -> (?, 128, 16)

            DM_64 = derivative_matrix(64)

            C4 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C4')(F3)
            M4 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C4)
            AVG4_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X1)
            DRV4_X1 = tf.reshape(tf.matmul(tf.reshape(AVG4_X1, [-1, 64]), DM_64), [-1, 64, 1])
            MAX4_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX3_X1)
            MIN4_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN3_X1)
            AVG4_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X2)
            DRV4_X2 = tf.reshape(tf.matmul(tf.reshape(AVG4_X2, [-1, 64]), DM_64), [-1, 64, 1])
            F4 = tf.concat([M4, MAX4_X1, AVG4_X1, DRV4_X1, MIN4_X1, AVG4_X2, DRV4_X2], axis=2)
            # Conv -> (?, 64, 16)

            DM_32 = derivative_matrix(32)

            C5 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C5')(F4)
            M5 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C5)
            AVG5_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X1)
            DRV5_X1 = tf.reshape(tf.matmul(tf.reshape(AVG5_X1, [-1, 32]), DM_32), [-1, 32, 1])
            MAX5_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX4_X1)
            MIN5_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN4_X1)
            AVG5_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X2)
            DRV5_X2 = tf.reshape(tf.matmul(tf.reshape(AVG5_X2, [-1, 32]), DM_32), [-1, 32, 1])
            F5 = tf.concat([M5, MAX5_X1, AVG5_X1, DRV5_X1, MIN5_X1, AVG5_X2, DRV5_X2], axis=2)

            DM_16 = derivative_matrix(16)

            # Conv -> (?, 32, 32)
            C6 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C6')(F5)
            M6 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C6)
            AVG6_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X1)
            DRV6_X1 = tf.reshape(tf.matmul(tf.reshape(AVG6_X1, [-1, 16]), DM_16), [-1, 16, 1])
            MAX6_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX5_X1)
            MIN6_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN5_X1)
            AVG6_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X2)
            DRV6_X2 = tf.reshape(tf.matmul(tf.reshape(AVG6_X2, [-1, 16]), DM_16), [-1, 16, 1])
            F6 = tf.concat([M6, MAX6_X1, AVG6_X1, DRV6_X1, MIN6_X1, AVG6_X2, DRV6_X2], axis=2)
            # Conv -> (?, 16, 32)

            DM_8 = derivative_matrix(8)

            C7 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C7')(F6)
            M7 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C7)
            AVG7_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X1)
            DRV7_X1 = tf.reshape(tf.matmul(tf.reshape(AVG7_X1, [-1, 8]), DM_8), [-1, 8, 1])
            MAX7_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX6_X1)
            MIN7_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN6_X1)
            AVG7_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X2)
            DRV7_X2 = tf.reshape(tf.matmul(tf.reshape(AVG7_X2, [-1, 8]), DM_8), [-1, 8, 1])
            F7 = tf.concat([M7, MAX7_X1, AVG7_X1, DRV7_X1, MIN7_X1, AVG7_X2, DRV7_X2], axis=2)
            # Conv -> (?, 8, 64)

            C8 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C8')(F7)
            M8 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C8)
            AVG8_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X1)
            MAX8_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX7_X1)
            MIN8_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN7_X1)
            AVG8_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X2)
            F8 = tf.concat([M8, MAX8_X1, AVG8_X1, MIN8_X1, AVG8_X2], axis=2)
            # Conv -> (?, 4, 64)

            flat = tf.layers.Flatten()(F8)

            if d in (0, 1):
                self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat), 0, name='Output')
            else:
                FC1 = tf.layers.Dense(d, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
                FC2 = tf.layers.Dense(d, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC1)
                self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC2), 0, name='Output')

            # Simplified cost/loss function
            self.cost_cnn = tf.reduce_mean(tf.square(self.Output_cnn - self.it_y_cnn))  # - hypothesis_std

            # Minimize
            self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_cnn)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'))

    def build_isc(self, init_isc):

        print('init isc length : %d' % len(init_isc))

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):
            init_w = tf.constant_initializer(init_isc[:, 0])
            init_b = tf.constant_initializer(init_isc[:, 1])
            SELECTOR = tf.one_hot(name="individual_selector", indices=tf.reshape(self.it_i, [-1]),
                                  depth=len(init_isc))
            WI = tf.get_variable(name="weight_individual", shape=(len(init_isc), 1), initializer=init_w)
            BI = tf.get_variable(name="bias_individual", shape=(len(init_isc), 1), initializer=init_b)
            self.Output_scaled = self.Output_cnn * tf.matmul(SELECTOR, WI) + tf.matmul(SELECTOR, BI)

            # Simplified cost/loss function
            self.cost_isc = tf.reduce_mean(tf.square(self.Output_scaled - self.it_y_cnn))  # - hypothesis_std

        var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        # Minimize
        self.train_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_isc, var_list=var_all)
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'), max_to_keep=len(init_isc))

    def predict(self, x1_test, x2_test):
        len_d = x1_test.shape[0]
        feed_dict = {self.X1: x1_test, self.X2: x2_test, self.Y: np.zeros((len_d, 1)), self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_test.shape[0] / batch_size)

        print('Prediction Started!')
        avg_cost = 0
        result = list()
        for _ in range(num_batches):
            cost_val, hy_val = self.sess.run([self.cost_cnn, self.Output_cnn], feed_dict={self.training: False})
            avg_cost += cost_val / num_batches
            result.extend(hy_val)

        return result

    def train(self, x1_data, x2_data, y_data, epochs):
        len_d = x1_data.shape[0]
        feed_dict = {self.X1: x1_data, self.X2: x2_data, self.Y: y_data, self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_data.shape[0] / batch_size)

        print('Learning Started!')
        unchanged = last_cost = 0
        effective = True
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_cnn, self.Output_cnn, self.train_cnn],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))
            if last_cost == avg_cost:
                unchanged += 1
            else:
                unchanged = 0
            if unchanged > 10:
                effective = False
                break
            last_cost = avg_cost

        return effective

    def train_isc(self, x1_data, x2_data, y_data, i_data, epochs):
        feed_dict = {self.X1: x1_data, self.X2: x2_data, self.Y: y_data, self.I: i_data}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_data.shape[0] / batch_size)

        print('Learning Started!')
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_isc, self.Output_scaled, self.train_all],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))

        return

    def save_model(self, file_name):
        save_path = self.saver.save(self.sess, file_name)
        return save_path

    def restore_model(self, file_name):
        self.saver.restore(self.sess, file_name)
        return


class ModelDual2(ModelDualBase):

    def _build_net(self, f, k, d):

        assert len(f) == 4 and len(k) == 4 and d, 'Wrong filter vector or kernel vector.'

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):

            self.I = tf.placeholder(tf.int32, shape=[None, 1], name='Ind')

        with tf.variable_scope(self.name+'/cnn', reuse=tf.AUTO_REUSE):

            # input place holders
            self.X1 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave1')
            self.X2 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave2')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Target')

            self.training = tf.placeholder(dtype=tf.bool)

            dataset = tf.data.Dataset.from_tensor_slices((self.X1, self.X2, self.Y, self.I)).batch(batch_size).repeat()
            self.iterator = dataset.make_initializable_iterator()
            self.it_x1_cnn, self.it_x2_cnn, self.it_y_cnn, self.it_i = self.iterator.get_next()

            DM_1024 = derivative_matrix(1024)

            # Input
            AVG0_X1 = tf.reshape(self.it_x1_cnn, [-1, 1024, 1])
            DRV0_X1 = tf.reshape(tf.matmul(tf.reshape(AVG0_X1, [-1, 1024]), DM_1024), [-1, 1024, 1])
            AVG0_X2 = tf.reshape(self.it_x2_cnn, [-1, 1024, 1])
            DRV0_X2 = tf.reshape(tf.matmul(tf.reshape(AVG0_X2, [-1, 1024]), DM_1024), [-1, 1024, 1])
            F0 = tf.concat([AVG0_X1, DRV0_X1, AVG0_X2, DRV0_X2], axis=2)

            DM_512 = derivative_matrix(512)

            # L1 SigIn shape = (?, 1024, 1)
            C11 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C11')(F0)
            C12 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C12')(C11)
            M1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C12)
            AVG1_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            DRV1_X1 = tf.reshape(tf.matmul(tf.reshape(AVG1_X1, [-1, 512]), DM_512), [-1, 512, 1])
            MAX1_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            MIN1_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            AVG1_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X2)
            DRV1_X2 = tf.reshape(tf.matmul(tf.reshape(AVG1_X2, [-1, 512]), DM_512), [-1, 512, 1])
            F1 = tf.concat([M1, MAX1_X1, AVG1_X1, DRV1_X1, MIN1_X1, AVG1_X2, DRV1_X2], axis=2)
            # Conv -> (?, 512, 8)

            DM_256 = derivative_matrix(256)

            C21 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C21')(F1)
            C22 = tf.layers.Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C22')(C21)
            M2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C22)
            AVG2_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X1)
            DRV2_X1 = tf.reshape(tf.matmul(tf.reshape(AVG2_X1, [-1, 256]), DM_256), [-1, 256, 1])
            MAX2_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX1_X1)
            MIN2_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN1_X1)
            AVG2_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X2)
            DRV2_X2 = tf.reshape(tf.matmul(tf.reshape(AVG2_X2, [-1, 256]), DM_256), [-1, 256, 1])
            F2 = tf.concat([M2, MAX2_X1, AVG2_X1, DRV2_X1, MIN2_X1, AVG2_X2, DRV2_X2], axis=2)
            # Conv -> (?, 256, 8)

            DM_128 = derivative_matrix(128)

            C31 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C31')(F2)
            C32 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C32')(C31)
            M3 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C32)
            AVG3_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X1)
            DRV3_X1 = tf.reshape(tf.matmul(tf.reshape(AVG3_X1, [-1, 128]), DM_128), [-1, 128, 1])
            MAX3_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX2_X1)
            MIN3_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN2_X1)
            AVG3_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X2)
            DRV3_X2 = tf.reshape(tf.matmul(tf.reshape(AVG3_X2, [-1, 128]), DM_128), [-1, 128, 1])
            F3 = tf.concat([M3, MAX3_X1, AVG3_X1, DRV3_X1, MIN3_X1, AVG3_X2, DRV3_X2], axis=2)
            # Conv -> (?, 128, 16)

            DM_64 = derivative_matrix(64)

            C41 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C41')(F3)
            C42 = tf.layers.Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C42')(C41)
            M4 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C42)
            AVG4_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X1)
            DRV4_X1 = tf.reshape(tf.matmul(tf.reshape(AVG4_X1, [-1, 64]), DM_64), [-1, 64, 1])
            MAX4_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX3_X1)
            MIN4_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN3_X1)
            AVG4_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X2)
            DRV4_X2 = tf.reshape(tf.matmul(tf.reshape(AVG4_X2, [-1, 64]), DM_64), [-1, 64, 1])
            F4 = tf.concat([M4, MAX4_X1, AVG4_X1, DRV4_X1, MIN4_X1, AVG4_X2, DRV4_X2], axis=2)
            # Conv -> (?, 64, 16)

            DM_32 = derivative_matrix(32)

            C51 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C51')(F4)
            C52 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C52')(C51)
            M5 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C52)
            AVG5_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X1)
            DRV5_X1 = tf.reshape(tf.matmul(tf.reshape(AVG5_X1, [-1, 32]), DM_32), [-1, 32, 1])
            MAX5_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX4_X1)
            MIN5_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN4_X1)
            AVG5_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X2)
            DRV5_X2 = tf.reshape(tf.matmul(tf.reshape(AVG5_X2, [-1, 32]), DM_32), [-1, 32, 1])
            F5 = tf.concat([M5, MAX5_X1, AVG5_X1, DRV5_X1, MIN5_X1, AVG5_X2, DRV5_X2], axis=2)

            DM_16 = derivative_matrix(16)

            # Conv -> (?, 32, 32)
            C61 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C61')(F5)
            C62 = tf.layers.Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C61')(C61)
            M6 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C62)
            AVG6_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X1)
            DRV6_X1 = tf.reshape(tf.matmul(tf.reshape(AVG6_X1, [-1, 16]), DM_16), [-1, 16, 1])
            MAX6_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX5_X1)
            MIN6_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN5_X1)
            AVG6_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X2)
            DRV6_X2 = tf.reshape(tf.matmul(tf.reshape(AVG6_X2, [-1, 16]), DM_16), [-1, 16, 1])
            F6 = tf.concat([M6, MAX6_X1, AVG6_X1, DRV6_X1, MIN6_X1, AVG6_X2, DRV6_X2], axis=2)
            # Conv -> (?, 16, 32)

            DM_8 = derivative_matrix(8)

            C71 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C71')(F6)
            C72 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C72')(C71)
            M7 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C72)
            AVG7_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X1)
            DRV7_X1 = tf.reshape(tf.matmul(tf.reshape(AVG7_X1, [-1, 8]), DM_8), [-1, 8, 1])
            MAX7_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX6_X1)
            MIN7_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN6_X1)
            AVG7_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X2)
            DRV7_X2 = tf.reshape(tf.matmul(tf.reshape(AVG7_X2, [-1, 8]), DM_8), [-1, 8, 1])
            F7 = tf.concat([M7, MAX7_X1, AVG7_X1, DRV7_X1, MIN7_X1, AVG7_X2, DRV7_X2], axis=2)
            # Conv -> (?, 8, 64)

            C81 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C81')(F7)
            C82 = tf.layers.Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C82')(C81)
            M8 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C82)
            AVG8_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X1)
            MAX8_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX7_X1)
            MIN8_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN7_X1)
            AVG8_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X2)
            F8 = tf.concat([M8, MAX8_X1, AVG8_X1, MIN8_X1, AVG8_X2], axis=2)
            # Conv -> (?, 4, 64)

            flat = tf.layers.Flatten()(F8)

            if d in (0, 1):
                self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu,
                                                         kernel_initializer=tf.contrib.layers.xavier_initializer())(
                    flat), 0, name='Output')
            else:
                FC1 = tf.layers.Dense(d, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
                FC2 = tf.layers.Dense(d, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC1)
                self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC2), 0, name='Output')

            # Simplified cost/loss function
            self.cost_cnn = tf.reduce_mean(tf.square(self.Output_cnn - self.it_y_cnn))  # - hypothesis_std

            # Minimize
            self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_cnn)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'))


class Model_DUAL_1:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):

            self.I = tf.placeholder(tf.int32, shape=[None, 1], name='Ind')

        with tf.variable_scope(self.name+'/cnn', reuse=tf.AUTO_REUSE):

            # input place holders
            self.X1 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave1')
            self.X2 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave2')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Target')

            self.training = tf.placeholder(dtype=tf.bool)

            dataset = tf.data.Dataset.from_tensor_slices((self.X1, self.X2, self.Y, self.I)).batch(batch_size).repeat()
            self.iterator = dataset.make_initializable_iterator()
            self.it_x1_cnn, self.it_x2_cnn, self.it_y_cnn, self.it_i = self.iterator.get_next()

            DM_1024 = derivative_matrix(1024)

            # Input
            AVG0_X1 = tf.reshape(self.it_x1_cnn, [-1, 1024, 1])
            DRV0_X1 = tf.reshape(tf.matmul(tf.reshape(AVG0_X1, [-1, 1024]), DM_1024), [-1, 1024, 1])
            AVG0_X2 = tf.reshape(self.it_x2_cnn, [-1, 1024, 1])
            DRV0_X2 = tf.reshape(tf.matmul(tf.reshape(AVG0_X2, [-1, 1024]), DM_1024), [-1, 1024, 1])
            F0 = tf.concat([AVG0_X1, DRV0_X1, AVG0_X2, DRV0_X2], axis=2)

            DM_512 = derivative_matrix(512)

            # L1 SigIn shape = (?, 1024, 1)
            C11 = tf.layers.Conv1D(filters=16, kernel_size=13, strides=1, padding='same', activation='relu', name='C11')(F0)
            C12 = tf.layers.Conv1D(filters=16, kernel_size=13, strides=1, padding='same', activation='relu', name='C12')(C11)
            M1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C12)
            AVG1_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            DRV1_X1 = tf.reshape(tf.matmul(tf.reshape(AVG1_X1, [-1, 512]), DM_512), [-1, 512, 1])
            MAX1_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            MIN1_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            AVG1_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X2)
            DRV1_X2 = tf.reshape(tf.matmul(tf.reshape(AVG1_X2, [-1, 512]), DM_512), [-1, 512, 1])
            F1 = tf.concat([M1, MAX1_X1, AVG1_X1, DRV1_X1, MIN1_X1, AVG1_X2, DRV1_X2], axis=2)
            # Conv -> (?, 512, 8)

            DM_256 = derivative_matrix(256)

            C21 = tf.layers.Conv1D(filters=16, kernel_size=13, strides=1, padding='same', activation='relu', name='C21')(F1)
            C22 = tf.layers.Conv1D(filters=16, kernel_size=13, strides=1, padding='same', activation='relu', name='C22')(C21)
            M2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C22)
            AVG2_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X1)
            DRV2_X1 = tf.reshape(tf.matmul(tf.reshape(AVG2_X1, [-1, 256]), DM_256), [-1, 256, 1])
            MAX2_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX1_X1)
            MIN2_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN1_X1)
            AVG2_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X2)
            DRV2_X2 = tf.reshape(tf.matmul(tf.reshape(AVG2_X2, [-1, 256]), DM_256), [-1, 256, 1])
            F2 = tf.concat([M2, MAX2_X1, AVG2_X1, DRV2_X1, MIN2_X1, AVG2_X2, DRV2_X2], axis=2)
            # Conv -> (?, 256, 8)

            DM_128 = derivative_matrix(128)

            C31 = tf.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu', name='C31')(F2)
            C32 = tf.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu', name='C32')(C31)
            M3 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C32)
            AVG3_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X1)
            DRV3_X1 = tf.reshape(tf.matmul(tf.reshape(AVG3_X1, [-1, 128]), DM_128), [-1, 128, 1])
            MAX3_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX2_X1)
            MIN3_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN2_X1)
            AVG3_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X2)
            DRV3_X2 = tf.reshape(tf.matmul(tf.reshape(AVG3_X2, [-1, 128]), DM_128), [-1, 128, 1])
            F3 = tf.concat([M3, MAX3_X1, AVG3_X1, DRV3_X1, MIN3_X1, AVG3_X2, DRV3_X2], axis=2)
            # Conv -> (?, 128, 16)

            DM_64 = derivative_matrix(64)

            C41 = tf.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu', name='C41')(F3)
            C42 = tf.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu', name='C42')(C41)
            M4 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C42)
            AVG4_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X1)
            DRV4_X1 = tf.reshape(tf.matmul(tf.reshape(AVG4_X1, [-1, 64]), DM_64), [-1, 64, 1])
            MAX4_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX3_X1)
            MIN4_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN3_X1)
            AVG4_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X2)
            DRV4_X2 = tf.reshape(tf.matmul(tf.reshape(AVG4_X2, [-1, 64]), DM_64), [-1, 64, 1])
            F4 = tf.concat([M4, MAX4_X1, AVG4_X1, DRV4_X1, MIN4_X1, AVG4_X2, DRV4_X2], axis=2)
            # Conv -> (?, 64, 16)

            DM_32 = derivative_matrix(32)

            C51 = tf.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='C51')(F4)
            C52 = tf.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='C52')(C51)
            M5 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C52)
            AVG5_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X1)
            DRV5_X1 = tf.reshape(tf.matmul(tf.reshape(AVG5_X1, [-1, 32]), DM_32), [-1, 32, 1])
            MAX5_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX4_X1)
            MIN5_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN4_X1)
            AVG5_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X2)
            DRV5_X2 = tf.reshape(tf.matmul(tf.reshape(AVG5_X2, [-1, 32]), DM_32), [-1, 32, 1])
            F5 = tf.concat([M5, MAX5_X1, AVG5_X1, DRV5_X1, MIN5_X1, AVG5_X2, DRV5_X2], axis=2)

            DM_16 = derivative_matrix(16)

            # Conv -> (?, 32, 32)
            C61 = tf.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='C61')(F5)
            C62 = tf.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='C61')(C61)
            M6 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C62)
            AVG6_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X1)
            DRV6_X1 = tf.reshape(tf.matmul(tf.reshape(AVG6_X1, [-1, 16]), DM_16), [-1, 16, 1])
            MAX6_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX5_X1)
            MIN6_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN5_X1)
            AVG6_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X2)
            DRV6_X2 = tf.reshape(tf.matmul(tf.reshape(AVG6_X2, [-1, 16]), DM_16), [-1, 16, 1])
            F6 = tf.concat([M6, MAX6_X1, AVG6_X1, DRV6_X1, MIN6_X1, AVG6_X2, DRV6_X2], axis=2)
            # Conv -> (?, 16, 32)

            DM_8 = derivative_matrix(8)

            C71 = tf.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='C71')(F6)
            C72 = tf.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='C72')(C71)
            M7 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C72)
            AVG7_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X1)
            DRV7_X1 = tf.reshape(tf.matmul(tf.reshape(AVG7_X1, [-1, 8]), DM_8), [-1, 8, 1])
            MAX7_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX6_X1)
            MIN7_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN6_X1)
            AVG7_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X2)
            DRV7_X2 = tf.reshape(tf.matmul(tf.reshape(AVG7_X2, [-1, 8]), DM_8), [-1, 8, 1])
            F7 = tf.concat([M7, MAX7_X1, AVG7_X1, DRV7_X1, MIN7_X1, AVG7_X2, DRV7_X2], axis=2)
            # Conv -> (?, 8, 64)

            C81 = tf.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='C81')(F7)
            C82 = tf.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='C82')(C81)
            M8 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C82)
            AVG8_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X1)
            MAX8_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX7_X1)
            MIN8_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN7_X1)
            AVG8_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X2)
            F8 = tf.concat([M8, MAX8_X1, AVG8_X1, MIN8_X1, AVG8_X2], axis=2)
            # Conv -> (?, 4, 64)

            flat = tf.layers.Flatten()(F8)
            FC1 = tf.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
            FC2 = tf.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC1)
            self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC2), 0, name='Output')

            # Simplified cost/loss function
            self.cost_cnn = tf.reduce_mean(tf.square(self.Output_cnn - self.it_y_cnn))  # - hypothesis_std

            # Minimize
            self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_cnn)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'))

    def build_isc(self, init_isc):

        print('init isc length : %d' % len(init_isc))

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):
            init_w = tf.constant_initializer(init_isc[:, 0])
            init_b = tf.constant_initializer(init_isc[:, 1])
            SELECTOR = tf.one_hot(name="individual_selector", indices=tf.reshape(self.it_i, [-1]),
                                  depth=len(init_isc))
            WI = tf.get_variable(name="weight_individual", shape=(len(init_isc), 1), initializer=init_w)
            BI = tf.get_variable(name="bias_individual", shape=(len(init_isc), 1), initializer=init_b)
            self.Output_scaled = self.Output_cnn * tf.matmul(SELECTOR, WI) + tf.matmul(SELECTOR, BI)

            # Simplified cost/loss function
            self.cost_isc = tf.reduce_mean(tf.square(self.Output_scaled - self.it_y_cnn))  # - hypothesis_std

        var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        # Minimize
        self.train_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_isc, var_list=var_all)
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'), max_to_keep=len(init_isc))

    def predict(self, x1_test, x2_test):
        len_d = x1_test.shape[0]
        feed_dict = {self.X1: x1_test, self.X2: x2_test, self.Y: np.zeros((len_d, 1)), self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_test.shape[0] / batch_size)

        print('Prediction Started!')
        avg_cost = 0
        result = list()
        for _ in range(num_batches):
            cost_val, hy_val = self.sess.run([self.cost_cnn, self.Output_cnn], feed_dict={self.training: False})
            avg_cost += cost_val / num_batches
            result.extend(hy_val)

        return result

    def train(self, x1_data, x2_data, y_data, epochs):
        len_d = x1_data.shape[0]
        feed_dict = {self.X1: x1_data, self.X2: x2_data, self.Y: y_data, self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_data.shape[0] / batch_size)

        print('Learning Started!')
        unchanged = last_cost = 0
        effective = True
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_cnn, self.Output_cnn, self.train_cnn],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))
            if last_cost == avg_cost:
                unchanged += 1
            else:
                unchanged = 0
            if unchanged > 10:
                effective = False
                break
            last_cost = avg_cost

        return effective

    def train_isc(self, x1_data, x2_data, y_data, i_data, epochs):
        feed_dict = {self.X1: x1_data, self.X2: x2_data, self.Y: y_data, self.I: i_data}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_data.shape[0] / batch_size)

        print('Learning Started!')
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_isc, self.Output_scaled, self.train_all],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))

        return

    def save_model(self, file_name):
        save_path = self.saver.save(self.sess, file_name)
        return save_path

    def restore_model(self, file_name):
        self.saver.restore(self.sess, file_name)
        return


class Model_DUAL_2:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):

            self.I = tf.placeholder(tf.int32, shape=[None, 1], name='Ind')

        with tf.variable_scope(self.name+'/cnn', reuse=tf.AUTO_REUSE):

            # input place holders
            self.X1 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave1')
            self.X2 = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave2')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Target')

            self.training = tf.placeholder(dtype=tf.bool)

            dataset = tf.data.Dataset.from_tensor_slices((self.X1, self.X2, self.Y, self.I)).batch(batch_size).repeat()
            self.iterator = dataset.make_initializable_iterator()
            self.it_x1_cnn, self.it_x2_cnn, self.it_y_cnn, self.it_i = self.iterator.get_next()

            DM_1024 = derivative_matrix(1024)

            # Input
            AVG0_X1 = tf.reshape(self.it_x1_cnn, [-1, 1024, 1])
            DRV0_X1 = tf.reshape(tf.matmul(tf.reshape(AVG0_X1, [-1, 1024]), DM_1024), [-1, 1024, 1])
            F0_X1 = tf.concat([AVG0_X1, DRV0_X1], axis=2)

            DM_512 = derivative_matrix(512)

            # L1 SigIn shape = (?, 1024, 1)
            C11_X1 = tf.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu', name='C11')(F0_X1)
            C12_X1 = tf.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu', name='C12')(C11_X1)
            M1_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C12_X1)
            AVG1_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            DRV1_X1 = tf.reshape(tf.matmul(tf.reshape(AVG1_X1, [-1, 512]), DM_512), [-1, 512, 1])
            MAX1_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            MIN1_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(AVG0_X1)
            F1_X1 = tf.concat([M1_X1, MAX1_X1, AVG1_X1, DRV1_X1, MIN1_X1], axis=2)
            # Conv -> (?, 512, 8)

            DM_256 = derivative_matrix(256)

            C21_X1 = tf.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu', name='C21')(F1_X1)
            C22_X1 = tf.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu', name='C22')(C21_X1)
            M2_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C22_X1)
            AVG2_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X1)
            DRV2_X1 = tf.reshape(tf.matmul(tf.reshape(AVG2_X1, [-1, 256]), DM_256), [-1, 256, 1])
            MAX2_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX1_X1)
            MIN2_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN1_X1)
            F2_X1 = tf.concat([M2_X1, MAX2_X1, AVG2_X1, DRV2_X1, MIN2_X1], axis=2)
            # Conv -> (?, 256, 8)

            DM_128 = derivative_matrix(128)

            C31_X1 = tf.layers.Conv1D(filters=16, kernel_size=7, strides=1, padding='same', activation='relu', name='C31')(F2_X1)
            C32_X1 = tf.layers.Conv1D(filters=16, kernel_size=7, strides=1, padding='same', activation='relu', name='C32')(C31_X1)
            M3_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C32_X1)
            AVG3_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X1)
            DRV3_X1 = tf.reshape(tf.matmul(tf.reshape(AVG3_X1, [-1, 128]), DM_128), [-1, 128, 1])
            MAX3_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX2_X1)
            MIN3_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN2_X1)
            F3_X1 = tf.concat([M3_X1, MAX3_X1, AVG3_X1, DRV3_X1, MIN3_X1], axis=2)
            # Conv -> (?, 128, 16)

            DM_64 = derivative_matrix(64)

            C41_X1 = tf.layers.Conv1D(filters=16, kernel_size=7, strides=1, padding='same', activation='relu', name='C41')(F3_X1)
            C42_X1 = tf.layers.Conv1D(filters=16, kernel_size=7, strides=1, padding='same', activation='relu', name='C42')(C41_X1)
            M4_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C42_X1)
            AVG4_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X1)
            DRV4_X1 = tf.reshape(tf.matmul(tf.reshape(AVG4_X1, [-1, 64]), DM_64), [-1, 64, 1])
            MAX4_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX3_X1)
            MIN4_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN3_X1)
            F4_X1 = tf.concat([M4_X1, MAX4_X1, AVG4_X1, DRV4_X1, MIN4_X1], axis=2)
            # Conv -> (?, 64, 16)

            DM_32 = derivative_matrix(32)

            C51_X1 = tf.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name='C51')(F4_X1)
            C52_X1 = tf.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name='C52')(C51_X1)
            M5_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C52_X1)
            AVG5_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X1)
            DRV5_X1 = tf.reshape(tf.matmul(tf.reshape(AVG5_X1, [-1, 32]), DM_32), [-1, 32, 1])
            MAX5_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX4_X1)
            MIN5_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN4_X1)
            F5_X1 = tf.concat([M5_X1, MAX5_X1, AVG5_X1, DRV5_X1, MIN5_X1], axis=2)

            DM_16 = derivative_matrix(16)

            # Conv -> (?, 32, 32)
            C61_X1 = tf.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name='C61')(F5_X1)
            C62_X1 = tf.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name='C61')(C61_X1)
            M6_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C62_X1)
            AVG6_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X1)
            DRV6_X1 = tf.reshape(tf.matmul(tf.reshape(AVG6_X1, [-1, 16]), DM_16), [-1, 16, 1])
            MAX6_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX5_X1)
            MIN6_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN5_X1)
            F6_X1 = tf.concat([M6_X1, MAX6_X1, AVG6_X1, DRV6_X1, MIN6_X1], axis=2)
            # Conv -> (?, 16, 32)

            DM_8 = derivative_matrix(8)

            C71_X1 = tf.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='C71')(F6_X1)
            C72_X1 = tf.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='C72')(C71_X1)
            M7_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C72_X1)
            AVG7_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X1)
            DRV7_X1 = tf.reshape(tf.matmul(tf.reshape(AVG7_X1, [-1, 8]), DM_8), [-1, 8, 1])
            MAX7_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX6_X1)
            MIN7_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN6_X1)
            F7_X1 = tf.concat([M7_X1, MAX7_X1, AVG7_X1, DRV7_X1, MIN7_X1], axis=2)
            # Conv -> (?, 8, 64)

            C81_X1 = tf.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='C81')(F7_X1)
            C82_X1 = tf.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='C82')(C81_X1)
            M8_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C82_X1)
            AVG8_X1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7_X1)
            MAX8_X1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(MAX7_X1)
            MIN8_X1 = -tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(-MIN7_X1)
            F8_X1 = tf.layers.Flatten()(tf.concat([M8_X1, MAX8_X1, AVG8_X1, MIN8_X1], axis=2))
            # Conv -> (?, 4, 64)

            AVG0_X2 = tf.reshape(self.it_x2_cnn, [-1, 1024, 1])
            DRV0_X2 = tf.reshape(tf.matmul(tf.reshape(AVG0_X2, [-1, 1024]), DM_1024), [-1, 1024, 1])
            F0_X2 = tf.concat([AVG0_X2, DRV0_X2], axis=2)

            C11_X2 = tf.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu', name='C11_X2')(F0_X2)
            C12_X2 = tf.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu', name='C12_X2')(C11_X2)
            M1_X2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C12_X2)
            AVG1_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0_X2)
            DRV1_X2 = tf.reshape(tf.matmul(tf.reshape(AVG1_X2, [-1, 512]), DM_512), [-1, 512, 1])
            F1_X2 = tf.concat([M1_X2, DRV1_X2], axis=2)

            C21_X2 = tf.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu', name='C21_X2')(F1_X2)
            C22_X2 = tf.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu', name='C22_X2')(C21_X2)
            M2_X2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C22_X2)
            AVG2_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1_X2)
            DRV2_X2 = tf.reshape(tf.matmul(tf.reshape(AVG2_X2, [-1, 256]), DM_256), [-1, 256, 1])
            F2_X2 = tf.concat([M2_X2, DRV2_X2], axis=2)

            C31_X2 = tf.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same', activation='relu', name='C31_X2')(F2_X2)
            C32_X2 = tf.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same', activation='relu', name='C32_X2')(C31_X2)
            M3_X2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C32_X2)
            AVG3_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2_X2)
            DRV3_X2 = tf.reshape(tf.matmul(tf.reshape(AVG3_X2, [-1, 128]), DM_128), [-1, 128, 1])
            F3_X2 = tf.concat([M3_X2, DRV3_X2], axis=2)

            C41_X2 = tf.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same', activation='relu', name='C41_X2')(F3_X2)
            C42_X2 = tf.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same', activation='relu', name='C42_X2')(C41_X2)
            M4_X2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C42_X2)
            AVG4_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3_X2)
            DRV4_X2 = tf.reshape(tf.matmul(tf.reshape(AVG4_X2, [-1, 64]), DM_64), [-1, 64, 1])
            F4_X2 = tf.concat([M4_X2, DRV4_X2], axis=2)

            C51_X2 = tf.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu', name='C51_X2')(F4_X2)
            C52_X2 = tf.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu', name='C52_X2')(C51_X2)
            M5_X2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C52_X2)
            AVG5_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4_X2)
            DRV5_X2 = tf.reshape(tf.matmul(tf.reshape(AVG5_X2, [-1, 32]), DM_32), [-1, 32, 1])
            F5_X2 = tf.concat([M5_X2, DRV5_X2], axis=2)

            C61_X2 = tf.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu', name='C61_X2')(F5_X2)
            C62_X2 = tf.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu', name='C62_X2')(C61_X2)
            M6_X2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C62_X2)
            AVG6_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5_X2)
            DRV6_X2 = tf.reshape(tf.matmul(tf.reshape(AVG6_X2, [-1, 16]), DM_16), [-1, 16, 1])
            F6_X2 = tf.concat([M6_X2, DRV6_X2], axis=2)

            C71_X2 = tf.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu', name='C71_X2')(F6_X2)
            C72_X2 = tf.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu', name='C72_X2')(C71_X2)
            M7_X2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C72_X2)
            AVG7_X2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6_X2)
            DRV7_X2 = tf.reshape(tf.matmul(tf.reshape(AVG7_X2, [-1, 8]), DM_8), [-1, 8, 1])
            F7_X2 = tf.concat([M7_X2, DRV7_X2], axis=2)

            C81_X2 = tf.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu', name='C81_X2')(F7_X2)
            C82_X2 = tf.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu', name='C82_X2')(C81_X2)
            M8_X2 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(C82_X2)
            F8_X2 = tf.layers.Flatten()(M8_X2)

            flat = tf.concat([F8_X1, F8_X2], 1)
            FC1 = tf.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
            FC2 = tf.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC1)
            self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(FC2), 0, name='Output')

            # Simplified cost/loss function
            self.cost_cnn = tf.reduce_mean(tf.square(self.Output_cnn - self.it_y_cnn))  # - hypothesis_std

            # Minimize
            self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_cnn)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'))

    def build_isc(self, init_isc):

        print('init isc length : %d' % len(init_isc))

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):
            init_w = tf.constant_initializer(init_isc[:, 0])
            init_b = tf.constant_initializer(init_isc[:, 1])
            SELECTOR = tf.one_hot(name="individual_selector", indices=tf.reshape(self.it_i, [-1]),
                                  depth=len(init_isc))
            WI = tf.get_variable(name="weight_individual", shape=(len(init_isc), 1), initializer=init_w)
            BI = tf.get_variable(name="bias_individual", shape=(len(init_isc), 1), initializer=init_b)
            self.Output_scaled = self.Output_cnn * tf.matmul(SELECTOR, WI) + tf.matmul(SELECTOR, BI)

            # Simplified cost/loss function
            self.cost_isc = tf.reduce_mean(tf.square(self.Output_scaled - self.it_y_cnn))  # - hypothesis_std

        var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        # Minimize
        self.train_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_isc, var_list=var_all)
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'), max_to_keep=len(init_isc))

    def predict(self, x1_test, x2_test):
        len_d = x1_test.shape[0]
        feed_dict = {self.X1: x1_test, self.X2: x2_test, self.Y: np.zeros((len_d, 1)), self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_test.shape[0] / batch_size)

        print('Prediction Started!')
        avg_cost = 0
        result = list()
        for _ in range(num_batches):
            cost_val, hy_val = self.sess.run([self.cost_cnn, self.Output_cnn], feed_dict={self.training: False})
            avg_cost += cost_val / num_batches
            result.extend(hy_val)

        return result

    def train(self, x1_data, x2_data, y_data, epochs):
        len_d = x1_data.shape[0]
        feed_dict = {self.X1: x1_data, self.X2: x2_data, self.Y: y_data, self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_data.shape[0] / batch_size)

        print('Learning Started!')
        unchanged = last_cost = 0
        effective = True
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_cnn, self.Output_cnn, self.train_cnn],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))
            if last_cost == avg_cost:
                unchanged += 1
            else:
                unchanged = 0
            if unchanged > 10:
                effective = False
                break
            last_cost = avg_cost

        return effective

    def train_isc(self, x1_data, x2_data, y_data, i_data, epochs):
        feed_dict = {self.X1: x1_data, self.X2: x2_data, self.Y: y_data, self.I: i_data}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x1_data.shape[0] / batch_size)

        print('Learning Started!')
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_isc, self.Output_scaled, self.train_all],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))

        return

    def save_model(self, file_name):
        save_path = self.saver.save(self.sess, file_name)
        return save_path

    def restore_model(self, file_name):
        self.saver.restore(self.sess, file_name)
        return


class Model_CNN_3:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):

            self.I = tf.placeholder(tf.int32, shape=[None, 1], name='Ind')

        with tf.variable_scope(self.name+'/cnn', reuse=tf.AUTO_REUSE):

            # input place holders
            self.X = tf.placeholder(tf.float32, shape=[None, 1024, 1], name='InputWave')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Target')

            self.training = tf.placeholder(dtype=tf.bool)

            dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y, self.I)).batch(batch_size).repeat()
            self.iterator = dataset.make_initializable_iterator()
            self.it_x_cnn, self.it_y_cnn, self.it_i = self.iterator.get_next()

            # Input
            AVG0 = tf.reshape(self.it_x_cnn, [-1, 1024, 1])
            DRV0 = tf.reshape(tf.matmul(tf.reshape(AVG0, [-1, 1024]), derivative_matrix(1024)), [-1, 1024, 1])
            F0 = tf.concat([AVG0, DRV0], axis=2)

            # L1 SigIn shape = (?, 1024, 1)
            L11 = tf.layers.Conv1D(filters=128, kernel_size=12, strides=1, padding='same', activation=tf.nn.relu, name='L11')(F0)
            L12 = tf.layers.Conv1D(filters=128, kernel_size=12, strides=2, padding='same', activation=tf.nn.relu, name='L12')(L11)

            #AVG1 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG0)
            #DRV1 = tf.reshape(tf.matmul(tf.reshape(AVG1, [-1, 512]), derivative_matrix(512)), [-1, 512, 1])
            #F1 = tf.concat([M1, MAX1, AVG1, DRV1, MIN1], axis=2)
            # Conv -> (?, 512, 8)

            L21 = tf.layers.Conv1D(filters=128, kernel_size=12, strides=1, padding='same', activation=tf.nn.relu, name='L21')(L12)
            L22 = tf.layers.Conv1D(filters=128, kernel_size=12, strides=2, padding='same', activation=tf.nn.relu, name='L22')(L21)
            #AVG2 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1)
            #DRV2 = tf.reshape(tf.matmul(tf.reshape(AVG2, [-1, 256]), derivative_matrix(256)), [-1, 256, 1])
            #F2 = tf.concat([M2, MAX2, AVG2, DRV2, MIN2], axis=2)
            # Conv -> (?, 256, 8)

            L31 = tf.layers.Conv1D(filters=256, kernel_size=6, strides=1, padding='same', activation=tf.nn.relu, name='L31')(L22)
            L32 = tf.layers.Conv1D(filters=256, kernel_size=6, strides=2, padding='same', activation=tf.nn.relu, name='L32')(L31)
            #AVG3 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2)
            #DRV3 = tf.reshape(tf.matmul(tf.reshape(AVG3, [-1, 128]), derivative_matrix(128)), [-1, 128, 1])
            #F3 = tf.concat([M3, MAX3, AVG3, DRV3, MIN3], axis=2)
            # Conv -> (?, 128, 16)

            L41 = tf.layers.Conv1D(filters=256, kernel_size=6, strides=1, padding='same', activation=tf.nn.relu, name='L41')(L32)
            L42 = tf.layers.Conv1D(filters=256, kernel_size=6, strides=2, padding='same', activation=tf.nn.relu, name='L42')(L41)
            #AVG4 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3)
            #DRV4 = tf.reshape(tf.matmul(tf.reshape(AVG4, [-1, 64]), derivative_matrix(64)), [-1, 64, 1])
            #F4 = tf.concat([M4, MAX4, AVG4, DRV4, MIN4], axis=2)
            # Conv -> (?, 64, 16)

            L51 = tf.layers.Conv1D(filters=512, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name='L21')(L42)
            L52 = tf.layers.Conv1D(filters=512, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name='L22')(L51)
            #AVG5 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4)
            #DRV5 = tf.reshape(tf.matmul(tf.reshape(AVG5, [-1, 32]), derivative_matrix(32)), [-1, 32, 1])
            #F5 = tf.concat([M5, MAX5, AVG5, DRV5, MIN5], axis=2)

            # Conv -> (?, 32, 32)
            L61 = tf.layers.Conv1D(filters=512, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name='L61')(L52)
            L62 = tf.layers.Conv1D(filters=512, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name='L62')(L61)
            #AVG6 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5)
            #DRV6 = tf.reshape(tf.matmul(tf.reshape(AVG6, [-1, 16]), derivative_matrix(16)), [-1, 16, 1])
            #F6 = tf.concat([M6, MAX6, AVG6, DRV6, MIN6], axis=2)
            # Conv -> (?, 16, 32)

            L71 = tf.layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='L71')(L62)
            L72 = tf.layers.Conv1D(filters=1024, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='L72')(L71)
            #AVG7 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6)
            #DRV7 = tf.reshape(tf.matmul(tf.reshape(AVG7, [-1, 8]), derivative_matrix(8)), [-1, 8, 1])
            #F7 = tf.concat([M7, MAX7, AVG7, DRV7, MIN7], axis=2)
            # Conv -> (?, 8, 64)

            L81 = tf.layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='L81')(L72)
            L82 = tf.layers.Conv1D(filters=1024, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='L82')(L81)
            #AVG8 = tf.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7)
            #F8 = tf.concat([M8, MAX8, AVG8, MIN8], axis=2)
            # Conv -> (?, 4, 64)

            flat = tf.layers.Flatten()(L82)
            D1 = tf.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
            D2 = tf.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(D1)
            self.Output_cnn = tf.add(tf.layers.Dense(1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())(D2), 0, name='Output')

            # Simplified cost/loss function
            self.cost_cnn = tf.reduce_mean(tf.square(self.Output_cnn - self.it_y_cnn))  # - hypothesis_std

            # Minimize
            self.train_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_cnn)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'))

    def build_isc(self, init_isc):

        with tf.variable_scope(self.name+'/isc', reuse=tf.AUTO_REUSE):
            init_w = tf.constant_initializer(init_isc[:, 0])
            init_b = tf.constant_initializer(init_isc[:, 1])
            SELECTOR = tf.one_hot(name="individual_selector", indices=tf.reshape(self.it_i, [-1]),
                                  depth=len(init_isc))
            WI = tf.get_variable(name="weight_individual", shape=(len(init_isc), 1), initializer=init_w)
            BI = tf.get_variable(name="bias_individual", shape=(len(init_isc), 1), initializer=init_b)
            self.Output_scaled = self.Output_cnn * tf.matmul(SELECTOR, WI) + tf.matmul(SELECTOR, BI)

            # Simplified cost/loss function
            self.cost_isc = tf.reduce_mean(tf.square(self.Output_scaled - self.it_y_cnn))  # - hypothesis_std

        var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        # Minimize
        self.train_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_isc, var_list=var_all)
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+'/cnn'), max_to_keep=len(init_isc))

    def predict(self, x_test):
        len_d = x_test.shape[0]
        feed_dict = {self.X: x_test, self.Y: np.zeros((len_d, 1)), self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x_test.shape[0] / batch_size)

        print('Prediction Started!')
        avg_cost = 0
        result = list()
        for _ in range(num_batches):
            cost_val, hy_val = self.sess.run([self.cost_cnn, self.Output_cnn], feed_dict={self.training: False})
            avg_cost += cost_val / num_batches
            result.extend(hy_val)

        return result

    def train(self, x_data, y_data, epochs):
        len_d = x_data.shape[0]
        feed_dict = {self.X: x_data, self.Y: y_data, self.I: np.zeros((len_d, 1))}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x_data.shape[0] / batch_size)

        unchanged = last_cost = 0
        effective = True
        print('Learning Started!')
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_cnn, self.Output_cnn, self.train_cnn],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))
            if last_cost == avg_cost:
                unchanged += 1
            else:
                unchanged = 0
            if unchanged > 10:
                effective = False
                break
            last_cost = avg_cost

        return effective

    def train_isc(self, x_data, y_data, i_data, epochs):
        feed_dict = {self.X: x_data, self.Y: y_data, self.I: i_data}
        self.sess.run([self.iterator.initializer], feed_dict=feed_dict)
        num_batches = ceil(x_data.shape[0] / batch_size)

        print('Learning Started!')
        for epoch in range(epochs):
            avg_cost = 0
            for _ in range(num_batches):
                cost_val, hy_val, _ = self.sess.run([self.cost_isc, self.Output_scaled, self.train_all],
                                                    feed_dict={self.training: True})
                avg_cost += cost_val / num_batches
            print('%s, epoch=%04d, cost=%.3f' % (datetime.datetime.now().isoformat(), epoch + 1, avg_cost))

        return

    def save_model(self, file_name):
        save_path = self.saver.save(self.sess, file_name)
        return save_path

    def restore_model(self, file_name):
        self.saver.restore(self.sess, file_name)
        return


class ModelInitISC:

    def __init__(self, sess, name, bias=True):
        self.sess = sess
        self.name = name
        self.bias = bias
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=[None, 1], name='X')
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

            self.W = tf.get_variable(name="weight_initial", shape=[1], initializer=tf.initializers.ones)

            if self.bias:
                self.B = tf.get_variable(name="bias_initial", shape=[1], initializer=tf.initializers.zeros)
                self.hypothesis = self.W * self.X + self.B
            else:
                self.hypothesis = self.W * self.X

            # Simplified cost/loss function
            self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))  # - hypothesis_std

            var_isc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            # Minimize
            self.train = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.cost, var_list=var_isc)

    def get_isc(self, x_data, y_data):
        self.sess.run(tf.global_variables_initializer())
        feed_dict = {self.X: x_data, self.Y: y_data}

        epoch = 0
        min_cost = 0
        min_cost_epoch = 0
        min_cost_isc = []
        is_effective = True

        # Start populating the filename queue.
        while is_effective:
            if self.bias:
                cost_val, _, tmp_w, tmp_b = self.sess.run([self.cost, self.train, self.W, self.B], feed_dict=feed_dict)
            else:
                cost_val, _, tmp_w = self.sess.run([self.cost, self.train, self.W], feed_dict=feed_dict)
            if min_cost == 0 or cost_val < min_cost:
                min_cost = cost_val
                min_cost_epoch = epoch
                if self.bias:
                    min_cost_isc = self.sess.run([self.W, self.B])
                else:
                    min_cost_isc = self.sess.run([self.W])
                    min_cost_isc.append(0)

            elif epoch > min_cost_epoch + 100:
                is_effective = False
            epoch += 1
            if not epoch % 100:
                if self.bias:
                    print('Epoch=%03d, cost=%.4f, weight=%.2f, bias=%.2f' % (epoch + 1, cost_val, tmp_w, tmp_b))
                else:
                    print('Epoch=%03d, cost=%.4f, weight=%.2f, bias=0' % (epoch + 1, cost_val, tmp_w))

        return min_cost_isc


def get_models(sess, max_to_keep=5):
    model_class = list()
    model_class.append(ModelSingleBase(sess, 'a0', max_to_keep=max_to_keep))
    model_class.append(ModelSingle2(sess, 'a1', f=(16, 32, 64, 128), k=(9, 7, 5, 3), d=32))
    model_class.append(ModelSingleBase(sess, 'a2', f=(16, 32, 64, 128), k=(9, 7, 5, 3), d=32))
    model_class.append(ModelSingle2(sess, 'a3', f=(16, 32, 64, 128), k=(9, 7, 5, 3), d=32))
    model_class.append(ModelDualBase(sess, 'd0', d=32))
    model_class.append(ModelDualBase(sess, 'd1', f=(16, 32, 64, 128), k=(9, 7, 5, 3), d=32))
    model_class.append(ModelDualBase(sess, 'd2', f=(16, 32, 64, 128), k=(9, 7, 5, 3), d=32))
    return model_class


def get_model_info(model_name):
    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()
    query = "SELECT model_path, model_type FROM model WHERE model_name='%s'" % model_name
    cursor.execute(query)
    query_results = cursor.fetchall()
    assert len(query_results) == 1, 'Wrong model: %s' % model_name
    db.close()

    return query_results[0][0], query_results[0][1]


#origin -> init
def get_r_inter_npz(model_name, npzs, target_dir, vals, ref='VG_SV', model_type=0, epoches=500, init=False, normalize_abp=True):

    sess = tf.Session()
    model_class = get_models(sess)

    title = list()
    result_r_table = list()

    title.append('dataset')

    tmp_npzs = list()
    for npz in npzs:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    npzs = tmp_npzs

    max_pcr = 0

    # Launch the graph in a session.
    for train_data_npz in npzs:
        # Initializes global variables in the graph.
        train_data = np.load(train_data_npz, allow_pickle=True)
        d = cf.load_npz(train_data, ref=ref, to3d=True, trim=True, normalize_abp=normalize_abp)
        col_dict = d['col_dict']

        if len(d['timestamp']):
            result_row = list()
            result_row.append(train_data_npz)
            logging.info('%s, building a model with data %s.' % (datetime.datetime.now().isoformat(), train_data_npz))
            sess.run(tf.global_variables_initializer())
            if model_type in [0, 1, 2, 3]:
                if init:
                    model_class[model_type].restore_model(os.path.join(model_dir, model_name, 'init'))
                effective = model_class[model_type].train(d['abp'], d['feature'][:, col_dict[ref]], epoches)
            elif model_type in [4, 5, 6]:
                if init:
                    model_class[model_type].restore_model(os.path.join(model_dir, model_name, 'init'))
                effective = model_class[model_type].train(d['abp'], d['ppg'], d['feature'][:, col_dict[ref]], epoches)
            else:
                assert False

            # Start populating the filename queue.
            for test_data_npz in vals:
                test_data = np.load(test_data_npz, allow_pickle=True)
                d_test = cf.load_npz(test_data, ref=ref, to3d=True, trim=True, normalize_abp=normalize_abp)
                if len(d_test['timestamp']):
                    if test_data_npz not in title:
                        title.append(test_data_npz)
                    if effective:
                        if model_type in [0, 1, 2, 3]:
                            predicted = model_class[model_type].predict(d_test['abp'])
                        elif model_type in [4, 5, 6]:
                            predicted = model_class[model_type].predict(d_test['abp'], d_test['ppg'])
                        else:
                            assert False
                        predicted = cf.smoothing_result(predicted, d_test['timestamp'], type='datetime')
                        error_pcr = pearsonr(d_test['feature'][:, col_dict[ref]], predicted)[0][0]
                        error_rms = sqrt(mean_squared_error(d_test['feature'][:, col_dict[ref]], predicted))
                        result_row.append(error_pcr if not np.isnan(error_pcr) else 0)
                    else:
                        result_row.append(0)
            if len(result_row) > 1:
                result_row.append(sum(result_row[1:])/len(result_row[1:]))
            else:
                result_row.append(result_row[0])
            if result_row[-1] > max_pcr:
                max_pcr = result_row[-1]
                print('%s, better model, file=%s, pcr=%.3f' % (datetime.datetime.now().isoformat(), train_data_npz, max_pcr))
                if model_type in [0, 1, 2, 3]:
                    model_class[model_type].save_model(os.path.join(model_dir, model_name, 'origin'))
                elif model_type in [4, 5, 6]:
                    model_class[model_type].save_model(os.path.join(model_dir, model_name, 'origin'))
                else:
                    assert False

            result_r_table.append(result_row)

    title.append('average')

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    query = "DELETE FROM quality_index WHERE model_name ='%s'" % model_name
    cursor.execute(query)
    query = "DELETE FROM model WHERE model_name ='%s'" % model_name
    cursor.execute(query)
    db.commit()

    query = "INSERT INTO model (model_name, model_path, model_type) VALUES ('%s','%s',%d)" % (model_name, os.path.join(model_dir, model_name), model_type)
    cursor.execute(query)
    db.commit()

    with open(os.path.join(target_dir, model_name + '.csv'), 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title)
        for i, row in enumerate(result_r_table):
            query = 'INSERT INTO quality_index (model_name, file_name, file_path, quality_index) VALUES '
            query += '(\'%s\', \'%s\', \'%s\', %f)' % (model_name, os.path.basename(row[0]), os.path.dirname(row[0]), row[-1])
            cursor.execute(query)
            row[0] = os.path.basename(row[0])
            cyclewriter.writerow(row)

    db.commit()


def build_model(model_name, npzs, ref='VG_SV', model_type=0, epoches=1000, normalize_abp=False, restore=False):

    sess = tf.Session()
    model_class = get_models(sess)

    tf.set_random_seed(random.randint(0, 1000))

    title = list()

    title.append('dataset')

    tmp_npzs = list()
    for npz in npzs:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    npzs = tmp_npzs

    X1 = list()
    X2 = list()
    Y = list()

    # Launch the graph in a session.
    for train_data_npz in npzs:
        # Initializes global variables in the graph.
        train_data = np.load(train_data_npz, allow_pickle=True)
        d = cf.load_npz(train_data, ref=ref, to3d=True, trim=True, normalize_abp=normalize_abp)
        col_dict = d['col_dict']

        if len(d['timestamp']):
            Y.append(d['feature'][:, col_dict[ref]])
            X1.append(d['abp'])
            if model_type in [4, 5, 6]:
                X2.append(d['ppg'])

    X1 = np.concatenate(X1)
    Y = np.concatenate(Y)
    if model_type in [4, 5, 6]:
        X2 = np.concatenate(X2)

    sess.run(tf.global_variables_initializer())
    if restore:
        model_class[model_type].restore_model(os.path.join(model_dir, model_name, 'init'))

    if model_type in [0, 1, 2, 3]:
        effective = model_class[model_type].train(X1, Y, epoches)
    elif model_type in [4, 5, 6]:
        effective = model_class[model_type].train(X1, X2, Y, epoches)
    else:
        assert False

    if effective:
        model_class[model_type].save_model(os.path.join(model_dir, model_name, 'init'))


def build_scale_model(model_name, npzs, ref='VG_SV', model_type=0, epoches=1000, resume=False):

    sess = tf.Session()
    model_class = get_models(sess)

    tf.set_random_seed(random.randint(0, 1000))

    title = list()

    title.append('dataset')

    tmp_npzs = list()
    for npz in npzs:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    npzs = tmp_npzs

    X1 = list()
    X2 = list()
    Y = list()

    # Launch the graph in a session.
    for train_data_npz in npzs:
        # Initializes global variables in the graph.
        train_data = np.load(train_data_npz, allow_pickle=True)
        d = cf.load_npz(train_data, ref=ref, to3d=True, trim=True)
        col_dict = d['col_dict']

        if len(d['timestamp']):
            selected = random.sample(range(len(d['timestamp'])), min(len(d['timestamp']), 1024))
            for i in selected:
                Y.append(d['feature'][i, col_dict[ref]])
                X1.append(d['abp'][i])
                if model_type in [4, 5, 6]:
                    X2.append(d['ppg'][i])

    X1 = np.array(X1)
    Y = np.array(Y)
    if model_type in [4, 5, 6]:
        X2 = np.array(X2)

    sess.run(tf.global_variables_initializer())
    if resume:
        model_class[model_type].restore_model(os.path.join(model_dir, 'scale', model_name))
    if model_type in [0, 1, 2, 3]:
        effective = model_class[model_type].train(X1, Y, epoches)
    elif model_type in [4, 5, 6]:
        effective = model_class[model_type].train(X1, X2, Y, epoches)
    else:
        assert False

    if effective:
        model_class[model_type].save_model(os.path.join(model_dir, 'scale', model_name))

    return


def save_figure_simulation(d, sv_ev_predicted, svv_predicted, npz):
    formatter = DateFormatter('%H:%M')
    fig_size = 9
    plt.figure(figsize=(fig_size, fig_size*1.414))
    col_dict = d['col_dict']

    dataset = os.path.splitext(os.path.basename(npz))[0]

    ax = plt.subplot(3, 1, 1)
    plt.title(dataset)
    ax.xaxis.set_major_formatter(formatter)
    plt.ylabel('ml')
    plt.grid(linestyle='solid')
    plt.plot(d['timestamp'], d['feature'][:, col_dict['EV_SV']], color='red')
    plt.plot(d['timestamp'], sv_ev_predicted, color='blue')
    plt.legend(['SV, EV1000', 'SV, Model'], loc='upper right')

    ax = plt.subplot(3, 1, 2)
    ax.xaxis.set_major_formatter(formatter)
    plt.grid(linestyle='solid')
    plt.ylabel('mmHg')
    plt.plot(d['timestamp'], d['feature'][:, col_dict['GE_ART1_SBP']], color='olive')
    plt.plot(d['timestamp'], d['feature'][:, col_dict['GE_ART1_DBP']], color='cyan')
    plt.legend(['SBP', 'DBP'], loc='upper right')

    ax = plt.subplot(3, 1, 3)
    ax.xaxis.set_major_formatter(formatter)
    plt.grid(linestyle='solid')
    plt.ylabel('%')
    plt.plot(d['timestamp'], d['feature'][:, col_dict['GE_PPV']], color='red')
    plt.plot(d['timestamp'], d['feature'][:, col_dict['EV_SVV']], color='orange')
    plt.plot(d['timestamp'], svv_predicted, color='blue')
    plt.legend(['GE_PPV', 'EV_SVV', 'SVV, Model'], loc='upper right')

    plt.savefig(os.path.join('/home/shmoon/Result/simulation', dataset))
    plt.close()

    return


def test_ev_model(npzs):

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    model_class = get_models(sess)

    tmp_npzs = list()
    for npz in npzs:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    npzs = tmp_npzs

    title = list()
    title.append('dataset')
    title.append('CORR(EV_SV)')
    title.append('CORR(EV_SVV)')

    result_r_table = list()

    for npz in npzs:
        train_data = np.load(npz, allow_pickle=True)
        d = cf.load_npz(train_data, ref='EV_SV', to3d=True, trim=True)
        col_dict = d['col_dict']

        if len(d['timestamp']):
            model_class[1].restore_model(os.path.join(model_dir, 'scale', 'sv_ev_01'))
            sv_ev_predicted = cf.smoothing_result(model_class[1].predict(d['abp']), d['timestamp'], type='datetime')
            model_class[1].restore_model(os.path.join(model_dir, 'scale', 'svv_01'))
            svv_predicted = cf.smoothing_result(model_class[1].predict(d['abp']), d['timestamp'], type='datetime')

            tmp_row = list()
            tmp_row.append(os.path.basename(npz))
            tmp_row.append(pearsonr(sv_ev_predicted, d['feature'][:, col_dict['EV_SV']])[0][0])
            tmp_row.append(pearsonr(svv_predicted, d['feature'][:, col_dict['EV_SVV']])[0][0])

            save_figure_simulation(d, sv_ev_predicted, svv_predicted, os.path.splitext(os.path.basename(npz))[0])

            result_r_table.append(tmp_row)

    with open(os.path.join(validation_dir, 'simul_ev1000.csv'), 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title)
        for i, row in enumerate(result_r_table):
            cyclewriter.writerow(row)

    return


def test_scale_model(model_name, npzs, ref='VG_SV', model_type=0):

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    query = 'SELECT a.id, a.filename, a.bed, a.op_date, a.sex, a.age, a.height, a.weight, a.echo_date, a.esv, a.edv, '
    query += 'a.SV_DEMO_MODEL, (a.edv-a.esv) SV_ECHO FROM echo_data AS a INNER JOIN '
    query += '(SELECT filename, MAX(echo_date) echo_date FROM echo_data WHERE echo_date < op_date GROUP BY filename) AS b '
    query += 'ON a.filename = b.filename AND a.echo_date = b.echo_date'

    cursor.execute(query)
    echo_result = cursor.fetchall()

    echo_dict = dict()
    for row in echo_result:
        echo_dict[row[1]] = row

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    model_class = get_models(sess)

    tmp_npzs = list()
    for npz in npzs:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    npzs = tmp_npzs

    title = list()
    title.append('dataset')
    title.append('AVG(VG_SV)')
    title.append('AVG(EV_SV)')
    title.append('COR(EV_SV)')
    title.append('AVG(SM_SV)')
    title.append('COR(SM_SV)')
    title.append('AVG(SV_DEMO)')
    title.append('AVG(SV_ECHO)')
    title.append('DIFF(OP_ECHO)')

    result_r_table = list()

    for npz in npzs:
        train_data = np.load(npz, allow_pickle=True)
        d = cf.load_npz(train_data, ref=ref, to3d=True, trim=True)
        col_dict = d['col_dict']

        if len(d['timestamp']):

            i = 0
            while i < len(d['timestamp']) and (d['timestamp'][i] - d['timestamp'][0]).seconds < 1800:
                i += 1
            print(npz, i, len(d['timestamp']))
            if model_type in [0, 1, 2, 3]:
                model_class[model_type].restore_model(os.path.join(model_dir, 'scale', model_name))
                sv_scale_estimator = model_class[model_type].predict(d['abp'][:i])
            elif model_type in [4, 5, 6]:
                model_class[model_type].restore_model(os.path.join(model_dir, 'scale', model_name))
                sv_scale_estimator = model_class[model_type].predict(d['abp'][:i], d['ppg'][:i])
            else:
                assert False

            sv_scale_estimator = cf.smoothing_result(sv_scale_estimator, d['timestamp'], type='datetime')

            tmp_row = list()
            tmp_row.append(os.path.basename(npz))
            tmp_row.append(trim_mean(d['feature'][:i, col_dict[ref], :], 0.05)[0])
            tmp_row.append(trim_mean(d['feature'][:i, col_dict['EV_SV'], :], 0.05)[0])
            tmp_row.append(pearsonr(d['feature'][:i, col_dict[ref], :], d['feature'][:i, col_dict['EV_SV'], :])[0][0])
            tmp_row.append(trim_mean(sv_scale_estimator, 0.05)[0])
            tmp_row.append(pearsonr(d['feature'][:i, col_dict[ref], :], sv_scale_estimator)[0][0])

            if os.path.basename(npz) in echo_dict:
                tmp_row.append(echo_dict[os.path.basename(npz)][11])
                tmp_row.append(echo_dict[os.path.basename(npz)][12])
                tmp_row.append((echo_dict[os.path.basename(npz)][3] - echo_dict[os.path.basename(npz)][8]).days)
            else:
                tmp_row.append(None)
                tmp_row.append(None)
                tmp_row.append(None)

            result_r_table.append(tmp_row)

    with open(os.path.join(validation_dir, model_name + '.csv'), 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title)
        for i, row in enumerate(result_r_table):
            cyclewriter.writerow(row)

    return


def model_init_isc(model_name, ref='VG_SV', smoothing=False, normalize_abp=False, bias=True):

    formatter = DateFormatter('%H:%M')

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()
    query = "SELECT model_path, model_type FROM model WHERE model_name='%s'" % model_name
    cursor.execute(query)
    query_results = cursor.fetchall()
    assert len(query_results) == 1, 'Wrong model: %s' % model_name
    model_path = query_results[0][0]
    model_type = query_results[0][1]

    query = "SELECT file_name, file_path FROM quality_index WHERE model_name='%s' AND quality_index > 0 ORDER BY quality_index DESC" % model_name
    cursor.execute(query)
    training_files = cursor.fetchall()
    assert len(query_results), 'Wrong model: %s' % model_name

    sess = tf.Session()
    model_class = get_models(sess)

    # Initializes global variables in the graph.

    for npz in training_files:
        t_data = np.load(os.path.join(npz[1], npz[0]), allow_pickle=True)
        d = cf.load_npz(t_data, ref=ref, to3d=True, trim=True, normalize_abp=normalize_abp)
        col_dict = d['col_dict']

        sv_reference = d['feature'][:, col_dict[ref], :]
        model_class[model_type].restore_model(os.path.join(model_path, 'origin'))
        if model_type in [0, 1, 2, 3]:
            sv_unscaled = model_class[model_type].predict(d['abp'])
        elif model_type in [4, 5, 6]:
            sv_unscaled = model_class[model_type].predict(d['abp'], d['ppg'])
        else:
            assert False
        if smoothing:
            sv_unscaled = cf.smoothing_result(sv_unscaled, d['timestamp'], type='datetime')

        m_isc = ModelInitISC(sess, 'init_isc', bias=bias)
        tmp_isc = m_isc.get_isc(sv_unscaled, sv_reference)
        sv_scaled = list()
        for i in sv_unscaled:
            sv_scaled.append(i*tmp_isc[0]+tmp_isc[1])
        query = "UPDATE quality_index SET weight=%f, bias=%f " % (tmp_isc[0], tmp_isc[1])
        query += "WHERE model_name='%s' AND file_name='%s' AND file_path='%s'" % (model_name, npz[0], npz[1])
        cursor.execute(query)
        db.commit()
        del m_isc

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_formatter(formatter)
        ax.plot(d['timestamp'], sv_reference, color='red')
        ax.plot(d['timestamp'], sv_unscaled, color='grey')
        ax.plot(d['timestamp'], sv_scaled, color='blue')
        plt.savefig(os.path.join(model_path, 'fig', os.path.splitext(npz[0])[0]))
        '''

    db.close()
    return


def model_combine_npz(model_name, ref='VG_SV', normalize_abp=False, bias=True, train='all'):

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    query = "SELECT model_path, model_type FROM model WHERE model_name='%s'" % model_name
    cursor.execute(query)
    query_results = cursor.fetchall()
    assert len(query_results) == 1, 'Wrong model: %s' % model_name
    model_path = query_results[0][0]
    model_type = query_results[0][1]

    query = "SELECT file_name, file_path, weight, bias FROM quality_index WHERE model_name='%s' and quality_index > 0 ORDER BY quality_index DESC" % model_name
    cursor.execute(query)
    query_results = cursor.fetchall()
    db.close()
    assert len(query_results), 'Quality index doesn\'t exists.'

    sess = tf.Session()
    model_class = get_models(sess)
    model_class[model_type].restore_model(os.path.join(model_path, 'origin'))

    init_isc = list()
    input_abp = list()
    input_ppg = list()
    input_i = list()
    sv_unscaled = list()
    sv_reference = list()

    for npz in query_results:
        t_data = np.load(os.path.join(npz[1], npz[0]), allow_pickle=True)
        d = cf.load_npz(t_data, ref=ref, to3d=True, trim=True, normalize_abp=normalize_abp)
        if d['timestamp'].shape[0]:
            col_dict = d['col_dict']
            input_abp.append(d['abp'])
            input_i.append(np.full((d['abp'].shape[0], 1), len(input_i), dtype=np.int32))
            init_isc.append([npz[2], npz[3]])
            sv_reference.append(d['feature'][:, col_dict[ref], :])
            if model_type in [0, 1, 2, 3]:
                sv_unscaled.append(np.array(model_class[model_type].predict(d['abp']), ndmin=2))
            elif model_type in [4, 5, 6]:
                sv_unscaled.append(np.array(model_class[model_type].predict(d['abp'], d['ppg']), ndmin=2))
                input_ppg.append(d['ppg'])
            else:
                assert False

    init_isc = np.array(init_isc, dtype=np.float32)
    model_class[model_type].build_isc(init_isc, bias=bias)

    assert len(input_abp) == len(input_i) == len(sv_reference), 'Different lengths.'
    for i in range(len(input_abp)):
        tmp_abp = np.concatenate(input_abp[:i+1])
        tmp_i = np.concatenate(input_i[:i+1])
        tmp_sv = np.concatenate(sv_reference[:i+1])
        assert len(tmp_abp) == len(tmp_i) == len(tmp_sv), 'Different lengths.'
        sess.run(tf.global_variables_initializer())
        if model_type in [0, 1, 2, 3]:
            model_class[model_type].restore_model(os.path.join(model_path, 'origin'))
            model_class[model_type].train_isc(tmp_abp, tmp_sv, tmp_i, 50, train=train)
            model_class[model_type].save_model(os.path.join(model_path, 'm_%02d' % (i+1)))
        elif model_type in [4, 5, 6]:
            tmp_ppg = np.concatenate(input_ppg[:i+1])
            model_class[model_type].restore_model(os.path.join(model_path, 'origin'))
            model_class[model_type].train_isc(tmp_abp, tmp_ppg, tmp_sv, tmp_i, 50, train=train)
            model_class[model_type].save_model(os.path.join(model_path, 'm_%02d' % (i+1)))
        else:
            assert False

    return


def model_combine_npz_all(model_name, ref='VG_SV', normalize_abp=False, bias=True, train='all'):

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    query = "SELECT model_path, model_type FROM model WHERE model_name='%s'" % model_name
    cursor.execute(query)
    query_results = cursor.fetchall()
    assert len(query_results) == 1, 'Wrong model: %s' % model_name
    model_path = query_results[0][0]
    model_type = query_results[0][1]

    query = "SELECT file_name, file_path, weight, bias FROM quality_index WHERE model_name='%s' and quality_index > 0 ORDER BY quality_index DESC" % model_name
    cursor.execute(query)
    query_results = cursor.fetchall()
    db.close()
    assert len(query_results), 'Quality index doesn\'t exists.'

    sess = tf.Session()
    model_class = get_models(sess, max_to_keep=10)
    model_class[model_type].restore_model(os.path.join(model_path, 'origin'))

    init_isc = list()
    input_abp = list()
    input_ppg = list()
    input_i = list()
    sv_reference = list()

    for npz in query_results:
        t_data = np.load(os.path.join(npz[1], npz[0]), allow_pickle=True)
        d = cf.load_npz(t_data, ref=ref, to3d=True, trim=True, normalize_abp=normalize_abp)
        if d['timestamp'].shape[0]:
            col_dict = d['col_dict']
            input_abp.append(d['abp'])
            input_i.append(np.full((d['abp'].shape[0], 1), len(input_i), dtype=np.int32))
            init_isc.append([npz[2], npz[3]])
            sv_reference.append(d['feature'][:, col_dict[ref], :] / npz[2])

    assert len(input_abp) == len(input_i) == len(sv_reference), 'Different lengths.'
    tmp_abp = np.concatenate(input_abp)
    tmp_i = np.concatenate(input_i)
    tmp_sv = np.concatenate(sv_reference)
    assert len(tmp_abp) == len(tmp_i) == len(tmp_sv), 'Different lengths.'
    sess.run(tf.global_variables_initializer())
    if model_type in [0, 1, 2, 3]:
        model_class[model_type].restore_model(os.path.join(model_path, 'origin'))
        for epoch_phase in range(5):
            model_class[model_type].train(tmp_abp, tmp_sv, 50)
            model_class[model_type].save_model(os.path.join(model_path, 'ma_%03d' % (epoch_phase*50+50)))
    elif model_type in [4, 5, 6]:
        tmp_ppg = np.concatenate(input_ppg)
        model_class[model_type].restore_model(os.path.join(model_path, 'origin'))
        for epoch_phase in range(10):
            model_class[model_type].train(tmp_abp, tmp_ppg, tmp_sv, 50)
            model_class[model_type].save_model(os.path.join(model_path, 'ma_%03d' % (epoch_phase*50+50)))
    else:
        assert False

    return


def validate_model(model_name, vdns, model_list='inc', ref='VG_SV', normalize_abp=False):

    formatter = DateFormatter('%H:%M')

    model_path, model_type = get_model_info(model_name)

    models = list()

    if model_list == 'inc':
        model_name_re = re.compile('m_[0-9]{2}')
    elif model_list == 'all':
        model_name_re = re.compile('ma_[0-9]{3}')
    elif model_list == 'init':
        models.append('origin')
        models.append('init')
    else:
        assert False, 'Unknown model type %s.' % model_list

    if model_list in ('inc', 'all'):
        for file in os.listdir(model_path):
            split_file = os.path.splitext(file)
            if len(split_file) > 1:
                if split_file[1] == '.index' and model_name_re.match(split_file[0]):
                    models.append(split_file[0])
        models.sort()

    sess = tf.Session()
    model_class = get_models(sess)

    title = list()
    result_r_table = list()

    title.append('dataset')
    title.append('EV1000')
    title.extend(models)

    tmp_npzs = list()
    for npz in vdns:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    vdls = tmp_npzs

    for npz in vdls:
        t_data = np.load(npz, allow_pickle=True)
        d = cf.load_npz(t_data, ref=ref, to3d=True, trim=True, normalize_abp=normalize_abp)
        col_dict = d['col_dict']

        #assert len(d['timestamp']), 'Length 0 validation set. %s' % npz
        if len(d['timestamp']):
            tmp_row = list()
            tmp_row.append(npz)

            sv_reference = d['feature'][:, col_dict[ref], :]
            sv_ev = d['feature'][:, col_dict['EV_SV'], :]
            tmp_row.append(cf.pcr(sv_reference, sv_ev)[0])

            for m in models:
                if model_type in [0, 1, 2, 3]:
                    model_class[model_type].restore_model(os.path.join(model_path, m))
                    sv_unscaled = model_class[model_type].predict(d['abp'])
                elif model_type in [4, 5, 6]:
                    model_class[model_type].restore_model(os.path.join(model_path, m))
                    sv_unscaled = model_class[model_type].predict(d['abp'], d['ppg'])
                else:
                    assert False
                sv_unscaled = cf.smoothing_result(sv_unscaled, d['timestamp'], type='datetime')
                tmp_row.append(cf.pcr(sv_reference, sv_unscaled)[0])

            result_r_table.append(tmp_row)

    with open(os.path.join(validation_dir, model_name + '_' + model_list + '.csv'), 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title)
        for i, row in enumerate(result_r_table):
            cyclewriter.writerow(row)

    return


def eval_model(model_names, npzs, output_file, ref='VG_SV'):

    #formatter = DateFormatter('%H:%M')

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    title = list()
    title.append('dataset')
    title.append('EV1000')

    models = list()
    for model_name in model_names:
        query = "SELECT model_name, model_path, model_type FROM model WHERE model_name='%s'" % model_name
        cursor.execute(query)
        query_results = cursor.fetchall()
        assert len(query_results) == 1, 'Wrong model: %s' % model_name
        models.append(query_results[0])
        title.append(model_name)

    db.close()

    sess = tf.Session()
    model_class = get_models(sess)

    # Initializes global variables in the graph.
    tmp_npzs = list()
    for npz in npzs:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    npzs = tmp_npzs

    result_r_table = list()

    for npz in npzs:
        t_data = np.load(npz, allow_pickle=True)
        d = cf.load_npz(t_data, ref=ref, to3d=True, trim=True)
        col_dict = d['col_dict']

        if len(d['timestamp']):
            tmp_row = list()
            tmp_row.append(os.path.basename(npz))
            sv_reference = d['feature'][:, col_dict[ref], :]
            tmp_row.append(pearsonr(sv_reference, d['feature'][:, col_dict['EV_SV'], :])[0][0])
            for m in models:
                model_class[m[2]].restore_model(os.path.join(m[1], 'rep'))
                if m[2] in [0, 1, 2, 3]:
                    sv_unscaled = model_class[m[2]].predict(d['abp'])
                elif m[2] in [4, 5, 6]:
                    sv_unscaled = model_class[m[2]].predict(d['abp'], d['ppg'])
                else:
                    assert False
                sv_unscaled = cf.smoothing_result(sv_unscaled, d['timestamp'], type='datetime')
                tmp_row.append(pearsonr(sv_reference, sv_unscaled)[0][0])
            result_r_table.append(tmp_row)

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_formatter(formatter)
        ax.plot(d['timestamp'], sv_reference, color='red')
        ax.plot(d['timestamp'], sv_unscaled, color='grey')
#        ax.plot(d['timestamp'], sv_scaled, color='blue')
        plt.savefig(os.path.join(model_path, 'fig', os.path.splitext(os.path.basename(npz))[0]))
        '''

    with open(output_file, 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title)
        for i, row in enumerate(result_r_table):
            cyclewriter.writerow(row)

    return


'''
def evaluation_stat(model_dir, npzs, result_file, ref='VG_SV', num_wave_channel=1):

    models = list()

    model_name_re = re.compile('m_[0-9]{2}')

    for file in os.listdir(model_dir):
        split_file = os.path.splitext(file)
        if len(split_file) > 1:
            if split_file[1] == '.index' and model_name_re.match(split_file[0]):
                models.append(split_file[0])
    models.sort()

    results = list()
    title = list()

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    model_class = list()
    model_class.append(ModelABP(sess, 'm1'))
    model_class.append(Model_2(sess, 'm2'))

    tmp_npzs = list()
    for npz in npzs:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    npzs = tmp_npzs

    title.append('filename')
    title.append('EV1000')
    for m in models:
        title.append(m)

    for npz in npzs:
        t_data = np.load(npz)
        d = cf.load_npz(t_data, ref='VG_SV', to3d=True, trim=True)
        if len(d['timestamp']):
            col_dict = d['col_dict']
            tmp_row = list()
            tmp_row.append(os.path.basename(npz))
            tmp_row.append(pearsonr(d['feature'][:, col_dict[ref]], d['feature'][:, col_dict['EV_SV']])[0][0])
            for model in models:
                sess.run(tf.global_variables_initializer())
                if num_wave_channel == 1:
                    model_class[0].restore_model(os.path.join(model_dir, model))
                    pred = cf.smoothing_result(model_class[0].predict(d['abp']), d['timestamp'], type='datetime')
                elif num_wave_channel == 2:
                    model_class[1].restore_model(os.path.join(model_dir, model))
                    pred = cf.smoothing_result(model_class[1].predict(d['abp'], d['ppg']), d['timestamp'], type='datetime')
                else:
                    assert False
                error_pcr = pearsonr(d['feature'][:, col_dict[ref]], pred)[0][0]
                tmp_row.append(error_pcr if not np.isnan(error_pcr) else 0)
            results.append(tmp_row)

    with open(result_file, 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title)
        for r in results:
            cyclewriter.writerow(r)

    return
'''

