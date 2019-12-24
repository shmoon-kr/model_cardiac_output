import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import permute_dimensions, dot


class Derivative1D(tf.keras.layers.Layer):

    def __init__(self, padding='valid', data_format='channels_last', **kwargs):
        self.padding = padding
        self.derivative_matrix = None
        self.data_format = data_format
        super(Derivative1D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3, 'Wrong input shape. Is it an 1D vector?'
        assert self.padding in ('valid', 'same'), 'Wrong padding %s.' % self.data_format
        assert self.data_format in ('channels_first', 'channels_last'), 'Wrong data_format %s.' % self.data_format

        input_len = input_shape[1] if self.data_format == 'channels_last' else input_shape[2]
        matrix_shape = [input_len, input_len-1] if self.padding == 'valid' else [input_len, input_len]

        drv_mat = np.zeros(matrix_shape, dtype=np.float32)

        for i in range(input_len - 1):
            drv_mat[i, i] = -1
            drv_mat[i+1, i] = 1
        if self.padding == 'same':
            i = input_len - 1
            drv_mat[i-1, i] = -1
            drv_mat[i, i] = 1

        # Create a trainable weight variable for this layer.
        self.derivative_matrix = tf.constant(drv_mat)

        super(Derivative1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        reshaped_input = permute_dimensions(x, (0, 2, 1)) if self.data_format == 'channels_last' else x
        y = dot(reshaped_input, self.derivative_matrix)
        if self.data_format == 'channels_last':
            y = permute_dimensions(y, (0, 2, 1))

        return y

    def compute_output_shape(self, input_shape):
        if self.padding == 'same':
            return input_shape
        elif self.data_format == 'channels_first':
            return input_shape[0], input_shape[1], input_shape[2] - 1
        elif self.data_format == 'channels_last':
            return input_shape[0], input_shape[1] - 1, input_shape[2]
        else:
            assert False
