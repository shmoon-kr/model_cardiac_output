import numpy as np
from keras import backend as K
from keras.layers import Layer


class Derivative1D(Layer):

    def __init__(self, padding='valid', data_format='channels_last', **kwargs):
        self.padding = padding
        self.data_format = K.normalize_data_format(data_format)
        super(Derivative1D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3, 'Wrong input shape. Is it 1D?'
        assert self.padding in ('valid', 'left', 'right'), 'Wrong padding %s.' % self.data_format
        assert self.data_format in ('channels_first', 'channels_last'), 'Wrong data_format %s.' % self.data_format

        input_len = input_shape[1] if self.data_format == 'channels_last' else input_shape[2]
        matrix_shape = [input_len, input_len-1] if self.padding == 'valid' else [input_len, input_len]

        drv_mat = np.zeros(matrix_shape, dtype=np.int32)

        for i in range(input_len - 1):
            if self.padding in ('valid', 'right'):
                drv_mat[i, i] = -1
                drv_mat[i+1, i] = 1
            else:
                drv_mat[i, i+1] = -1
                drv_mat[i+1, i+1] = 1
        if self.padding == 'left':
            drv_mat[0, 0] = -1
            drv_mat[1, 0] = 1
        elif self.padding == 'right':
            i = input_len - 1
            drv_mat[i-1, i] = -1
            drv_mat[i, i] = 1

        # Create a trainable weight variable for this layer.
        self.derivative_matrix = K.constant(drv_mat)

        super(Derivative1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        reshaped_input = K.permute_dimensions(x, (0, 2, 1)) if self.data_format == 'channels_last' else x
        y = K.dot(reshaped_input, self.derivative_matrix)
        if self.data_format == 'channels_last':
            y = K.permute_dimensions(y, (0, 2, 1))

        return y

    def compute_output_shape(self, input_shape):
        if self.padding in ('left', 'right'):
            return input_shape
        elif self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], input_shape[2]-1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1]-1, input_shape[2])
        else:
            assert False
