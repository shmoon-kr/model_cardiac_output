import keras
import os
import datetime
import CommonFunction as cf

from keras.models import Sequential, Input, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Dropout, Reshape, SpatialDropout1D, Activation, Lambda
import tensorflow as tf
import numpy as np
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# cnn_01
trn0 = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 2, 1), end_date=datetime.date(2018, 2, 28))
trn1 = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 3, 1), end_date=datetime.date(2018, 3, 31))
trn2 = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 4, 1), end_date=datetime.date(2018, 4, 30))
trn3 = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 5, 1), end_date=datetime.date(2018, 5, 31))
vdns = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 6, 1), end_date=datetime.date(2018, 6, 30))


def semi_vggnet1000_stride_1c():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu', input_shape=(300, 1)))
    model.add(Conv1D(64, 12, strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12, strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8, strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model


def train_model(training_data):
    ref = 'VG_SV'
    input_abp = list()
    sv_reference = list()

    for npz in training_data:
        t_data = np.load(npz, allow_pickle=True)
        d = cf.load_npz(t_data, ref=ref, to3d=True, trim=True, normalize_abp=True, input_len=300)
        if d['timestamp'].shape[0]:
            col_dict = d['col_dict']
            input_abp.append(d['abp'])
            sv_reference.append(d['feature'][:, col_dict[ref], :] / npz[2])
        print(npz)
    return


train_model(trn0)
