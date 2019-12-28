# python3 ./ModelsJM.py init -m sv_model_p1 -t VG_SV -b JCM -g 0 -d trn0 trn1
# python3 ./ModelsJM.py dq -m sv_model_p0 -g 1 -v vdns -d trn0 trn1
# python3 ./ModelsJM.py com -m sv_model_p0 -g 1 -v vdns
# python3 ./ModelsJM.py step -m sv_model_p0 -g 1 -v vdns
# python3 ./ModelsJM.py -t EV_SVV -m JCM -d trn0 trn1 trn2 trn3 -g 1 -f svv_jcm -r

import os
import csv
import shutil
import datetime
import argparse
import MySQLdb
import CommonFunction as cf
import tensorflow as tf
import numpy as np
from math import ceil, sqrt
from scipy.stats import pearsonr, trim_mean, gmean
from tensorflow.python.keras import optimizers, backend as K
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Input, Model
from tensorflow.python.keras.layers import Conv1D, MaxPool1D, AveragePooling1D, Dense, Dropout, Reshape, Concatenate, Flatten, Lambda
from derivative import Derivative1D


model_dir = '/home/shmoon/model/production'
max_individual = 100

parser = argparse.ArgumentParser(description='Build a production model.')
parser.add_argument('command')
parser.add_argument('-d', nargs='+', dest='d', help='Specify dataset.')
parser.add_argument('-m', dest='model', help='Specify a model.')
parser.add_argument('-b', dest='base', help='Specify a base model.')
parser.add_argument('-g', dest='gpu', help='Specify GPUs.')
parser.add_argument('-t', dest='target', help='Target variable.')
parser.add_argument('-i', dest='isc', action='store_true', help='Use individual scale coefficient.')
parser.add_argument('-f', dest='file', help='New model file.')
parser.add_argument('-r', dest='resume', help='Model file to continue.')
parser.add_argument('-v', dest='vdn', help='Validation data set.')


# cnn_01
trn0 = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 2, 1), end_date=datetime.date(2018, 2, 28))
trn1 = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 3, 1), end_date=datetime.date(2018, 3, 31))
trn2 = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 4, 1), end_date=datetime.date(2018, 4, 30))
trn3 = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 5, 1), end_date=datetime.date(2018, 5, 31))
vdns = cf.enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 6, 1), end_date=datetime.date(2018, 6, 30))


def ModelJM():
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
    model.add(Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))

    return model


def ModelJCM(f=(8, 16, 32, 64), k=(12, 6, 4, 3), d=None, ind=max_individual):

    ABP = Input(shape=(1024, 1))
    DRV0 = Derivative1D(padding='same')(ABP)
    F0 = Concatenate(axis=2)([ABP, DRV0])

    # L1 SigIn shape = (?, 1024, 1)
    L1 = Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C1')(F0)
    M1 = MaxPool1D(pool_size=2, strides=2, padding='same')(L1)
    AVG1 = AveragePooling1D(pool_size=2, strides=2, padding='same')(ABP)
    DRV1 = Derivative1D(padding='same')(AVG1)
    MAX1 = MaxPool1D(pool_size=2, strides=2, padding='same')(ABP)
    MIN1 = Lambda(lambda x: -x)(MaxPool1D(pool_size=2, strides=2, padding='same')(Lambda(lambda x: -x)(ABP)))
    F1 = Concatenate(axis=2)([M1, MAX1, AVG1, DRV1, MIN1])
    # Conv -> (?, 512, 8)

    L2 = Conv1D(filters=f[0], kernel_size=k[0], strides=1, padding='same', activation='relu', name='C2')(F1)
    M2 = MaxPool1D(pool_size=2, strides=2, padding='same')(L2)
    AVG2 = AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG1)
    DRV2 = Derivative1D(padding='same')(AVG2)
    MAX2 = MaxPool1D(pool_size=2, strides=2, padding='same')(MAX1)
    MIN2 = Lambda(lambda x: -x)(MaxPool1D(pool_size=2, strides=2, padding='same')(Lambda(lambda x: -x)(MIN1)))
    F2 = Concatenate(axis=2)([M2, MAX2, AVG2, DRV2, MIN2])
    # Conv -> (?, 256, 8)

    L3 = Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C3')(F2)
    M3 = MaxPool1D(pool_size=2, strides=2, padding='same')(L3)
    AVG3 = AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG2)
    DRV3 = Derivative1D(padding='same')(AVG3)
    MAX3 = MaxPool1D(pool_size=2, strides=2, padding='same')(MAX2)
    MIN3 = Lambda(lambda x: -x)(MaxPool1D(pool_size=2, strides=2, padding='same')(Lambda(lambda x: -x)(MIN2)))
    F3 = Concatenate(axis=2)([M3, MAX3, AVG3, DRV3, MIN3])
    # Conv -> (?, 128, 16)

    L4 = Conv1D(filters=f[1], kernel_size=k[1], strides=1, padding='same', activation='relu', name='C4')(F3)
    M4 = MaxPool1D(pool_size=2, strides=2, padding='same')(L4)
    AVG4 = AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG3)
    DRV4 = Derivative1D(padding='same')(AVG4)
    MAX4 = MaxPool1D(pool_size=2, strides=2, padding='same')(MAX3)
    MIN4 = Lambda(lambda x: -x)(MaxPool1D(pool_size=2, strides=2, padding='same')(Lambda(lambda x: -x)(MIN3)))
    F4 = Concatenate(axis=2)([M4, MAX4, AVG4, DRV4, MIN4])
    # Conv -> (?, 64, 16)

    L5 = Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C5')(F4)
    M5 = MaxPool1D(pool_size=2, strides=2, padding='same')(L5)
    AVG5 = AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG4)
    DRV5 = Derivative1D(padding='same')(AVG5)
    MAX5 = MaxPool1D(pool_size=2, strides=2, padding='same')(MAX4)
    MIN5 = Lambda(lambda x: -x)(MaxPool1D(pool_size=2, strides=2, padding='same')(Lambda(lambda x: -x)(MIN4)))
    F5 = Concatenate(axis=2)([M5, MAX5, AVG5, DRV5, MIN5])

    # Conv -> (?, 32, 32)
    L6 = Conv1D(filters=f[2], kernel_size=k[2], strides=1, padding='same', activation='relu', name='C6')(F5)
    M6 = MaxPool1D(pool_size=2, strides=2, padding='same')(L6)
    AVG6 = AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG5)
    DRV6 = Derivative1D(padding='same')(AVG6)
    MAX6 = MaxPool1D(pool_size=2, strides=2, padding='same')(MAX5)
    MIN6 = Lambda(lambda x: -x)(MaxPool1D(pool_size=2, strides=2, padding='same')(Lambda(lambda x: -x)(MIN5)))
    F6 = Concatenate(axis=2)([M6, MAX6, AVG6, DRV6, MIN6])
    # Conv -> (?, 16, 32)

    L7 = Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C7')(F6)
    M7 = MaxPool1D(pool_size=2, strides=2, padding='same')(L7)
    AVG7 = AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG6)
    DRV7 = Derivative1D(padding='same')(AVG7)
    MAX7 = MaxPool1D(pool_size=2, strides=2, padding='same')(MAX6)
    MIN7 = Lambda(lambda x: -x)(MaxPool1D(pool_size=2, strides=2, padding='same')(Lambda(lambda x: -x)(MIN6)))
    F7 = Concatenate(axis=2)([M7, MAX7, AVG7, DRV7, MIN7])
    # Conv -> (?, 8, 64)

    L8 = Conv1D(filters=f[3], kernel_size=k[3], strides=1, padding='same', activation='relu', name='C8')(F7)
    M8 = MaxPool1D(pool_size=2, strides=2, padding='same')(L8)
    AVG8 = AveragePooling1D(pool_size=2, strides=2, padding='same')(AVG7)
    MAX8 = MaxPool1D(pool_size=2, strides=2, padding='same')(MAX7)
    MIN8 = Lambda(lambda x: -x)(MaxPool1D(pool_size=2, strides=2, padding='same')(Lambda(lambda x: -x)(MIN7)))
    F8 = Concatenate(axis=2)([M8, MAX8, AVG8, MIN8])
    # Conv -> (?, 4, 64)
    flat = Flatten()(F8)

    if d is not None:
        flat = Dense(d, activation='relu', kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
        flat = Dense(d, activation='relu', kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)

    OUTPUT_UNSCALED = Dense(1, activation='relu', kernel_initializer=tf.initializers.glorot_uniform())(flat)

    IND = Input(shape=(ind, ), dtype='float32')
    WI = Dense(1, activation='relu', kernel_initializer='ones', use_bias=False)
    OUTPUT_SCALED = Lambda(lambda x: x[0] * x[1])((WI(IND), OUTPUT_UNSCALED))

    return Model(inputs=ABP, outputs=OUTPUT_UNSCALED), Model(inputs=(ABP, IND), outputs=OUTPUT_SCALED), WI


def ModelInitISC(ind=max_individual):
    UNSCALED = Input(shape=(1, ), dtype='float32')
    IND = Input(shape=(ind, ), dtype='float32')
    WI = Dense(1, activation='relu', kernel_initializer='ones', use_bias=False)
    SCALED = Lambda(lambda x: x[0] * x[1])((WI(IND), UNSCALED))
    return Model(inputs=(UNSCALED, IND), outputs=SCALED), WI


def GetInitISC(sv_unscaled, ind, sv_scaled):
    model_init_isc, model_init_isc_w = ModelInitISC()
    adam = optimizers.Adam(lr=1e-2, clipnorm=1.)
    model_init_isc.compile(adam, loss='mean_squared_error', metrics=['mse'])
    model_init_isc.summary()

    sess = K.get_session()
    sess.run(tf.global_variables_initializer())

    last_mse = None
    while True:
        mse = model_init_isc.train_on_batch((sv_unscaled, ind), sv_scaled)[0]
        if last_mse is None:
            last_mse = mse
        elif mse >= last_mse:
            break
        else:
            last_mse = mse

    return model_init_isc_w.get_weights()


def get_model_info(name):
    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    query = "SELECT id, name, base, ref, param FROM model_k WHERE name ='%s'" % name
    cursor.execute(query)
    db.close()
    model_info = cursor.fetchall()
    if not len(model_info):
        return None
    assert len(model_info) == 1, 'Multiple models %s exist.' % name
    return {'id': model_info[0][0], 'name': model_info[0][1], 'base': model_info[0][2], 'ref': model_info[0][3],
            'param': model_info[0][4]}


def train_model_jm(training_data):
    ref = 'VG_SV'
    input_abp = list()
    sv_reference = list()

    model = ModelJM()
    model.build()
    adam = optimizers.Adam(lr=0.000002, clipnorm=1.)
    model.compile(adam, loss='mean_squared_error', metrics=['mse'])
    model.summary()

    for npz in training_data:
        t_data = np.load(npz, allow_pickle=True)
        d = cf.load_npz(t_data, ref=ref, to3d=True, trim=True, normalize_abp=True, input_len=300)
        if d['timestamp'].shape[0]:
            col_dict = d['col_dict']
            input_abp.append(d['abp'])
            sv_reference.append(d['feature'][:, col_dict[ref], :])

    tmp_abp = np.concatenate(input_abp)
    tmp_sv = np.concatenate(sv_reference)
    assert len(tmp_abp) == len(tmp_sv), 'Different lengths.'

    model.fit(tmp_abp, tmp_sv, epochs=1000, batch_size=1024)
    model.save('/home/shmoon/model_production/sv_model_k/m_0.h5')

    return


def build_init_model(model_name, base, init_data, ref):
    if base == 'JCM':
        model_unscaled, model_scaled, model_isc = ModelJCM(ind=100)
        adam = optimizers.Adam(lr=1e-4, clipnorm=1.)
        model_unscaled.compile(adam, loss='mean_squared_error', metrics=['mse'])
    else:
        assert False

    sess = K.get_session()

    for d_init in init_data:
        train_data = np.load(d_init, allow_pickle=True)
        d = cf.load_npz(train_data, ref=ref, to3d=True, trim=True, normalize_abp=True)
        col_dict = d['col_dict']

        if len(d['timestamp']):
            sess.run(tf.global_variables_initializer())
            model_unscaled.fit(d['abp'], d['feature'][:, col_dict[ref]], epochs=3, batch_size=1024)
            if model_unscaled.history.history['loss'][0] > model_unscaled.history.history['loss'][-1]:
                model_path = os.path.join(model_dir, model_name)
                if os.path.isdir(model_path) and not os.path.islink(model_path):
                    shutil.rmtree(model_path)
                elif os.path.exists(model_path):
                    os.remove(model_path)
                os.mkdir(model_path)
                model_unscaled.save_weights(os.path.join(model_dir, model_name, 'init.h5'))

                db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
                cursor = db.cursor()

                query = "DELETE quality_index_k FROM model_k INNER JOIN quality_index_k "
                query += "ON model_k.id = quality_index_k.model_id WHERE model_k.name='%s'" % model_name
                cursor.execute(query)

                query = "DELETE FROM model_k WHERE name='%s'" % model_name
                cursor.execute(query)

                query = "INSERT INTO model_k (name, base, ref, param) VALUES ('%s','%s','%s',NULL)" % (
                model_name, base, ref)
                cursor.execute(query)

                db.commit()
                db.close()

                return True

    return False


def evaluate_data_quality(model_name, data, val_data, epochs=500):
    title = list()
    result_r_table = list()

    model_info = get_model_info(model_name)

    if model_info['base'] == 'JCM':
        model_jcm_unscaled, model_jcm_scaled, model_jcm_isc = ModelJCM(ind=100)
        adam_jcm = optimizers.Adam(lr=1e-4, clipnorm=1.)
        model_jcm_unscaled.compile(adam_jcm, loss='mean_squared_error', metrics=['mse'])
        model = model_jcm_unscaled
    else:
        assert False

    title.append('dataset')
    tmp_npzs = list()
    for npz in data:
        d = np.load(npz)
        if 'ppg' in d:
            tmp_npzs.append(npz)
    npzs = tmp_npzs
    max_pcr = 0

    sess = K.get_session()

    for train_data_npz in data:
        # Initializes global variables in the graph.
        train_data = np.load(train_data_npz, allow_pickle=True)
        d = cf.load_npz(train_data, ref=model_info['ref'], to3d=True, trim=True, normalize_abp=True)
        col_dict = d['col_dict']

        if len(d['timestamp']):
            result_row = list()
            result_row.append(train_data_npz)

            sess.run(tf.global_variables_initializer())
            model.load_weights(os.path.join(model_dir, model_name, 'init.h5'))
            model.fit(d['abp'], d['feature'][:, col_dict[model_info['ref']]], epochs=epochs, batch_size=1024)

            # Start populating the filename queue.
            for test_data_npz in val_data:
                test_data = np.load(test_data_npz, allow_pickle=True)
                d_test = cf.load_npz(test_data, ref=model_info['ref'], to3d=True, trim=True, normalize_abp=True)
                if len(d_test['timestamp']):
                    if os.path.basename(test_data_npz) not in title:
                        title.append(os.path.basename(test_data_npz))
                    predicted = model.predict(d_test['abp'])
                    predicted = cf.smoothing_result(predicted, d_test['timestamp'], type='datetime')
                    error_pcr = pearsonr(d_test['feature'][:, col_dict[model_info['ref']]], predicted)[0][0]
                    result_row.append(error_pcr if not np.isnan(error_pcr) else 0)
            if len(result_row) > 1:
                result_row.append(sum(result_row[1:]) / len(result_row[1:]))
            else:
                result_row.append(result_row[0])
            if result_row[-1] > max_pcr:
                max_pcr = result_row[-1]
                print('%s, better model, file=%s, pcr=%.3f' % (
                datetime.datetime.now().isoformat(), train_data_npz, max_pcr))
                model.save(os.path.join(model_dir, model_name, 'origin.h5'))

            result_r_table.append(result_row)

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    query = "DELETE FROM quality_index_k WHERE model_id = %d" % model_info['id']
    cursor.execute(query)
    db.commit()

    with open(os.path.join(model_dir, model_name, 'quality.csv'), 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title)
        for i, row in enumerate(result_r_table):
            query = 'INSERT INTO quality_index_k (model_id, file_name, file_path, quality_index) VALUES '
            query += '(\'%d\', \'%s\', \'%s\', %f)' % (
                model_info['id'], os.path.basename(row[0]), os.path.dirname(row[0]), row[-1])
            cursor.execute(query)
            row[0] = os.path.basename(row[0])
            cyclewriter.writerow(row)
    db.commit()

    return


def build_combine_model(model_name, val_data, epochs=100, validation_epochs=5):

    model_info = get_model_info(model_name)

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    query = "SELECT file_path, file_name, quality_index FROM quality_index_k WHERE model_id =%d ORDER BY quality_index DESC" % model_info['id']
    cursor.execute(query)

    training_data = cursor.fetchall()
    #training_data = training_data[:3]
    #val_data = val_data[:3]

    title = list()
    title.append('dataset')

    model_jcm_unscaled, model_jcm_scaled, model_jcm_isc = ModelJCM(ind=100)
    adam_jcm = optimizers.Adam(lr=1e-4, clipnorm=1.)
    model_jcm_unscaled.compile(adam_jcm, loss='mean_squared_error', metrics=['mse'])
    model_jcm_scaled.compile(adam_jcm, loss='mean_squared_error', metrics=['mse'])

    sess = K.get_session()
    sess.run(tf.global_variables_initializer())
    model_jcm_unscaled.load_weights(os.path.join(model_dir, model_info['name'], 'origin.h5'))

    training_abp = list()
    training_ind = list()
    training_sv = list()
    training_sv_scaled = list()

    for i, npz_training_data in enumerate(training_data):
        tmp_train_data = np.load(os.path.join(npz_training_data[0], npz_training_data[1]), allow_pickle=True)
        d = cf.load_npz(tmp_train_data, ref=model_info['ref'], to3d=True, trim=True, normalize_abp=True)
        col_dict = d['col_dict']
        training_abp.append(d['abp'])
        training_sv.append(cf.smoothing_result(model_jcm_unscaled.predict(d['abp'], batch_size=1024), d['timestamp'], type='datetime'))
        training_ind.append([i]*len(d['timestamp']))
        training_sv_scaled.append(d['feature'][:, col_dict[model_info['ref']]])

    total_training_sv = np.concatenate(training_sv)
    total_training_ind = to_categorical(np.concatenate(training_ind), num_classes=max_individual)
    total_training_sv_scaled = np.concatenate(training_sv_scaled)

    isc_weights = GetInitISC(total_training_sv, total_training_ind, total_training_sv_scaled)

    del total_training_sv, total_training_ind, total_training_sv_scaled

    global_optimum_pcr = 0
    validation_data_x = list()
    validation_data_ref = list()
    validation_data_ts = list()
    tmp_val_list = list()

    for j, val_data_single in enumerate(val_data):
        tmp_val_data = np.load(val_data_single, allow_pickle=True)
        d_val = cf.load_npz(tmp_val_data, ref=model_info['ref'], to3d=True, trim=True, normalize_abp=True)
        col_dict = d_val['col_dict']
        if len(d_val['timestamp']):
            tmp_val_list.append(val_data_single)
            title.append(os.path.basename(val_data_single))
            validation_data_x.append(d_val['abp'])
            validation_data_ref.append(d_val['feature'][:, col_dict[model_info['ref']]])
            validation_data_ts.append(d_val['timestamp'])
    title.append('average')

    val_data = tmp_val_list
    result_r_table = np.empty((len(training_data), len(val_data)))

    for i in range(len(training_data)):
        # Initializes global variables in the graph.
        tmp_training_abp = np.concatenate(training_abp[:i+1])
        tmp_training_ind = to_categorical(np.concatenate(training_ind[:i+1]), num_classes=max_individual)
        tmp_training_sv_scaled = np.concatenate(training_sv_scaled[:i+1])

        sess.run(tf.global_variables_initializer())
        model_jcm_unscaled.load_weights(os.path.join(model_dir, model_info['name'], 'origin.h5'))
        model_jcm_isc.set_weights(isc_weights)

        total_elapsed_epochs = 0
        local_optimum_pcr = None

        while total_elapsed_epochs < epochs:
            model_jcm_scaled.fit((tmp_training_abp, tmp_training_ind), tmp_training_sv_scaled, epochs=validation_epochs,
                                 batch_size=1024)
            total_elapsed_epochs += validation_epochs
            tmp_local_pcr = list()
            for j, val_data_single in enumerate(val_data):
                predicted = model_jcm_unscaled.predict(validation_data_x[j], batch_size=1024)
                predicted = cf.smoothing_result(predicted, validation_data_ts[j], type='datetime')
                tmp_local_pcr.append(pearsonr(validation_data_ref[j], predicted)[0][0])
            if np.mean(tmp_local_pcr) > np.mean(local_optimum_pcr) if local_optimum_pcr is not None else True:
                local_optimum_pcr = tmp_local_pcr

        result_r_table[i] = local_optimum_pcr
        if np.mean(local_optimum_pcr) > global_optimum_pcr:
            global_optimum_pcr = np.mean(local_optimum_pcr)
            model_jcm_unscaled.save(os.path.join(model_dir, model_info['name'], 'optimum.h5'))

    with open(os.path.join(model_dir, model_info['name'], 'combine.csv'), 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title)
        for i in range(len(training_data)):
            tmp_row = ['m_%03d' % (i+1)]
            tmp_row.extend(result_r_table[i])
            tmp_row.append(np.mean(result_r_table[i]))
            cyclewriter.writerow(tmp_row)

    return


def build_isc_step(model_name, val_data, total_steps=100, step_size=10, num_data=None):

    model_info = get_model_info(model_name)

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()

    query = "SELECT file_path, file_name, quality_index FROM quality_index_k WHERE model_id ='%d' ORDER BY quality_index DESC" % model_info['id']
    if num_data is not None:
        query += ' LIMIT %d' % num_data
    cursor.execute(query)

    training_data = cursor.fetchall()

    title_pcr = list()
    title_pcr.append('dataset')
    title_isc = list()
    title_isc.append('dataset')

    model_unscaled, model_scaled, model_isc = ModelJCM(ind=100)
    adam = optimizers.Adam(lr=1e-5, clipnorm=1.)
    model_unscaled.compile(adam, loss='mean_squared_error', metrics=['mse'])
    model_scaled.compile(adam, loss='mean_squared_error', metrics=['mse'])

    sess = K.get_session()
    sess.run(tf.global_variables_initializer())
    model_unscaled.load_weights(os.path.join(model_dir, model_name, 'origin.h5'))

    training_abp = list()
    training_ind = list()
    training_sv = list()
    training_sv_scaled = list()

    for i, npz_training_data in enumerate(training_data):
        tmp_train_data = np.load(os.path.join(npz_training_data[0], npz_training_data[1]), allow_pickle=True)
        d = cf.load_npz(tmp_train_data, ref=model_info['ref'], to3d=True, trim=True, normalize_abp=True)
        col_dict = d['col_dict']
        training_abp.append(d['abp'])
        training_sv.append(cf.smoothing_result(model_unscaled.predict(d['abp'], batch_size=1024), d['timestamp'], type='datetime'))
        training_ind.append([i]*len(d['timestamp']))
        training_sv_scaled.append(d['feature'][:, col_dict[model_info['ref']]])
        title_isc.append(npz_training_data[1])

    training_abp = np.concatenate(training_abp)
    training_sv = np.concatenate(training_sv)
    training_ind = to_categorical(np.concatenate(training_ind), num_classes=max_individual)
    training_sv_scaled = np.concatenate(training_sv_scaled)

    init_isc = GetInitISC(training_sv, training_ind, training_sv_scaled)
    model_isc.set_weights(init_isc)

    isc_history = np.zeros((total_steps, len(training_data)))
    pcr_history = np.zeros((total_steps, len(val_data)))

    val_abp = list()
    val_sv_scaled = list()
    val_ts = list()

    for i, npz in enumerate(val_data):
        title_pcr.append(os.path.basename(npz))
        tmp_val_data = np.load(npz, allow_pickle=True)
        d = cf.load_npz(tmp_val_data, ref=model_info['ref'], to3d=True, trim=True, normalize_abp=True)
        col_dict = d['col_dict']
        val_abp.append(d['abp'])
        val_sv_scaled.append(d['feature'][:, col_dict[model_info['ref']]])
        val_ts.append(d['timestamp'])

    row_title = list()
    for i in range(total_steps):
        row_title.append('step_%04d' % (i*step_size+step_size))
        model_scaled.fit((training_abp, training_ind), training_sv_scaled, epochs=step_size, batch_size=1024)
        tmp_isc = model_isc.get_weights()
        for j, npz in enumerate(val_data):
            t = cf.smoothing_result(model_unscaled.predict(val_abp[j], batch_size=1024), val_ts[j], type='datetime')
            pcr_history[i, j] = pearsonr(val_sv_scaled[j], t)[0][0]
        isc_history[i] = tmp_isc[0][:len(training_data), 0]

    with open(os.path.join(model_dir, model_name, 'history_pcr.csv'), 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title_pcr + ['average'])
        for i in range(len(row_title)):
            tmp_row = [row_title[i]]
            tmp_row.extend(pcr_history[i])
            tmp_row.append(np.mean(pcr_history[i]))
            cyclewriter.writerow(tmp_row)

    with open(os.path.join(model_dir, model_name, 'history_isc.csv'), 'w', newline='') as csvfile:
        cyclewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cyclewriter.writerow(title_isc + ['average'])
        for i in range(len(row_title)):
            tmp_row = [row_title[i]]
            tmp_row.extend(isc_history[i, :])
            tmp_row.append(gmean(isc_history[i, :]))
            cyclewriter.writerow(tmp_row)

    return False


def build_single_model(model_name, new_model_file, data, saved_file=None):
    model_info = get_model_info(model_name)

    if model_info['base'] == 'JCM':
        model_unscaled, model_scaled, model_isc = ModelJCM(ind=100)
        adam = optimizers.Adam(lr=1e-4, clipnorm=1.)
        model_unscaled.compile(adam, loss='mean_squared_error', metrics=['mse'])
    else:
        assert False

    x = list()
    y = list()

    for npz in data:
        t_data = np.load(npz, allow_pickle=True)
        d = cf.load_npz(t_data, ref=model_info['ref'], to3d=True, trim=True, normalize_abp=True, input_len=1024)
        if d['timestamp'].shape[0]:
            col_dict = d['col_dict']
            x.append(d['abp'])
            y.append(d['feature'][:, col_dict[args.target], :])

    x = np.concatenate(x)
    y = np.concatenate(y)
    assert len(x) == len(y), 'Different lengths.'

    if saved_file is None:
        saved_file = 'init'
    model_unscaled.load_weights(os.path.join(model_dir, model_info['name'], '%s.h5' % saved_file))
    model_unscaled.fit(x, y, epochs=1000, batch_size=1024)
    model_unscaled.save(os.path.join(model_dir, model_info['name'], '%s.h5' % new_model_file))
    return


if __name__ == "__main__":
    # execute only if run as a script
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data = list()
    if args.d is not None:
        for d in args.d:
            if d == 'trn0':
                data.extend(trn0)
            elif d == 'trn1':
                data.extend(trn1)
            elif d == 'trn2':
                data.extend(trn2)
            elif d == 'trn3':
                data.extend(trn3)
            elif d == 'vdns':
                data.extend(vdns)

    val_data = list()
    if args.vdn is not None:
        if args.vdn == 'trn0':
            val_data = trn0
        elif args.vdn == 'trn1':
            val_data = trn1
        elif args.vdn == 'trn2':
            val_data = trn2
        elif args.vdn == 'trn3':
            val_data = trn3
        elif args.vdn == 'vdns':
            val_data = vdns

    if args.command == 'init':
        assert args.target in ('VG_SV', 'EV_SVV'), 'Wrong target %s' % args.target
        assert args.base in ('JCM', 'JM'), 'Wrong base model %s' % args.model
        build_init_model(args.model, args.base, data, args.target)
    elif args.command == 'dq':
        assert args.model is not None, 'No model name was specified.' % args.model
        evaluate_data_quality(args.model, data, val_data)
    elif args.command == 'com':
        assert args.model is not None, 'No model name was specified.' % args.model
        build_combine_model(args.model, val_data)
    elif args.command == 'step':
        assert args.model is not None, 'No model name was specified.' % args.model
        build_isc_step(args.model, val_data, num_data=53)
    elif args.command == 'sng':
        assert args.model is not None, 'No model name was specified.'
        assert args.file is not None, 'No file name was specified.'
        build_single_model(args.model, args.file, data, args.resume)
    else:
        assert False, 'Unknown command %s.' % args.action
