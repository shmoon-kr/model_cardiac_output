from scipy import stats
from scipy import signal
from matplotlib.dates import DateFormatter
import re
import math
import cmath
import os.path
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import MySQLdb


vital_dir = '/mnt/Data/CloudStation'
prep_dir = '/home/shmoon/Preprocessed/all'
npz_dir = '/home/shmoon/Preprocessed/npz'

svv_model = '/home/shmoon/model/svv_01/origin.ckpt'

prep_params = dict()
prep_params['abp_device'] = 'Bx50'
prep_params['abp_channel'] = 'IBP1'
prep_params['ppg_device'] = 'Bx50'
prep_params['ppg_channel'] = 'PLETH'
prep_params['window_size'] = 10.24
prep_params['vigilance_delay'] = 600
prep_params['sampling_rate'] = 100
prep_params['agg_room'] = ['D-02', 'D-06']
prep_params['sng_room'] = ['D-01', 'D-03', 'D-04', 'D-05']
prep_params['interval'] = {
    'Bx50': 6.0, 'Vigilance': 5.0, 'CardioQ': 6.0, 'EV1000': 5.0
}
prep_params['channels'] = (
    ('Bx50', 'HR'),
    ('Bx50', 'ART1_SBP'),
    ('Bx50', 'ART1_DBP'),
    ('Bx50', 'ART1_MBP'),
    ('Bx50', 'CVP2'),
    ('Bx50', 'ETCO2'),
    ('Bx50', 'RR_VENT'),
    ('Bx50', 'RR_CO2'),
    ('Bx50', 'MV'),
    ('Bx50', 'TV_INSP'),
    ('Bx50', 'TV_EXP'),
    ('Bx50', 'PPV'),
    ('Bx50', 'BT1'),
    ('Vigilance', 'CO'),
    ('Vigilance', 'CI'),
    ('Vigilance', 'SVR'),
    ('Vigilance', 'SVRI'),
    ('Vigilance', 'SV'),
    ('Vigilance', 'SVI'),
    ('Vigilance', 'EDV'),
    ('Vigilance', 'EDVI'),
    ('Vigilance', 'ESV'),
    ('Vigilance', 'ESVI'),
    ('CardioQ', 'CO'),
    ('CardioQ', 'CI'),
    ('CardioQ', 'SV'),
    ('CardioQ', 'SVI'),
    ('CardioQ', 'MD'),
    ('CardioQ', 'SD'),
    ('CardioQ', 'FTc'),
    ('CardioQ', 'FTp'),
    ('CardioQ', 'MA'),
    ('CardioQ', 'PV'),
    ('EV1000', 'CO'),
    ('EV1000', 'CI'),
    ('EV1000', 'SVV'),
    ('EV1000', 'SVR'),
    ('EV1000', 'SVRI'),
    ('EV1000', 'SV'),
    ('EV1000', 'SVI'),
)
prep_params['db_columns'] = list()
prep_params['effective'] = list()
prep_params['effective'].append([prep_params['abp_device'], prep_params['abp_channel']])
prep_params['effective'].append([prep_params['ppg_device'], prep_params['ppg_channel']])
for c in prep_params['channels']:
    prep_params['effective'].append(list(c))

eval_models = [
    '/home/shmoon/model/cnn_01/m_rep.ckpt',
    '/home/shmoon/model/cnn_02/m_rep.ckpt',
    '/home/shmoon/model/cnn_03/m_rep.ckpt',
    '/home/shmoon/model/cnn_04/m_rep.ckpt',
    '/home/shmoon/model/cnn_05/m_rep.ckpt',
    '/home/shmoon/model/svv_01/origin.ckpt'
]

res_cols = [
    'SV_MODEL_01',
    'SV_MODEL_02',
    'SV_MODEL_03',
    'SV_MODEL_04',
    'SV_MODEL_05',
    'SVV_MODEL_01'
]

svv_test_npz = [
    '/home/shmoon/Preprocessed/npz/D-02_180221.npz',
    '/home/shmoon/Preprocessed/npz/D-02_180226.npz',
    '/home/shmoon/Preprocessed/npz/D-02_180228.npz',
    '/home/shmoon/Preprocessed/npz/D-06_180220.npz',
    '/home/shmoon/Preprocessed/npz/D-06_180222.npz',
    '/home/shmoon/Preprocessed/npz/D-06_180226.npz',
    '/home/shmoon/Preprocessed/npz/D-06_180227.npz'
]

cq_qualified = [
    '/home/shmoon/Preprocessed/npz/D-01_180410_073102.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181128_021858.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181219_081248.npz',
    '/home/shmoon/Preprocessed/npz/D-03_180316_093652.npz',
    '/home/shmoon/Preprocessed/npz/D-01_190118_081220.npz',
    '/home/shmoon/Preprocessed/npz/D-01_190122_080004.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181012_080410.npz',
    '/home/shmoon/Preprocessed/npz/D-03_180713_081416.npz',
    '/home/shmoon/Preprocessed/npz/D-01_190109_081854.npz',
    '/home/shmoon/Preprocessed/npz/D-03_190118_140123.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181204_080148.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181012_080241.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180803_080507.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181011_080749.npz',
    '/home/shmoon/Preprocessed/npz/D-03_180717_080236.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181122_080115.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181121_081339.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181101_131024.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180919_084903.npz',
    '/home/shmoon/Preprocessed/npz/D-03_180720_080129.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181126_130140.npz',
    '/home/shmoon/Preprocessed/npz/D-03_180206_123826.npz',
    '/home/shmoon/Preprocessed/npz/D-03_190116_123044.npz',
    '/home/shmoon/Preprocessed/npz/D-03_190122_121340.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180918_090217.npz',
    '/home/shmoon/Preprocessed/npz/D-01_190131_081241.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181030_080838.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180914_081806.npz',
    '/home/shmoon/Preprocessed/npz/D-03_180706_130213.npz',
    '/home/shmoon/Preprocessed/npz/D-01_190124_080700.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181016_122413.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180913_103757.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181123_080318.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180911_075818.npz',
    '/home/shmoon/Preprocessed/npz/D-03_180320_125828.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181126_081009.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181211_081727.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181026_090844.npz',
    '/home/shmoon/Preprocessed/npz/D-03_190111_141201.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180822_080422.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181128_080800.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181213_081316.npz',
    '/home/shmoon/Preprocessed/npz/D-01_181015_080848.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181207_133838.npz',
    '/home/shmoon/Preprocessed/npz/D-01_190116_081107.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180404_070041.npz',
    '/home/shmoon/Preprocessed/npz/D-01_180904_081956.npz',
    '/home/shmoon/Preprocessed/npz/D-03_181019_131834.npz'
]

model_colors = ['blue', 'darkgreen', 'magenta', 'maroon', 'darkviolet']


def get_col_dict(title):
    col_dict = dict()
    for i in range(title.shape[0]):
        col_dict[title[i]] = i
    return col_dict


def rmse(array1, array2):
    assert len(array1) == len(array2), 'Arrays have different lengths.'
    n = 0
    ss = 0
    for i in range(len(array1)):
        if not np.isnan(array1[i]) and not np.isnan(array2[i]):
            n += 1
            ss += (array1[i]-array2[i]) * (array1[i]-array2[i])
    return math.sqrt(ss / n) if n else 0


def pcr(array1, array2):
    assert len(array1) == len(array2), 'Arrays have different lengths.'
    l1 = list()
    l2 = list()
    for i in range(len(array1)):
        if not np.isnan(array1[i]) and not np.isnan(array2[i]):
            l1.append(array1[i])
            l2.append(array2[i])
    return stats.pearsonr(l1, l2)[0] if len(l1) else 0


def compare_npz(file1, file2):
    npz1 = np.load(file1)
    npz2 = np.load(file2)

    assert npz1['timestamp'].shape == npz2['timestamp'].shape
    assert npz1['feature'].shape == npz2['feature'].shape
    assert npz1['abp'].shape == npz2['abp'].shape
    assert npz1['title'].shape == npz2['title'].shape

    for i in range(npz1['title'].shape[0]):
        assert npz1['title'][i] == npz2['title'][i]

    for i in range(npz1['timestamp'].shape[0]):
        assert npz1['timestamp'][i] == npz2['timestamp'][i]
        for j in range(npz1['feature'].shape[1]):
            if npz1['title'][j] not in ['DT_ABP_START', 'DT_ABP_END', 'T_Cycle_Begin', 'T_P_max', 'T_dias',
                                        'T_Cycle_End', 'Area_Total', 'Area_Systolic']:
                assert npz1['feature'][i][j] == npz2['feature'][i][j], '%d, %s' % (i, npz1['title'][j])


def enumerate_vital(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 2, 1), end_date=datetime.date.today(),
                    method='.csv'):

    assert method in ['.npz', '.csv'], 'Unknown method.'
    vital_files = []
    for bed in bed_list:
        dates = os.listdir(os.path.join(vital_dir, bed))
        dates.sort()
        for t_date in dates:
            if start_date.strftime('%y%m%d') <= t_date <= end_date.strftime('%y%m%d'):
                vfiles = os.listdir(os.path.join(vital_dir, bed, t_date))
                vfiles.sort()
                for vfile in vfiles:
                    vital_files.append((os.path.join(vital_dir, bed, t_date, vfile), bed, t_date))
    p_dict = dict()
    for vfile in vital_files:
        if vfile[1] in prep_params['agg_room']:
            if method == '.csv':
                prep_file = os.path.join(prep_dir, '%s_%s.csv' % (vfile[1], vfile[2]))
            elif method == '.npz':
                prep_file = os.path.join(npz_dir, '%s_%s.npz' % (vfile[1], vfile[2]))
        else:
            if method == '.csv':
                prep_file = os.path.join(prep_dir, os.path.splitext(os.path.basename(vfile[0]))[0] + '.csv')
            elif method == '.npz':
                prep_file = os.path.join(npz_dir, os.path.splitext(os.path.basename(vfile[0]))[0] + '.npz')
        if prep_file not in p_dict:
            p_dict[prep_file] = list()
        p_dict[prep_file].append(vfile[0])

    return vital_files, p_dict


def enumerate_npz(bed_list=['D-02', 'D-06'], start_date=datetime.date(2018, 2, 1), end_date=datetime.date.today()):

    npz_files = []
    files = os.listdir(npz_dir)
    files.sort()
    for file in files:
        if file.endswith('.npz'):
            f_sp = os.path.splitext(file)[0].split('_')
            if f_sp[0] in bed_list and start_date.strftime('%y%m%d') <= f_sp[1] <= end_date.strftime('%y%m%d'):
                npz_files.append(os.path.join(npz_dir, file))
    return npz_files


def enumerate_prep(bed_list=[], start_date=None, end_date=None, method='.npz'):

    query = 'SELECT file_name, file_path FROM prep_file WHERE file_name LIKE \'%%%s\'' % method
    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()
    try:
        cursor.execute(query)
    except Exception as e:
        logging.exception(e)
    prep_all = cursor.fetchall()
    db.close()

    r = list()
    for prep in prep_all:
        p_prep = os.path.splitext(prep[0])[0].split('_')
        t_bed = p_prep[0]
        t_date = p_prep[1]
        if t_bed in bed_list if len(bed_list) else True:
            if start_date is None and end_date is None:
                r.append(os.path.join(prep[1], prep[0]))
            elif end_date is None and t_date >= start_date.strftime('%y%m%d'):
                r.append(os.path.join(prep[1], prep[0]))
            elif start_date is None and t_date <= end_date.strftime('%y%m%d'):
                r.append(os.path.join(prep[1], prep[0]))
            elif start_date.strftime('%y%m%d') <= t_date <= end_date.strftime('%y%m%d'):
                r.append(os.path.join(prep[1], prep[0]))

    return r


def get_prep_info(prepfile):

    db = MySQLdb.connect(host='localhost', user='shmoon', password='ibsntxmes', db='sv_trend_model')
    cursor = db.cursor()
    query = 'SELECT id, file_name, file_path, reference, is_vg, is_cq, is_ev FROM prep_file '
    query += 'WHERE file_name=\'%s\' AND file_path=\'%s\'' % (os.path.basename(prepfile), os.path.dirname(prepfile))
    try:
        cursor.execute(query)
    except Exception as e:
        logging.exception(e)
    result_prep = cursor.fetchall()
    assert len(result_prep) == 1, 'prep file %s could not be found in database.' % prepfile
    db.close()
    return result_prep[0]


def get_prep_title(prepfile):

    assert os.path.exists(prepfile), 'Preprocessed file %s doesn\'t exists.' % prepfile
    with open(prepfile, 'r') as fp:
        title = fp.readline().rstrip('\n').split(',')
    n_non_wave = 0
    wave = re.compile('W[0-9]{4}')
    col_dict = dict()
    for i, col in enumerate(title):
        col_dict[col] = i
        if not wave.match(col):
            n_non_wave += 1

    return title, col_dict, n_non_wave


def load_npz(t_data, ref='EV_SVV', to3d=False, trim=True, normalize_abp=False, input_len=1024):

    if ref == 'CQ_SV' or trim:
        trim = datetime.timedelta(seconds=1800)
    else:
        trim = datetime.timedelta(seconds=0)

    t = {'title': np.array(t_data['title']), 'timestamp': np.array(t_data['timestamp']),
         'feature': np.array(t_data['feature']), 'abp': np.array(t_data['abp']),
         'abp_valid': np.array(t_data['abp_valid'])}
    try:
        t['ppg'] = np.array(t_data['ppg'])
        t['ppg_valid'] = np.array(t_data['ppg_valid'])
    except KeyError:
        pass

    col_dict = dict()
    for i in range(t['title'].shape[0]):
        col_dict[t['title'][i]] = i
    t['col_dict'] = col_dict

    assert 0 <= input_len <= 1024, 'Wrong input vector length.'
    if input_len < 1024:
        t['abp'] = t['abp'][:, -input_len:]
        if 'ppg' in t.keys():
            t['ppg'] = t['ppg'][:, -input_len:]

    if normalize_abp:
        displacement = np.quantile(t['abp'], 0.01, axis=1)
        for i in range(t['abp'].shape[0]):
            t['abp'][i, :] = t['abp'][i, :] - displacement[i]

    if ref == 'CQ_SV':
        t['feature'][:, col_dict['CQ_SV']] = smoothing_result(t['feature'][:, col_dict['CQ_SV']], t['timestamp'],
                                                              propotiontocut=0.15, windowsize=150, side=2,
                                                              type='datetime')

    contaminated = list()
    if t['timestamp'].shape[0]:
        time_begin = t['timestamp'][0]
        time_end = t['timestamp'][-1]
        for i in range(t['timestamp'].shape[0]):
            if not time_begin + trim <= t['timestamp'][i] <= time_end - trim:
                contaminated.append(i)
            elif ref == 'EV_SVV' and not t['feature'][i, col_dict['EV_SVV']]:
                contaminated.append(i)
            elif ref == 'VG_SV' and not t['feature'][i, col_dict['VG_SV']]:
                contaminated.append(i)
            elif ref == 'CQ_SV' and t['feature'][i, col_dict['CQ_SV']] < 10:
                contaminated.append(i)
            elif ref == 'EV_SV' and not t['feature'][i, col_dict['EV_SV']]:
                contaminated.append(i)
            elif t['feature'][i, col_dict['T_Cycle_Begin']] < 30 or not t['feature'][i, col_dict['T_Cycle_End']]:
                contaminated.append(i)
            elif t['feature'][i, col_dict['T_Cycle_Begin']] >= t['feature'][i, col_dict['T_Cycle_End']]:
                contaminated.append(i)
            elif not t['abp_valid'][i]:
                contaminated.append(i)
            elif not t['ppg_valid'][i] if 'ppg_valid' in t.keys() else False:
                contaminated.append(i)
    else:
        assert False, 'Npz file contains no data.'

    t['abp'] = np.delete(t['abp'], contaminated, 0)
    try:
        t['ppg'] = np.delete(t['ppg'], contaminated, 0)
    except KeyError:
        pass

    t['feature'] = np.delete(t['feature'], contaminated, 0)
    t['timestamp'] = np.delete(t['timestamp'], contaminated, 0)

    if ref == 'CQ_SV':
        contaminated = list()
        p_prev = 0
        p_next = 0
        for i in range(t['timestamp'].shape[0]):
            while t['timestamp'][p_prev] < t['timestamp'][i] - datetime.timedelta(seconds=300):
                p_prev += 1
            while t['timestamp'][p_next] < t['timestamp'][i] + datetime.timedelta(seconds=300) if p_next < t['timestamp'].shape[0] else False:
                p_next += 1
            if i - p_prev < 60:
                contaminated.append(i)
            elif p_next - i < 60:
                contaminated.append(i)
        t['abp'] = np.delete(t['abp'], contaminated, 0)
        t['ppg'] = np.delete(t['ppg'], contaminated, 0)
        t['feature'] = np.delete(t['feature'], contaminated, 0)
        t['timestamp'] = np.delete(t['timestamp'], contaminated, 0)

    if to3d:
        t['abp'] = t['abp'][:, :, np.newaxis]
        if 'ppg' in t.keys():
            t['ppg'] = t['ppg'][:, :, np.newaxis]
        t['feature'] = t['feature'][:, :, np.newaxis]

    return t


def remove_dummy_preprocessed(dataset_prep, dataset_desc):
    i = 0
    while i < len(dataset_prep):
        if not os.path.exists(dataset_prep[i]):
            dataset_prep.pop(i)
            dataset_desc.pop(i)
        else:
            i += 1
    return


def find_contaminated_dataset(dataset, trim=0, preprocess_type='VG'):

    r = []
    if not len(dataset):
        return r
    time_begin = dataset[0][0]
    time_end = dataset[-1][0]
    for i in range(dataset.shape[0]):
        if preprocess_type == 'VG':
            if dataset[i][1] == 0 or dataset[i][11] == 0 or dataset[i][12] == 0 or dataset[i][16] < 30:
                r.append(i)
            elif int(dataset[i][16]) >= int(dataset[i][19]):
                r.append(i)
            elif dataset[i][16] == 0 or dataset[i][19] == 0:
                r.append(i)
            elif dataset[i][0] - time_begin < trim or time_end - dataset[i][0] < trim:
                r.append(i)
        elif preprocess_type == 'CQ':
            if dataset[i][1] == 0 or dataset[i][16] < 30:
                r.append(i)
            elif int(dataset[i][16]) >= int(dataset[i][19]):
                r.append(i)
            elif dataset[i][16] == 0 or dataset[i][19] == 0:
                r.append(i)
            elif dataset[i][0] - time_begin < trim or time_end - dataset[i][0] < trim:
                r.append(i)

    return r


#Adjust Length of Wave Data
def length_adjustment(wave, length):
    r = []
    for i in range(length):
        if i >= len(wave):
            r.append(0)
        else:
            r.append(wave[i])
    return r


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


#Resampling Wave Data
def resampling(wave, sampling_rate_before, sampling_rate_after):
    r = []
    p_before = 0
    while p_before < len(wave):
        if p_before / sampling_rate_before == len(r) / sampling_rate_after:
            r.append(wave[p_before])
            p_before = p_before + 1
        elif p_before / sampling_rate_before < len(r) / sampling_rate_after:
            p_before = p_before + 1
        else:
            last_value = wave[p_before-1]
            current_value = wave[p_before]
            last_time_before = (p_before-1)/sampling_rate_before
            current_time_before = (p_before)/sampling_rate_before
            current_time_after = len(r) / sampling_rate_after
            weighted_average = ( last_value * (current_time_after - last_time_before) + current_value * ( current_time_before - current_time_after ) ) * sampling_rate_before
            r.append(round(weighted_average, 5))
    return r


#Adjust Length of Wave Data
def length_adjustment(wave, length):
    r = []
    for i in range(length):
        if i >= len(wave):
            r.append(0)
        else:
            r.append(wave[i])
    return r


# Build a dataset for DNN.
# The last column is SV and the rest of the columns are sampled waveform data.

def slice_wave(wave, time, time_delay=0, window_size=10.24, srate=100):
    dt_start = time - time_delay - window_size
    dt_end = time - time_delay
    for i in range(len(wave)):
        if wave[i][0] <= dt_start and wave[i][0] + len(wave[i][1])/srate >= dt_end:
            p_end = int( ( dt_end - wave[i][0] ) * srate )
            r = [ wave[i][0] + (p_end-window_size*srate) / srate, wave[i][0] + p_end / srate ]
            r.extend(wave[i][1][p_end-int(window_size*srate):p_end])
            return r
    return [0] * (int(window_size * srate)+2)


def get_BA_error(data1, data2):
    x=[]
    y=[]
    for i in range(len(data1)):
        x.append(np.mean([data1[i],data2[i]]))
        y.append(data2[i]-data1[i])
    std = np.std(y)
    avg = np.mean(y)
    error = 0
    for i in range(len(data1)):
        if y[i] > avg + 2 * std or y[i] < avg - 2 * std:
            error += 1
    error_rate = error / len(data1)
    return error_rate


def save_figure_Q(data1, data2, title, filename, error_range=10, error_type='box'):
    n_target = 0
    n_error = 0
    absolute_error = []
    fig_range = 50

    for i in range(len(data1)):
        absolute_error.append(abs(data1[i]-data2[i]))
        if error_type == 'box':
            if data1[i] * data2[i] <= 0:
                if error_range == 0 or abs(data1[i]) >= error_range or abs(data2[i]) >= error_range:
                    n_target += 1
                    n_error += 1
            else:
                if error_range == 0 or abs(data1[i]) >= error_range or abs(data2[i]) >= error_range:
                    n_target += 1
        elif error_type == 'linear':
            n_target += 1
            if abs(data1[i] - data2[i]) > error_range:
                n_error += 1

    if n_target > 0:
        plt.figure(figsize=(16 * 0.75, 9 * 0.75))
        plt.title(title + ' (total=%d, target=%d, error=%d, er=%.2f%%, ae=%.2f+-%.2f%%)'%(len(data1), n_target, n_error, 100*n_error/n_target, np.mean(absolute_error), np.std(absolute_error)))
        plt.xlabel('dSVpac in %')
        plt.ylabel('dSVpred in %')
        plt.scatter(data1, data2, s=1)
        plt.plot([-fig_range, fig_range], [0, 0], color='black')
        plt.plot([0, 0], [-fig_range, fig_range], color='black')
        plt.xlim(-fig_range, fig_range)
        plt.ylim(-fig_range, fig_range)

        if error_type == 'box' and error_range != 0:
            plt.plot([-error_range, -error_range, error_range, error_range, -error_range ],
                     [-error_range, error_range, error_range, -error_range, -error_range], linestyle='dashed', color='red')
        elif error_type == 'linear' and error_range != 0:
            plt.plot([-fig_range, fig_range-error_range], [-fig_range+error_range, fig_range], linestyle='dashed', color='red')
            plt.plot([-fig_range+error_range, fig_range], [-fig_range, fig_range-error_range], linestyle='dashed', color='red')

        plt.savefig(filename, dpi=1200)


def save_figure_polar(data1, data2, title, filename, error_range=10):
    n_target = 0

    theta = []
    theta_t = []
    r = []
    color = []
    rlim = 20

    for i in range(len(data1)):
        r.append(abs(data1[i]+data2[i])/2)
        theta.append(cmath.phase(complex(data1[i], data2[i])))
        if r[-1] >= error_range:
            n_target += 1
            color.append('black')
            if data2[i] >= -data1[i]:
                theta_t.append(cmath.phase(complex(data1[i], data2[i])))
            else:
                theta_t.append(cmath.phase(complex(-data1[i], -data2[i])))

        else:
            color.append('grey')

    mean_theta = np.mean(theta_t)
    dev = []
    for i in theta_t:
        dev.append(abs(i-mean_theta))
    dev.sort()

    if n_target > 0:
        theta_domain = np.arange(0, 2*math.pi, 0.01)
        loa = 1.96 * np.std(theta_t)
#        loa = dev[int(len(dev) * 0.95) - 1]
        plt.figure(figsize=(16 * 0.75, 9 * 0.75))
        pp = plt.subplot(111, polar=True)
        pp.set_theta_offset(-math.pi/4)
        pp.set_rlim(0, rlim)
        plt.title(title + ' (total=%d, target=%d, angular bias and 95%% radial limit of agreement=%.1f+-%.1f)'%(len(data1), n_target, mean_theta/math.pi*180-45, loa/math.pi*180))
        pp.scatter(theta, r, color=color, s=2)
        pp.plot([np.mean(theta_t), np.mean(theta_t)+math.pi], [rlim, rlim], color='red')
        pp.plot([np.mean(theta_t+loa), np.mean(theta_t+loa)+math.pi], [rlim, rlim], color='red', linestyle='dotted', alpha=0.5)
        pp.plot([np.mean(theta_t-loa), np.mean(theta_t-loa)+math.pi], [rlim, rlim], color='red', linestyle='dotted', alpha=0.5)
        pp.plot(theta_domain, len(theta_domain)*[error_range], color='orange', linestyle='dashed')

        plt.savefig(filename, dpi=1200)


def save_figure_cost(data, filename):
    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
    plt.xlabel('training epochs')
    plt.ylabel('cost')
    plt.yscale('log')
    plt.plot(data, marker='o', color='black', markersize=2, markevery=5)
    plt.savefig(filename)


def save_figure_BA(data1, data2, title, filename, range_type='default'):

    x = []
    y = []
    for i in range(len(data1)):
        for j in range(len(data1[i])):
            x.append(np.mean([data1[i][j], data2[i][j]]))
            y.append(data2[i][j]-data1[i][j])
    std = np.std(y)
    avg = np.mean(y)
    error = 0
    for i in range(len(y)):
        if y[i] > avg + 2 * std or y[i] < avg - 2 * std:
            error += 1
    error_rate = 100 * error / len(y)

    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
    plt.title(title + " (error = %.2f%%, sigma = %2.2f, bias = %2.2f)"%(error_rate, std, avg))
    plt.xlabel('SV (pred+target)/2 in ml')
    plt.ylabel('SV pred-target in ml')
    plt.axhline(y=avg+2*std, label='+2sd', linestyle='dotted')
    plt.axhline(y=avg, label='average' , linestyle='dashed')
    plt.axhline(y=avg-2*std, label='-2sd', linestyle='dotted')
    plt.scatter(x, y, s=1)
    if range_type == 'sv_normalized':
        plt.xlim(20, 200)
        plt.ylim(-100, 100)
    plt.savefig(filename, dpi=1200)
    plt.close()

    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
#    plt.title(title + " (error = %.2f%%, sigma = %2.2f, bias=%2.2f)" % (error_rate, std, avg))
#    plt.xlabel('SV (pred+target)/2 in ml')
#    plt.ylabel('SV pred-target in ml')
    plt.axhline(y=avg + 2 * std, label='+2sd', linestyle='dotted')
    plt.axhline(y=avg, label='average', linestyle='dashed')
    plt.axhline(y=avg - 2 * std, label='-2sd', linestyle='dotted')

    for i in range(len(data1)):
        x = []
        y = []
        for j in range(len(data1[i])):
            x.append(np.mean([data1[i][j], data2[i][j]]))
            y.append(data2[i][j]-data1[i][j])
        plt.scatter(x, y, s=1)

    if range_type == 'sv_normalized':
        plt.xlim(20, 200)
        plt.ylim(-100, 100)
    plt.savefig(filename+"_marked", dpi=1200)
    plt.close()

    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
#    for i in range(len(data1)):
    for i in range(30):
        x = []
        y = []
        for j in range(len(data1[i])):
            x.append(np.mean([data1[i][j], data2[i][j]]))
            y.append(data2[i][j]-data1[i][j])
        plt.subplot(6, 5, i+1)
        plt.scatter(x, y, s=1)
        if range_type == 'sv_normalized':
            plt.xlim(20, 200)
            plt.ylim(-100, 100)

    plt.savefig(filename+"_each", dpi=1200)

    return error_rate


def save_figure_scatter(data1, data2, title, filename, range_type='default'):
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'grey']
    markers = ['o', 'v', '^', '<', '>', '8']
    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
    plt.title(title)
    plt.xlabel('Target SV in ml ')
    plt.ylabel('Predicted SV in ml')
    for i in range(len(data1)):
        plt.scatter(data1[i], data2[i], s=1, c=colors[0], marker=markers[0])
    if range_type == 'sv_normalized':
        plt.xlim(20, 180)
        plt.ylim(20, 180)
    plt.savefig(filename, dpi=1200)
    plt.close()

    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
#    plt.title(title)
#    plt.xlabel('Target SV in ml ')
#    plt.ylabel('Predicted SV in ml')
    for i in range(len(data1)):
        plt.scatter(data1[i], data2[i], s=1)
    if range_type == 'sv_normalized':
        plt.xlim(20, 180)
        plt.ylim(20, 180)
    plt.savefig(filename+"_marked", dpi=1200)
    plt.close()

    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
#    for i in range(len(data1)):
    for i in range(30):
        plt.subplot(6, 5, i+1)
        plt.scatter(data1[i], data2[i], s=1)
        if range_type == 'sv_normalized':
            plt.xlim(20, 180)
            plt.ylim(20, 180)
    plt.savefig(filename+"_each", dpi=1200)


def save_figure_result(timestamp, lines, colors, legends, title, filename):
    timestamp_converted = []
    for i in range(len(timestamp)):
        timestamp_converted.append((datetime.datetime.utcfromtimestamp(timestamp[i]) + datetime.timedelta(hours=9)))
        #timestamp_converted.append((datetime.datetime.utcfromtimestamp(timestamp[i]) + datetime.timedelta(hours=9)).time())
    plt.figure(figsize=(16*0.75, 9*0.75))
    plt.xlabel('Time')
    plt.ylabel('SV in ml')
    plt.title(title)
    for i in range(len(lines)):
        plt.plot(timestamp_converted, lines[i], color=colors[i])
    plt.legend(legends, loc='upper right')
    plt.grid(linestyle = 'solid')
    plt.savefig(filename)
    plt.close()


def save_figure_comparison(timestamp, lines, colors, legends, title, filename):
    timestamp_converted = []
    for i in range(len(timestamp)):
        timestamp_converted.append((datetime.datetime.utcfromtimestamp(timestamp[i]) + datetime.timedelta(hours=9)))
        #timestamp_converted.append((datetime.datetime.utcfromtimestamp(timestamp[i]) + datetime.timedelta(hours=9)).time())
    plt.figure(figsize=(16*0.75, 9*0.75))

    plt.subplot(2, 1, 1)
    plt.title(title)
    plt.ylabel('ml')
    plt.grid(linestyle = 'solid')
    for i in range(len(lines[0])):
        plt.plot(timestamp_converted, lines[0][i], color = colors[0][i])
    plt.legend(legends[0], loc='upper right')

    plt.subplot(2, 1, 2)
    plt.ylabel('mmHg')
    plt.grid(linestyle = 'solid')
    for i in range(len(lines[1])):
        plt.plot(timestamp_converted, lines[1][i], color = colors[1][i])
    plt.legend(legends[1], loc='upper right')

    plt.savefig(filename)
    plt.close()


def abp_feature_extraction(row):
    wave = row[22:]
    dap = wave[int(row[19])]
    sap = wave[int(row[16])]
    map = (sap + 2*dap) / 3
    hr = 6000 / ( int(row[19]) - int(row[16]) )
    pp = sap - dap
    t_sap = int(row[17] - row[16])
    lz = pp / (sap + dap)
    hd = sap * 0.67 + dap * 0.33 - dap
    t_max_dtdp = 0
    max_dtdp = 0
    for i in range(int(row[16]), int(row[19])):
        if wave[i]-wave[i-1] > max_dtdp:
            t_max_dtdp = i-int(row[16])
            max_dtdp = wave[i]-wave[i-1]
    peakarea = 0
    for i in range(int(row[16]), int(row[17])):
        peakarea = peakarea + wave[i] - wave[int(row[16])]
    return [map, hr, pp, max_dtdp, t_max_dtdp, peakarea, t_sap, lz, hd]

# [map, hr, pp, max_dtdp, t_max_dtdp, peakarea, t_sap, lz, hd]


def smoothing_result(before_smoothing, timestamp, propotiontocut=0.05, windowsize=300, side=1, type='unixtime'):
    r = np.array(before_smoothing, dtype=np.float32)
    p_start = 0
    p_end = 0
    for i in range(len(r)):
        if type == 'unixtime':
            while timestamp[p_start] + windowsize <= timestamp[i]:
                p_start += 1
        elif type == 'datetime':
            while timestamp[p_start] - timestamp[i] <= datetime.timedelta(seconds=-windowsize):
                p_start += 1
        else:
            assert False, 'Unknown timestamp type.'
        if side == 1:
            r[i] = stats.trim_mean(before_smoothing[p_start:i + 1], propotiontocut)
        if side == 2:
            if type == 'unixtime':
                while timestamp[p_end] - windowsize <= timestamp[i] if p_end < len(r) else False:
                    p_end += 1
            elif type == 'datetime':
                while timestamp[p_end] - timestamp[i] <= datetime.timedelta(seconds=windowsize) if p_end < len(r) else False:
                    p_end += 1
            else:
                assert False, 'Unknown timestamp type.'

            r[i] = stats.trim_mean(before_smoothing[p_start:p_end], propotiontocut)
    return r


def normalize_result(result, mu=0.0, sigma=1.0):
    mean = np.nanmean(result)
    std = np.nanstd(result)
    r = []
    for i in range(len(result)):
        r.append((result[i] - mean) / std * sigma + mu)
    return r


def feature_extraction_vital(wave_param, fig_filename=''):
    wave = np.array(wave_param)
    for i in range(len(wave)):
        if wave[i] <= 30 or wave[i] >= 300:
            raise ValueError('Value is not arterial blood pressure. Out of range.')
    maxima = signal.find_peaks_cwt(wave, np.arange(1,30))
    minima = signal.find_peaks_cwt(-wave, np.arange(1,30))
    if len(maxima) == 0 or len(minima) == 0:
        raise ValueError('Wavelet function is not working.')
    max_val = wave[maxima[0]]
    for p in range(1, len(maxima)):
        if wave[maxima[p]] > max_val:
            max_val = wave[maxima[p]]
    min_val = wave[minima[0]]
    for p in range(1, len(minima)):
        if wave[minima[p]] > min_val:
            min_val = wave[minima[p]]
    quartile = [min_val, min_val*0.75+max_val*0.25, min_val*0.5+max_val*0.5, min_val*0.25+max_val*0.75, max_val]
    p_max = []
    p_foot = []
    p_dias = []
    for p in range(0, len(maxima)):
        if wave[maxima[p]] > quartile[3]:
            p_max.append(maxima[p])
    for p1 in range(0, len(p_max)):
        p2 = 0
        tmp_p_min = -1
        while p2 < len(minima) and minima[p2] < p_max[p1]:
            if wave[minima[p2]] < quartile[1]:
                tmp_p_min = minima[p2]
            p2 = p2 + 1
        p_foot.append(tmp_p_min)
        tmp_p_min = -1
        if p1 == len(p_max)-1:
            p_dias.append(tmp_p_min)
        else:
            while p2 < len(minima) and tmp_p_min == -1 and minima[p2] < p_max[p1+1]:
                if wave[minima[p2]] < quartile[3]:
                    tmp_p_min = minima[p2]
                p2 = p2 + 1
            p_dias.append(tmp_p_min)
    if len(p_max) < 2:
        raise ValueError('Cycle was not found.')
    if p_foot[-2] == -1:
        raise ValueError('Cycle was not found.')
    start_p_max = -2
    if len(wave) - p_max[-1] < 20: # If a found cycle is too close to end
        if len(p_max) < 3:
            raise ValueError('Cycle was not found.')
        start_p_max = -3
    A_total = sum(wave[p_foot[start_p_max]:p_foot[start_p_max+1]])/100.0
    A_sys = 0
    if p_dias[start_p_max] != -1:
        A_sys = sum(wave[p_foot[start_p_max]:p_dias[start_p_max]])/100.0
    if fig_filename != '':
        plt.plot(wave)
        plt.axvline(p_max[start_p_max], color='black', linestyle='dashed')
        plt.axvline(p_dias[start_p_max], color='black', linestyle='dashed')
        plt.axvline(p_foot[start_p_max], color='red', linestyle='dashed')
        plt.axvline(p_foot[start_p_max+1], color='red', linestyle='dashed')
        plt.savefig(fig_filename)
        plt.close()
    return [p_foot[start_p_max], p_max[start_p_max], p_dias[start_p_max], p_foot[start_p_max+1], A_total, A_sys]


def read_prep(prepfile, trim=0, svv_cleansing=False):

    print('Reading prep file %s.' % prepfile)

    title, col_dict, n_non_wave = get_prep_title(prepfile)

    xy = np.loadtxt(prepfile, delimiter=',', ndmin=2, skiprows=1, dtype=np.float64)
    prep_info = get_prep_info(prepfile)

    contaminated = list()
    if xy.shape[0]:
        time_begin = xy[0][col_dict['dt']]
        time_end = xy[-1][col_dict['dt']]
        for i in range(xy.shape[0]):
            if xy[i, col_dict['dt']] < time_begin+trim or xy[i, col_dict['dt']] > time_end-trim:
                contaminated.append(i)
#                print('trim', time_begin, time_end, xy[i, col_dict['dt']])
            elif prep_info[4] and not xy[i, col_dict['VG_SV']]:
                contaminated.append(i)
#                print('VG')
            elif prep_info[5] and not xy[i, col_dict['CQ_SV']]:
                contaminated.append(i)
#                print('CQ')
            elif prep_info[6] and not xy[i, col_dict['EV_SV']]:
                contaminated.append(i)
#                print('EV')
            elif svv_cleansing and prep_info[6] and not xy[i, col_dict['EV_SVV']]:
                contaminated.append(i)
#                print('EV')
            elif xy[i, col_dict['T_Cycle_Begin']] < 30 or not xy[i, col_dict['T_Cycle_End']]:
                contaminated.append(i)
#                print('Invalid Cycle 1')
            elif xy[i, col_dict['T_Cycle_Begin']] >= xy[i, col_dict['T_Cycle_End']]:
                contaminated.append(i)
#                print('Invalid Cycle 2')
    else:
        assert False, 'Preprocessed file %s isn\'t valid.' % prepfile

    refined_xy = np.delete(xy, contaminated, 0)
#    print(xy.shape, refined_xy.shape)

    x_val = np.empty([refined_xy.shape[0], len(title)-n_non_wave, 1], dtype=np.float32)
    x_val[:, :, 0] = refined_xy[:, n_non_wave:]
    etc_val = np.array(refined_xy[:, :n_non_wave])

    return x_val, etc_val


def read_preprocessed_data(filename, input_type, trim=0, preprocess_type='VG'):

    print(filename, input_type)
    xy = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=np.float64)
    contaminated = find_contaminated_dataset(xy, trim=trim, preprocess_type=preprocess_type)
    refined_xy = np.delete(xy, contaminated, 0)

    num_features = 9
    max_cycle_size = 150
    if input_type == "raw_256_subsample":
        size_x_val = 256
    elif input_type == "raw_1024":
        size_x_val = 1024
    elif ( input_type == "cycle" ):
        size_x_val = max_cycle_size
        slice_size = 1024
    else:
        print("Unknown Input Type.")
        exit(1)

    xval = np.empty([refined_xy.shape[0], size_x_val, 1], dtype=np.float32)
    tmp_xval = np.empty([size_x_val], dtype=np.float32)
    ftval = np.empty([refined_xy.shape[0], num_features], dtype=np.float32)
    if input_type == "raw_256_subsample":
        etcval = np.empty([refined_xy.shape[0], refined_xy.shape[1] - size_x_val*2], dtype=np.float64)
    else:
        etcval = np.empty([refined_xy.shape[0], refined_xy.shape[1] - size_x_val], dtype=np.float64)
    contaminated = []
    for ix in range(refined_xy.shape[0]):
        # Adding Target Response Variable
        if input_type == "raw_256_subsample":
            etcval[ix] = refined_xy[ix][:-size_x_val*2]
            for i in range(size_x_val):
                p = (i-size_x_val)*2
#                tmp_xval[i] = np.mean(refined_xy[ix][p:p+2])
                tmp_xval[i] = ( refined_xy[ix][p] + refined_xy[ix][p+1] ) / 2
            xval[ix, :, 0] = tmp_xval
            try:
                etcval[ix][-6:] = feature_extraction_vital(tmp_xval)
                ftval[ix] = abp_feature_extraction(list(etcval[ix]) + list(xval[ix, :, 0]))
            except ValueError as e:
                contaminated.append(ix)
            except ZeroDivisionError as e:
                contaminated.append(ix)

        elif input_type == "cycle":
            etcval[ix] = refined_xy[ix][:-size_x_val]
            ftval[ix] = abp_feature_extraction(refined_xy[ix])
            cycle_start = int(refined_xy[ix][10])
            cycle_end = int(refined_xy[ix][13])
            xval[ix] = length_adjustment(
                refined_xy[ix][(-slice_size + cycle_start):(-slice_size + cycle_end)], max_cycle_size)
        else:
            etcval[ix] = refined_xy[ix][:-size_x_val]
            ftval[ix] = abp_feature_extraction(refined_xy[ix])
            xval[ix, :, 0] = refined_xy[ix][-size_x_val:]

    if input_type == "raw_256_subsample":
        return np.delete(xval, contaminated, axis=0), np.delete(ftval, contaminated, axis=0), np.delete(etcval, contaminated, axis=0)
    else:
        return xval, ftval, etcval


def get_svv(sv, timestamp):

    assert len(sv) == len(timestamp), 'Lengths of SV and timestamp are different. (%d, %d)' % (len(sv), len(timestamp))

    p = 0
    respiratory_cycle = 25
    svv = list()

    for i in range(len(sv)):
        while timestamp[p] + datetime.timedelta(seconds=respiratory_cycle) < timestamp[i]:
            p += 1
        svv.append((max(sv[p:i+1])-min(sv[p:i+1]))/np.mean(sv[p:i+1])*100)

    return svv


def get_lz(abp):

    lz = list()
    for i in range(abp.shape[0]):
        min_val = max_val = abp[i, -200, 0]
        for j in range(200):
            if abp[i, j - 200, 0] > max_val:
                max_val = abp[i, j - 200, 0]
            if abp[i, j - 200, 0] < min_val:
                min_val = abp[i, j - 200, 0]
        lz.append((max_val-min_val)/(max_val+min_val))

    return lz
