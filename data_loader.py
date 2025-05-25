# data_loader.py - 数据加载文件

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def splid_window(data, order):
    """划分时间窗口"""
    X = []
    label = []
    for i in range(len(data) - order):
        X.append(data[i:i + order, :])
        label.append(data[i + order, :])
    X = np.array(X)
    label = np.array(label)
    X = np.swapaxes(X, axis1=1, axis2=2)
    return X, label


def load_taiex(order, year):
    """加载TAIEX数据集"""
    df = pd.read_csv('TAIEX.csv', index_col='date', usecols=['date', 'open', 'high', 'low', 'close'])

    if year == 2000:
        df_year = df['2000-01-01':'2000-12-30']
        train_data = df_year['2000-01-01':'2000-10-31'].values
        test_data = df_year['2000-11-01':'2000-12-30'].values
    elif year == 2001:
        df_year = df['2001-01-01':'2001-12-30']
        train_data = df_year['2001-01-01':'2001-10-31'].values
        test_data = df_year['2001-11-01':'2001-12-30'].values
    elif year == 2002:
        df_year = df['2002-01-01':'2002-12-30']
        train_data = df_year['2002-01-01':'2002-10-31'].values
        test_data = df_year['2002-11-01':'2002-12-30'].values
    elif year == 2003:
        df_year = df['2003-01-01':'2003-12-30']
        train_data = df_year['2003-01-01':'2003-10-31'].values
        test_data = df_year['2003-11-01':'2003-12-30'].values
    else:  # 2004
        df_year = df['2004-01-01':'2004-12-30']
        train_data = df_year['2004-01-01':'2004-10-31'].values
        test_data = df_year['2004-11-01':'2004-12-30'].values

    scaler = MinMaxScaler()
    train_len = len(train_data)
    test_len = len(test_data)
    scaled_data = scaler.fit_transform(np.concatenate((train_data, test_data), axis=0))
    all_X, all_label = splid_window(scaled_data, order)

    train_X = all_X[:train_len, ...]
    train_label = all_label[:train_len, ...]
    test_X = all_X[train_len - order: train_len + test_len, ...]
    test_label = all_label[train_len - order: train_len + test_len, ...]

    return np.array(train_X), np.array(train_label), np.array(test_X), np.array(test_label), scaler


def load_sse(order, year):
    """加载SSE数据集"""
    df = pd.read_csv('SSE.csv', index_col='date', usecols=['date', 'open', 'high', 'low', 'close'])

    year_str = str(year)
    df_year = df[f'{year_str}-01-01':f'{year_str}-12-30']
    train_data = df_year[f'{year_str}-01-01':f'{year_str}-10-31'].values
    test_data = df_year[f'{year_str}-11-01':f'{year_str}-12-30'].values

    scaler = MinMaxScaler()
    train_len = len(train_data)
    test_len = len(test_data)
    scaled_data = scaler.fit_transform(np.concatenate((train_data, test_data), axis=0))
    all_X, all_label = splid_window(scaled_data, order)

    train_X = all_X[:train_len, ...]
    train_label = all_label[:train_len, ...]
    test_X = all_X[train_len - order: train_len + test_len, ...]
    test_label = all_label[train_len - order: train_len + test_len, ...]

    return np.array(train_X), np.array(train_label), np.array(test_X), np.array(test_label), scaler


def load_traffic(order):
    """加载交通数据集"""
    df = pd.read_csv('traffic.csv', usecols=[1, 2, 3, 4, 5, 6])
    df_10 = pd.DataFrame()
    for i in range(len(df) // 10):
        avg = df.iloc[i * 10:(i + 1) * 10, :].mean(axis=0)
        df_10 = pd.concat([df_10, pd.DataFrame([avg])], ignore_index=True)

    data = df_10.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    all_data, all_label = splid_window(scaled_data, order)
    total_len = len(all_data)
    train_X = all_data[:int(total_len * 0.7), ...]
    train_label = all_label[:int(total_len * 0.7), ...]
    test_X = all_data[int(total_len * 0.7):, ...]
    test_label = all_label[int(total_len * 0.7):, ...]

    return np.array(train_X), np.array(train_label), np.array(test_X), np.array(test_label), scaler


def load_temp(order):
    """加载温度数据集"""
    all_cols = ['1:Date', '2:Time', '3:Temperature_Comedor_Sensor', '4:Temperature_Habitacion_Sensor',
                '5:Weather_Temperature', '6:CO2_Comedor_Sensor', '7:CO2_Habitacion_Sensor',
                '8:Humedad_Comedor_Sensor', '9:Humedad_Habitacion_Sensor', '10:Lighting_Comedor_Sensor',
                '11:Lighting_Habitacion_Sensor', '12:Precipitacion', '13:Meteo_Exterior_Crepusculo',
                '14:Meteo_Exterior_Viento', '15:Meteo_Exterior_Sol_Oest', '16:Meteo_Exterior_Sol_Est',
                '17:Meteo_Exterior_Sol_Sud', '18:Meteo_Exterior_Piranometro', '19:Exterior_Entalpic_1',
                '20:Exterior_Entalpic_2', '21:Exterior_Entalpic_turbo', '22:Temperature_Exterior_Sensor',
                '23:Humedad_Exterior_Sensor', '24:Day_Of_Week']

    df = pd.read_csv('temp.txt', delimiter=' ')[all_cols]
    df['1:Date'] = pd.Categorical(df['1:Date']).codes
    df['2:Time'] = pd.Categorical(df['2:Time']).codes

    data = df.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    all_data, all_label = splid_window(scaled_data, order)
    total_len = len(all_data)
    train_X = all_data[:int(total_len * 0.7), ...]
    train_label = all_label[:int(total_len * 0.7), ...]
    test_X = all_data[int(total_len * 0.7):, ...]
    test_label = all_label[int(total_len * 0.7):, ...]

    return np.array(train_X), np.array(train_label), np.array(test_X), np.array(test_label), scaler


def load_epc(order):
    """加载EPC电力消耗数据集"""
    cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    df = pd.read_csv('epc.csv', usecols=cols)

    data = df.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    all_data, all_label = splid_window(scaled_data, order)
    total_len = len(all_data)
    train_X = all_data[:int(total_len * 0.7), ...]
    train_label = all_label[:int(total_len * 0.7), ...]
    test_X = all_data[int(total_len * 0.7):, ...]
    test_label = all_label[int(total_len * 0.7):, ...]

    return np.array(train_X), np.array(train_label), np.array(test_X), np.array(test_label), scaler


def load_eeg(order, obj=1):
    """加载EEG脑电数据集"""
    use_cols = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
                'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
                'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']

    test_lengths = {1: 23900, 2: 58375, 3: 24453, 4: 49323, 5: 45594, 6: 29916}
    test_len = test_lengths[obj]
    df = pd.read_csv(f'eeg/subj{obj}_series1_data.csv', usecols=use_cols)

    data = df.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    all_data, all_label = splid_window(scaled_data, order)
    train_X = all_data[:-test_len, ...]
    train_label = all_label[:-test_len, ...]
    test_X = all_data[-test_len:, ...]
    test_label = all_label[-test_len:, ...]

    return np.array(train_X), np.array(train_label), np.array(test_X), np.array(test_label), scaler


def get_data_loader(dataset_name):
    """获取数据加载器"""
    loaders = {
        'taiex': load_taiex,
        'sse': load_sse,
        'traffic': load_traffic,
        'temp': load_temp,
        'epc': load_epc,
        'eeg': load_eeg
    }
    return loaders.get(dataset_name.lower())