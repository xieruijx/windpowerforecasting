# 生成用于曲线拟合的数据集
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def DataMatrix1(name_wind, l_hours=24, h_hours=47):    
    df_power = pd.read_csv('data/Data4Training/WindPowerData96w_' + name_wind + '.csv')
    df_power['DAY'] = pd.to_datetime(df_power['DAY'])
    df_power.set_index('DAY', inplace=True)
    df_power['Forecast1'] = False
    id_columns = [f'{i}' for i in range(0, 96, 4)]

    for index_power, _ in df_power.iterrows():
        index_weather = index_power - pd.Timedelta(days=1)
        input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index_weather.strftime('%Y%m%d') + '08.csv'
        if os.path.exists(input_filename):
            df_power.loc[index_power, 'Forecast1'] = True

    df_weather = pd.read_csv(input_filename)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    num_input_period = sum((df_weather['time'] >= index_weather + pd.Timedelta(hours=l_hours)) & (df_weather['time'] <= index_weather + pd.Timedelta(hours=h_hours)))

    matrix_input = np.zeros((sum(df_power['Forecast1']), num_input_period))
    matrix_output = np.zeros((sum(df_power['Forecast1']), 24))
    index_matrix = 0
    for index_power, _ in df_power.iterrows():
        if df_power.loc[index_power, 'Forecast1'] == True:
            index_weather = index_power - pd.Timedelta(days=1)
            input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index_weather.strftime('%Y%m%d') + '08.csv'
            df_weather = pd.read_csv(input_filename)
            df_weather['time'] = pd.to_datetime(df_weather['time'])
            timeindex_weather = (df_weather['time'] >= index_weather + pd.Timedelta(hours=l_hours)) & (df_weather['time'] <= index_weather + pd.Timedelta(hours=h_hours))

            matrix_input[index_matrix, :] = df_weather.loc[timeindex_weather, 'a100'].to_numpy()
            matrix_output[index_matrix, :] = df_power.loc[index_power, id_columns].to_numpy()
            
            index_matrix = index_matrix + 1

    vector_input = matrix_input.reshape((-1,))
    vector_output = matrix_output.reshape((-1,))

    np.save('data/MediateData/DataVector1_' + name_wind + '_input.npy', vector_input)
    np.save('data/MediateData/DataVector1_' + name_wind + '_output.npy', vector_output)

    return vector_input, vector_output

def DataMatrix4(name_wind, l_hours=96, h_hours=119):    
    df_power = pd.read_csv('data/Data4Training/WindPowerData96w_' + name_wind + '.csv')
    df_power['DAY'] = pd.to_datetime(df_power['DAY'])
    df_power.set_index('DAY', inplace=True)
    df_power['Forecast4'] = False
    id_columns = [f'{i}' for i in range(96)]

    for index_power, _ in df_power.iterrows():
        index_weather = index_power - pd.Timedelta(days=4)
        input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index_weather.strftime('%Y%m%d') + '08.csv'
        if os.path.exists(input_filename):
            df_power.loc[index_power, 'Forecast4'] = True

    df_weather = pd.read_csv(input_filename)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    num_input_period = sum((df_weather['time'] >= index_weather + pd.Timedelta(hours=l_hours)) & (df_weather['time'] <= index_weather + pd.Timedelta(hours=h_hours)))

    matrix_input = np.zeros((sum(df_power['Forecast4']), num_input_period, len(name_features)))
    matrix_output = np.zeros((sum(df_power['Forecast4']), 96))
    index_matrix = 0
    for index_power, _ in df_power.iterrows():
        if df_power.loc[index_power, 'Forecast4'] == True:
            index_weather = index_power - pd.Timedelta(days=1)
            input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index_weather.strftime('%Y%m%d') + '08.csv'
            df_weather = pd.read_csv(input_filename)
            df_weather['time'] = pd.to_datetime(df_weather['time'])
            timeindex_weather = (df_weather['time'] >= index_weather + pd.Timedelta(hours=l_hours)) & (df_weather['time'] <= index_weather + pd.Timedelta(hours=h_hours))

            matrix_input[index_matrix, :, :] = df_weather.loc[timeindex_weather, name_features].to_numpy() / (np.ones((num_input_period, 1)) @ np.array(a_input).reshape((1, -1))) + np.ones((num_input_period, 1)) @ np.array(b_input).reshape((1, -1))
            matrix_output[index_matrix, :] = df_power.loc[index_power, id_columns].to_numpy() / a_output + b_output
            
            index_matrix = index_matrix + 1

    xmax = np.max(np.max(matrix_input, axis=0), axis=0)
    xmin = np.min(np.min(matrix_input, axis=0), axis=0)
    print(xmax)
    print(xmin)

    ymax = np.max(np.max(matrix_output, axis=0), axis=0)
    ymin = np.min(np.min(matrix_output, axis=0), axis=0)
    print(ymax)
    print(ymin)

    np.save('data/MediateData/DataMatrix4_' + name_wind + '_input.npy', matrix_input)
    np.save('data/MediateData/DataMatrix4_' + name_wind + '_output.npy', matrix_output)

    return matrix_input, matrix_output

def main():
    print('*************************Wind Farm D*************************')
    vector_input, vector_output = DataMatrix1('D')
    predict = np.minimum(np.maximum(vector_input * 4 - 160, 0), 380)

    plt.figure(figsize=(8, 6))
    plt.plot(vector_input, vector_output, ',')
    plt.xlabel('a100')
    plt.ylabel('power')

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(vector_input)), vector_input, label='a100')
    plt.plot(range(len(vector_input)), vector_output, label='power')
    plt.plot(range(len(vector_input)), predict, label='predict')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.xlim(1000, 3000)

    # DataMatrix4('D', a_input, b_input, a_output, b_output)

    print('*************************Wind Farm E*************************')
    vector_input, vector_output = DataMatrix1('E')
    predict = np.minimum(np.maximum(vector_input * 83 / 13 - 83 * 20 / 13, 0), 830)

    plt.figure(figsize=(8, 6))
    plt.plot(vector_input, vector_output, ',')
    plt.xlabel('a100')
    plt.ylabel('power')

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(vector_input)), vector_input, label='a100')
    plt.plot(range(len(vector_input)), vector_output, label='power')
    plt.plot(range(len(vector_input)), predict, label='predict')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.xlim(1000, 3000)

    print('*************************Wind Farm F*************************')
    vector_input, vector_output = DataMatrix1('F')
    predict = np.minimum(np.maximum(vector_input * 35 / 13 - 35 * 20 / 13, 0), 350)

    plt.figure(figsize=(8, 6))
    plt.plot(vector_input, vector_output, ',')
    plt.xlabel('a100')
    plt.ylabel('power')

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(vector_input)), vector_input, label='a100')
    plt.plot(range(len(vector_input)), vector_output, label='power')
    plt.plot(range(len(vector_input)), predict, label='predict')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.xlim(1000, 3000)

if __name__ == "__main__":
    main()

    plt.show()
