# 生成用于训练的数据集
import pandas as pd
import numpy as np
import os

def DataMatrix1(name_wind, a_input, b_input, a_output, b_output, name_features=['u100', 'v100', 'sp', 'tcw', 'a100', 't100'], l_hours=15, h_hours=45):    
    df_power = pd.read_csv('data/Data4Training/WindPowerData96w_' + name_wind + '.csv')
    df_power['DAY'] = pd.to_datetime(df_power['DAY'])
    df_power.set_index('DAY', inplace=True)
    df_power['Forecast1'] = False
    id_columns = [f'{i}' for i in range(96)]

    for index_power, _ in df_power.iterrows():
        index_weather = index_power - pd.Timedelta(days=1)
        input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index_weather.strftime('%Y%m%d') + '14.csv'
        if os.path.exists(input_filename):
            df_power.loc[index_power, 'Forecast1'] = True

    df_weather = pd.read_csv(input_filename)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    num_input_period = sum((df_weather['time'] >= index_weather + pd.Timedelta(hours=l_hours)) & (df_weather['time'] <= index_weather + pd.Timedelta(hours=h_hours)))

    matrix_input = np.zeros((sum(df_power['Forecast1']), num_input_period, len(name_features)))
    matrix_output = np.zeros((sum(df_power['Forecast1']), 96))
    index_matrix = 0
    for index_power, _ in df_power.iterrows():
        if df_power.loc[index_power, 'Forecast1'] == True:
            index_weather = index_power - pd.Timedelta(days=1)
            input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index_weather.strftime('%Y%m%d') + '14.csv'
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

    np.save('data/MediateData/DataMatrix1_14_' + name_wind + '_input.npy', matrix_input)
    np.save('data/MediateData/DataMatrix1_14_' + name_wind + '_output.npy', matrix_output)

    return matrix_input, matrix_output

def DataMatrix4(name_wind, a_input, b_input, a_output, b_output, name_features=['u100', 'v100', 'sp', 'tcw', 'a100', 't100'], l_hours=87, h_hours=116):    
    df_power = pd.read_csv('data/Data4Training/WindPowerData96w_' + name_wind + '.csv')
    df_power['DAY'] = pd.to_datetime(df_power['DAY'])
    df_power.set_index('DAY', inplace=True)
    df_power['Forecast4'] = False
    id_columns = [f'{i}' for i in range(96)]

    for index_power, _ in df_power.iterrows():
        index_weather = index_power - pd.Timedelta(days=4)
        input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index_weather.strftime('%Y%m%d') + '14.csv'
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
            index_weather = index_power - pd.Timedelta(days=4)
            input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index_weather.strftime('%Y%m%d') + '14.csv'
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

    np.save('data/MediateData/DataMatrix4_14_' + name_wind + '_input.npy', matrix_input)
    np.save('data/MediateData/DataMatrix4_14_' + name_wind + '_output.npy', matrix_output)

    return matrix_input, matrix_output

def main():
    print('*************************Wind Farm D*************************')
    a_input=[51, 51, 4700, 91, 940, 3.15]
    b_input=[0.57, 0.57, -20.94, -0.09, 0, 0]
    a_output=410
    b_output=0.01
    DataMatrix1('D', a_input, b_input, a_output, b_output)
    DataMatrix4('D', a_input, b_input, a_output, b_output)

    print('*************************Wind Farm E*************************')
    a_input=[51, 51, 4700, 91, 940, 3.15]
    b_input=[0.57, 0.57, -20.94, -0.09, 0, 0]
    a_output=900
    b_output=0.01
    DataMatrix1('E', a_input, b_input, a_output, b_output)
    DataMatrix4('E', a_input, b_input, a_output, b_output)

    print('*************************Wind Farm F*************************')
    a_input=[51, 51, 4700, 91, 940, 3.15]
    b_input=[0.57, 0.57, -20.94, -0.09, 0, 0]
    a_output=390
    b_output=0.02
    DataMatrix1('F', a_input, b_input, a_output, b_output)
    DataMatrix4('F', a_input, b_input, a_output, b_output)

if __name__ == "__main__":
    main()
