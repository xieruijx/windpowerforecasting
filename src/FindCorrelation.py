# 检查风电功率和风速的关系
import pandas as pd
import matplotlib.pyplot as plt

def WeatherMatrix(name_wind, name_feature):    
    df_power = pd.read_csv('data/Data4Training/WindPowerData96w_' + name_wind + '.csv')
    df_power['DAY'] = pd.to_datetime(df_power['DAY'])
    df_power.set_index('DAY', inplace=True)

    id_columns = [f'{i}' for i in range(96)]
    id_columns_h = [f'{i}' for i in range(0, 96, 4)]
    df_output = pd.DataFrame(index=df_power.index, columns=id_columns)

    for index_df, _ in df_power.iterrows():
        index = index_df - pd.Timedelta(days=1)
        input_filename = 'data/Data4Training/Weather_' + name_wind + '/' + index.strftime('%Y%m%d') + '08.csv'
        df_weather = pd.read_csv(input_filename)
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        index_time = (df_weather['time'] < index_df + pd.Timedelta(days=1)) & (df_weather['time'] >= index_df)
        df_output.loc[index_df, id_columns_h] = list(df_weather.loc[index_time, name_feature])

    df_output.to_csv('data/Data4Training/WeatherData96w_' + name_wind + '_' + name_feature + '.csv')

    return df_output, df_power

def main():
    print('*************************Wind Farm D*************************')
    df_u100, df_power = WeatherMatrix('D', 'u100')
    df_v100, df_power = WeatherMatrix('D', 'v100')
    df_sp, df_power = WeatherMatrix('D', 'sp')
    df_tcw, df_power = WeatherMatrix('D', 'tcw')

    plt.figure(figsize=(8, 6))
    plt.plot(df_u100['92'] * df_u100['92'] + df_v100['92'] * df_v100['92'], df_power['92'], 'o')
    plt.title('Correlation')
    plt.xlabel('u^2 + v^2')
    plt.ylabel('Power')

    print('*************************Wind Farm E*************************')
    df_u100, df_power = WeatherMatrix('E', 'u100')
    df_v100, df_power = WeatherMatrix('E', 'v100')
    df_sp, df_power = WeatherMatrix('E', 'sp')
    df_tcw, df_power = WeatherMatrix('E', 'tcw')

    plt.figure(figsize=(8, 6))
    plt.plot(df_u100['0'] * df_u100['0'] + df_v100['0'] * df_v100['0'], df_power['0'], 'o')
    plt.title('Correlation')
    plt.xlabel('u^2 + v^2')
    plt.ylabel('Power')

    print('*************************Wind Farm F*************************')
    df_u100, df_power = WeatherMatrix('F', 'u100')
    df_v100, df_power = WeatherMatrix('F', 'v100')
    df_sp, df_power = WeatherMatrix('F', 'sp')
    df_tcw, df_power = WeatherMatrix('F', 'tcw')

    plt.figure(figsize=(8, 6))
    plt.plot(df_u100['0'] * df_u100['0'] + df_v100['0'] * df_v100['0'], df_power['0'], 'o')
    plt.title('Correlation')
    plt.xlabel('u^2 + v^2')
    plt.ylabel('Power')

    plt.figure(figsize=(8, 6))
    plt.plot(df_sp['0'], df_power['0'], 'o')
    plt.title('Correlation')
    plt.xlabel('sp')
    plt.ylabel('Power')

    plt.figure(figsize=(8, 6))
    plt.plot(df_tcw['0'], df_power['0'], 'o')
    plt.title('Correlation')
    plt.xlabel('tcw')
    plt.ylabel('Power')


if __name__ == "__main__":
    main()
    plt.show()
