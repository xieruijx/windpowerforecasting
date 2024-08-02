import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def Forecast(file_weather, name_wind, longitude, latitude, today, a, b, c):

    df_weather = pd.read_csv(file_weather)
    df_weather = df_weather.loc[(df_weather['longitude'] == longitude) & (df_weather['latitude'] == latitude), ['time', 'u100', 'v100', 'sp', 'tcw']]
    df_weather['a100'] = (df_weather['v100'] ** 2) + (df_weather['u100'] ** 2)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather['time'] = df_weather['time'] + pd.Timedelta(hours=8)

    index_weather = pd.to_datetime(today)
    input_period = (df_weather['time'] >= index_weather + pd.Timedelta(hours=24))

    df_weather = df_weather.loc[input_period]
    df_weather['index'] = 0
    for index_row, row in df_weather.iterrows():
        delta = row['time'] - index_weather
        df_weather.loc[index_row, 'index'] = delta.total_seconds() // 900 - 96

    f = interp1d(df_weather['index'], df_weather['a100'], kind='cubic')
    a100 = f(range(897))
    a100l = np.zeros((960,))
    a100l[:897] = a100
    output = np.minimum(np.maximum(a * a100l + b, 0), c)
    return output, df_weather['index'], df_weather['a100'], a100l

def output_file(name_wind, output, today_str):
    today = pd.to_datetime(today_str)
    tomorrow = today + pd.Timedelta(days=1)
    file_name = 'data/Output/' + name_wind + '_DQ_' + tomorrow.strftime('%Y%m%d') + '.csv'

    time = pd.date_range(tomorrow, tomorrow + pd.Timedelta(minutes=14385), freq='15min')
    PRED_DQ = output
    df = pd.DataFrame({})
    df['time'] = time
    # df['time'] = df['time'].dt.strftime('%Y/%m/%d %H:%M')
    df['PRED_DQ'] = PRED_DQ
    df.to_csv(file_name, float_format='%.3f', index=False)

def main():
    today_str = '20240731'
    file_weather = 'data/Input4Forecast/2024073100.csv'

    print('*************************Wind Farm D*************************')
    longitude = 111.5
    latitude = 21.4
    output, index, a100, a100l = Forecast(file_weather, 'D', longitude, latitude, pd.to_datetime(today_str), a=4, b=-160, c=380)
    output_file('D', output, today_str)
    plt.figure(figsize=(8, 6))
    plt.plot(index, a100, label='a100')
    plt.plot(range(960), a100l, label='interp')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('value')

    print('*************************Wind Farm E*************************')
    longitude = 111.6
    latitude = 21.3
    output, index, a100, a100l = Forecast(file_weather, 'E', longitude, latitude, pd.to_datetime(today_str), a=83/13, b=-83*20/13, c=830)
    output_file('E', output, today_str)
    plt.figure(figsize=(8, 6))
    plt.plot(index, a100, label='a100')
    plt.plot(range(960), a100l, label='interp')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('value')

    print('*************************Wind Farm F*************************')
    longitude = 111.5
    latitude = 21.3
    output, index, a100, a100l = Forecast(file_weather, 'F', longitude, latitude, pd.to_datetime(today_str), a=35/13, b=-35*20/13, c=350)
    output_file('F', output, today_str)
    plt.figure(figsize=(8, 6))
    plt.plot(index, a100, label='a100')
    plt.plot(range(960), a100l, label='interp')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('value')

if __name__ == "__main__":
    main()
    plt.show()