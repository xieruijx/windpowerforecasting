# 选取对应风电场的气象数据，改UTC时区

import pandas as pd
import os

def ProcessWeather(name, longitude, latitude):
    foldername = 'data/Data4Training/Weather_' + name + '/'
    df = pd.read_csv('data/Data4Training/WindPowerData96_' + name + '.csv')
    df['DAY'] = pd.to_datetime(df['DAY'])
    df.set_index('DAY', inplace=True)
    df['Weather'] = False
    for index_df, _ in df.iterrows():
        index = index_df - pd.Timedelta(days=1)
        input_filename = 'data/HistoricalWeatherForecastData/Type2/' + index.strftime('%Y%m%d') + '00.csv'
        if os.path.exists(input_filename):
            df_input = pd.read_csv(input_filename)
            if not 'tcw' in df_input.columns:
                index1y = index + pd.Timedelta(days=365)
                df_input_tcw = pd.read_csv('data/HistoricalWeatherForecastData/Type2/' + index1y.strftime('%Y%m%d') + '00.csv')
                df_input['tcw'] = df_input_tcw['tcw']
            df_output = df_input.loc[(df_input['longitude'] == longitude) & (df_input['latitude'] == latitude), ['time', 'u100', 'v100', 'sp', 'tcw']]
            df_output['a100'] = (df_output['u100'] ** 2) + (df_output['v100'] ** 2)
            df_output['time'] = pd.to_datetime(df_output['time'])
            df_output['time'] = df_output['time'] + pd.Timedelta(hours=8)
            df_output.set_index('time', inplace=True)
            df_output.to_csv(foldername + index.strftime('%Y%m%d') + '08.csv')
            if index in df.index:
                df.loc[index, 'Weather'] = True
    df_new = df.loc[df['Weather'] == True, [f'{i}' for i in range(96)]]
    df_new.to_csv('data/Data4Training/WindPowerData96w_' + name + '.csv')

def main():
    print('*************************Wind Farm D*************************')
    longitude = 111.5
    latitude = 21.4
    ProcessWeather('D', longitude, latitude)

    print('*************************Wind Farm E*************************')
    longitude = 111.6
    latitude = 21.3
    ProcessWeather('E', longitude, latitude)

    print('*************************Wind Farm F*************************')
    longitude = 111.5
    latitude = 21.3
    ProcessWeather('F', longitude, latitude)

    # date_range = pd.date_range(start='2023-01-01', end='2024-06-30', freq='D')
    # for single_date in date_range:
    #     filename = 'data/HistoricalWeatherForecastData/Type2/' + single_date.strftime('%Y%m%d') + '00.csv'
    #     if os.path.exists(filename):
    #         df = pd.read_csv(filename)
    #         if not 'tcw' in df.columns:
    #             print(filename)

if __name__ == "__main__":
    main()
