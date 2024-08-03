# 检查数据，留下有全天96个数据的天和数据，存成每行一天96个点的形式
import pandas as pd

def CheckDayData(df):
    date_range = pd.date_range(start='2023-01-01', end='2024-06-30', freq='D')

    all_times = pd.date_range(start='2023-01-01 00:00', end='2024-06-30 23:45', freq='15min')

    df['HAPPEN_TIME_str'] = df['HAPPEN_TIME'].dt.strftime('%Y-%m-%d %H:%M')

    valid_dates = pd.Series(False, index=date_range)

    for single_date in date_range:
        current_date_times = all_times[(all_times >= single_date) & (all_times < single_date + pd.Timedelta(days=1))]
        
        current_date_times_str = current_date_times.strftime('%Y-%m-%d %H:%M')
        
        if (current_date_times_str.isin(df['HAPPEN_TIME_str'])).all():
            valid_dates[single_date] = True

    for index_row, row in df.iterrows():
        if row['ACTIVE_POWER'] in [0, -0.04182, 0.01926, 0.05377]:
            index_day = row['HAPPEN_TIME'].date()
            valid_dates[index_day] = False

    result_df = pd.DataFrame(valid_dates, columns=['Valid Dates'])

    return result_df

def Processdf(df):
    df['HAPPEN_TIME'] = pd.to_datetime(df['HAPPEN_TIME'])
    valid_dates = CheckDayData(df)

    id_columns = ['DAY'] + [f'{i}' for i in range(96)]
    df_new = pd.DataFrame(columns=id_columns)
    df_new['DAY'] = valid_dates.index[valid_dates['Valid Dates']]
    df_new['DAY'] = pd.to_datetime(df_new['DAY'])
    df_new.set_index('DAY', inplace=True)

    for _, row in df.iterrows():
        day = pd.to_datetime(row['HAPPEN_TIME'].date())
        if pd.DatetimeIndex([day]).isin(df_new.index):
            id = row['HAPPEN_TIME'].hour * 4 + (row['HAPPEN_TIME'].minute // 15)
            df_new.loc[day, str(id)] = row['ACTIVE_POWER']

    return df_new

print('*************************Wind Farm D*************************')
df_D = pd.read_csv('data/Data4Training/WindPowerData_Dmore.csv')
print('MIN ACTIVE_POWER: {}, MAX ACTIVE_POWER: {}'.format(min(df_D['ACTIVE_POWER']), max(df_D['ACTIVE_POWER'])))
df_Dnew = Processdf(df_D)
df_Dnew.to_csv('data/Data4Training/WindPowerData96_D.csv')

print('*************************Wind Farm E*************************')
df_E = pd.read_csv('data/Data4Training/WindPowerData_Emore.csv')
print('MIN ACTIVE_POWER: {}, MAX ACTIVE_POWER: {}'.format(min(df_E['ACTIVE_POWER']), max(df_E['ACTIVE_POWER'])))
df_Enew = Processdf(df_E)
df_Enew.to_csv('data/Data4Training/WindPowerData96_E.csv')

print('*************************Wind Farm F*************************')
df_F = pd.read_csv('data/Data4Training/WindPowerData_F.csv')
print('MIN ACTIVE_POWER: {}, MAX ACTIVE_POWER: {}'.format(min(df_F['ACTIVE_POWER']), max(df_F['ACTIVE_POWER'])))
df_Fnew = Processdf(df_F)
df_Fnew.to_csv('data/Data4Training/WindPowerData96_F.csv')
