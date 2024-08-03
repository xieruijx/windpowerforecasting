import pandas as pd
import os

df_D = pd.read_csv('data/Data4Training/WindPowerData_Dmore.csv')
df_D['HAPPEN_TIME'] = pd.to_datetime(df_D['HAPPEN_TIME'])
df_D.set_index('HAPPEN_TIME', inplace=True)

df_E = pd.read_csv('data/Data4Training/WindPowerData_Emore.csv')
df_E['HAPPEN_TIME'] = pd.to_datetime(df_E['HAPPEN_TIME'])
df_E.set_index('HAPPEN_TIME', inplace=True)

df_F = pd.read_csv('data/Data4Training/WindPowerData_F.csv')
df_F['HAPPEN_TIME'] = pd.to_datetime(df_F['HAPPEN_TIME'])
df_F.set_index('HAPPEN_TIME', inplace=True)

input_directory = "data/WindFarmRealData/Add"
for filename in os.listdir(input_directory):
    input_string = os.path.join(input_directory, filename)
    if '.csv' in input_string:
        df_input = pd.read_csv(input_string)
    if '.xlsx' in input_string:
        df_input = pd.read_excel(input_string)
    df_input['HAPPEN_TIME'] = pd.to_datetime(df_input['HAPPEN_TIME'])
    df_input.set_index('HAPPEN_TIME', inplace=True)

    for index_row, row in df_input.iterrows():
        new_row = {}
        new_row['ACTIVE_POWER'] = row['ACTIVE_POWER']
        new_row['LIMIT_POWER'] = row['LIMIT_POWER']
        new_row['LIMIT_BOOL'] = False
        if row['SITEID'] == 'D':
            df_D.loc[index_row] = new_row
        elif row['SITEID'] == 'E':
            df_E.loc[index_row] = new_row
        elif row['SITEID'] == 'F':
            df_F.loc[index_row] = new_row

df_D = df_D.sort_index()
df_D.to_csv('data/Data4Training/WindPowerData_Dmore.csv')

df_E = df_E.sort_index()
df_E.to_csv('data/Data4Training/WindPowerData_Emore.csv')

df_F = df_F.sort_index()
df_F.to_csv('data/Data4Training/WindPowerData_F.csv')