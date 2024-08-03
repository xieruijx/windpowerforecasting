# 将风电场DE各个风机的功率读出来，形成一个新的csv

import pandas as pd

chunksize = 10000

# csv_file_path = 'data/WindFarmRealData/Type2/DE/POWCAL_FJPOW/POWCAL_FJPOW_20230101-20231230.csv'



# reader = pd.read_csv(csv_file_path, chunksize=chunksize)

# id_columns = [f'#{i}' for i in range(1, 200)]
# all_columns = ['HAPPEN_TIME'] + id_columns
# final_df = pd.DataFrame(columns=all_columns)

# start = pd.to_datetime('2023-01-01 00:15:00')
# end = pd.to_datetime('2023-12-30 23:45:00')
# final_df['HAPPEN_TIME'] = pd.date_range(start=start, end=end, freq='15min')

# num = 0
# for chunk in reader:
#     chunk['HAPPEN_TIME'] = pd.to_datetime(chunk['HAPPEN_TIME'])
#     chunk['ACTIVE_POWER'] = chunk['ACTIVE_POWER'].apply(pd.to_numeric, errors='coerce')
    
#     for _, row in chunk.iterrows():
#         index_finaldf = final_df.loc[final_df['HAPPEN_TIME'] == row['HAPPEN_TIME']].index
        
#         final_df.loc[index_finaldf, row['FANNUMBER']] = row['ACTIVE_POWER']

#     num = num + 1
#     print('Finish {} rounds'.format(num))

# final_df.set_index('HAPPEN_TIME', inplace=True)
# final_df.to_csv('data/Data4Training/WindTurbineData_DE.csv')

# csv_file_path = 'data/WindFarmRealData/Type2/D/POWCAL_FJPOW/POWCAL_FJPOW_20231230-20240208.csv'

# reader = pd.read_csv(csv_file_path, chunksize=chunksize)

# num = 0
# for chunk in reader:
#     chunk['HAPPEN_TIME'] = pd.to_datetime(chunk['HAPPEN_TIME'])
#     chunk['ACTIVE_POWER'] = chunk['ACTIVE_POWER'].apply(pd.to_numeric, errors='coerce')
    
#     for _, row in chunk.iterrows():
#         final_df.loc[row['HAPPEN_TIME'], row['FANNUMBER']] = row['ACTIVE_POWER']

#     num = num + 1
#     print('Finish {} rounds'.format(num))

# final_df.to_csv('data/Data4Training/WindTurbineData_DE.csv')

# csv_file_path = 'data/WindFarmRealData/Type2/E/POWCAL_FJPOW/POWCAL_FJPOW_20231229-20240208.csv'

# reader = pd.read_csv(csv_file_path, chunksize=chunksize)

# num = 0
# for chunk in reader:
#     chunk['HAPPEN_TIME'] = pd.to_datetime(chunk['HAPPEN_TIME'])
#     chunk['ACTIVE_POWER'] = chunk['ACTIVE_POWER'].apply(pd.to_numeric, errors='coerce')
    
#     for _, row in chunk.iterrows():
#         final_df.loc[row['HAPPEN_TIME'], row['FANNUMBER']] = row['ACTIVE_POWER']

#     num = num + 1
#     print('Finish {} rounds'.format(num))

# final_df.to_csv('data/Data4Training/WindTurbineData_DE.csv')

id_columns = [f'{i}' for i in range(1, 200)]
all_columns = ['HAPPEN_TIME'] + id_columns
final_df = pd.DataFrame(columns=all_columns)
start = pd.to_datetime('2024-03-15 00:00:00')
end = pd.to_datetime('2024-06-28 23:45:00')
final_df['HAPPEN_TIME'] = pd.date_range(start=start, end=end, freq='15min')
final_df.set_index('HAPPEN_TIME', inplace=True)

csv_file_path = 'data/WindFarmRealData/Type2/D/POWCAL_FJPOW/POWCAL_FJPOW_20240315-20240628.csv'

reader = pd.read_csv(csv_file_path, chunksize=chunksize)

num = 0
for chunk in reader:
    chunk['HAPPEN_TIME'] = pd.to_datetime(chunk['HAPPEN_TIME'])
    chunk['ACTIVE_POWER'] = chunk['ACTIVE_POWER'].apply(pd.to_numeric, errors='coerce')
    
    for _, row in chunk.iterrows():
        final_df.loc[row['HAPPEN_TIME'], str(row['FANNUMBER'])] = row['ACTIVE_POWER']

    num = num + 1
    print('Finish {} rounds'.format(num))

final_df.to_csv('data/Data4Training/WindTurbineData_newDE.csv')


csv_file_path = 'data/WindFarmRealData/Type2/E/POWCAL_FJPOW/POWCAL_FJPOW_20240315-20240628.csv'

reader = pd.read_csv(csv_file_path, chunksize=chunksize)

num = 0
for chunk in reader:
    chunk['HAPPEN_TIME'] = pd.to_datetime(chunk['HAPPEN_TIME'])
    chunk['ACTIVE_POWER'] = chunk['ACTIVE_POWER'].apply(pd.to_numeric, errors='coerce')
    
    for _, row in chunk.iterrows():
        final_df.loc[row['HAPPEN_TIME'], str(row['FANNUMBER'])] = row['ACTIVE_POWER']

    num = num + 1
    print('Finish {} rounds'.format(num))

final_df.to_csv('data/Data4Training/WindTurbineData_newDE.csv')