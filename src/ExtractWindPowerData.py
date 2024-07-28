# 把风电功率实测数据提取出来（不包含新加数据），每个风电场一个csv，包括时间标、风电和限电与否
# 只包含D和E的原始数据，没有把DE的数据拆开
import pandas as pd
import matplotlib.pyplot as plt

def mergedf(list_df):
    merged_df = pd.concat(list_df, ignore_index=True)
    merged_df = merged_df[['HAPPEN_TIME', 'ACTIVE_POWER', 'LIMIT_POWER']]
    merged_df['HAPPEN_TIME'] = pd.to_datetime(merged_df['HAPPEN_TIME'])
    merged_df['ACTIVE_POWER'] = merged_df['ACTIVE_POWER'].apply(pd.to_numeric, errors='coerce')
    merged_df['LIMIT_POWER'] = merged_df['LIMIT_POWER'].apply(pd.to_numeric, errors='coerce')
    merged_df.set_index('HAPPEN_TIME', inplace=True)
    merged_df = merged_df.sort_index()
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    print('MIN ACTIVE_POWER: {}, MAX ACTIVE_POWER: {}'.format(min(merged_df['ACTIVE_POWER']), max(merged_df['ACTIVE_POWER'])))
    print('MIN LIMIT_POWER: {}, MAX LIMIT_POWER: {}'.format(min(merged_df['LIMIT_POWER']), max(merged_df['LIMIT_POWER'])))

    return merged_df

def main():
    # Wind farm A
    print('*************************Wind Farm A*************************')
    df1 = pd.read_csv('data/WindFarmRealData/Type1/A/POWCAL_SITEPOW/POWCAL_SITEPOW_20220810-20230910.csv')
    df2 = pd.read_csv('data/WindFarmRealData/Type1/A/POWCAL_SITEPOW/POWCAL_SITEPOW_20231001-20240627.csv')
    merged_A = mergedf([df1, df2])
    merged_A['LIMIT_BOOL'] = merged_A['LIMIT_POWER'] < 297
    merged_A.to_csv('data/Data4Training/WindPowerData_A.csv')

    print('Proportion of limiting power: {}'.format(sum(merged_A['LIMIT_BOOL']) / len(merged_A)))

    # print('Data types in the merged dataframe:')
    # print(merged_A.dtypes)
    # plt.figure(figsize=(8, 6))
    # plt.hist(merged_A['LIMIT_POWER'], bins=50, alpha=0.7)
    # plt.title('Histogram for LIMIT_POWER in A')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(len(merged_A[~merged_A['LIMIT_BOOL']])), merged_A['ACTIVE_POWER'][~merged_A['LIMIT_BOOL']])
    # plt.title('ACTIVE_POWER in A')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # Wind farm B
    print('*************************Wind Farm B*************************')
    df1 = pd.read_csv('data/WindFarmRealData/Type1/B/POWCAL_SITEPOW/POWCAL_SITEPOW_20220812-20230912.csv')
    df2 = pd.read_csv('data/WindFarmRealData/Type1/B/POWCAL_SITEPOW/POWCAL_SITEPOW_20231001-20240627.csv')
    merged_B = mergedf([df1, df2])
    merged_B['LIMIT_BOOL'] = merged_B['LIMIT_POWER'] < 396
    merged_B.to_csv('data/Data4Training/WindPowerData_B.csv')

    print('Proportion of limiting power: {}'.format(sum(merged_B['LIMIT_BOOL']) / len(merged_B)))

    # print('Data types in the merged dataframe:')
    # print(merged_B.dtypes)
    # plt.figure(figsize=(8, 6))
    # plt.hist(merged_B['LIMIT_POWER'], bins=50, alpha=0.7)
    # plt.title('Histogram for LIMIT_POWER in B')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(len(merged_B[~merged_B['LIMIT_BOOL']])), merged_B['ACTIVE_POWER'][~merged_B['LIMIT_BOOL']])
    # plt.title('ACTIVE_POWER in B')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # Wind farm C
    print('*************************Wind Farm C*************************')
    df1 = pd.read_csv('data/WindFarmRealData/Type1/C/POWCAL_SITEPOW/POWCAL_SITEPOW_20220810-20230912.csv')
    df2 = pd.read_csv('data/WindFarmRealData/Type1/C/POWCAL_SITEPOW/POWCAL_SITEPOW_20231001-20240627.csv')
    merged_C = mergedf([df1, df2])
    merged_C['ACTIVE_POWER'] = abs(merged_C['ACTIVE_POWER'])
    merged_C['LIMIT_POWER'] = abs(merged_C['LIMIT_POWER'])
    merged_C = merged_C[merged_C.index >= pd.to_datetime('2022-12-10 00:00:00')]
    time_index = merged_C.index < pd.to_datetime('2023-12-13 00:00:00')
    merged_C.loc[time_index, 'ACTIVE_POWER'] = merged_C['ACTIVE_POWER'][time_index] / 52.8 * 56.1
    merged_C.loc[time_index, 'LIMIT_POWER'] = merged_C['LIMIT_POWER'][time_index] / 52.8 * 56.1
    time_index = merged_C.index < pd.to_datetime('2023-11-06 12:00:00')
    merged_C.loc[time_index, 'ACTIVE_POWER'] = merged_C['ACTIVE_POWER'][time_index] / 56.1 * 85.8
    merged_C.loc[time_index, 'LIMIT_POWER'] = merged_C['LIMIT_POWER'][time_index] / 56.1 * 85.8
    print('Revise ACTIVE_POWER and LIMIT_POWER')
    print('MIN ACTIVE_POWER: {}, MAX ACTIVE_POWER: {}'.format(min(merged_C['ACTIVE_POWER']), max(merged_C['ACTIVE_POWER'])))
    print('MIN LIMIT_POWER: {}, MAX LIMIT_POWER: {}'.format(min(merged_C['LIMIT_POWER']), max(merged_C['LIMIT_POWER'])))
    merged_C['LIMIT_BOOL'] = merged_C['LIMIT_POWER'] < 84.9
    merged_C.to_csv('data/Data4Training/WindPowerData_C.csv')

    print('Proportion of limiting power: {}'.format(sum(merged_C['LIMIT_BOOL']) / len(merged_C)))

    # print('Data types in the merged dataframe:')
    # print(merged_C.dtypes)
    # plt.figure(figsize=(8, 6))
    # plt.hist(merged_C['LIMIT_POWER'], bins=100, alpha=0.7)
    # plt.title('Histogram for LIMIT_POWER in C')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(len(merged_C)), merged_C['ACTIVE_POWER'])
    # plt.title('ACTIVE_POWER in C')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # Wind farm D
    print('*************************Wind Farm D*************************')
    df1 = pd.read_csv('data/WindFarmRealData/Type2/D/POWCAL_SITEPOW/POWCAL_SITEPOW_20240315-20240628.csv')
    merged_D = mergedf([df1])
    merged_D['LIMIT_BOOL'] = False
    merged_D.to_csv('data/Data4Training/WindPowerData_D.csv')

    print('Proportion of limiting power: {}'.format(sum(merged_D['LIMIT_BOOL']) / len(merged_D)))
    # print('Data types in the merged dataframe:')
    # print(merged_D.dtypes)
    # plt.figure(figsize=(8, 6))
    # plt.hist(merged_D['ACTIVE_POWER'], bins=50, alpha=0.7)
    # plt.title('Histogram for ACTIVE_POWER in D')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(len(merged_D)), merged_D['ACTIVE_POWER'])
    # plt.title('ACTIVE_POWER in D')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # Wind farm E
    print('*************************Wind Farm E*************************')
    df1 = pd.read_csv('data/WindFarmRealData/Type2/E/POWCAL_SITEPOW/POWCAL_SITEPOW_20240315-20240628.csv')
    merged_E = mergedf([df1])
    merged_E['LIMIT_BOOL'] = False
    merged_E.to_csv('data/Data4Training/WindPowerData_E.csv')

    print('Proportion of limiting power: {}'.format(sum(merged_E['LIMIT_BOOL']) / len(merged_E)))

    # print('Data types in the merged dataframe:')
    # print(merged_E.dtypes)
    # plt.figure(figsize=(8, 6))
    # plt.hist(merged_E['ACTIVE_POWER'], bins=50, alpha=0.7)
    # plt.title('Histogram for ACTIVE_POWER in E')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(len(merged_E)), merged_E['ACTIVE_POWER'])
    # plt.title('ACTIVE_POWER in E')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # merged_D_reindexed = merged_D.reindex(merged_E.index, method='ffill')
    # merged_E_reindexed = merged_E.reindex(merged_D.index, method='ffill')
    # DE_sum = merged_D_reindexed['ACTIVE_POWER'] + merged_E_reindexed['ACTIVE_POWER']
    # plt.figure(figsize=(8, 6))
    # plt.hist(DE_sum, bins=50, alpha=0.7)
    # plt.title('Histogram for ACTIVE_POWER in D and E')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # plt.figure(figsize=(8, 6))
    # y = merged_D_reindexed['ACTIVE_POWER'] / DE_sum
    # y = y[DE_sum > 50]
    # plt.plot(range(len(y)), y)
    # plt.title('Proportion of D in D and E')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(len(merged_D_reindexed['ACTIVE_POWER'])), merged_D_reindexed['ACTIVE_POWER'])
    # plt.plot(range(len(merged_E_reindexed['ACTIVE_POWER'])), merged_E_reindexed['ACTIVE_POWER']/2)
    # plt.xlim(9770, 10000)
    # plt.title('Proportion of D in D and E')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # Wind farm F
    print('*************************Wind Farm F*************************')
    df1 = pd.read_csv('data/WindFarmRealData/Type2/F/POWCAL_SITEPOW/POWCAL_SITEPOW_20230101-20240125.csv')
    df2 = pd.read_csv('data/WindFarmRealData/Type2/F/POWCAL_SITEPOW/POWCAL_SITEPOW_20240315-20240628.csv')
    merged_F = mergedf([df1, df2])
    merged_F['LIMIT_BOOL'] = False
    merged_F.to_csv('data/Data4Training/WindPowerData_F.csv')

    print('Proportion of limiting power: {}'.format(sum(merged_F['LIMIT_BOOL']) / len(merged_F)))

    print('Data types in the merged dataframe:')
    print(merged_F.dtypes)
    plt.figure(figsize=(8, 6))
    plt.hist(merged_F['ACTIVE_POWER'], bins=50, alpha=0.7)
    plt.title('Histogram for ACTIVE_POWER in F')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(merged_F[~merged_F['LIMIT_BOOL']])), merged_F['ACTIVE_POWER'][~merged_F['LIMIT_BOOL']])
    plt.title('ACTIVE_POWER in F')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Wind farm DE
    print('*************************Wind Farm DE*************************')
    df1 = pd.read_csv('data/WindFarmRealData/Type2/DE/POWCAL_SITEPOW/POWCAL_SITEPOW_20230101-20231229.csv')
    merged_DE = mergedf([df1])
    # df2 = pd.read_csv('data/WindFarmRealData/Type2/E/POWCAL_SITEPOW/POWCAL_SITEPOW_20231229-20240208.csv')
    # merged_DE = mergedf([df1, df2])
    merged_DE['LIMIT_BOOL'] = False
    merged_DE.to_csv('data/Data4Training/WindPowerData_DE.csv')

    print('Proportion of limiting power: {}'.format(sum(merged_DE['LIMIT_BOOL']) / len(merged_DE)))

    # print('Data types in the merged dataframe:')
    # print(merged_DE.dtypes)
    # plt.figure(figsize=(8, 6))
    # plt.hist(merged_DE['ACTIVE_POWER'], bins=50, alpha=0.7)
    # plt.title('Histogram for ACTIVE_POWER in DE')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(len(merged_DE)), merged_DE['ACTIVE_POWER'])
    # plt.title('ACTIVE_POWER in DE')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    plt.show()

if __name__ == "__main__":
    main()
