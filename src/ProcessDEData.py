# 根据DE风机功率，把两个场站功率分开

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_DE = pd.read_csv('data/Data4Training/WindPowerData_DE.csv')
df_DE['HAPPEN_TIME'] = pd.to_datetime(df_DE['HAPPEN_TIME'])

df_turbine = pd.read_csv('data/Data4Training/WindTurbineData_DE.csv')
df_turbine['HAPPEN_TIME'] = pd.to_datetime(df_turbine['HAPPEN_TIME'])
df_turbine = df_turbine.loc[df_turbine['HAPPEN_TIME'] <= pd.to_datetime('2023-12-29 13:30:00')]

id_DE = [f'#{i}' for i in range(1, 200)]
id_D = [f'#{i}' for i in range(1, 63)]
id_E = [f'#{i}' for i in range(63, 200)]
df_turbine['sum_DE'] = df_turbine[id_DE].sum(axis=1) / 1000 * 1.1746715 + 3.13640512
df_turbine['sum_D'] = df_turbine[id_D].sum(axis=1) / 1000 
df_turbine['sum_E'] = df_turbine[id_E].sum(axis=1) / 1000 

merged_df = pd.merge(df_DE, df_turbine, on='HAPPEN_TIME', how='inner')
# sum_power = merged_df['sum_DE'].to_numpy()
# DE_power = merged_df['ACTIVE_POWER'].to_numpy()
# coefficients = np.polyfit(sum_power, DE_power, 1)
# print(coefficients)

# plt.figure(figsize=(8, 6))
# plt.plot(range(len(merged_df)), merged_df['ACTIVE_POWER'])
# plt.plot(range(len(merged_df)), merged_df['sum_DE'])
# plt.plot(range(len(merged_df)), merged_df['sum_D'] + merged_df['sum_E'])
# plt.plot(range(len(merged_df)), merged_df['sum_D'])
# plt.title('ACTIVE_POWER')
# plt.xlabel('Time')
# plt.ylabel('Value')

df_Dadd = pd.DataFrame(columns=['HAPPEN_TIME', 'ACTIVE_POWER', 'LIMIT_POWER', 'LIMIT_BOOL'])
df_Dadd['HAPPEN_TIME'] = merged_df['HAPPEN_TIME']
df_Dadd['ACTIVE_POWER'] = merged_df['sum_D']
df_Dadd['LIMIT_POWER'] = 0
df_Dadd['LIMIT_BOOL'] = False
df_D = pd.read_csv('data/Data4Training/WindPowerData_D.csv')
df_D = pd.concat([df_Dadd, df_D], ignore_index=True)
df_D.set_index('HAPPEN_TIME', inplace=True)
df_D.to_csv('data/Data4Training/WindPowerData_Dmore.csv')

# plt.figure(figsize=(8, 6))
# plt.plot(range(len(df_D)), df_D['ACTIVE_POWER'])
# plt.title('ACTIVE_POWER')
# plt.xlabel('Time')
# plt.ylabel('Value')

df_Eadd = pd.DataFrame(columns=['HAPPEN_TIME', 'ACTIVE_POWER', 'LIMIT_POWER', 'LIMIT_BOOL'])
df_Eadd['HAPPEN_TIME'] = merged_df['HAPPEN_TIME']
df_Eadd['ACTIVE_POWER'] = merged_df['sum_E']
df_Eadd['LIMIT_POWER'] = 0
df_Eadd['LIMIT_BOOL'] = False
df_E = pd.read_csv('data/Data4Training/WindPowerData_E.csv')
df_E = pd.concat([df_Eadd, df_E], ignore_index=True)
df_E.set_index('HAPPEN_TIME', inplace=True)
df_E.to_csv('data/Data4Training/WindPowerData_Emore.csv')

# plt.figure(figsize=(8, 6))
# plt.plot(range(len(df_E)), df_E['ACTIVE_POWER'])
# plt.title('ACTIVE_POWER')
# plt.xlabel('Time')
# plt.ylabel('Value')

plt.show()