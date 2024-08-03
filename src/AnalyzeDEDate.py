import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_D = pd.read_csv('data/Data4Training/WindPowerData_D.csv')
df_D['HAPPEN_TIME'] = pd.to_datetime(df_D['HAPPEN_TIME'])

df_E = pd.read_csv('data/Data4Training/WindPowerData_E.csv')
df_E['HAPPEN_TIME'] = pd.to_datetime(df_E['HAPPEN_TIME'])

df_turbine = pd.read_csv('data/Data4Training/WindTurbineData_newDE.csv')
df_turbine['HAPPEN_TIME'] = pd.to_datetime(df_turbine['HAPPEN_TIME'])

id_D = [f'{i}' for i in range(1, 63)]
id_E = [f'{i}' for i in range(63, 200)]
df_turbine['sum_D'] = df_turbine[id_D].sum(axis=1) / 1000 
df_turbine['sum_E'] = df_turbine[id_E].sum(axis=1) / 1000 

merged_df_D = pd.merge(df_D, df_turbine, on='HAPPEN_TIME', how='inner')

merged_df_D = merged_df_D.loc[merged_df_D['sum_D'] - merged_df_D['ACTIVE_POWER'] >= 0]

coefficients = np.polyfit(merged_df_D['sum_D'], merged_df_D['sum_D'] - merged_df_D['ACTIVE_POWER'], 2)
x_fit = np.linspace(min(merged_df_D['ACTIVE_POWER']), max(merged_df_D['ACTIVE_POWER']), 100) 
y_fit = coefficients[0] * x_fit * x_fit + coefficients[1] * x_fit + coefficients[2]
print(coefficients)

plt.figure(figsize=(8, 6))
plt.plot(merged_df_D['sum_D'], merged_df_D['sum_D'] - merged_df_D['ACTIVE_POWER'], ',')
plt.plot(x_fit, y_fit)
plt.xlabel('ACTIVE_POWER')
plt.ylabel('SUM - ACTIVE_POWER')

merged_df_E = pd.merge(df_E, df_turbine, on='HAPPEN_TIME', how='inner')

merged_df_E = merged_df_E.loc[merged_df_E['sum_E'] - merged_df_E['ACTIVE_POWER'] >= 0]

coefficients = np.polyfit(merged_df_E['sum_E'], merged_df_E['sum_E'] - merged_df_E['ACTIVE_POWER'], 2)
x_fit = np.linspace(min(merged_df_E['ACTIVE_POWER']), max(merged_df_E['ACTIVE_POWER']), 100) 
y_fit = coefficients[0] * x_fit * x_fit + coefficients[1] * x_fit + coefficients[2]
print(coefficients)

plt.figure(figsize=(8, 6))
plt.plot(merged_df_E['sum_E'], merged_df_E['sum_E'] - merged_df_E['ACTIVE_POWER'], ',')
plt.plot(x_fit, y_fit)
plt.xlabel('ACTIVE_POWER')
plt.ylabel('SUM - ACTIVE_POWER')

plt.show()