# windpowerforecasting

The order:
src/ExtractWindPowerData.py
src/ExtractWindTurbineData.py
src/AnalyzeDEData.py
src/ProcessDEData.py
(data/WindFarmRealData/AddFilePower.py
 data/HistoricalWeatherForecastData/AddFile.py
 src/ProcessAddData.py)
src/ExamineWindPowerData.py
src/ExtractHistoricalWeatherForecastData.py
src/FindCorrelation.py
src/ProcessDataforMLP.py
src/TrainingMLP_spa.py, src/TrainingMLP_uvspa.py, src/TrainingMLP_a.py 前面到这里都是训练
data/Input4Forecast/AddFileInput.py 把当天天气预报“日期00.csv”放到data/Input4Forecast里面
Forecasting.py 第100行改成当天日期，运行完以后data/Output里面文件可上传提交
