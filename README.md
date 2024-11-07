# windpowerforecasting
2024年能源行业人工智能应用算法大赛算法题3-风电场功率预测建模与多实景验证

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
data/Input4Forecast/AddFileInput.py 把平台下载的当天天气预报改名成“日期00.csv”（比如2024080400.csv）放到data/Input4Forecast里面
src/Forecasting.py 第99行today_str改成当天日期，运行完以后data/Output里面文件可上传提交
