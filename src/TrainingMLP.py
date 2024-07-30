# 训练MLP网络用于预测
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()  
        current_size = num_inputs
        
        for size in hidden_layers:
            self.layers.append(nn.Linear(current_size, size))
            current_size = size
        
        self.layers.append(nn.Linear(current_size, num_outputs))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
class Day1Loss(nn.Module):
    def __init__(self, cap):
        super(Day1Loss, self).__init__()
        self.cap = cap

    def forward(self, y_pred, y_true):
        relative_error = torch.abs(y_pred - y_true) / torch.clamp(y_true, min=0.2*self.cap)
        
        loss = torch.mean(relative_error)
        return loss

class Day4Loss(nn.Module):
    def __init__(self, cap):
        super(Day4Loss, self).__init__()
        self.cap = cap

    def forward(self, y_pred, y_true):
        relative_error = torch.abs(y_pred - y_true) / torch.clamp(y_true, min=0.2*self.cap)
        
        loss = torch.mean(relative_error ** 2)
        return loss

def TrainMLP(name_wind, name_day, cap, hidden_layers=[32], batch_size=64, num_epochs=500, weight_decay=1e-4, index_features=[0, 1, 2, 3, 4]):
    name_file = 'data/MediateData/DataMatrix' + name_day + '_' + name_wind + '_'

    matrix_input = np.load(name_file + 'input.npy')
    matrix_input = matrix_input[:, :, index_features]
    matrix_output = np.load(name_file + 'output.npy')

    num_samples = matrix_input.shape[0]
    num_inputs = matrix_input.shape[1] * matrix_input.shape[2]
    num_outputs = matrix_output.shape[1]

    X = np.zeros((num_samples, num_inputs))
    for i in range(num_samples):
        X[i, :] = np.reshape(matrix_input[i, :, :], (-1,))
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(matrix_output, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(num_inputs, num_outputs, hidden_layers)

    if name_day == '1':
        criterion = Day1Loss(cap)
    elif name_day == '4':
        criterion = Day4Loss(cap)
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.9f}')

            model.eval()
            with torch.no_grad():
                predictions = model(X_test)
                test_loss = criterion(predictions, y_test)
                print(f'Test Loss: {test_loss.item():.9f}')

    torch.save(model.state_dict(), name_file + 'mlp_model.pth')

    return matrix_input, matrix_output

def Forecast1(file_weather, name_wind, longitude, latitude, a_input, b_input, a_output, b_output, today, hidden_layers=[32], name_features=['u100', 'v100', 'sp', 'tcw', 'a100'], l_hours=21, h_hours=51):

    df_weather = pd.read_csv(file_weather)
    df_weather = df_weather.loc[(df_weather['longitude'] == longitude) & (df_weather['latitude'] == latitude), ['time', 'u100', 'v100', 'sp', 'tcw']]
    df_weather['a100'] = (df_weather['v100'] ** 2) + (df_weather['u100'] ** 2)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather['time'] = df_weather['time'] + pd.Timedelta(hours=8)
    index_weather = pd.to_datetime(today)
    input_period = (df_weather['time'] >= index_weather + pd.Timedelta(hours=l_hours)) & (df_weather['time'] <= index_weather + pd.Timedelta(hours=h_hours))
    num_input_period = sum(input_period)
    df_weather = df_weather.loc[input_period, name_features]
    df_weather.to_csv('data/MediateData/df_weather_' + name_wind + '1.csv')
    input_scaled = df_weather.to_numpy() / (np.ones((num_input_period, 1)) @ np.array(a_input).reshape((1, -1))) + np.ones((num_input_period, 1)) @ np.array(b_input).reshape((1, -1))
    input_scaled = input_scaled.reshape((-1,))

    num_inputs = len(input_scaled)
    num_outputs = 96

    model = MLP(num_inputs, num_outputs, hidden_layers)
    model.load_state_dict(torch.load('data/MediateData/DataMatrix1_' + name_wind + '_' + 'mlp_model.pth'))

    model.eval()
    with torch.no_grad():
        output_scaled = model(torch.tensor(input_scaled, dtype=torch.float32)).numpy()
    output = np.maximum((output_scaled - b_output) * a_output, 0)
    return output

def Forecast4(file_weather, name_wind, longitude, latitude, a_input, b_input, a_output, b_output, today, hidden_layers=[32], name_features=['u100', 'v100', 'sp', 'tcw', 'a100'], l_hours=93, h_hours=122):

    df_weather = pd.read_csv(file_weather)
    df_weather = df_weather.loc[(df_weather['longitude'] == longitude) & (df_weather['latitude'] == latitude), ['time', 'u100', 'v100', 'sp', 'tcw']]
    df_weather['a100'] = (df_weather['v100'] ** 2) + (df_weather['u100'] ** 2)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather['time'] = df_weather['time'] + pd.Timedelta(hours=8)
    index_weather = pd.to_datetime(today)
    input_period = (df_weather['time'] >= index_weather + pd.Timedelta(hours=l_hours)) & (df_weather['time'] <= index_weather + pd.Timedelta(hours=h_hours))
    num_input_period = sum(input_period)
    df_weather = df_weather.loc[input_period, name_features]
    df_weather.to_csv('data/MediateData/df_weather_' + name_wind + '4.csv')
    input_scaled = df_weather.to_numpy() / (np.ones((num_input_period, 1)) @ np.array(a_input).reshape((1, -1))) + np.ones((num_input_period, 1)) @ np.array(b_input).reshape((1, -1))
    input_scaled = input_scaled.reshape((-1,))

    num_inputs = len(input_scaled)
    num_outputs = 96

    model = MLP(num_inputs, num_outputs, hidden_layers)
    model.load_state_dict(torch.load('data/MediateData/DataMatrix4_' + name_wind + '_' + 'mlp_model.pth'))

    model.eval()
    with torch.no_grad():
        output_scaled = model(torch.tensor(input_scaled, dtype=torch.float32)).numpy()
    output = np.maximum((output_scaled - b_output) * a_output, 0)
    return output

def output_file(name_wind, output1, output4, today_str):
    today = pd.to_datetime(today_str)
    tomorrow = today + pd.Timedelta(days=1)
    file_name = 'data/Output/' + name_wind + '_DQ_' + tomorrow.strftime('%Y%m%d') + '.csv'

    time = pd.date_range(tomorrow, tomorrow + pd.Timedelta(minutes=14385), freq='15min')
    PRED_DQ = np.concatenate((output1, output1, output1, output4, output4, output4, output4, output4, output4, output4))
    df = pd.DataFrame({})
    df['time'] = time
    # df['time'] = df['time'].dt.strftime('%Y/%m/%d %H:%M')
    df['PRED_DQ'] = PRED_DQ
    df.to_csv(file_name, float_format='%.3f', index=False)

def main():
    today_str = '20240729'
    file_weather = 'data/Input4Forecast/2024072900.csv'

    print('*************************Wind Farm D*************************')
    longitude = 111.5
    latitude = 21.4
    a_input=[51, 51, 4700, 91, 940]
    b_input=[0.57, 0.57, -20.94, -0.09, 0]
    a_output=410
    b_output=0.01
    cap = 399.9 / a_output + b_output
    hidden_layers=[128, 128, 128]
    TrainMLP('D', '1', cap, hidden_layers=hidden_layers, num_epochs=500, weight_decay=1e-3)
    output1 = Forecast1(file_weather, 'D', longitude, latitude, a_input, b_input, a_output, b_output, pd.to_datetime(today_str), hidden_layers=hidden_layers)
    TrainMLP('D', '4', cap, hidden_layers=hidden_layers, num_epochs=130, weight_decay=1e-4)
    output4 = Forecast4(file_weather, 'D', longitude, latitude, a_input, b_input, a_output, b_output, pd.to_datetime(today_str), hidden_layers=hidden_layers)
    output_file('D', output1, output4, today_str)

    # # print('*************************Wind Farm E*************************')
    longitude = 111.6
    latitude = 21.3
    a_input=[51, 51, 4700, 91, 940]
    b_input=[0.57, 0.57, -20.94, -0.09, 0]
    a_output=900
    b_output=0.01
    cap = 906.35 / a_output + b_output
    hidden_layers=[128, 128, 128]
    TrainMLP('E', '1', cap, hidden_layers=hidden_layers, num_epochs=500, weight_decay=1e-3)
    output1 = Forecast1(file_weather, 'E', longitude, latitude, a_input, b_input, a_output, b_output, pd.to_datetime(today_str), hidden_layers=hidden_layers)
    TrainMLP('E', '4', cap, hidden_layers=hidden_layers, num_epochs=130, weight_decay=1e-4)
    output4 = Forecast4(file_weather, 'E', longitude, latitude, a_input, b_input, a_output, b_output, pd.to_datetime(today_str), hidden_layers=hidden_layers)
    output_file('E', output1, output4, today_str)

    # # print('*************************Wind Farm F*************************')
    longitude = 111.5
    latitude = 21.3
    a_input=[51, 51, 4700, 91, 940]
    b_input=[0.57, 0.57, -20.94, -0.09, 0]
    a_output=390
    b_output=0.02
    cap = 399.25 / a_output + b_output
    hidden_layers=[128, 128, 128]
    TrainMLP('F', '1', cap, hidden_layers=hidden_layers, num_epochs=500, weight_decay=1e-3)
    output1 = Forecast1(file_weather, 'F', longitude, latitude, a_input, b_input, a_output, b_output, pd.to_datetime(today_str), hidden_layers=hidden_layers)
    TrainMLP('F', '4', cap, hidden_layers=hidden_layers, num_epochs=130, weight_decay=1e-4)
    output4 = Forecast4(file_weather, 'F', longitude, latitude, a_input, b_input, a_output, b_output, pd.to_datetime(today_str), hidden_layers=hidden_layers)
    output_file('F', output1, output4, today_str)

if __name__ == "__main__":
    main()
