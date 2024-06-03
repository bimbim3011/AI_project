from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

def create_lstm_model(input_shape):
    model = nn.Sequential(
        nn.LSTM(input_shape[1], 50, batch_first=True),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    return model

def prepare_lstm_data(X, y):
    X_lstm = np.array(X).astype(np.float32)
    y_lstm = np.array(y).astype(np.float32).reshape(-1, 1)
    X_lstm = torch.from_numpy(X_lstm).unsqueeze(-1)
    y_lstm = torch.from_numpy(y_lstm)
    return X_lstm, y_lstm

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def split_data(df):
    df.sort_values(by='date', inplace=True)
    X = df.drop(columns=['date', 'symbol', 'Close'])  # Giả định 'Close' là cột mục tiêu
    y = df['Close']

    train_size = int(len(df) * 0.7)
    dev_size = int(len(df) * 0.1)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_dev = X[train_size:train_size + dev_size]
    y_dev = y[train_size:train_size + dev_size]
    X_test = X[train_size + dev_size:]
    y_test = y[train_size + dev_size:]
    return X_train, y_train, X_dev, y_dev, X_test, y_test
