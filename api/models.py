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
    model = LSTMModel(input_size=input_shape[1])
    return model

def prepare_lstm_data(X, y):
    X_lstm = np.array(X).astype(np.float32)
    y_lstm = np.array(y).astype(np.float32).reshape(-1, 1)
    X_lstm = torch.from_numpy(X_lstm).unsqueeze(-1)
    y_lstm = torch.from_numpy(y_lstm)
    return X_lstm, y_lstm

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

def split_data(df):
    X = df.drop(['date', 'close'], axis=1)
    y = df['close']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, y_train, X_dev, y_dev, X_test, y_test
