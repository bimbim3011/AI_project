from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import numpy as np
import logging
import pandas as pd
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
    model = LSTMModel(input_shape[1])
    return model

def prepare_lstm_data(X, y):
    X_lstm = np.array(X).astype(np.float32)
    y_lstm = np.array(y).astype(np.float32).reshape(-1, 1)
    X_lstm = torch.from_numpy(X_lstm).unsqueeze(1)
    y_lstm = torch.from_numpy(y_lstm)
    return X_lstm, y_lstm

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def split_data(df):
    try:
        # Kiểm tra xem cột 'listing_last_trading_date' có chứa giá trị hợp lệ không
        if df['listing_last_trading_date'].isnull().all():
            raise ValueError("Cột 'listing_last_trading_date' không có giá trị hợp lệ")

        df.sort_values(by='listing_last_trading_date', inplace=True)  # Giả sử 'listing_last_trading_date' là cột ngày
        X = df.drop(columns=['listing_last_trading_date', 'symbol', 'match_match_price'])  # Giả sử 'match_match_price' là cột mục tiêu
        y = df['match_match_price']

        if X.empty or y.empty:
            raise ValueError("Dataset is empty after removing necessary columns")

        train_size = int(len(df) * 0.7)
        dev_size = int(len(df) * 0.1)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_dev = X[train_size:train_size + dev_size]
        y_dev = y[train_size:train_size + dev_size]
        X_test = X[train_size + dev_size:]
        y_test = y[train_size + dev_size:]
        return X_train, y_train, X_dev, y_dev, X_test, y_test
    except IndexError as e:
        logging.error(f"IndexError in split_data: {e}")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
    except Exception as e:
        logging.error(f"Unexpected error in split_data: {e}")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()

