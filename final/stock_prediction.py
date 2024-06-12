import pandas as pd
from vnstock import stock_historical_data
from models import LSTMModel, train_random_forest, prepare_lstm_data, train_lstm
from utils import preprocess_data, split_data
import torch
import torch.nn as nn
import logging

def fetch_and_preprocess_data(ticker, start_date, end_date):
    df = stock_historical_data(symbol=ticker, start_date=start_date, end_date=end_date)
    print(df.columns)  # In ra các cột để kiểm tra
    if 'time' in df.columns:
        df['Date'] = pd.to_datetime(df['time'], format='mixed')
    else:
        raise KeyError("The dataframe does not contain a 'time' column")
    df.set_index('Date', inplace=True)

    # Chuyển đổi tên các cột thành chữ thường
    df.columns = [col.lower() for col in df.columns]

    # Tiền xử lý dữ liệu
    df = preprocess_data(df)

    return df

def predict_stock_price(ticker, start_date, end_date, model_type='LSTM'):
    df = fetch_and_preprocess_data(ticker, start_date, end_date)
    train_data, dev_data, test_data = split_data(df)
    
    if model_type == 'LSTM':
        model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
        X_train, y_train = prepare_lstm_data(train_data)
        X_dev, y_dev = prepare_lstm_data(dev_data)
        X_test, y_test = prepare_lstm_data(test_data)
        
        train_lstm(model, X_train, y_train, epochs=10, batch_size=1, lr=0.001)
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).squeeze().numpy()

        # Đánh giá hiệu suất mô hình
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test.numpy(), predictions)
    elif model_type == 'RF':
        model = train_random_forest(train_data)
        X_test = test_data.drop(columns=['return', 'close'])  # Loại bỏ cột 'return' và 'close' từ test_data
        y_test = test_data['return']
        predictions = model.predict(X_test)

        # Đánh giá hiệu suất mô hình
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, predictions)
    else:
        raise ValueError("Model type must be 'LSTM' hoặc 'RF'")

    print(f'Mean Squared Error: {mse}')
    
    return predictions

# Mean Squared Error (MSE) trên dữ liệu kiểm tra
# Cuối cùng, sau khi mô hình đã được huấn luyện, bạn sử dụng mô hình này để dự đoán giá cổ phiếu trên dữ liệu kiểm tra (test data).
# Mean Squared Error (MSE) được tính trên tập dữ liệu kiểm tra để đánh giá hiệu suất của mô hình. Trong ví dụ của bạn, MSE trên dữ liệu kiểm tra là 0.0002584976318757981. Giá trị này càng nhỏ thì mô hình càng chính xác.

# Tóm lại, các giá trị epoch và loss cung cấp thông tin về quá trình huấn luyện và hiệu suất của mô hình. 
# Việc giảm dần loss qua các epoch là dấu hiệu tích cực cho thấy mô hình đang học tốt. 
# Giá trị MSE cuối cùng trên dữ liệu kiểm tra là một chỉ số quan trọng để đánh giá độ chính xác của mô hình.