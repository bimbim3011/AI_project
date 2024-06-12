# main.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import create_lstm_model, prepare_lstm_data, train_random_forest, split_data
from fetch_data import fetch_and_select_data
from preprocessing import preprocess_data
from feature_engineering import feature_engineering
import uvicorn
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import pandas as pd

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gốc
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(symbols: str, model_type: str = "lstm"):
    try:
        symbols_list = symbols.split(",")
        symbols_list = [symbol.strip().upper() for symbol in symbols_list if symbol.strip()]
        if not symbols_list:
            raise ValueError("Không có mã hợp lệ được cung cấp")
        
        logging.info(f"Lấy dữ liệu cho các mã: {symbols_list}")
        all_data = []
        for symbol in symbols_list:
            df = fetch_and_select_data(symbol)
            if df.empty:
                logging.warning(f"Không có dữ liệu trả về cho {symbol}")
                continue
            all_data.append(df)

        if not all_data:
            raise ValueError("Không có dữ liệu được lấy cho bất kỳ mã nào")

        df = pd.concat(all_data, ignore_index=True)
        
        logging.info(f"Các cột dữ liệu lấy về: {df.columns}")
        logging.info(f"Dữ liệu lấy về head: {df.head()}")
        
        df = preprocess_data(df)
        logging.info(f"Dữ liệu sau khi tiền xử lý các cột: {df.columns}")
        logging.info(f"Dữ liệu sau khi tiền xử lý head: {df.head()}")
        
        df = feature_engineering(df)
        logging.info(f"Dữ liệu sau kỹ thuật đặc trưng các cột: {df.columns}")
        logging.info(f"Dữ liệu sau kỹ thuật đặc trưng head: {df.head()}")

        predictions_dict = {}
        dev_errors = {}

        for symbol in symbols_list:
            df_symbol = df[df['symbol'] == symbol]
            if df_symbol.empty:
                logging.warning(f"Không có dữ liệu để huấn luyện cho {symbol}")
                continue

            predictions = None
            try:
                if model_type == "lstm":
                    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(df_symbol)
                    if X_train.empty or y_train.empty:
                        logging.warning(f"Data split resulted in empty datasets for {symbol}")
                        continue

                    lstm_model = create_lstm_model((X_train.shape[1],))
                    X_train_lstm, y_train_lstm = prepare_lstm_data(X_train, y_train)
                    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

                    for epoch in range(10):
                        lstm_model.train()
                        optimizer.zero_grad()
                        output = lstm_model(X_train_lstm)
                        loss = F.mse_loss(output, y_train_lstm)
                        loss.backward()
                        optimizer.step()

                    lstm_model.eval()
                    with torch.no_grad():
                        X_dev_lstm, y_dev_lstm = prepare_lstm_data(X_dev, y_dev)
                        dev_predictions = lstm_model(X_dev_lstm).numpy()
                        dev_error = mean_squared_error(y_dev_lstm, dev_predictions)
                        dev_errors[symbol] = dev_error

                        X_test_lstm, _ = prepare_lstm_data(X_test, y_test)
                        predictions = lstm_model(X_test_lstm).numpy()
                else:
                    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(df_symbol)
                    if X_train.empty or y_train.empty:
                        logging.warning(f"Data split resulted in empty datasets for {symbol}")
                        continue

                    model = train_random_forest(X_train, y_train)
                    dev_predictions = model.predict(X_dev)
                    dev_error = mean_squared_error(y_dev, dev_predictions)
                    dev_errors[symbol] = dev_error

                    predictions = model.predict(X_test)

                if predictions is not None:
                    predictions_dict[symbol] = predictions.tolist()
            except Exception as e:
                logging.error(f"Error during model training or prediction for {symbol}: {e}")
                continue

            if predictions is not None:
                predictions_dict[symbol] = predictions.tolist()

        if not predictions_dict:
            raise ValueError("Không có dự đoán nào được tạo cho bất kỳ mã nào.")

        return {"predictions": predictions_dict}
    except Exception as e:
        logging.error(f"Lỗi trong dự đoán: {e}")
        return {"Error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app)
