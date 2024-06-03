import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import create_lstm_model, prepare_lstm_data, train_random_forest, split_data
from fetch_data import fetch_stock_data
from preprocessing import preprocess_data
from feature_engineering import feature_engineering
import uvicorn
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(symbols: str, start_date: str, end_date: str, model_type: str = "lstm"):
    try:
        symbols_list = symbols.split(",")  # Tách chuỗi thành danh sách mã cổ phiếu
        df = fetch_stock_data(symbols_list, start_date, end_date)
        df_preprocessed = preprocess_data(df)
        df_features = feature_engineering(df_preprocessed)

        predictions_dict = {}
        dev_errors = {}

        for symbol in symbols_list:
            df_symbol = df_features[df_features['symbol'] == symbol]
            if model_type == "lstm":
                X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(df_symbol)
                lstm_model = create_lstm_model((X_train.shape,))
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
                    dev_error = mean_squared_error(y_dev_lstm.numpy(), dev_predictions)
                    dev_errors[symbol] = dev_error

                    X_test_lstm, _ = prepare_lstm_data(X_test, y_test)
                    predictions = lstm_model(X_test_lstm).numpy()
            else:
                X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(df_symbol)
                model = train_random_forest(X_train, y_train)
                dev_predictions = model.predict(X_dev)
                dev_error = mean_squared_error(y_dev, dev_predictions)
                dev_errors[symbol] = dev_error

                predictions = model.predict(X_test)

            predictions_dict[symbol] = predictions.tolist()

        return {"predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return {"Error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app)
