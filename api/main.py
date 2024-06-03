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
def predict(symbol: str, start_date: str, end_date: str, model_type: str = "lstm"):
    try:
        df = fetch_stock_data(symbol, start_date, end_date)
        df_preprocessed = preprocess_data(df)
        df_features = feature_engineering(df_preprocessed)

        if model_type == "lstm":
            X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(df_features)
            lstm_model = create_lstm_model((X_train.shape[0], X_train.shape[1]))
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
                X_test_lstm, _ = prepare_lstm_data(X_test, y_test)
                predictions = lstm_model(X_test_lstm).numpy()
        else:
            X_train, y_train, X_test, y_test = split_data(df_features)[:4]
            model = train_random_forest(X_train, y_train)
            predictions = model.predict(X_test)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app)
