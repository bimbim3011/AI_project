# main.py
from utils import fetch_historical_data, fetch_current_data, preprocess_data
from backtesting import backtest_strategy, example_strategy
from stock_prediction import predict_stock_price

def main():
    ticker = input("Enter stock ticker (e.g., VNM): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    
    # Lấy và tiền xử lý dữ liệu lịch sử
    print("Fetching historical data...")
    try:
        historical_data = fetch_historical_data(ticker, start_date, end_date)
        historical_data = preprocess_data(historical_data)
        print(historical_data.head())
    except KeyError as e:
        print(f"Error fetching historical data: {e}")
    
    # Lấy dữ liệu hiện tại
    print("Fetching current data...")
    try:
        current_data = fetch_current_data(ticker)
        print(current_data.head())
    except KeyError as e:
        print(f"Error fetching current data: {e}")
    
    # Kiểm tra chiến lược
    print("Backtesting strategy...")
    try:
        backtest_results = backtest_strategy(ticker, start_date, end_date, example_strategy)
        print(backtest_results[['Cumulative Market', 'Cumulative Strategy']].tail())
    except Exception as e:
        print(f"Error during backtesting: {e}")
    
    # Dự đoán giá cổ phiếu
    print("Predicting stock price...")
    try:
        predictions = predict_stock_price(ticker, start_date, end_date, model_type='LSTM')
        print(predictions)
    except Exception as e:
        print(f"Error predicting stock price: {e}")

if __name__ == "__main__":
    main()
