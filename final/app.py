import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils import fetch_historical_data, preprocess_data, fetch_current_data
from backtesting import backtest_strategy, example_strategy
from stock_prediction import predict_stock_price


# Hàm để chuyển đổi ngày thành chuỗi
def date_to_str(date):
    return date.strftime('%Y-%m-%d')

# Hàm để kiểm tra tính hợp lệ của mã chứng khoán
def is_valid_ticker(ticker):
    try:
        df = fetch_historical_data(ticker, '2020-01-01', '2023-01-01')
        return not df.empty
    except Exception:
        return False
    
# Hàm để hiển thị dữ liệu và dự đoán trên Streamlit
def main():
    st.title("Stock Analysis and Prediction")

    ticker = st.text_input("Enter stock ticker (e.g., VNM):", value="VNM")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    model_type = st.selectbox("Select model type for prediction", options=["LSTM", "RF"])

    start_date_str = date_to_str(start_date)
    end_date_str = date_to_str(end_date)

    if st.button("Fetch and Preprocess Data"):
        if is_valid_ticker(ticker):
            st.write("Fetching historical data...")
            try:
                historical_data = fetch_historical_data(ticker, start_date_str, end_date_str)
                historical_data = preprocess_data(historical_data)
                st.write(historical_data.head())

                # Plot interactive chart with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['close'], mode='lines', name='Close Price'))
                fig.update_layout(title=f"Historical Data for {ticker}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)
            except KeyError as e:
                st.error(f"Error fetching historical data: {e}")

            # st.write("Fetching current data...")
            # try:
            #     current_data = fetch_current_data(ticker)
            #     st.write(current_data.head())
            # except KeyError as e:
            #     st.error(f"Error fetching current data: {e}")
        else:
            st.error(f"Invalid stock ticker: {ticker}")

    if st.button("Backtest Strategy"):
        if is_valid_ticker(ticker):
            st.write("Backtesting strategy...")
            try:
                # Cumulative Market (Lũy kế Thị trường), Cumulative Strategy (Lũy kế Chiến lược)
                # so sánh hiệu suất của một chiến lược giao dịch cụ thể với hiệu suất của thị trường chung
                backtest_results = backtest_strategy(ticker, start_date_str, end_date_str, example_strategy)
                st.write(backtest_results[['Cumulative Market', 'Cumulative Strategy']].tail())
                
                # fig, ax = plt.subplots()
                # backtest_results[['Cumulative Market', 'Cumulative Strategy']].plot(ax=ax)
                # st.pyplot(fig)
                
                # Plot interactive backtesting results
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results['Cumulative Market'], mode='lines', name='Cumulative Market'))
                fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results['Cumulative Strategy'], mode='lines', name='Cumulative Strategy'))
                fig.update_layout(title="Backtesting Results", xaxis_title="Date", yaxis_title="Cumulative Returns")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during backtesting: {e}")
        else:
            st.error(f"Invalid stock ticker: {ticker}")

    if st.button("Predict Stock Price"):
        if is_valid_ticker(ticker):
            st.write("Predicting stock price...")
            try:
                predictions = predict_stock_price(ticker, start_date_str, end_date_str, model_type=model_type)
                st.write(predictions)

                # Lấy và tiền xử lý dữ liệu lại để vẽ biểu đồ
                historical_data = fetch_historical_data(ticker, start_date_str, end_date_str)
                historical_data = preprocess_data(historical_data)

                # Loại bỏ các nhãn chỉ mục trùng lặp
                historical_data = historical_data.loc[~historical_data.index.duplicated(keep='first')]

                # Căn chỉnh độ dài của các dự đoán với chỉ mục
                pred_index = historical_data.index[-len(predictions):]
                historical_data.loc[pred_index, 'Predicted Close'] = predictions

                # fig, ax = plt.subplots()
                # historical_data['close'].plot(ax=ax, label='Actual Close')
                # historical_data['Predicted Close'].plot(ax=ax, label='Predicted Close')
                # plt.legend()
                # st.pyplot(fig)

                # Plot interactive chart with predictions
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['close'], mode='lines', name='Actual Close'))
                fig.add_trace(go.Scatter(x=pred_index, y=predictions, mode='lines', name='Predicted Close'))
                fig.update_layout(title=f"Predicted vs Actual Close Price for {ticker}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error predicting stock price: {e}")
        else:
            st.error(f"Invalid stock ticker: {ticker}")

if __name__ == "__main__":
    main()
