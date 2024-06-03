import yfinance as yf
import pandas as pd

def fetch_stock_data(symbols, start_date, end_date):
    df_list = []
    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date)
        df.reset_index(inplace=True)  # Đảm bảo rằng cột 'Date' là một cột thông thường, không phải index
        df['symbol'] = symbol  # Thêm cột để xác định mã cổ phiếu
        df_list.append(df)
    combined_df = pd.concat(df_list)
    combined_df.rename(columns={'Date': 'date'}, inplace=True)  # Đổi tên cột Date thành date nếu cần
    return combined_df
