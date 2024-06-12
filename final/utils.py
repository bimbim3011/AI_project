import pandas as pd
from vnstock import stock_historical_data, stock_intraday_data

def fetch_historical_data(ticker, start_date, end_date):
    df = stock_historical_data(symbol=ticker, start_date=start_date, end_date=end_date)
    print(df.columns)  # In ra các cột để kiểm tra
    if 'time' in df.columns:
        df['Date'] = pd.to_datetime(df['time'], format='mixed')
    else:
        raise KeyError("The dataframe does not contain a 'time' column")
    df.set_index('Date', inplace=True)
    return df

def fetch_current_data(ticker):
    df = stock_intraday_data(symbol=ticker, page=0, page_size=5000)
    print(df.columns)  # In ra các cột để kiểm tra
    if 'time' in df.columns:
        df['Date'] = pd.to_datetime(df['time'], format='mixed')
    else:
        raise KeyError("The dataframe does not contain a 'time' column")
    df.set_index('Date', inplace=True)
    return df

def preprocess_data(df):
    # Chọn các cột liên quan và loại bỏ NaN
    df = df[['close', 'open', 'high', 'low', 'volume']].dropna()
    
    # Tính toán tỷ lệ phần trăm thay đổi
    df['return'] = df['close'].pct_change()
    
    # Chuẩn hóa các đặc trưng
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[['close', 'open', 'high', 'low', 'volume']] = scaler.fit_transform(df[['close', 'open', 'high', 'low', 'volume']])
    
    df = df.dropna()
    return df

def split_data(df):
    train_size = int(len(df) * 0.7)
    dev_size = int(len(df) * 0.15)
    
    train_data = df[:train_size]
    dev_data = df[train_size:train_size + dev_size]
    test_data = df[train_size + dev_size:]
    
    return train_data, dev_data, test_data