import pandas as pd

def preprocess_data(df):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort data by date
    df.sort_values('date', inplace=True)
    
    df.dropna(inplace=True)  # Loại bỏ các hàng có giá trị thiếu
    df.ffill(inplace=True)  # Thực hiện forward fill để điền giá trị thiếu
    
    # Select relevant columns and extract additional features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Normalize the 'close' price
    df['close'] = (df['close'] - df['close'].mean()) / df['close'].std()
    
    # Select final columns
    df = df[['date', 'close', 'year', 'month', 'day', 'day_of_week']]
    
    df = df.dropna()  # Ví dụ: loại bỏ các hàng có giá trị thiếu

    return df
