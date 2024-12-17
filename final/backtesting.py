import pandas as pd
from utils import fetch_historical_data, preprocess_data

def backtest_strategy(ticker, start_date, end_date, strategy_func):
    df = fetch_historical_data(ticker, start_date, end_date)
    df = preprocess_data(df)
    signals = strategy_func(df)
    results = calculate_returns(df, signals)
    return results

# Cumulative Market (Lũy kế Thị trường), Cumulative Strategy (Lũy kế Chiến lược)
def calculate_returns(df, signals):
    df['Strategy'] = df['return'] * signals.shift(1)
    df['Cumulative Market'] = (1 + df['return']).cumprod()
    df['Cumulative Strategy'] = (1 + df['Strategy']).cumprod()
    return df

def example_strategy(df):
    signals = pd.Series(index=df.index)
    signals[df['close'] > df['close'].rolling(window=20).mean()] = 1
    signals[df['close'] <= df['close'].rolling(window=20).mean()] = 0
    return signals