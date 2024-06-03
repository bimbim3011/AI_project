
def feature_engineering(df):
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    df['close_lag3'] = df['close'].shift(3)
    df['close_rolling_mean'] = df['close'].rolling(window=5).mean()
    df['close_rolling_std'] = df['close'].rolling(window=5).std()
    
    df.dropna(inplace=True)
    return df
