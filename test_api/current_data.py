import pandas as pd
from vnstock import stock_intraday_data

def fetch_current_data(ticker):
    df = stock_intraday_data(symbol=ticker, page_num=0, page_size=5000)
    return df