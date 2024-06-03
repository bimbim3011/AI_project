import requests
import pandas as pd

def fetch_stock_data(symbol, start_date, end_date):
    url = f"https://finfo-api.vndirect.com.vn/v4/stock_prices"
    params = {
        "symbol": symbol,
        "startDate": start_date,
        "endDate": end_date
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'data' in data:
        df = pd.DataFrame(data['data'])
        return df
    else:
        return None
